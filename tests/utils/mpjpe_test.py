from propose.utils.mpjpe import mpjpe, pa_mpjpe

import torch
import numpy.testing as npt

from unittest import TestCase


class MPJPETests(TestCase):
    def test_mpjpe(self):
        self.assertAlmostEqual(
            mpjpe(torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6])).item(),
            5.196152210235596,
        )

    def test_pa_mpjpe(self):
        self.assertAlmostEqual(
            pa_mpjpe(
                torch.Tensor([[[1, 2, 3], [3, 4, 5]]]),
                torch.Tensor([[[1, 2, 3], [3, 4, 5]]]),
            ).item(),
            0.0,
        )

    def test_against_wehrbein(self):
        p_pred = torch.randn((200, 17, 3)) / 0.0036
        p_true = torch.randn((1, 17, 3)) / 0.0036

        r1 = err_3dpe_parallel(
            p_true.repeat(200, 1, 1), p_pred, return_sum=False, joints=17
        )

        p_pred = p_pred.permute(1, 0, 2)
        p_true = p_true.permute(1, 0, 2)

        r2 = pa_mpjpe(p_true, p_pred, dim=0)

        npt.assert_allclose(r1, r2)


### Code for testing the above functions ###
# Original code from:
# https://github.com/twehrbein/Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows/


def procrustes_torch_parallel(p_gt, p_pred):
    # p_gt and p_pred need to be of shape (-1, 3, #joints)
    # care: run on cpu! way faster than on gpu

    mu_gt = p_gt.mean(dim=2)
    mu_pred = p_pred.mean(dim=2)

    # print(mu_gt)
    # print(mu_pred)

    X0 = p_gt - mu_gt[:, :, None]
    Y0 = p_pred - mu_pred[:, :, None]

    ssX = (X0**2.0).sum(dim=(1, 2))
    ssY = (Y0**2.0).sum(dim=(1, 2))

    # centred Frobenius norm
    normX = torch.sqrt(ssX)
    normY = torch.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX[:, None, None]
    Y0 /= normY[:, None, None]

    # optimum rotation matrix of Y
    A = torch.bmm(X0, Y0.transpose(1, 2))

    try:
        U, s, V = torch.svd(A, some=True)
    except:
        print("ERROR IN SVD, could not converge")
        print("SVD INPUT IS:")
        print(A)
        print(A.shape)
        exit()

    T = torch.bmm(V, U.transpose(1, 2))

    # Make sure we have a rotation
    detT = torch.det(T)
    sign = torch.sign(detT)
    V[:, :, -1] *= sign[:, None]
    s[:, -1] *= sign
    T = torch.bmm(V, U.transpose(1, 2))

    traceTA = s.sum(dim=1)

    # optimum scaling of Y
    b = traceTA * normX / normY

    # standardised distance between X and b*Y*T + c
    d = 1 - traceTA**2

    # transformed coords
    scale = normX * traceTA
    Z = (
        scale[:, None, None] * torch.bmm(Y0.transpose(1, 2), T) + mu_gt[:, None, :]
    ).transpose(1, 2)

    # transformation matrix
    c = mu_gt - b[:, None] * (torch.bmm(mu_pred[:, None, :], T)).squeeze()

    # transformation values
    tform = {"rotation": T, "scale": b, "translation": c}
    return d, Z, tform


def err_3dpe_parallel(p_ref, p, return_sum=True, return_poses=False, joints=17):
    p_ref, p = p_ref.view((-1, 3, joints)), p.view((-1, 3, joints))
    d, Z, tform = procrustes_torch_parallel(p_ref.clone(), p)

    if return_sum:
        err = torch.sum(
            torch.mean(torch.sqrt(torch.sum((p_ref - Z) ** 2, dim=1)), dim=1)
        ).item()
    else:
        err = torch.mean(torch.sqrt(torch.sum((p_ref - Z) ** 2, dim=1)), dim=1)
    if not return_poses:
        return err
    else:
        return err, Z
