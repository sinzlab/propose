import torch
import torch.nn as nn

from tqdm import tqdm

from torch_geometric.loader.dataloader import Collater

from propose.utils.mpjpe import mpjpe


def supervised_trainer(
        dataloader, flow, optimizer=None, lr_scheduler=None, epochs=100, device="cpu", use_wandb=False, use_mode=True,
):
    """
    Train a flow in a supervised way
    :param dataloader: dataloader for the supervised training
    :param flow: flow to be trained
    :param optimizer: optimizer to be used
    :param lr_scheduler: lr_scheduler to be used
    :param epochs: number of epochs
    :param device: device to be used
    :param use_wandb: whether to use wandb
    :param use_mode: whether to use the mode loss
    :return: None
    """
    if use_wandb:
        import wandb

    if optimizer is None:
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.001, weight_decay=1e-5)

    if lr_scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                                  verbose=True, threshold=5e-2)

    collater = Collater(follow_batch=None, exclude_keys=None)
    for epoch in range(epochs):
        pbar = tqdm(
            dataloader,
            desc=f"Epoch: {epoch + 1}/{epochs} | NLLoss: 0 | RecLoss: 0 | Batch",
        )
        epoch_loss = []
        for data, x_data, action in pbar:
            batch = collater([data, x_data])
            batch.to(device)
            data.to(device)
            x_data.to(device)

            optimizer.zero_grad()

            loss = -flow(batch)
            n_posterior = data["x"].x.shape[0]

            prior_loss = loss[n_posterior:].mean()
            posterior_loss = loss[:n_posterior].mean()

            mse_mode_pose = torch.Tensor([0])
            if use_mode:
                scaling = 0.0036  # the std with which the data was normalized

                gt_pose = data['x']['x'].reshape(1, -1, 2) / scaling
                mode_pose = flow.mode_sample(gt_pose)['x']['x'].reshape(1, -1, 2) / scaling

                mse_mode_pose = mpjpe(mode_pose, gt_pose)

            loss = 0.5 * prior_loss + 0.5 * posterior_loss + 0.1 * mse_mode_pose
            loss.backward()

            epoch_loss.append(loss.item())

            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)

            if use_wandb:
                wandb.log(
                    {"Prior Loss": prior_loss.item(), "Posterior Loss": posterior_loss.item(), "Loss": loss.item(),
                     "Mode Error": mse_mode_pose.item()})

            optimizer.step()
            pbar.set_description(
                f"Epoch: {epoch + 1}/{epochs} | Loss {loss.item():.4f} | Prior Loss {prior_loss.item():.4f} | "
                f"Posterior Loss {posterior_loss.item():.4f} | Mode Error {mse_mode_pose.item():.4f} | Batch "
            )

        mean_loss = torch.mean(torch.Tensor(epoch_loss))
        lr_scheduler.step(mean_loss)
