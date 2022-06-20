import pickle

import numpy as np

from pathlib import Path

from torch.utils.data import Dataset
from propose.poses.human36m import Human36mPose

from torch_geometric.data import HeteroData
from torch_geometric.loader.dataloader import Collater

import torch
import torch.distributions as D

from tqdm import tqdm

CHUNK_SIZE = 3500


class Human36mDataset(Dataset):
    def __init__(
        self,
        dirname: str,
        num_samples: int = None,
        occlusion_fractions: list[float] = None,
        test: bool = False,
        hardsubset: bool = False,
        fully_connected: bool = False,
        mpii: bool = False,
        return_matrix: bool = False,
        num_context_samples: int = None,
    ) -> None:
        """
        :param dirname: directory containing the data
        :param num_samples: number of samples to be used
        :param occlusion_fractions: occlusion fractions to be used
        :param test: whether to use the test set
        :param hardsubset: whether to use the hard subset
        :param fully_connected: whether to use the fully connected dataset
        :param mpii: whether to use the mpii dataset
        :param return_matrix: whether to return as an matrix
        """

        self.return_matrix = return_matrix

        self.desc = ""

        if num_context_samples is None:
            num_context_samples = 1
            sample_context = False
        else:
            sample_context = True

        if occlusion_fractions is None:
            occlusion_fractions = [0.2, 0.4, 0.6, 0.8]

        file = Path(dirname) / "poses.pkl"

        with open(file, "rb") as f:
            dataset = pickle.load(f)

        n_poses = len(dataset["poses3d"])
        # index that selects random num_samples from the dataset
        self.hardsubset = hardsubset
        self.test = test
        if num_samples is not None:
            self._idx = np.random.choice(n_poses, num_samples, replace=False)
        elif test:
            self.desc = "test"
            stride = 16
            self._idx = np.arange(n_poses)[::stride]
        elif hardsubset:
            self.desc = "hardsubset"
            hard_frames = np.array(dataset["occlusions"])
            self._idx = hard_frames.any(1)
            self.hard_frames = hard_frames[self._idx]

            self.hard_frames = self.hard_frames[:n_poses]
            self._idx = self._idx[:n_poses]
        else:
            self._idx = np.arange(n_poses)

        self.actions = np.array(dataset["actions"])[self._idx]
        self.cameras = np.array(dataset["cameras"])[self._idx]
        self.subjects = np.array(dataset["subjects"])[self._idx]
        self.occlusions = np.array(dataset["occlusions"])[self._idx]
        self.center3d = torch.Tensor(np.concatenate(dataset["center3d"])[self._idx])
        self.gaussfits = torch.Tensor(np.concatenate(dataset["gaussfits"])[self._idx])

        pose = Human36mPose(np.zeros((1, 17, 3)))

        poses3d = torch.Tensor(dataset["poses3d"][self._idx]) * 0.0036
        poses2d = torch.Tensor(dataset["poses2d"][self._idx]) * 0.0139
        poses2d[..., 0] *= -1

        n_joints = poses3d.shape[1]
        edges = torch.LongTensor(pose.edges).T

        if fully_connected:
            adj = np.triu(np.ones((n_joints, n_joints)), 0)
            edges = torch.LongTensor(np.where(adj == 1))

        context_node_idx = torch.arange(0, n_joints * num_context_samples)
        target_node_idx = torch.arange(0, n_joints).repeat_interleave(
            num_context_samples
        )

        context_edges = torch.stack([context_node_idx, target_node_idx], dim=1).long().T

        if mpii:
            self.occlusions = self.occlusions[:, 1:]

        edges, root_edges, context_edges = self.remove_root_edges(
            edges, context_edges, num_context_samples
        )
        n_joints -= 1

        self.data = []
        self.base_data = []
        self.n_augs = len(occlusion_fractions) + 1

        for i in tqdm(range(len(poses3d)), desc=f"Preparing {self.desc} dataset"):
            mask = np.ones(16).astype(bool)
            if self.hardsubset:
                mask = self.hard_frames[i] == False

            if mpii:
                mask = ~self.occlusions[i]
                mask = np.insert(mask, 9, False)

            if sample_context:
                mask = np.repeat(mask, num_context_samples)

            datas = []
            data = HeteroData()

            # base nodes
            data["x"].x = poses3d[i, 1:]  # + torch.randn(n_joints, 3) * 0.01
            data["x", "->", "x"].edge_index = edges
            data["x", "<-", "x"].edge_index = edges
            # context nodes
            data["c"].x = poses2d[i, 1:]  # + torch.randn(poses2d[i, 1:].shape) * 0.01

            if sample_context:
                gaussfit = self.gaussfits[i]
                data["c"].x = self._sample_context(gaussfit, num_context_samples)

            data["c", "->", "x"].edge_index = context_edges[:, mask]
            # root nodes
            data["r"].x = poses3d[i, :1]
            data["r", "->", "x"].edge_index = root_edges
            data["r", "<-", "x"].edge_index = root_edges

            datas.append(data)

            for p in occlusion_fractions:
                mask = ~self.occlusions[i]
                mask = np.insert(mask, 9, False)

                mask[: int(p * context_edges.shape[-1])] = 0

                # shuffle mask
                np.random.shuffle(mask)
                mask = mask.astype(bool)

                if mpii:
                    mask = ~self.occlusions[i]
                    mask = np.insert(mask, 9, False)
                    rand_idx = np.random.choice(
                        np.arange(0, len(mask)), int(len(mask) * p), replace=False
                    )
                    mask[rand_idx] = False
                    # mask[9] = False

                if sample_context:
                    mask = np.repeat(mask, num_context_samples)

                data = HeteroData()

                data["x"].x = poses3d[i, 1:]  # + torch.randn(n_joints, 3) * 0.01
                data["x", "->", "x"].edge_index = edges
                data["x", "<-", "x"].edge_index = edges

                data["c"].x = poses2d[i, 1:]  # + torch.randn(poses2d[i].shape) * 0.01
                if sample_context:
                    gaussfit = self.gaussfits[i]
                    data["c"].x = self._sample_context(gaussfit, num_context_samples)

                data["c", "->", "x"].edge_index = context_edges[:, mask]

                data["r"].x = poses3d[i, :1]
                data["r", "->", "x"].edge_index = root_edges
                data["r", "<-", "x"].edge_index = root_edges

                datas.append(data)

            data = datas

            base_data = HeteroData()
            # base nodes
            base_data["x"].x = poses3d[i, 1:]
            base_data["x", "->", "x"].edge_index = edges
            base_data["x", "<-", "x"].edge_index = edges

            base_data["r"].x = poses3d[i, :1]
            base_data["r", "->", "x"].edge_index = root_edges
            base_data["r", "<-", "x"].edge_index = root_edges

            self.data += data
            self.base_data.append(base_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.return_matrix:
            return (
                self.data[item]["x"]["x"],
                self.base_data[item // self.n_augs]["x"]["x"].reshape(-1, 16 * 3),
                {
                    "action": self.actions[item // self.n_augs],
                    "camera": self.cameras[item // self.n_augs],
                    "subject": self.subjects[item // self.n_augs],
                    "center3d": self.center3d[item // self.n_augs],
                },
            )

        return (
            self.data[item],
            self.base_data[item // self.n_augs],
            {
                "action": self.actions[item // self.n_augs],
                "camera": self.cameras[item // self.n_augs],
                "subject": self.subjects[item // self.n_augs],
                "occlusion": torch.Tensor(self.occlusions[item // self.n_augs]),
                "center3d": self.center3d[item // self.n_augs],
            },
        )  # returns: full data, base data

    def remove_root_edges(self, edges, context_edges, num_context_samples):
        full_edges = edges[:, torch.where(edges[0] != 0)[0]]
        context_edges = context_edges[:, torch.where(context_edges[1] != 0)[0]]
        root_edges = edges[:, torch.where(edges[0] == 0)[0]]

        full_edges -= 1
        context_edges[0] -= num_context_samples
        context_edges[1] -= 1
        root_edges[1] -= 1

        return full_edges, root_edges, context_edges

    def _sample_context(self, gaussfit, num_context_samples):
        mean = torch.stack([gaussfit[:, 1], gaussfit[:, 2]], dim=1)
        cov = torch.stack([gaussfit[:, 3], gaussfit[:, 5]], dim=1).unsqueeze(
            2
        ) ** 2 * torch.eye(2).repeat(16, 1, 1)

        c_dist = D.MultivariateNormal(mean, covariance_matrix=cov)
        samples = c_dist.sample((num_context_samples,))
        return samples.view(samples.shape[0] * samples.shape[1], samples.shape[2])


class NewestHumanDataset(Dataset):
    def __init__(
        self,
        dirname: str,
        num_samples: int = None,
        occlusion_fractions: list[float] = None,
        stride: int = None,
        hardsubset: bool = False,
    ):
        if occlusion_fractions is None:
            occlusion_fractions = [0.2, 0.4, 0.6, 0.8]

        file = Path(dirname) / "poses.pkl"

        with open(file, "rb") as f:
            dataset = pickle.load(f)

        n_poses = len(dataset["poses3d"])
        # index that selects random num_samples from the dataset

        self.hardsubset = hardsubset

        if num_samples is not None:
            self._idx = np.random.choice(n_poses, num_samples, replace=False)
        elif stride is not None:
            self._idx = np.arange(0, n_poses, stride)
        elif hardsubset:
            hard_frames = np.load(Path(dirname) / "hard_subset_indices.npy")
            self._idx = hard_frames.any(1)
            self.hard_frames = hard_frames[self._idx]
        else:
            self._idx = np.arange(n_poses)

        self.actions = np.array(dataset["actions"])[self._idx]

        pose = Human36mPose(np.zeros((1, 17, 3)))

        poses3d = torch.Tensor(dataset["poses3d"][self._idx]) * 0.0036
        poses2d = torch.Tensor(dataset["poses2d"][self._idx]) * 0.0139
        # poses2d = poses3d[..., [0, 2]]

        n_joints = poses3d.shape[1]
        edges = torch.LongTensor(pose.edges).T
        context_edges = torch.arange(0, n_joints).repeat(2).reshape(2, n_joints).long()

        collater = Collater(follow_batch=None, exclude_keys=None)

        self.data = []
        self.base_data = []
        for i in tqdm(range(len(poses3d))):
            mask = np.ones(17).astype(bool)
            if self.hardsubset:
                mask[1:] = self.hard_frames[i] == False

            datas = []
            data = HeteroData()
            # base nodes
            data["x"].x = poses3d[i]
            data["x", "->", "x"].edge_index = edges
            data["x", "<-", "x"].edge_index = edges
            # context nodes
            data["c"].x = poses2d[i]
            data["c", "->", "x"].edge_index = context_edges[:, mask]

            datas.append(data)

            for p in occlusion_fractions:
                mask = np.ones(17)
                mask[: int(p * 17)] = 0
                # shuffle mask
                np.random.shuffle(mask)
                mask = mask.astype(bool)

                data = HeteroData()

                data["x"].x = poses3d[i]
                data["x", "->", "x"].edge_index = edges
                data["x", "<-", "x"].edge_index = edges

                data["c"].x = poses2d[i]
                data["c", "->", "x"].edge_index = context_edges[:, mask]

                datas.append(data)

            data = collater(datas)

            base_data = HeteroData()
            # base nodes
            base_data["x"].x = poses3d[i]
            base_data["x", "->", "x"].edge_index = edges
            base_data["x", "<-", "x"].edge_index = edges

            self.data.append((data, base_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return (
            self.data[item][0],
            self.data[item][1],
            self.actions[item],
        )  # returns: full data, base data


class NewestHumanDatasetNoRoot(Dataset):
    def __init__(
        self,
        dirname: str,
        num_samples: int = None,
        occlusion_fractions: list[float] = None,
        test: bool = False,
        hardsubset: bool = False,
        fully_connected: bool = False,
        mpii: bool = False,
    ):
        self.desc = ""
        print("Loading dataset...")
        print("full pose 2D", "full pose 3D")

        if occlusion_fractions is None:
            occlusion_fractions = [0.2, 0.4, 0.6, 0.8]

        file = Path(dirname) / "poses.pkl"

        with open(file, "rb") as f:
            dataset = pickle.load(f)

        n_poses = len(dataset["poses3d"])
        # index that selects random num_samples from the dataset
        self.hardsubset = hardsubset
        self.test = test
        if num_samples is not None:
            self._idx = np.random.choice(n_poses, num_samples, replace=False)
        elif test:
            self.desc = "test"
            stride = 16
            self._idx = np.arange(n_poses)[::stride]
        elif hardsubset:
            self.desc = "hardsubset"
            # hard_frames = np.load(Path(dirname) / 'hard_subset_indices.npy')
            # print(hard_frames.shape, n_poses)
            hard_frames = np.array(dataset["occlusions"])
            self._idx = hard_frames.any(1)
            self.hard_frames = hard_frames[self._idx]

            self.hard_frames = self.hard_frames[:n_poses]
            self._idx = self._idx[:n_poses]
        else:
            self._idx = np.arange(n_poses)

        self.actions = np.array(dataset["actions"])[self._idx]
        self.cameras = np.array(dataset["cameras"])[self._idx]
        self.subjects = np.array(dataset["subjects"])[self._idx]
        self.occlusions = np.array(dataset["occlusions"])[self._idx]
        # self.image_paths = np.array(dataset["image_paths"])[self._idx]
        self.center3d = torch.Tensor(np.concatenate(dataset["center3d"])[self._idx])

        pose = Human36mPose(np.zeros((1, 17, 3)))

        poses3d = torch.Tensor(dataset["poses3d"][self._idx]) * 0.0036  # - 0.0659
        # poses3d = poses3d[..., [0, 2]]
        # print(dataset["poses2d"][self._idx].std(0).shape)
        poses2d = torch.Tensor(dataset["poses2d"][self._idx]) * 0.0139  # - 0.1250
        poses2d[..., 0] *= -1
        # poses2d = poses3d[..., [0, 2]]

        n_joints = poses3d.shape[1]
        edges = torch.LongTensor(pose.edges).T

        if fully_connected:
            adj = np.triu(np.ones((n_joints, n_joints)), 0)
            edges = torch.LongTensor(np.where(adj == 1))

        context_edges = torch.arange(0, n_joints).repeat(2).reshape(2, n_joints).long()
        if mpii:
            # self.occlusions[:, 9] = True
            self.occlusions = self.occlusions[:, 1:]
            # print(context_edges.shape, self.occlusions.shape)
            # context_edges = context_edges[:, ~self.occlusions]

        edges, root_edges, context_edges = self.remove_root_edges(edges, context_edges)
        n_joints -= 1

        collater = Collater(follow_batch=None, exclude_keys=None)

        self.data = []
        self.base_data = []
        self.n_augs = len(occlusion_fractions) + 1
        print(self.n_augs)
        for i in tqdm(range(len(poses3d)), desc=f"Preparing {self.desc} dataset"):
            mask = np.ones(16).astype(bool)
            if self.hardsubset:
                mask = self.hard_frames[i] == False

            if mpii:
                # mask[9] = False
                # mask = np.delete(mask, 9)
                # print(mask)
                # print(self.occlusions.shape)
                mask = ~self.occlusions[i]
                mask = np.insert(mask, 9, False)

            datas = []
            data = HeteroData()

            # base nodes
            data["x"].x = poses3d[i, 1:]  # + torch.randn(n_joints, 3) * 0.01
            data["x", "->", "x"].edge_index = edges
            data["x", "<-", "x"].edge_index = edges
            # context nodes
            data["c"].x = poses2d[i, 1:]  # + torch.randn(poses2d[i, 1:].shape) * 0.01
            data["c", "->", "x"].edge_index = context_edges[:, mask]
            # root nodes
            data["r"].x = poses3d[i, :1]
            data["r", "->", "x"].edge_index = root_edges
            data["r", "<-", "x"].edge_index = root_edges

            datas.append(data)

            for p in occlusion_fractions:
                mask = ~self.occlusions[i]
                mask = np.insert(mask, 9, False)

                mask[: int(p * context_edges.shape[-1])] = 0

                # shuffle mask
                np.random.shuffle(mask)
                mask = mask.astype(bool)

                if mpii:
                    mask = ~self.occlusions[i]
                    mask = np.insert(mask, 9, False)
                    rand_idx = np.random.choice(
                        np.arange(0, len(mask)), int(len(mask) * p), replace=False
                    )
                    mask[rand_idx] = False
                    # mask[9] = False

                data = HeteroData()

                data["x"].x = poses3d[i, 1:]  # + torch.randn(n_joints, 3) * 0.01
                data["x", "->", "x"].edge_index = edges
                data["x", "<-", "x"].edge_index = edges

                data["c"].x = poses2d[i, 1:]  # + torch.randn(poses2d[i].shape) * 0.01
                data["c", "->", "x"].edge_index = context_edges[:, mask]

                data["r"].x = poses3d[i, :1]
                data["r", "->", "x"].edge_index = root_edges
                data["r", "<-", "x"].edge_index = root_edges

                datas.append(data)

            # data = collater(datas)
            data = datas

            base_data = HeteroData()
            # base nodes
            base_data["x"].x = poses3d[i, 1:]
            base_data["x", "->", "x"].edge_index = edges
            base_data["x", "<-", "x"].edge_index = edges

            base_data["r"].x = poses3d[i, :1]
            base_data["r", "->", "x"].edge_index = root_edges
            base_data["r", "<-", "x"].edge_index = root_edges

            self.data += data
            self.base_data.append(base_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # return self.data[item]['x']['x'], self.base_data[item // self.n_augs]['x']['x'].reshape(-1, 16 * 3), {
        #     'action': self.actions[item // self.n_augs],
        #     'camera': self.cameras[item // self.n_augs],
        #     'subject': self.subjects[item // self.n_augs],
        #  #   'center3d': self.center3d[item // self.n_augs]
        # }
        return (
            self.data[item],
            self.base_data[item // self.n_augs],
            {
                "action": self.actions[item // self.n_augs],
                "camera": self.cameras[item // self.n_augs],
                "subject": self.subjects[item // self.n_augs],
                "occlusion": torch.Tensor(self.occlusions[item // self.n_augs]),
                # 'image_path': self.image_paths[item // self.n_augs],
                "center3d": self.center3d[item // self.n_augs],
            },
        )  # returns: full data, base data

    def remove_root_edges(self, edges, context_edges):
        full_edges = edges[:, torch.where(edges[0] != 0)[0]]
        context_edges = context_edges[:, 1:]
        root_edges = edges[:, torch.where(edges[0] == 0)[0]]

        full_edges -= 1
        context_edges -= 1
        root_edges[1] -= 1

        return full_edges, root_edges, context_edges
