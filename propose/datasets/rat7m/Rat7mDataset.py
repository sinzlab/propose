import pickle
import imageio
import torch

from neuralpredictors.data.datasets import TransformDataset
from propose.poses.rat7m import Rat7mPose

CHUNK_SIZE = 3500


class Rat7mDataset(TransformDataset):
    def __init__(self, *data_keys: list[str], dirname: str, data_key: str, transforms: list):
        self.dirname = dirname
        self.data_key = data_key

        super().__init__(*data_keys, transforms=transforms)

        self.poses_path = f'{dirname}/poses/{data_key}.npy'
        self.cameras_path = f'{dirname}/cameras/{data_key}.pickle'
        self.image_dir = f'{dirname}/images/{data_key}'

        self.poses = Rat7mPose.load(self.poses_path)

        with open(self.cameras_path) as f:
            self.cameras = pickle.load(f)
            self.camera_keys = list(self.cameras.keys())

    def __len__(self):
        return len(self.poses) * len(self.cameras)

    def __getitem__(self, item):
        pose_idx = item % len(self.poses)
        pose = self.poses[pose_idx]

        camera_idx = item // len(self.poses)
        camera_key = self.camera_keys[camera_idx]
        camera = self.cameras[camera_key]

        chunk = camera.frames[pose_idx] // CHUNK_SIZE * CHUNK_SIZE

        image_path = f'{self.image_dir}/{self.data_key}-{camera_key.lower()}-{chunk}.pickle'

        image = imageio.imread(image_path)

        return camera, pose, torch.tensor(image)
