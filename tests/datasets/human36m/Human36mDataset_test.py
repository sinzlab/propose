from unittest.mock import patch

from neuralpredictors.data.transforms import ToTensor
from torch_geometric.loader import DataLoader

import propose.datasets.rat7m.transforms as tr
from propose.datasets.human36m.Human36mDataset import Human36mDataset

# @patch('propose.datasets.human36m.Human36mDataset.Human36mPose')
# def test_works_with_data_loader():
#
#     dataset = Human36mDataset(
#         dirname="../data/human36m/processed/"
#     )
#
#     # print(dataset[0])
#     #
#     dataloader = DataLoader(dataset, batch_size=10)
#
#     batch = next(iter(dataloader))
#     #
#     print(batch)
#     #
#     # assert batch.poses.shape[0] == 10
