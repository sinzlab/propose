from propose.datasets.human36m.Human36mDataset import Human36mDataset

from torch_geometric.loader import DataLoader

from propose.models.flows import CondGraphFlow
from propose.training import supervised_trainer
from propose.utils.reproducibility import set_random_seed

import torch

import os


def human36m(use_wandb: bool = False, config: dict = None):
    """
    Train a CondGraphFlow on the Human36m dataset.
    :param use_wandb: Whether to use wandb for logging.
    :param config: A dictionary of configuration parameters.
    """
    set_random_seed(config["seed"])

    if use_wandb:
        import wandb

        wandb.init(project="propose_human36m", entity=os.environ["WANDB_USER"])

        wandb.config = config

    dataset = Human36mDataset(**config["dataset"])

    dataloader = DataLoader(
        dataset, batch_size=config["train"]["batch_size"], shuffle=True
    )

    lr = config["train"]["lr"]
    weight_decay = config["train"]["weight_decay"]

    flow = CondGraphFlow(**config["model"])
    if torch.cuda.is_available():
        flow.to("cuda:0")

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)

    supervised_trainer(
        dataloader,
        flow,
        optimizer,
        epochs=config["train"]["epochs"],
        device=flow.device,
        use_wandb=use_wandb,
    )

    torch.save(flow.state_dict(), "/results/model.pt")
