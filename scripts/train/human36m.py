from propose.datasets.human36m.Human36mDataset import Human36mDataset

from torch_geometric.loader import DataLoader

from propose.models.flows import CondGraphFlow
from propose.models.nn.embedding import embeddings
from propose.training import supervised_trainer
from propose.utils.reproducibility import set_random_seed

import torch

import os

import time


def human36m(use_wandb: bool = False, config: dict = None):
    """
    Train a CondGraphFlow on the Human36m dataset.
    :param use_wandb: Whether to use wandb for logging.
    :param config: A dictionary of configuration parameters.
    """
    set_random_seed(config["seed"])

    if "cuda_accelerated" not in config:
        config["cuda_accelerated"] = torch.cuda.is_available()

    if use_wandb:
        import wandb

        wandb.init(
            project="propose_human36m",
            entity=os.environ["WANDB_USER"],
            config=config,
            job_type="training",
            name=f"{config['experiment_name']}_{time.strftime('%d/%m/%Y::%H:%M:%S')}",
            tags=config["tags"] if "tags" in config else None,
            group=config["group"] if "group" in config else None,
        )

    dataset = Human36mDataset(**config["dataset"])

    dataloader = DataLoader(
        dataset, batch_size=config["train"]["batch_size"], shuffle=True
    )

    embedding_net = None
    if config["embedding"]:
        embedding_net = embeddings[config["embedding"]["name"]](
            **config["embedding"]["config"]
        )

    flow = CondGraphFlow(**config["model"], embedding_net=embedding_net)

    if "use_pretrained" in config:
        artifact = wandb.run.use_artifact(
            f'ppierzc/propose_human36m/{config["use_pretrained"]}', type="model"
        )
        artifact_dir = artifact.download()
        flow.load_state_dict(torch.load(artifact_dir + "/model.pt"))

    if config["cuda_accelerated"]:
        flow.to("cuda:0")

    optimizer = torch.optim.Adam(flow.parameters(), **config["train"]["optimizer"])

    lr_scheduler = None
    if config["train"]["lr_scheduler"]:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config["train"]["lr_scheduler"], verbose=True
        )

    supervised_trainer(
        dataloader,
        flow,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=config["train"]["epochs"],
        device=flow.device,
        use_wandb=use_wandb,
    )
