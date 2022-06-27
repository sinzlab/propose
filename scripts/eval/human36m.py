from propose.datasets.human36m.Human36mDataset import Human36mDataset
from torch_geometric.loader import DataLoader

from propose.utils.reproducibility import set_random_seed
from propose.utils.mpjpe import mpjpe

from propose.models.flows import CondGraphFlow

import torch

import os

import time
from tqdm import tqdm
import numpy as np

import wandb


def evaluate(flow, test_dataloader, temperature=1.0):
    mpjpes = []

    iter_dataloader = iter(test_dataloader)
    for _ in tqdm(range(len(test_dataloader))):
        batch, _, action = next(iter_dataloader)
        batch.cuda()
        samples = flow.sample(200, batch, temperature=temperature)

        true_pose = batch["x"].x.cpu().numpy().reshape(-1, 16, 1, 3)
        sample_poses = samples["x"].x.detach().cpu().numpy().reshape(-1, 16, 200, 3)

        true_pose = np.insert(true_pose, 0, 0, axis=1)
        sample_poses = np.insert(sample_poses, 0, 0, axis=1)

        m = mpjpe(true_pose / 0.0036, sample_poses / 0.0036, dim=1)
        m = np.min(m, axis=-1)

        m = m.tolist()

        mpjpes += [m]

    return mpjpes


def mpjpe_experiment(flow, config, **kwargs):
    test_dataset = Human36mDataset(
        **config["dataset"],
        **kwargs,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, pin_memory=False, num_workers=0
    )
    test_res = evaluate(flow, test_dataloader)

    return np.concatenate(test_res).mean(), test_dataset, test_dataloader


def human36m(use_wandb: bool = False, config: dict = None):
    """
    Train a CondGraphFlow on the Human36m dataset.
    :param use_wandb: Whether to use wandb for logging.
    :param config: A dictionary of configuration parameters.
    """
    set_random_seed(config["seed"])

    config["dataset"]["dirname"] = config["dataset"]["dirname"] + "/test"

    wandb.init(
        project="propose_human36m",
        entity=os.environ["WANDB_USER"],
        config=config,
        job_type="evaluation",
        name=f"{config['experiment_name']}_{time.strftime('%d/%m/%Y::%H:%M:%S')}",
        tags=config["tags"] if "tags" in config else None,
        group=config["group"] if "group" in config else None,
    )

    flow = CondGraphFlow.from_pretrained(
        f'ppierzc/propose_human36m/{config["experiment_name"]}:latest'
    )

    config["cuda_accelerated"] = False
    if torch.cuda.is_available():
        flow.to("cuda:0")
        config["cuda_accelerated"] = True

    flow.eval()

    # Test
    test_res, test_dataset, test_dataloader = mpjpe_experiment(
        flow,
        config,
        occlusion_fractions=[],
        test=True,
    )

    wandb.log({"test/best_mpjpe": test_res})

    # Hard
    hard_res, hard_dataset, hard_dataloader = mpjpe_experiment(
        flow,
        config,
        occlusion_fractions=[],
        hardsubset=True,
    )

    wandb.log({"hard/best_mpjpe": hard_res})

    # Occlusion Only
    mpjpes = []
    for i in tqdm(range(len(hard_dataset))):
        batch = hard_dataset[i][0]
        batch.cuda()
        samples = flow.sample(200, batch.cuda())

        true_pose = (
            batch["x"]
            .x.cpu()
            .numpy()
            .reshape(-1, 16, 1, 3)[:, np.insert(hard_dataset.occlusions[i], 9, False)]
        )
        sample_poses = (
            samples["x"]
            .x.detach()
            .cpu()
            .numpy()
            .reshape(-1, 16, 200, 3)[:, np.insert(hard_dataset.occlusions[i], 9, False)]
        )

        m = mpjpe(true_pose / 0.0036, sample_poses / 0.0036, dim=1)
        m = np.min(m, axis=-1)

        m = m.tolist()

        mpjpes += [m]

    wandb.log({"occl/best_mpjpe": np.nanmean(mpjpes)})

    # Temperature Evaluation
    temperatures = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for temperature in temperatures:
        test_res = evaluate(flow, test_dataloader, temperature=temperature)

        wandb.log(
            {
                "temperature/best_mpjpe": np.concatenate(test_res).mean(),
                "temperature/temperature": temperature,
            }
        )
