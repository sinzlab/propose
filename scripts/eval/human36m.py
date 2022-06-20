from propose.datasets.human36m.Human36mDataset import Human36mDataset
from torch_geometric.loader import DataLoader

from propose.utils.reproducibility import set_random_seed

from propose.models.flows import CondGraphFlow
from propose.models.nn.embedding import embeddings

import torch

import os

import time
from tqdm import tqdm
import numpy as np


def evaluate(flow, test_dataloader):
    mpjpes = []

    iter_dataloader = iter(test_dataloader)
    for _ in tqdm(range(len(test_dataloader))):
        batch, _, action = next(iter_dataloader)
        batch.cuda()
        samples = flow.sample(200, batch)

        true_pose = batch["x"].x.cpu().numpy().reshape(-1, 16, 1, 3)
        sample_poses = samples["x"].x.detach().cpu().numpy().reshape(-1, 16, 200, 3)

        true_pose = np.insert(true_pose, 0, 0, axis=1)
        sample_poses = np.insert(sample_poses, 0, 0, axis=1)

        m = (((true_pose / 0.0036 - sample_poses / 0.0036) ** 2).sum(-1) ** .5).mean(1)  # .min(-1) ** 0.5
        m = np.min(m, axis=-1)

        m = m.tolist()

        mpjpes += [m]

    return mpjpes


def human36m(use_wandb: bool = False, config: dict = None):
    """
    Train a CondGraphFlow on the Human36m dataset.
    :param use_wandb: Whether to use wandb for logging.
    :param config: A dictionary of configuration parameters.
    """
    set_random_seed(config["seed"])

    import wandb

    wandb.init(
        project="propose_human36m",
        entity=os.environ["WANDB_USER"],
        config=config,
        job_type="evaluation",
        name=f"{config['experiment_name']}_{time.strftime('%d/%m/%Y::%H:%M:%S')}",
        tags=config['tags'] if 'tags' in config else None,
        group=config['group'] if 'group' in config else None,
    )

    artifact = wandb.run.use_artifact(f'ppierzc/propose_human36m/{config["experiment_name"]}:best', type='model')
    artifact_dir = artifact.download()

    embedding_net = None
    if config["embedding"]:
        embedding_net = embeddings[config["embedding"]["name"]](
            **config["embedding"]["config"]
        )

    flow = CondGraphFlow(**config["model"], embedding_net=embedding_net)

    flow.load_state_dict(torch.load(artifact_dir + '/model.pt'))

    config['cuda_accelerated'] = False
    if torch.cuda.is_available():
        flow.to("cuda:0")
        config['cuda_accelerated'] = True

    flow.eval()

    config['dataset']['dirname'] = config['dataset']['dirname'] + "/test"

    # Test
    test_dataset = Human36mDataset(
        **config['dataset'],
        occlusion_fractions=[],
        test=True,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=False, num_workers=0)
    test_res = evaluate(flow, test_dataloader)

    wandb.log({
        'test/best_mpjpe': np.concatenate(test_res).mean()
    })

    # Hard
    hard_dataset = Human36mDataset(
        **config['dataset'],
        occlusion_fractions=[],
        hardsubset=True
    )

    hard_dataloader = DataLoader(hard_dataset, batch_size=1, shuffle=True, pin_memory=False, num_workers=0)
    hard_res = evaluate(flow, hard_dataloader)

    wandb.log({
        'hard/best_mpjpe': np.concatenate(hard_res).mean()
    })
