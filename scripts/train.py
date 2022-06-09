from pathlib import Path
from propose.datasets.human36m.preprocess import pickle_poses, pickle_cameras

import argparse

from train.human36m import human36m

import os
import yaml

parser = argparse.ArgumentParser(description="Arguments for running the scripts")

parser.add_argument(
    "--human36m",
    default=False,
    action="store_true",
    help="Run the training script for the Human 3.6m dataset",
)

parser.add_argument(
    "--wandb",
    default=False,
    action="store_true",
    help="Whether to use wandb for logging",
)

parser.add_argument(
    "--config",
    default="/configs/human36m/human36m_config.yaml",
    type=str,
    help="Experiment config file",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.wandb:
        if not os.environ["WANDB_API_KEY"]:
            raise ValueError(
                "Wandb API key not set. Please set the WANDB_API_KEY environment variable."
            )
        if not os.environ["WANDB_USER"]:
            raise ValueError(
                "Wandb user not set. Please set the WANDB_USER environment variable."
            )

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.human36m:
        human36m(use_wandb=args.wandb, config=config)
    else:
        print(
            "Not running any scripts as no arguments were passed. Run with --help for more information."
        )
