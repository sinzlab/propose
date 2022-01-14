from pathlib import Path
from propose.datasets.human36m.loaders import pickle_poses

import argparse

parser = argparse.ArgumentParser(description="Arguments for running the scripts")

parser.add_argument(
    "--human36m",
    default=False,
    action="store_true",
    help="Run the preprocess script for the Human 3.6m dataset",
)
parser.add_argument(
    "--rat7m",
    default=False,
    action="store_true",
    help="Run the preprocess script for the Rat 7m dataset",
)


def human36m():
    input_dir = Path("/data/human36m/raw/")
    output_dir = Path("/data/human36m/processed/")

    print("Preprocessing Human3.6M data...")
    pickle_poses(input_dir, output_dir)
    print("Done.")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.human36m:
        human36m()

    if args.rat7m:
        raise NotImplementedError(
            "Rat7m data preprocessing is not yet implemented. Look at the notebook preprocess_rat7m.ipynb for more information."
        )
