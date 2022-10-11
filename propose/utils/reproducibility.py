import numpy as np
import torch
import random
import pkg_resources


def set_random_seed(seed):
    """
    Sets all random seeds
    """

    # Base seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Cuda seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Make GPU operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_commit_hash() -> str:
    """
    Returns the current commit hash shortend to 7 characters.
    """
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()


def check_uncommited_changes() -> bool:
    """
    Checks if there are uncommited changes.
    """
    import subprocess

    return (
        subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        != ""
    )


def get_package_version(package_name: str) -> str:
    """
    Returns the version of the package.
    """
    return pkg_resources.get_distribution(package_name).version
