#!/usr/bin/env python
import subprocess
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

version = '0.0'


def _get_version_hash():
    """Talk to git and find out the tag/hash of our latest commit"""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        print("Couldn't run git to get a version number for setup.py")
        return version


setup(
    name="propose",
    version=_get_version_hash(),
    description="Probabilistic Pose Estimation",
    author="Paweł A. Pierzchlewicz",
    author_email="ppierzc@gmail.com",
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    package_data={
        "": [
            "*.yaml",
        ]
    },
)
