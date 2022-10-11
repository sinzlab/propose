#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path
import subprocess

here = path.abspath(path.dirname(__file__))

version = '0.0'


def _get_version_hash():
    """Talk to git and find out the tag/hash of our latest commit"""
    try:
        ver = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
        return ver
    except EnvironmentError:
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
