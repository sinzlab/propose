#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="propose",
    version="0.0",
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
