#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import re
import setuptools
import glob
from os import path
import torch
from torch.utils.cpp_extension import CppExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="zero",
    version="0.1.0",
    author="dutmmlab",
    python_requires=">=3.6",
    long_description=long_description,
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_namespace_packages(),
)