#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved
import setuptools

#     install_requires=[
#         'yolox',
#         'bytetrack'
#     ],
setuptools.setup(
    name="count",
    version="0.1.0",
    author="cong tou",
    python_requires=">=3.6",
    long_description="count algorithm",
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    packages=setuptools.find_namespace_packages(),
)