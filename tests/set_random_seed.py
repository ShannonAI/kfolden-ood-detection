#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: set_random_seed.py

import os
import sys

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import torch
from utils.random_seed import set_random_seed
# set_random_seed(2333)


def reset_random_seed():
    set_random_seed(4455)
    print(f"RESET torch seed is : {torch.initial_seed()}")


if __name__ == "__main__":
    set_random_seed(2333)
    print(f"torch seed is : {torch.initial_seed()}")
    reset_random_seed()



