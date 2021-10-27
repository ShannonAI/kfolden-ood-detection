#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: compute_loss.py

import os
import sys

repo_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print(repo_path)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import torch
from torch.nn.modules import CrossEntropyLoss
from utils.random_seed import set_random_seed
set_random_seed(2333)

def test_cross_entropy_ignore_idx(eps=1e-12):
    logits = torch.rand([4, 3])
    labels = torch.tensor([0, 1, 2, -100], dtype=torch.long)
    id_label_mask = torch.tensor([1, 1, 1, 0], dtype=torch.long)

    ce_loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
    data_loss = ce_loss_fct(logits.view(-1, 3), labels.view(-1))
    print(f"loss tensor is: {data_loss}")
    avg_loss = torch.sum(data_loss * id_label_mask) / torch.sum(id_label_mask.float() + eps)
    print(f"average loss is: {avg_loss}")


if __name__ == "__main__":
    test_cross_entropy_ignore_idx()


