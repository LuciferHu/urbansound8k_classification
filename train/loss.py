# -*- coding:utf-8 -*-
"""
列举一些损失函数可供选择
"""
import torch.nn.functional as F
import torch.nn


def nll_loss(output, target):
    # loss for log_softmax
    return F.nll_loss(output, target)


def cross_entropy(output, target):
    return F.cross_entropy(output, target)
