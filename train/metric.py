# -*- coding:utf-8 -*-
"""
计算多分类下macro-precision, macro-recall, micro-precision, micro-recall, accuracy, 混淆矩阵
"""
import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def avg_precision(output, target, num_classes, mode='macro'):
    """
    计算平均精确率，有两种模式，macro与micro
    :param output:
    :param target:
    :param num_classes:
    :param mode: 决定模式参数，默认macro
    :return:
    """
    with torch.no_grad():

        assert output.shape[0] == len(target)

        if mode == 'micro':
            return _precision_micro_agg(output, target)
        elif mode == 'macro':
            return _precision_macro_agg(output, target, num_classes)
        else:
            raise ValueError('Pass a valid type of aggregation to the precision metric.')


def avg_recall(output, target, num_classes, mode='macro'):
    """
    计算平均recall，有两种模式，macro与micro
    :param output:
    :param target:
    :param num_classes:
    :param mode: 决定模式参数，默认macro
    :return:
    """
    with torch.no_grad():

        assert output.shape[0] == len(target)

        if mode == 'micro':
            return _recall_micro_agg(output, target)
        elif mode == 'macro':
            return _recall_macro_agg(output, target, num_classes)
        else:
            raise ValueError('Pass a valid type of aggregation to the precision metric.')


def _precision_macro_agg(output, target, num_classes):
    preds = torch.argmax(output, dim=1)

    ret = torch.zeros(num_classes)

    for ind in target.unique():
        tp = torch.sum((preds == target) * (preds == ind)).item()
        fp = torch.sum((preds != target) * (preds == ind)).item()

        denom = (tp + fp)
        ret[ind] = tp / denom if denom > 0 else 0

    return ret.mean()


def _precision_micro_agg(output, target):
    preds = torch.argmax(output, dim=1)

    tp_cumsum = 0
    fp_cumsum = 0

    for ind in target.unique():
        tp = torch.sum((preds == target) * (preds == ind)).item()
        fp = torch.sum((preds != target) * (preds == ind)).item()

        tp_cumsum += tp
        fp_cumsum += fp

    return tp_cumsum / (tp_cumsum + fp_cumsum)


def _recall_macro_agg(output, target, num_classes):
    preds = torch.argmax(output, dim=1)

    ret = torch.zeros(num_classes)

    for ind in target.unique():
        tp = torch.sum((preds == target) * (target == ind)).item()
        fn = torch.sum((preds != target) * (target == ind)).item()

        denom = (tp + fn)
        ret[ind] = tp / denom if denom > 0 else 0

    return ret.mean()


def _recall_micro_agg(output, target):
    preds = torch.argmax(output, dim=1)

    tp_cumsum = 0
    fn_cumsum = 0

    for ind in target.unique():
        tp = torch.sum((preds == target) * (target == ind)).item()
        fn = torch.sum((preds != target) * (target == ind)).item()

        tp_cumsum += tp
        fn_cumsum += fn

    return tp_cumsum / (tp_cumsum + fn_cumsum)


def classification_metrics(num_classes):
    avg_p = lambda x, y: avg_precision(x, y, num_classes, mode='macro')
    avg_p.__name__ = 'avg_precision'

    avg_r = lambda x, y: avg_recall(x, y, num_classes, mode='macro')
    avg_r.__name__ = 'avg_recall'
    return [accuracy, avg_p, avg_r]


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
