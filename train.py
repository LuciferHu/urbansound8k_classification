# -*- coding:utf-8 -*-
import argparse
import torch
from data.data_manager import CSVDataManager
from train.models import ModelCalled
import collections
from train import Trainer
import train as net_utils
from parse_config import ConfigParser
import numpy as np
from utils import prepare_device


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train_main(config):
    """
    训练函数
    :param config: ConfigParser对象
    :return: None
    """
    logger = config.get_logger('train')    # 训练数据的日志对象

    data_manager = CSVDataManager(config['data_loader'])    # 将json文件中指示数据载入要求信息传入CSV管理器中，装载训练集与测试集
    classes = data_manager.classes    # 获取所有类别
    num_classes = len(classes)  # 知晓类别数量

    train_data = data_manager.get_loader('train')    # 得到训练集
    val_data = data_manager.get_loader('val')     # 得到验证集

    model_name = config['model']    # 从json文件中获取模型名称
    model = ModelCalled(model_name, num_classes=num_classes)     # 召唤模型
    logger.info(model)    # 记录模型的信息

    # 为多GPU训练做准备
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # if torch.cuda.is_available():  # 检测是否能用GPU运算
    #     model = model.cuda()  # 将模型转移到GPU上去

    loss = getattr(net_utils, config['loss'])    # 获取损失函数

    metrics = [getattr(net_utils, met) for met in config['metrics']]    # 多分类评价标准需要传进类别数

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())    # filter函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表

    optim_name = config['optimizer']['type']    # 优化器名字
    optim_args = config['optimizer']['args']    # 优化器参数
    optimizer = getattr(torch.optim, optim_name)(trainable_params, **optim_args)

    lr_name = config['lr_scheduler']['type']    # 学习率
    lr_args = config['lr_scheduler']['args']    # 学习率参数

    if lr_name == 'None':
        lr_scheduler = None
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)

    trainer = Trainer(model=model,
                      loss=loss,
                      metrics=metrics,
                      optimizer=optimizer,
                      config=config,
                      data_loader=train_data,
                      valid_data_loader=val_data,
                      lr_scheduler=lr_scheduler,
                      device=device
                      )
    trainer.train()


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="urbansound8k classification")
    # 以下为必要参数
    args.add_argument("-c", "--config", default=None, type=str,
                      help="config file path(Default: None)")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help='path to load latest checkpoint(Default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    train_main(config)
