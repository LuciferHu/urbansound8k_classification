# -*- coding:utf-8 -*-
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from tqdm import tqdm
from utils import inf_loop, MetricTracker


# 模型结构基于 https://github.com/victoresque/pytorch-template
class Trainer(BaseTrainer):
    """
    训练模型类
    """

    def __init__(self, model, loss, metrics, optimizer, config, data_loader, device,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config)  # 继承父类的初始化方法，初始化这些参数
        self.config = config    # ConfigParser类
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None  # 当验证集不为None时，开启验证模式
        self.lr_scheduler = lr_scheduler  # 学习率更新策略
        self.log_step = int(np.sqrt(data_loader.batch_size))  # 对batch_size开方，用于定点记录信息标志

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)  # 训练集的混淆矩阵实例化，index=['loss', metric_ftns]
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)  # 验证集的混淆矩阵实例化

    # def _eval_metrics(self, output, target):
    #     """
    #     验证矩阵
    #     :param out:
    #     :param target:
    #     :return:
    #     """
    #     acc_metrics = np.zeros(len(self.metrics))
    #     for i, metric in enumerate(self.metrics):
    #         acc_metrics[i] += metric(output, target)
    #
    #         # self.writer.add_scalar("%s"%metric.__name__, acc_metrics[i])
    #         # self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
    #     return acc_metrics

    def _train_epoch(self, epoch):
        """
        训练1epoch数据，即8498个数据
        :param epoch: 整型，目前训练轮次信息
        :return: log, 记录本轮的平均损失与混淆矩阵
        """
        self.model.train()  # 设置模型进入训练模式
        self.train_metrics.reset()  # 重置混淆矩阵
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)  # 数据装载到GPU上, data: batch_size*tensor

            self.optimizer.zero_grad()  # 优化器清除梯度信息
            output = self.model(data)
            loss = self.criterion(output, target)  # 计算loss
            loss.backward()  # 反向传播
            self.optimizer.step()  # 执行优化算法

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)    # tensorboard执行
            self.train_metrics.update('loss', loss.item())    # 一次batch就计算loss并保存

            for met in self.metric_ftns:  # 执行混淆矩阵的更新
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:  # 记录下一次epoch
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()
                ))    # 这是输出到控制台的内容，每个batch的loss都被显示出来
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))    # tensorboard内容

            if batch_idx == self.len_epoch:    # 当训练完所有数据
                break

        log = self.train_metrics.result()    # 训练集的混淆矩阵平均结果

        if self.do_validation:  # 进入验证模式
            val_log = self._valid_epoch(epoch)    # 验证集的混淆矩阵平均结果
            log.update(**{'val_' + k: v for k, v in val_log.items()})    # 训练结果再加上验证结果

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()  # 执行学习率更新

        return log  # 返回记录信息

    def _valid_epoch(self, epoch):
        """
        验证模式
        :param epoch: 目前训练的epoch
        :return: log, 验证集的log
        """
        self.model.eval()  # 设置模型进入验证模式
        self.valid_metrics.reset()  # 验证的混淆矩阵重置
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.cuda(), target.cuda()  # 数据装载到GPU上

                output = self.model(data)  # 模型输出
                loss = self.criterion(output, target)  # 计算loss

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')    # 记录到tensorboard
                self.valid_metrics.update('loss', loss.item())    # 验证集指标更新
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()    # 返回验证集结果

    def _progress(self, batch_idx):
        """
        用于统计进程信息
        :param batch_idx:
        :return:
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch

        return base.format(current, total, 100.0 * current / total)
