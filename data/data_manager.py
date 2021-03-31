# -*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data
from UrbanSound8K import genData
from data.data_sets import FolderDataset
from utils.util import load_audio


class CSVDataManager(object):
    """
    对CSV文件处理，针对csv中的数据项做训练集与测试集的切割，并返回DataLoader，
    使用方法：CSVDataManager(config).get_loader('train'), CSVDataManager(config).get_loader('train')
    生成的数据：data：batch_size*audio_series, tensor([[series], [series], ..., [series]]), target: classes, tensor([0, 3, ..., 2])
    """

    def __init__(self, config):
        self.splits = config['splits']
        # config['splits']
        # {
        #     "train": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     "val": [10]
        # }

        self.dir_path = config['path']
        # config['path']
        # "/home/richardhu/Documents/PycharmProjects/urbansound8k_classification/UrbanSound8K"

        self.loader_params = config['args']
        # config['args']
        # {
        #     "shuffle": True,
        #     "batch_size": 24,
        #     "num_workers": 4,
        #     "drop_last": True
        # }

        # metadata_df = genData.GenUS8k(self.dir_path).data_selected  # 获取csv中数据，读取UrbanSound8K.csv时选择
        metadata_df = genData.GenUS8k(self.dir_path).df_data    # 获取csv中数据，读取eight_labels_modifiy_classid.csv时选择
        self.metadata_df = metadata_df.sample(frac=1)  # 将数据随机打乱

        self.classes = self._get_classes()  # 获取所有类别，这是一个list:
        # ['siren'  'dog_bark' 'drilling' 'children_playing'
        # 'street_music' 'jackhammer' 'engine_idling' 'air_conditioner']

        self.data_splits = self._10kfold_split(self.metadata_df)  # 数据切分成训练集7691，测试集807，最后一个fold作为测试集

    def _get_classes(self):
        return self.metadata_df["class"].unique()

    def _10kfold_split(self, df):
        """
        切分数据集
        :param df: dataFrame, shape=(8498, 11)
        :return: ret: dict, {'train': [{'path': ..., 'class': ..., 'class_idx': ...}, ..., ]
                            'val': [{'path': ..., 'class': ..., 'class_idx': ...}, ..., ]
                            }
        """
        ret = {}
        for s, inds in self.splits.items():
            """s in 'train' and 'test', inds in [1, ..., 9] and [10]"""
            df_split = df[df['fold'].isin(inds)]
            ret[s] = []

            for row in df_split[['slice_file_name', 'class', 'classID', 'fold']].values:
                fold_mod = 'audio/fold%s' % row[-1]    # 寻找路径
                fname = os.path.join(self.dir_path, fold_mod, '%s' % row[0])
                ret[s].append({'path': fname, 'class': row[1], 'class_idx': row[2]})
        return ret

    def get_loader(self, name):
        """
        数据载入方法
        :param name: 'train' or 'val'
        :return: data.DataLoader()
        """
        assert name in self.data_splits
        dataset = FolderDataset(self.data_splits[name], transforms=None)

        return data.DataLoader(dataset=dataset, **self.loader_params)


if __name__ == "__main__":
    metadata = CSVDataManager().metadata_df
    classes = CSVDataManager().classes
    print(classes)
    ret = CSVDataManager().data_splits
    # val_data_loader = CSVDataManager().get_loader(name='val')
    print("train nums: {}, val nums: {}".format(len(ret['train']), len(ret['val'])))
    # print(ret['val'])
    # pass
    # for i, data in enumerate(val_data_loader):
    #     print("i: {}, data: {}".format(i, data))
    #     break
