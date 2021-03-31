# -*- coding:utf-8 -*-
import librosa
import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def load_audio(path, sr=22050, duration=4):
    """
    读取音频，对于不满4s的填充处理，多于4s的切割
    :param duration: 音频持续时长，默认为4s
    :param sr: 采样率，默认为22050Hz
    :param path: 路径
    :return: audio_arr: 倍填充完的音频数据，shape=(88200, )
    """
    input_length = sr * duration
    X, sr = librosa.load(path, sr=sr, duration=duration)
    dur = librosa.get_duration(y=X, sr=sr)

    if dur < duration:
        y = librosa.util.fix_length(X, input_length, mode='wrap')  # fix audio length, y: np.ndarr, shape=(88200,)
    else:
        y = X

    return y


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])     # 构建一个dataframe，有三列：total，counts， average
        self.reset()    # 重置dataframe，所有列都置0

    def reset(self):    # 重置函数
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):     # 更新tensorboard与_data中index=key中的三个colunm中的内容
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]    # 取平均

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


if __name__ == "__main__":
    metric_ftns = ["accuracy", "top_k_acc"]
    writer = None
    train_metrics = MetricTracker('loss', *metric_ftns,
                                  writer=writer)  # 训练集的混淆矩阵实例化
    print(train_metrics.result())    # 可以知晓到底有多少index
