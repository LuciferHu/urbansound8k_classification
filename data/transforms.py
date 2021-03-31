# -*- coding:utf-8 -*-
"""
本文件用于对载入的音频作变换，旨在为模型增加鲁棒性
"""
import numpy as np
import torch
from torchaudio.transforms import Spectrogram, MelSpectrogram
from torchvision import transforms
import librosa.feature as feature


class Melspectrogram(MelSpectrogram):
    """
    将一段音频转换成一个mel谱图，由于数据装载类CSVDataManager().get_loader()已将原始音频处理成一个个tensor，并pad到4s，传进来的其实是tensor
    被弃用，torchaudio有MelSpectrogram，使用起来比较方便
    """
    def __init__(self,
                 sample_rate=22050,
                 n_fft=2048,
                 hop_length=None,
                 n_mels=128,
                 center=False,
                 ):
        """
        初始化
        :param x: 音频序列，原始
        """
        super(Melspectrogram, self).__init__(sample_rate=sample_rate,
                                             n_fft=n_fft,
                                             hop_length= n_fft // 4,
                                             n_mels = n_mels,
                                            )    # 将父类方法继承回来

    def forward(self, x):
        """
        将传进来的时间序列信息转换成梅尔图
        :param x: audio series, 4s, tensor.shape=(88200,)
        :return: melspectrogram, tensor.shape=(nmels, t), t = duration * sr / hop_length
        """
        return feature.melspectrogram(y=x, sr=self.sample_rate,
                                      hop_length=self.hop_length,
                                      n_fft=self.n_fft,
                                      center=self.center,
                                      n_mels=self.n_mels)     # 返回的是梅尔图



