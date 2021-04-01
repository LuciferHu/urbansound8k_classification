# -*- coding:utf-8 -*-
"""
本文件用于对载入的音频作变换，旨在为模型增加鲁棒性
"""
import torch
import numpy as np
import torch.nn as nn
from torchaudio.transforms import Spectrogram, MelSpectrogram
from torchvision import transforms
import librosa.feature as feature


# class Melspectrogram(nn.modules):
#     """
#     将一段音频转换成一个mel谱图，由于数据装载类CSVDataManager().get_loader()已将原始音频处理成一个个tensor，并pad到4s，传进来的其实是tensor
#     被弃用，torchaudio有MelSpectrogram，使用起来比较方便
#     """
#     def __init__(self,
#                  sample_rate=22050,
#                  n_fft=2048,
#                  hop_length=None,
#                  n_mels=128,
#                  center=False,
#                  ):
#         """
#         初始化
#         :param x: 音频序列，原始
#         """
#         super().__init__()    # 将父类方法继承回来
#         self.sample_rate = sample_rate
#         self.hop_length = hop_length
#         self.n_fft = n_fft
#         self.center = center
#         self.n_mels = n_mels
#
#     def forward(self, x):
#         """
#         将传进来的时间序列信息转换成梅尔图
#         :param x: audio series, 4s, tensor.shape=(88200,)
#         :return: melspectrogram, tensor.shape=(nmels, t), t = duration * sr / hop_length
#         """
#         return feature.melspectrogram(y=x, sr=self.sample_rate,
#                                       hop_length=self.hop_length,
#                                       n_fft=self.n_fft,
#                                       center=self.center,
#                                       n_mels=self.n_mels)     # 返回的是梅尔图


class AddGaussNoise(object):
    """
    为装载进来的音频数据添加随机高斯噪声，只在训练时添加
    """

    def __init__(self, input_size=88200, mean=0.0, std=None, add_noise_probability=1.0):
        super(AddGaussNoise, self).__init__()
        """
        注：input_size = duration * sr, sr一定要与FolderDataset.sample_rate相等
        """
        assert isinstance(input_size, int)  # 指示input_size必须为int，也即音频序列的长度，input_size = sr * duration
        assert isinstance(mean, (int, float))  # mean为int或float
        assert isinstance(std, (int, float)) or std is None
        assert isinstance(add_noise_probability, float)

        self.input_size = input_size

        self.mean = mean

        if std is not None:  # 若外部声明std
            assert std > 0.0  # 标准差必须大于0
            self.std = std
        else:
            self.std = std

        assert 0.0 < add_noise_probability <= 1.0
        self.add_noise_probability = add_noise_probability

    def __call__(self, audio_series):
        """传进来的audio_series是tensor"""
        if np.random.random() > self.add_noise_probability:  # 若add_noise_probability为0，即不添加随机噪声
            return audio_series

        min_pixel_value = torch.min(audio_series)
        if self.std is None:
            std_factor = 0.03
            std = np.abs(min_pixel_value * std_factor)

        # 生成高斯随机噪声
        guass_mask = torch.from_numpy(np.random.normal(self.mean, std, size=self.input_size).astype("float32"))

        # 添加随机噪声
        noisy_audio_series = audio_series + guass_mask

        return noisy_audio_series


class AudioTransforms(object):
    def __init__(self, args):
        self.trans = transforms.Compose([AddGaussNoise(args)])

    def apply(self, data):
        audio_tensor = data
        return self.trans(audio_tensor)

    def __repr__(self):
        return self.trans.__repr__()


if __name__ == "__main__":
    args = 88200
    audio = torch.randn(size=(88200,))
    # print(audio)
    # print(audio.shape)
    noisy_audio = AudioTransforms(args).apply(audio)
    print(noisy_audio.shape)
