# -*- coding:utf-8 -*-
"""
本文件负责知道数据位置、标签
"""
import torch
import torch.utils.data as data
import librosa


class FolderDataset(data.Dataset):
    """
    自定义dataset，在装载时就已经将音频pad成4s，归一化并且统一到单声道
    """
    def __init__(self, data_arr, transforms=None,
                 sample_rate=16384,    # 与duration一起，使得mel图的shape=(128, 128)，记住MelSpectrogram里要设置fmax=8000
                 n_fft=2048,
                 hop_length=None,
                 n_mels=128,
                 center=False,
                 duration=4
                 ):
        self.sample_rate = sample_rate  # 音频采样率
        self.n_fft = n_fft  # stft帧长
        self.hop_length = n_fft // 4  # 帧移，默认为n_fft的1/4
        self.n_mels = n_mels  # 梅尔滤波器组数
        self.center = center  # 帧起始位置，默认使其在t * hop_length
        self.duration = duration  # 读取时长
        self.transforms = transforms    # 数据变换方法
        self.data_arr = data_arr    # 批数据
        # 载入数据的方法，这里使用librosa库

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, index):
        elem = self.data_arr[index]
        audio_tensor, label = self._load_file_data(elem['path']), elem['class_idx']

        if self.transforms is not None:
            audio, sr, label = self.transforms.apply(audio_tensor, label)
            return audio, sr, label

        return audio_tensor, label

    def _load_file_data(self, path):
        """
        载入一个文件夹的音频文件，要求时长在5s，对于不满足3s的，作自内容填充处理，对于超过3s的，截断
        :param path: 文件路径
        :param sr: 采样率，默认为22000Hz
        :param duration: 音频持续时长，默认为4s
        :return: data: 被填充完毕的5s音频
        """
        input_length = self.sample_rate * self.duration
        X, sr = librosa.load(path, sr=self.sample_rate, duration=self.duration, res_type='kaiser_best')  # read 4s
        dur = librosa.get_duration(y=X, sr=sr)

        if dur < self.duration:
            y = librosa.util.fix_length(X, input_length, mode='wrap')  # fix audio length, y: np.ndarr, shape=(88200,)
        else:
            y = X

        return torch.tensor(y)    # 将np.ndarray转换为tensor


class FolderRawDataset(data.Dataset):

    def __init__(self, data_arr, load_func, transforms=None):
        self.transforms = transforms
        self.data_arr = data_arr
        self.load_func = load_func

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, index):
        elem = self.data_arr[index]
        data, label = self.load_func(elem['path']), elem['class_idx']

        if self.transforms is not None:
            audio, sr, label = self.transforms.apply(data, label)
            return audio, sr, label

        return data, label


if __name__ == '__main__':
    pass