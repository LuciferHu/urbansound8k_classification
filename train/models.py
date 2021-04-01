# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import init
import sys
import Models
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.models.vgg import vgg16_bn

sys.path.append("..")


# print(Models.__dict__)
class ModelCalled(nn.Module):

    def __init__(self, arch, num_classes=10):
        super(ModelCalled, self).__init__()
        self.melspectrogram = MelSpectrogram(sample_rate=16384,   # 与FolderDataset.duration一起，使得mel图的shape=(128, 128)，记住设置fmax=8000
                                             n_fft=2048,
                                             hop_length=512,
                                             f_max=8000,
                                             n_mels=128)
        self.power2db = AmplitudeToDB(stype='power')
        self.model = Models.__dict__[arch](num_classes=num_classes)

        # self._initialize_weights()

    def forward(self, x):
        """
        应特别注意，传进来的是一个个音频，需要先对其转换成mel谱图，再传进网络
        :param x: audio series tensor, tensor([[series], [series], ..., [series]])
        :return:
        """
        x = self.melspectrogram(x)  # 转换成了梅尔图
        x = self.power2db(x)    # 转换一下单位，这样特征比较明显
        x = torch.unsqueeze(x, dim=1)    # 为了适应图片batch：[batch_size, channels, height, width]，扩张的是channel
        # print(x.shape)
        out = self.model.forward(x)  # 再对图调用CNN模型
        return out


if __name__ == "__main__":
    print("start...")
    net = ModelCalled("vgg16")
    # print(net)
    x = torch.randn([24, 66150])    # 3s
    # print(x[1])
    y = net(x)
    print(y.shape)
    print("end...")
