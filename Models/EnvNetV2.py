import torch
import torch.nn as nn
class EnvNetv2(nn.Module):
    # initialW = nn.init.kaiming_uniform()
    def __init__(self, n_classes):
        super(EnvNetv2, self).__init__()
        self.convstage1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 64), stride=(1, 2)),
            nn.BatchNorm2d(),
            nn.Conv2d(32, 64, (1, 16), stride=(1, 2)),
            nn.BatchNorm2d(),
            nn.MaxPool2d(1, 64),
        )
        self.convstage2 = nn.Sequential(
            nn.Conv2d(1, 32, (8, 8)),
            nn.BatchNorm2d(),
            nn.Conv2d(32, 32, (8, 8)),
            nn.BatchNorm2d(),
            nn.MaxPool2d(5, 3),
            nn.Conv2d(32, 64, (1, 4)),
            nn.BatchNorm2d(),
            nn.Conv2d(64, 64, (1, 4)),
            nn.BatchNorm2d(),
            nn.MaxPool2d(1, 2),
            nn.Conv2d(64, 128, (1, 2)),
            nn.BatchNorm2d(),
            nn.Conv2d(128, 128, (1, 2)),
            nn.BatchNorm2d(),
            nn.MaxPool2d(1, 2),
            nn.Conv2d(128, 256, (1, 2)),
            nn.BatchNorm2d(),
            nn.Conv2d(256, 256, (1, 2)),
            nn.BatchNorm2d(),
            nn.MaxPool2d(1, 2),
        )
        self.liner1 = nn.Sequential(
            nn.Linear(256 * 10 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.liner2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.liner3 = nn.Sequential(
            nn.Linear(4096, n_classes),
        )

    def __call__(self, x):
        h = self.convstage1(x)
        h = self.convstage2(h)
        h = h.permute(1,0,2)
        h = self.convstage2(h)
        h = self.liner1(h)
        h = self.liner2(h)
        h = self.liner3(h)
        return h
