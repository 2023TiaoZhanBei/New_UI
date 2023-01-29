import torch
import torch.nn as nn


# ResBlock
class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class TestCNNNet(nn.Module):
    def __init__(self, in_chs, num_classes):
        super(TestCNNNet, self).__init__()
        self.in_chs = in_chs
        self.num_classes = num_classes
        self.stage0 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_chs, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        # stage 1
        self.stage1 = nn.Sequential(
            ResBlock1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        # stage 2
        self.stage2 = nn.Sequential(
            ResBlock1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        # stage 3
        self.stage3 = nn.Sequential(
            ResBlock1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        # stage 4
        self.stage4 = nn.Sequential(
            ResBlock1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        )
        # classifier
        self.classifier = nn.Linear(256*126, self.num_classes)

    def forward(self, x):
        # stage 0
        out = self.stage0(x)
        # stage 1
        out = self.stage1(out)
        # stage 2
        out = self.stage2(out)
        # stage 3
        out = self.stage3(out)
        # stage 4
        out = self.stage4(out)
        # classifier
        out = self.classifier(out.view(out.size(0), -1))
        return out


if __name__ == '__main__':
    x = torch.randn(128, 30, 126)
    model = TestCNNNet(in_chs=30, num_classes=7)
    print(model(x).shape)
