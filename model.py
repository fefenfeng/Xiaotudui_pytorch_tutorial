# 完整训练
import torch
from torch import nn

# 搭建神经网络
class Feng(nn.Module):
    def __init__(self):
        super(Feng, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
#  模型只是为了简单示例，卷积后记得加激活啊
    def forward(self, x):
        x = self.model(x)
        return x

# 训练网络正确性
if __name__ == '__main__':
    feng = Feng()
    input = torch.ones((64, 3, 32, 32))  # 64的batch size，3通道，尺寸32*32
    output = feng(input)
    print(output.shape)