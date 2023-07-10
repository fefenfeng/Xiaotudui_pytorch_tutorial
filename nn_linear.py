import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Feng(nn.Module):
    def __init__(self):
        super(Feng, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

feng = Feng()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)  # 64 3 32 32
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    # print(output.shape)
    output = torch.flatten(imgs)  # 展平，把输入展成一行
    print(output.shape)
    output = feng(output)
    print(output.shape)
