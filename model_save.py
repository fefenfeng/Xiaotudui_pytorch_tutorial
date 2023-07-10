import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数，模型加保存路径
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐），把网络模型的参数保存成字典
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 方式1的陷阱
class Feng(nn.Module):
    def __init__(self):
        super(Feng, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

feng = Feng()
torch.save(feng, "feng_method1.pth")