from model_save import *  # 实际项目中存模型的文档中直接全部引用过来
import torchvision
from torch import nn

# 方式1-》保存方式1，加载模型
model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2，加载模型，先新建再加载状态
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)

# 方式1的陷阱
# class Feng(nn.Module):
#     def __init__(self):
#         super(Feng, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
# 不需要写feng = Feng()，但需要把类代码挪过来

model = torch.load('feng_method1.pth')
print(model)