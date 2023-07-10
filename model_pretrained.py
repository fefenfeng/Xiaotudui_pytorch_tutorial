import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=True)
vgg16_true = torchvision.models.vgg16(pretrained=True)
# True的时候预训练过，参数需要下载

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 在预训练的vgg16上最后加一层，将输出特征1000-->10，classifier就是加到内层中
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

print(vgg16_false)
# 修改最后一层，直接修改最后一层输出特征，改为10
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)


