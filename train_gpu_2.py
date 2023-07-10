# 完整训练
import torch.cuda
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader
import time

# 定义训练的设备
device = torch.device("cuda")
# device = torch.device("cuda:0")   # 单显卡没区别
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 准备训练数据集和测试数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))  # 字符串格式化，其实就是占位符
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
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
feng = Feng()
feng.to(device)  # 模型后面弄个cuda就行，网络模型转移到cuda上
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)   # 损失函数转移到cuda
# 优化器
learning_rate = 1e-2  # 0.01
optimizer = torch.optim.SGD(feng.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")
start_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    feng.train()    # train和eval转换模式，对dropout和batchNorm层有作用
    for data in train_dataloader:
        imgs, targets = data    # 取训练数据
        imgs = imgs.to(device)
        targets = targets.to(device)        # 数据转移到cuda
        outputs = feng(imgs)    # 数据放到网络中
        loss = loss_fn(outputs, targets)    # 计算损失函数

        # 优化器优化模型
        optimizer.zero_grad()       # 参数梯度归零
        loss.backward()     # 反向传播
        optimizer.step()    # 梯度更新

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    feng.eval()
    total_test_loss = 0     # 累计计算loss和acc
    total_accuracy = 0
    with torch.no_grad():       # 此处的代码不累计梯度
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)  # 数据转移到cuda
            outputs = feng(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # torch.save(feng, "feng_{}.pth".format(i))
    # torch.save(feng.state_dict(), "feng_{}.pth".format(i))
    # print("模型已保存")

writer.close()
