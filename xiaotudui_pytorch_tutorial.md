# 环境版本配置

- Anaconda: 3 5.2.0
- CUDA ToolKit 9.2
- pytorch 1.3
- python 3.6

本机环境配置：win11 python 3.10 conda 23.3.1pycharm pro 2023.1.3 cuda11.5.2 cuDNN8.3.2 pytorch1.12.1 tensorflow2.7.0

**编辑器**

pycharm + Jupyter notebook

在新环境中下载jupyter -- `conda install nb_conda`（听说3.9不行,反正需要ipykernel)

# Python学习中的两大法宝

- dir()：打开，看见
- help(): 说明书

python文件所有代码行为一个块，通用传播方便，适用于大型项目，缺点需要重新运行。

python控制台以每一行为块运行，显示每个变量属性报错更详细，不利于代码阅读修改。

Jupyter以任意行为块运行，利于代码阅读及修改，缺点环境需要配置。

# PyTorch加载数据

- Dataset: 提供一种方式去获取数据及其label

  **如何获取每一个数据及其label**

  **告诉我们总共有多少的数据**

  继承重写子类，重写`__getitem__`和`len`

- Dataloader: 为网络提供不同的数据形式

数据集label形式：文件夹为label，OCR数据集，label在图片名称上面

# TensorBoard

pytorch1.1之后从tensorflow移植过来

add_scalar 一般用于训练过程中显示train-loss

add_image加图像

![image-20230705204814917](C:/Users/FENG/AppData/Roaming/Typora/typora-user-images/image-20230705204814917.png)

打开log ：terminal输入`tensorboard --logdir=logs`,指定端口`tensorboard --logdir=logs --port=6007`

![image-20230705202353966](C:/Users/FENG/AppData/Roaming/Typora/typora-user-images/image-20230705202353966.png)

# Transforms

torchvision之中的Transform的结构和用法, transform.py工具箱

其中有很多类totensor（转化为tensor）数据类型

resize，compose等等......每个工具类我们都需要实例对象+调用对象方法

## 常见的Transforms

- 输入，输出，作用
- PIL，tensor，narrays
- Image.open(), ToTensor(), cv.imread()

# pytorch.torchvision中的数据集使用

torchvision.datasets以及和transforms一起的使用

![image-20230706195239011](C:/Users/FENG/AppData/Roaming/Typora/typora-user-images/image-20230706195239011.png)

# Dataloader

dataset, batch_size, shuffle, num_workers（几个进程加载数据，0就是仅有一个主进程）,drop last(batch_size除尽余下的舍不舍去)

# 神经网络的基本骨架 -nn.Module的使用

containers容器，Module所有神经网络的一个基本骨架，神经网络需要继承Module

主要写两个函数，一个init初始化继承父类super，一个forward前向传播

# Convolution layers

CONV1D, CONV2D, torch.nn.functional是nn的更细致功能和设置

dialation空洞卷积不经常用

# Pooling Layers

cell_mode floor向下取整 ceil向上取整

# 非线性激活

ReLU，Sigmoid

# 其余层

批量归一化，Recurrent layers, drop out, linear其实就是全连接层, embedding, distance functions, loss functions,shuffle, transformer