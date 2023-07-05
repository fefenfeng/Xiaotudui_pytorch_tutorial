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

- Dataloader: 为网络提供不同的数据形式

