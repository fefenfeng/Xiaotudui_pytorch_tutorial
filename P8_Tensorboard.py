from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# 其实用opencv读入图片就好，up主懒了鸽了，用的PIL直接numpy转数组
writer = SummaryWriter("logs")
image_path = "dataset/train/bees_image/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
# pil读出的文件是pil.jpg.image不符合要求，所以可能opencv最好
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)   # 三通道确实在最后一位，下面dataformats确实得改为HWC

writer.add_image("train", img_array, 2, dataformats='HWC')
# img_tensor需要为torch.Tensor，或者numpy.array,或string/blobname)
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)
    # writer.add_scalar("y=x",i,i)
writer.close()