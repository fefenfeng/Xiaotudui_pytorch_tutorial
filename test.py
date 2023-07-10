import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./imgs/airplane.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')   # png格式为4通道，除RGB三通道还有一个透明通道
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

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

    def forward(self, x):
        x = self.model(x)
        return x
model = torch.load("tudui_29_gpu.pth")
model = torch.load("tudui_29_gpu.pth", map_location=torch.device('cpu'))  # gpu上训练的模型，如果转移到cpu就搞一下
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))
