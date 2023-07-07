import torch
from torch import nn


class Fengmodule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    # def __init__(self):
    #     super().__init__()

    def forward(self, input):
        output = input + 1
        return output


fengmodule = Fengmodule()
x = torch.tensor(1.0)
output = fengmodule(x)
print(output)