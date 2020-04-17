import torch
from torch import nn
from torch.nn.functional import relu, sigmoid

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

    def forward(self, x):

        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = sigmoid(self.l3(x))
        return x



