import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 16)
        self.l2 = nn.Linear(16, 32)
        self.l3 = nn.Linear(32, 24)
        self.l4 = nn.Linear(24, 8)
        self.l5 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = F.relu(self.l2(x))
        x = self.dropout(x)
        x = F.relu(self.l3(x))
        x = self.dropout(x)
        x = F.relu(self.l4(x))
        x = F.sigmoid(self.l5(x))
        return x



