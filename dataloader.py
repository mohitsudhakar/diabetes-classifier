import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):

    def __init__(self, file):
        data = np.loadtxt(file, delimiter=',', dtype=np.float32)
        self._len = data.shape[0]
        self.x = torch.from_numpy(data[:, 0:-1])
        self.y = torch.from_numpy(data[:, -1])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self._len
