import pandas as pd

import torch


from torch.utils.data.dataset import Dataset

class EMNISTDataset(Dataset):

    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        pixels = row.iloc[1:].values
        image = pixels.reshape(28, 28)
        image = image.T
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)

        label = row.iloc[0]
        label = torch.tensor(label, dtype=torch.long)

        return image, label

