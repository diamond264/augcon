import torch
from torch.utils.data import Dataset

class EmptyDataset(Dataset):
    def __init__(self):
        # Initialize any necessary attributes here
        pass

    def __len__(self):
        # Return the length of the dataset (0 for empty)
        return 0

    def __getitem__(self, index):
        # Implement this method for indexing, even though the dataset is empty
        raise IndexError("Index out of range: dataset is empty")