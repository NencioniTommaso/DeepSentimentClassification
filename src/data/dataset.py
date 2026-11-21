import torch
from torch.utils.data import Dataset
from pathlib import Path

class IMDbDataset(Dataset):
    def __init__(self, root, train=True):
        self.data = []
        root_path = Path(root) / ("train" if train else "test")

        for label in ['pos', 'neg']:
            label_path = root_path / label
            for file in sorted(label_path.glob("*.txt")):           #iterate through all text files in the directory
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    self.data.append((text, 1 if label == 'pos' else 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample, label = self.data[index]
        return sample, label            # conveniently return the text and its corresponding label as a tuple, useful for DataLoader
