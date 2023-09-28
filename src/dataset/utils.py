from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

class CoinDataset(Dataset):
    def __init__(self, filename: List[str]) -> None:
        self.filename = filename
        self.transform = Compose([ToTensor()])

    def __len__(self) -> int:
        return len(self.filename)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(f"data/img/{self.filename[idx]}").convert("RGB")
        mask = Image.open(f"data/mask/{self.filename[idx]}").convert("RGB")
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask