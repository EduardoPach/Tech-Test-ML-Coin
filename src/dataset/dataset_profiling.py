import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from data_gradients.managers.segmentation_manager import SegmentationAnalysisManager 

class CoinDataset(Dataset):
    def __init__(self, filename: List[str]) -> None:
        self.filename = filename
        self.transform = Compose([ToTensor()])

    def __len__(self) -> int:
        return len(self.filename)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(f"data/img/{self.filename[idx]}.jpeg").convert("RGB")
        mask = Image.open(f"data/mask/{self.filename[idx]}.png").convert("RGB")
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask


def main() -> None:
    filenames = [f.split(".")[0] for f in os.listdir("data/img")]

    train_data = CoinDataset(filenames) # Your dataset iterable (torch dataset/dataloader/...)
    # val_data = ...    # Your dataset iterable (torch dataset/dataloader/...)
    class_names = ["coin", "coin", "coin"] # [<class-1>, <class-2>, ...]

    analyzer = SegmentationAnalysisManager(
        report_title="Testing Data-Gradients Segmentation",
        train_data=train_data,
        val_data=train_data,
        class_names=class_names,
        images_extractor=lambda data: data[0],
        labels_extractor=lambda data: data[1]
    )

    analyzer.run()

if __name__ == "__main__":
    main()