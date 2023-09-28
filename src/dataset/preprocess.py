import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def convert_to_binary_mask(mask: Image.Image) -> Image.Image:
    """Converts a mask to a binary mask.

    Args:
        mask (Image.Image): The mask to convert.

    Returns:
        Image.Image: The converted binary mask.
    """
    mask_array = np.array(mask).mean(axis=-1) # (H, W, 3) -> (H, W)
    mask_array[mask_array > 0] = 255
    return Image.fromarray(np.uint8(mask_array))

def main() -> None:
    """Reformats the raw unzipped dataset into an img/binary mask directory structure."""
    image_dirs = os.listdir("public_coin_dataset")

    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/img"):
        os.mkdir("data/img")
    if not os.path.exists("data/mask"):
        os.mkdir("data/mask")

    for idx, image_dir in enumerate(tqdm(image_dirs)):
        # Loading image and mask
        img = Image.open(f"public_coin_dataset/{image_dir}/original/coin.jpeg").convert("RGB")
        mask = Image.open(f"public_coin_dataset/{image_dir}/no_bg/coin.jpeg").convert("RGB")
        # Convert mask to binary
        binary_mask = convert_to_binary_mask(mask)
        # Save image mask pair
        img.save(f"data/img/{idx:04d}.jpeg")
        binary_mask.save(f"data/mask/{idx:04d}.png")

    

if __name__ == "__main__":
    main()