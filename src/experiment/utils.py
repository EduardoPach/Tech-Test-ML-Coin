import os
from typing import List, Tuple, Dict, Any

import cv2
import wandb
import numpy as np

TrainSet = Tuple[List[str], List[str]]
ValidSet = Tuple[List[str], List[str]]

def get_train_val_split(val_size: int = 20, random_seed: int=42) -> Tuple[TrainSet, ValidSet]:
    # NOTE this images were removed because they are either completely unrelated to the task or 
    # they are mislabeled.
    IGNORE_IMAGES = ["0017", "0075", "0146", "0135"]
    img_path = sorted([f"data/img/{f}" for f in os.listdir('data/img/')])
    mask_path = sorted([f"data/mask/{f}" for f in os.listdir('data/mask/')])
    # REMOVE IGNORED FILES
    img_path = [f for f in img_path if f.split("/")[-1].split(".")[0] not in IGNORE_IMAGES]
    mask_path = [f for f in mask_path if f.split("/")[-1].split(".")[0] not in IGNORE_IMAGES]
    # Set the random seed to 42 for reproducibility
    np.random.seed(random_seed)
    # Create an array with all idx
    idx = np.arange(len(img_path))
    # Select a few index for training
    train_idx = np.random.choice(idx, len(img_path) - val_size, replace=False)
    # Selecting the rest for validation
    valid_idx = np.setdiff1d(idx, train_idx)

    # Getting the paths for each set
    x_train = np.array(img_path)[train_idx].tolist()
    y_train = np.array(mask_path)[train_idx].tolist()
    x_valid = np.array(img_path)[valid_idx].tolist()
    y_valid = np.array(mask_path)[valid_idx].tolist()


    return (x_train, y_train), (x_valid, y_valid)

def calculate_iou(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) for a segmentation problem.

    Args:
        ground_truth (numpy.ndarray): The ground truth binary mask.
        prediction (numpy.ndarray): The predicted binary mask.

    Returns:
        float: The IoU score.
    """
    # Ensure both masks are binary (0 or 1)
    ground_truth = (ground_truth > 0).astype(np.uint8)
    prediction = (prediction > 0).astype(np.uint8)

    # Calculate the intersection and union
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)

    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou

def log_table(data: List[Dict[str, Any]]) -> None:
    """Logs a table to wandb containing the data.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        A list of dictionary containing the following keys:
            - id: str
            - image: np.ndarray
            - mask: np.ndarray
            - prediction: np.ndarray
            - iou: float
    """
    table = wandb.Table(columns=['ID', "IoU", "Image", "subset"])
    class_labels = {0: "background", 1: "coin"}

    for d in data:
        _id = d["id"]
        iou = d["iou"]
        ground_truth = d["mask"]
        prediction = d["prediction"]
        image = d["image"]
        subset = d["subset"]
        img = wandb.Image(
            image,
            masks={
                "ground_truth": {
                    "mask_data": np.where(ground_truth>0, 1, 0).astype(np.uint8),
                    "class_labels": class_labels,
                },
                "prediction": {
                    "mask_data": np.where(prediction>0, 1, 0).astype(np.uint8),
                    "class_labels": class_labels,
                },
            },
        )
        
        table.add_data(_id, iou, img, subset)

    wandb.log({"Table" : table})