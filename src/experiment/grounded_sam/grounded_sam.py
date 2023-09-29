import time
import argparse
from typing import List, Dict, Any

import cv2
import wandb
import numpy as np
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

from src.experiment.utils import calculate_iou, get_train_val_split, log_table


def generate_table_data(image_paths: str, ground_truths: np.ndarray, predictions: np.ndarray, ious: List[float], subset: str) -> List[Dict[str, Any]]:
    data = []
    for image_path, ground_truth, prediction, iou in zip(image_paths, ground_truths, predictions, ious):
        d = {}
        d["image"] = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        d["mask"] = ground_truth
        d["prediction"] = prediction
        d["iou"] = iou
        d["id"] = image_path.split("/")[-1].split(".")[0]
        d["subset"] = subset
        data.append(d)
    
    return data

def predict(model: GroundedSAM, image_paths: List[str]) -> List[np.ndarray]:
    pred_masks = []
    for image_path in image_paths:
        pred_mask = model.predict(image_path)
        pred_mask = pred_mask.mask.astype(np.uint8)

        # If GroundedSAM outputs more than one probable mask we take the average
        # across all masks
        if len(pred_mask.shape) == 3:
            pred_mask = pred_mask.mean(axis=0).astype(np.uint8)

        pred_masks.append(pred_mask)
    
    return pred_masks


def main(args: argparse.Namespace) -> None:
    (x_train, y_train), (x_val, y_val) = get_train_val_split(args.val_size, args.random_seed)
    model = GroundedSAM(ontology=CaptionOntology({args.prompt: "coin"}))

    start = time.time()
    pred_masks_train = predict(model, x_train)
    end = time.time()

    pred_masks_val = predict(model, x_val)

    ground_truths_train = [cv2.imread(mask, cv2.IMREAD_GRAYSCALE) for mask in y_train]
    ground_truths_val = [cv2.imread(mask, cv2.IMREAD_GRAYSCALE) for mask in y_val]

    ious_train = [calculate_iou(mask.copy(), pred_mask.copy()) for mask, pred_mask in zip(ground_truths_train, pred_masks_train)]
    iou_train = np.mean(ious_train)

    ious_val = [calculate_iou(mask.copy(), pred_mask.copy()) for mask, pred_mask in zip(ground_truths_val, pred_masks_val)]
    iou_val = np.mean(ious_val)

    throughput = len(x_train) / (end - start)
    latency = (end - start) / len(x_train)
    
    if args.wandb:
        with wandb.init(project="coin-segmentation", tags=["grounded-sam"], config=vars(args)) as run:
            run.log(
                {
                    "mIoU-train": iou_train,
                    "mIoU-val": iou_val,
                    "latency": latency, 
                    "throughput": throughput, 
                }
            )

            data_train = generate_table_data(x_train, ground_truths_train, pred_masks_train, ious_train, "train")
            data_val = generate_table_data(x_val, ground_truths_val, pred_masks_val, ious_val, "val")
            data = data_train + data_val
            log_table(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GroundedSAM")
    parser.add_argument("--val-size", type=int, default=20, help="Validation set size")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", type=str, default="all circle shaped objects that resemble a coin", help="Prompt")
    parser.add_argument("--wandb", action="store_true", help="Whether or not to use wandb")
    args = parser.parse_args()
    main(args)