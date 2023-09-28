import os
import time
import argparse
from typing import List

import cv2
import wandb
import optuna
import numpy as np

from src.experiment.utils import get_train_val_split, log_table
from src.experiment.classical_cv.utils import SegmentationPipeline, SegmentationOutput

def objective(trial: optuna.Trial, images: List[str], masks: List[str]) -> float:
    # Define the hyperparameters to be tuned
    params = {
        "blur_kernel_size": trial.suggest_categorical("blur_kernel_size", [i for i in range(2, 20) if i % 2 == 1]), # Suggest only odd numbers
        "blur_sigma": trial.suggest_int("blur_sigma", 0, 15),
        "canny_threshold_1": trial.suggest_int("canny_threshold_1", 0, 255),
        "canny_threshold_2": trial.suggest_int("canny_threshold_2", 0, 255),
        "clip_limit": trial.suggest_float("clip_limit", 0.0, 10.0),
        "tile_grid_size": trial.suggest_categorical("tile_grid_size", [8, 16, 32]),
        "brightness_alpha": trial.suggest_float("brightness_alpha", 0.0, 3.0),
        "brightness_beta": trial.suggest_int("brightness_beta", -100, 100),
        "sharpen_mode": trial.suggest_categorical("sharpen_mode", [None, "default", "gaussian", "laplacian"]),
        "pre_blur": trial.suggest_categorical("pre_blur", [True, False]),
        "kernel_size": trial.suggest_categorical("kernel_size", [i for i in range(2, 20) if i % 2 == 1]),
        "sigma_x": trial.suggest_int("sigma_x", 0, 15),
        "sigma_y": trial.suggest_int("sigma_y", 0, 15),
    }

    pipeline = SegmentationPipeline(**params)
    outputs = pipeline.segment(images, masks)

    # Calculate the mean IoU score
    iou = np.mean([out.iou for out in outputs])

    return iou

def generate_table_data(outputs: List[SegmentationOutput]) -> List[dict]:
    data = []
    for output in outputs:
        d = {}
        d["image"] = output.image
        d["mask"] = (output.ground_truth)
        d["prediction"] = output.prediction
        d["iou"] = output.iou
        d["id"] = output.image_path.split("/")[-1].split(".")[0]
        data.append(d)
    
    return data


def main(val_size: int, random_seed: int, n_trials: int) -> None:
    (x_train, y_train), (x_valid, y_valid) = get_train_val_split(val_size=val_size, random_seed=random_seed)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=n_trials)
    config = study.best_params
    config["val_size"] = val_size
    config["random_seed"] = random_seed
    config["n_trials"] = n_trials

    with wandb.init(project="coin-segmentation", tags=["classic-cv", "optuna"], config=config) as run:
        pipeline = SegmentationPipeline(**config)
        start = time.time()
        outputs_train = pipeline.segment(x_train, y_train)
        end = time.time()
        iou_train = np.mean([out.iou for out in outputs_train])
        throughput = len(x_train) / (end - start)

        outputs_val = pipeline.segment(x_valid, y_valid)
        iou_val = np.mean([out.iou for out in outputs_val])
            
        run.log({"latency": end - start, "throughput": throughput, "mIoU-train": iou_train, "mIoU-val": iou_val})
        data = generate_table_data(outputs_train+outputs_val)
        log_table(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument("--val-size", type=int, default=15, help="Validation set size")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-trials", type=int, default=300, help="Number of optuna trials")
    args = parser.parse_args()
    main(args.val_size, args.random_seed, args.n_trials)