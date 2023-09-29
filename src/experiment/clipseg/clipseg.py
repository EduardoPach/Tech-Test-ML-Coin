import time
import argparse
from typing import Tuple, List

import cv2
import wandb
import torch
import optuna
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from src.experiment.utils import get_train_val_split, calculate_iou, log_table


def load_model(model_id: str) -> Tuple[CLIPSegForImageSegmentation, CLIPSegProcessor]:
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id)

    return model, processor

def model_inference(model_id: str, images: List[Image.Image], prompt: str, threshold: int) -> List[np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(model_id)
    model.to(device)

    inputs = processor(text=[prompt]*len(images), images=images, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits

    pred_masks = [(pred.sigmoid().cpu().numpy()*255).astype(np.uint8) for pred in preds]
    pred_masks = [cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1] for mask in pred_masks]
    pred_masks = [cv2.resize(mask.copy(), (img.size[0], img.size[1]), interpolation=cv2.INTER_NEAREST) for mask, img in zip(pred_masks, images)]

    return pred_masks

def objective(trial: optuna.Trial, images: List[Image.Image], masks: List[np.ndarray], prompt: str) -> float:
    model_id = trial.suggest_categorical("model_id", ["CIDAS/clipseg-rd64-refined", "CIDAS/clipseg-rd16"])
    threshold = trial.suggest_int("threshold", 0, 255)

    pred_masks = model_inference(model_id, images, prompt, threshold)
    ious = [calculate_iou(mask, pred_mask) for mask, pred_mask in zip(masks, pred_masks)]

    return np.mean(ious)

def generate_table_data(
        images: List[Image.Image], 
        ground_truths: List[np.ndarray], 
        predictions: List[np.ndarray], 
        ious: List[float], 
        paths: List[str]
    ) -> List[dict]:
    data = []
    for image, ground_truth, prediction, iou, path in zip(images, ground_truths, predictions, ious, paths):
        d = {}
        d["image"] = np.array(image)
        d["mask"] = ground_truth
        d["prediction"] = prediction
        d["iou"] = iou
        d["id"] = path.split("/")[-1].split(".")[0]
        data.append(d)
    
    return data

def main(args: argparse.Namespace) -> None:
    # Getting data
    (x_train, y_train), (x_val, y_val) = get_train_val_split(args.val_size, args.random_seed)

    images_train = [Image.open(image) for image in x_train]
    masks_train = [cv2.imread(mask, cv2.IMREAD_GRAYSCALE) for mask in y_train]
    images_val = [Image.open(image) for image in x_val]
    masks_val = [cv2.imread(mask, cv2.IMREAD_GRAYSCALE) for mask in y_val]

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, images_train, masks_train, args.prompt), n_trials=args.n_trials)
    config = study.best_params
    config["val_size"] = args.val_size
    config["random_seed"] = args.random_seed
    config["n_trials"] = args.n_trials
    config["prompt"] = args.prompt
    config["is_gpu_available"] = torch.cuda.is_available()


    # with wandb.init(project="coin-segmentation", tags=["clipseg", "optuna"], config=config):
    #     start = time.time()
    #     outputs_train = model_inference(config["model_id"], images_train, args.prompt, config["threshold"])
    #     end = time.time()

    #     ious_train = [calculate_iou(mask, pred_mask) for mask, pred_mask in zip(masks_train, outputs_train)]
    #     iou_train = np.mean(ious_train)
    #     throughput = len(images_train) / (end - start)
    #     latency = (end - start) / len(images_train)

    #     outputs_val = model_inference(config["model_id"], images_val, args.prompt, config["threshold"])
    #     ious_val = [calculate_iou(mask, pred_mask) for mask, pred_mask in zip(masks_val, outputs_val)]
    #     iou_val = np.mean(ious_val)

        
    #     wandb.log(
    #         {
    #             "mIoU-train": iou_train,
    #             "mIoU-val": iou_val,
    #             "latency": latency, 
    #             "throughput": throughput, 
    #         }
    #     )

    #     images = images_train + images_val
    #     predictions = outputs_train + outputs_val
    #     ground_truths = masks_train + masks_val
    #     ious = ious_train + ious_val
    #     paths = x_train + x_val

    #     data = generate_table_data(images, ground_truths, predictions, ious, paths)

    #     log_table(data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna experiment for CLIPSeg")
    parser.add_argument("--val-size", type=int, default=140, help="Validation split")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", type=str, default="all circle shaped objects that resemble a coin", help="Prompt")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials")
    args = parser.parse_args()

    main(args)

