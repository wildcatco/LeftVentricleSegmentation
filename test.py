import argparse

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm

from dataset import Dataset
from utils.train_util import (
    get_preprocessing,
    get_tta_augmentation,
    get_validation_augmentation,
)


parser = argparse.ArgumentParser(description="Testing Segmentation Model")
parser.add_argument("checkpoint")

args = parser.parse_args()

DATASET_DIR = "data/echocardiography"
DEVICE = "cuda"

model = torch.load(args.checkpoint)

dataset = args.checkpoint.split("/")[-1].split("_")[0]
test_dataset = Dataset(
    mode="validation",
    view=dataset,
    data_dir=DATASET_DIR,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(),
)
test_dataset_flipped = Dataset(
    mode="validation",
    view=dataset,
    data_dir=DATASET_DIR,
    augmentation=get_tta_augmentation(),
    preprocessing=get_preprocessing(),
)

dice_total = 0
dice_total_tta = 0
for i in tqdm(range(len(test_dataset))):
    # normal test
    image, gt_mask = test_dataset[i]
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = model.predict(x_tensor)
    pr_mask = pr_mask.squeeze().unsqueeze(-1).cpu().numpy()

    # augmented test (flip)
    image_flipped, gt_mask_flipped = test_dataset_flipped[i]
    x_tensor_flipped = torch.from_numpy(image_flipped).to(DEVICE).unsqueeze(0)
    pr_mask_flipped = model.predict(x_tensor_flipped)
    pr_mask_flipped = pr_mask_flipped.squeeze().unsqueeze(-1).cpu().numpy()

    pr_mask_tta = (pr_mask + np.flip(pr_mask_flipped, axis=1)) / 2
    pr_mask_tta = pr_mask_tta.squeeze(-1).round()
    gt_mask = gt_mask.squeeze(0)

    pr_mask = pr_mask.round()
    dice = np.sum(pr_mask[gt_mask == 1]) * 2.0 / (np.sum(pr_mask) + np.sum(gt_mask))
    dice_total += dice
    dice_tta = (
        np.sum(pr_mask_tta[gt_mask == 1])
        * 2.0
        / (np.sum(pr_mask_tta) + np.sum(gt_mask))
    )
    dice_total_tta += dice_tta
dice = dice_total / len(test_dataset)
dice_tta = dice_total_tta / len(test_dataset)


print("\nF-Score 성능")
print(f"기존 성능\t\t:\t{dice}")
print(f"TTA 성능 (only flip)\t:\t{dice_tta}")
print(f"성능 향상\t\t:\t{(dice_tta - dice)}")
