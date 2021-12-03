import argparse

import numpy as np
import torch
from tqdm import tqdm as tqdm

from dataset import Dataset
from utils.test_utils import ensemble_predict
from utils.train_util import (
    get_preprocessing,
    get_tta_augmentation,
    get_validation_augmentation,
)

parser = argparse.ArgumentParser(description="Testing Segmentation Model")
parser.add_argument("checkpoints", nargs="+")

args = parser.parse_args()

DATASET_DIR = "data/echocardiography"
DEVICE = "cuda"
IMG_HEIGHT = 448
IMG_WIDTH = 640

models = []
for i in range(len(args.checkpoints)):
    models.append(torch.load(args.checkpoints[i]))

dataset = args.checkpoints[0].split("/")[-1].split("_")[0]
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

pred_ensemble = ensemble_predict(
    models, test_dataset, tta=True, test_dataset_flipped=test_dataset_flipped
)

gt = np.zeros((len(test_dataset), IMG_HEIGHT, IMG_WIDTH, 1))
for i in tqdm(range(len(test_dataset))):
    # normal test
    _, gt_mask = test_dataset[i]
    gt[i] = gt_mask.transpose(1, 2, 0)
dice_total = 0
for i in range(gt.shape[0]):
    pr_mask = pred_ensemble[i]
    gt_mask = gt[i]
    dice = np.sum(pr_mask[gt_mask == 1]) * 2.0 / (np.sum(pr_mask) + np.sum(gt_mask))
    dice_total += dice
dice = dice_total / gt.shape[0]

print("\nF-Score 성능")
print(f"앙상블 + TTA 성능\t:\t{dice}")
