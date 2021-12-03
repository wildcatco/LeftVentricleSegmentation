import argparse
import os
import pickle

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.utils.train import ValidEpoch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from dataset import Dataset
from utils.train_util import (
    TrainEpoch,
    get_preprocessing,
    get_training_augmentation,
    get_validation_augmentation,
)


parser = argparse.ArgumentParser(description="Training Segmentation Model")
parser.add_argument("--dataset", choices=["A2C", "A4C"])
parser.add_argument("--encoder", choices=["se_resnext50_32x4d"])
parser.add_argument("--num-workers", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--max-epochs", type=int)

args = parser.parse_args()

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("weights", exist_ok=True)

DATASET_DIR = "data/echocardiography"
DEVICE = "cuda"

model = smp.DeepLabV3Plus(
    encoder_name=args.encoder,
    encoder_weights="imagenet",
    classes=1,
    activation="sigmoid",
)

if args.encoder == "se_resnext50_32x4d":
    model.encoder.layer0.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

train_dataset = Dataset(
    mode="train",
    view=args.dataset,
    data_dir=DATASET_DIR,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(),
)

valid_dataset = Dataset(
    mode="validation",
    view=args.dataset,
    data_dir=DATASET_DIR,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=args.num_workers,
)
valid_loader = DataLoader(
    valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.Fscore(threshold=0.5),
]
optimizer = torch.optim.AdamW(
    params=model.parameters(), lr=args.lr, weight_decay=args.wd
)
T_0 = int((len(train_dataset) / args.batch_size * args.max_epochs))
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, last_epoch=-1)

train_epoch = TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    scheduler=scheduler,
    verbose=True,
)
valid_epoch = ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

max_score = 0
max_score_per_cycle = 0

training_logs = dict(
    lr=[],
    train_iou=[],
    valid_iou=[],
    train_fscore=[],
    valid_fscore=[],
    train_loss=[],
    valid_loss=[],
)

model_name = "_".join(
    [
        args.dataset,
        args.encoder,
        "lr",
        str(args.lr),
        "wd",
        str(args.wd),
        "epochs",
        str(args.max_epochs),
    ]
)

for i in range(0, args.max_epochs):

    print("\nEpoch: {}".format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    training_logs["lr"].append(optimizer.param_groups[0]["lr"])
    training_logs["train_iou"].append(train_logs["iou_score"])
    training_logs["valid_iou"].append(valid_logs["iou_score"])
    training_logs["train_fscore"].append(train_logs["fscore"])
    training_logs["valid_fscore"].append(valid_logs["fscore"])
    training_logs["train_loss"].append(train_logs["dice_loss"])
    training_logs["valid_loss"].append(valid_logs["dice_loss"])

    if max_score < valid_logs["iou_score"]:
        max_score = valid_logs["iou_score"]
        torch.save(model, f"results/checkpoints/{model_name}.pth")
        print("Model saved!")

with open(f"logs/{model_name}", "wb") as f:
    pickle.dump(training_logs, f)
