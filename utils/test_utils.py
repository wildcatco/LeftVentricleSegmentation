import copy

import numpy as np
import torch
from tqdm import tqdm as tqdm


def predict(model, test_dataset, tta=False, test_dataset_flipped=None, device="cuda"):
    out = np.zeros((len(test_dataset), 448, 640, 1))
    for i in tqdm(range(len(test_dataset))):
        # normal test
        image, gt_mask = test_dataset[i]
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().unsqueeze(-1).cpu().numpy()

        if not tta:
            out[i] = pr_mask
        else:
            # augmented test (flip)
            image_flipped, _ = test_dataset_flipped[i]
            x_tensor_flipped = torch.from_numpy(image_flipped).to(device).unsqueeze(0)
            pr_mask_flipped = model.predict(x_tensor_flipped)
            pr_mask_flipped = pr_mask_flipped.squeeze().unsqueeze(-1).cpu().numpy()

            pr_mask_tta = (pr_mask + np.flip(pr_mask_flipped, axis=1)) / 2
            out[i] = pr_mask_tta
    return out


def ensemble_predict(models, test_dataset, tta=False, test_dataset_flipped=None):
    for i, model in enumerate(models):
        pred = predict(model, test_dataset, tta, test_dataset_flipped)
        if i == 0:
            pred_ensemble = copy.deepcopy(pred)
        else:
            pred_ensemble += pred
    return (pred_ensemble / len(models)).round()
