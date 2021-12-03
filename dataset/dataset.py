import glob
import os

import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(self, mode, view, data_dir, augmentation=None, preprocessing=None):
        self.data_dir = data_dir
        self.mode = mode
        self.view = view
        self.dataset = self._load_dataset(mode, view, data_dir)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def _load_dataset(self, mode, view, data_dir):
        images = sorted(glob.glob(os.path.join(data_dir, mode, view, "*.png")))
        masks = sorted(glob.glob(os.path.join(data_dir, mode, view, "*.npy")))
        dataset = dict(images=images, masks=masks)
        return dataset

    def __getitem__(self, i):
        image = cv2.imread(self.dataset["images"][i])[:, :, :1]
        mask = np.load(self.dataset["masks"][i])
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.dataset["images"])
