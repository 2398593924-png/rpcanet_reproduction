import os
import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# 官方 utils/data.py 中 STARE 的归一化常数
STARE_MEAN = 98.16558837890625
STARE_STD = 52.33002853393555


class augmentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input.copy(), target.copy()


def random_crop(img, mask, patch_size, pos_prob=None):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w = img.shape

    cur_prob = random.random()
    if pos_prob == None or cur_prob > pos_prob or mask.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask > 0)
        if len(loc[0]) <= 1:
            idx = 0
        else:
            idx = random.randint(0, len(loc[0]) - 1)
        h_start = random.randint(max(0, loc[0][idx] - patch_size), min(loc[0][idx], h - patch_size))
        w_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], w - patch_size))

    h_end = h_start + patch_size
    w_end = w_start + patch_size
    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch


def PadImg(img, times = 32):
    if len(img.shape) == 2:
        h, w = img.shape
    elif len(img.shape) == 3:
        _, h, w = img.shape
    else:
        raise ValueError("Unexpected number of dimensions in image")
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img


def Normalized(img, mean = STARE_MEAN, std = STARE_STD):
    return (img - mean) / std


class VSDataset(Dataset):
    def __init__(self, dataset_path, mode = 'train', patch_size = 256, pos_prob = 0.5):
        self.image_path = os.path.join(dataset_path, "images")
        self.label_path = os.path.join(dataset_path, "labels")

        self.images = []
        self.labels = []
        for f in os.listdir(os.path.join(dataset_path, "images")):
            self.images.append(f)
        for f in os.listdir(os.path.join(dataset_path, "labels")):
            self.labels.append(f)

        self.images.sort()
        self.labels.sort()

        self.size = len(self.images)

        self.mode = mode
        self.patch_size = patch_size
        self.pos_prob = pos_prob
        self.tranform = augmentation()

        self.mean = STARE_MEAN
        self.std = STARE_STD

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_path, self.images[idx])).convert('I')
        mask = Image.open(os.path.join(self.label_path, self.labels[idx])).convert('I')

        img = np.array(img, dtype = np.float32)
        img = Normalized(img, self.mean, self.std)
        mask = np.array(mask, dtype = np.float32) / 255.0

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        if self.mode == 'train':
            img, mask = random_crop(img, mask, self.patch_size, pos_prob = self.pos_prob)
            img, mask = self.tranform(img, mask)
        else:
            img = PadImg(img)
            mask = PadImg(mask)

        img = img[np.newaxis, :]
        mask = mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        return img, mask
