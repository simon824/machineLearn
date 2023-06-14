import torch
import numpy as np
import cv2 as cv
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.images = []
        self.masks = []
        files = os.listdir(image_dir)
        mask_files = os.listdir(mask_dir)
        for i in range(len(mask_files)):
            img_file = os.path.join(image_dir, files[i])
            mask_file = os.path.join(mask_dir, mask_files[i])
            self.images.append(img_file)
            self.masks.append(mask_file)
    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        else:
            image_path = self.images[idx]
            mask_path = self.masks[idx]
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        # 输入图像
        img = np.float32(img) / 255.0
        img = np.expand_dims(img, 0)
        # 目标标签0-1
        mask[mask <= 128] = 0
        mask[mask > 128] = 1
        mask = np.expand_dims(mask, 0)
        sample = {"image": torch.from_numpy(img), "mask": torch.from_numpy(mask)}
        return sample

if __name__ == '__main__':
    image_dir = "D:/datasets/CrackForest-dataset/image"
    mask_dir = "D:/datasets/CrackForest-dataset/mask"
    ds = SegmentationDataset(image_dir, mask_dir)
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batch in enumerate(dataloader):
        print(i_batch, sample_batch["image"].size(), sample_batch["mask"])
        break


