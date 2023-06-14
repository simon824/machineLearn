import os

import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 5


def output_nums():
    return MAX_CAPTCHA * ALL_CHAR_SET_LEN


def encode(a):
    onehot = [0] * ALL_CHAR_SET_LEN
    idx = ALL_CHAR_SET.index(a)
    onehot[idx] = 1
    return onehot


class CapchaDataset(Dataset):
    def __init__(self, root_dir):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             transforms.Resize((32, 128))])
        img_files = os.listdir(root_dir)
        self.txt_labels = []
        self.encodes = []
        self.images = []
        for file_name in img_files:
            label = file_name[:-4]
            label_oh = []
            for i in label:
                label_oh += encode(i)
            self.images.append(os.path.join(root_dir, file_name))
            self.encodes.append(np.array(label_oh))
            self.txt_labels.append(label)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
        else:
            image_path = self.images[idx]
        img = cv2.imread(image_path)
        img = self.transform(img)
        h, w, c = img.shape
        # resize宽高
        # img = cv2.resize(img, (128, 32))
        # img = (np.float32(img) / 255.0 - 0.5) / 0.5
        # img = img.transpose((2, 0, 1))
        # img = torch.permute(img, (2, 0, 1))
        # sample = {"image": torch.from_numpy(img), "encode": self.encodes[idx], "label": self.txt_labels[idx]}
        sample = {"image": img, "encode": self.encodes[idx], "label": self.txt_labels[idx]}
        return sample


if __name__ == '__main__':
    ds = CapchaDataset("D:/datasets/Verification_Code/samples")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample["image"].size(), sample["label"], sample["encode"].shape)
        print(sample["encode"].reshape(5, -1))
        if i == 3:
            break

    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batch in enumerate(dataloader):
        print(i_batch, sample_batch["image"].size(), sample_batch["label"])
        break



