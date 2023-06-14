import torch
import numpy as np
import cv2 as cv
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


emotion_labels = ["neutral","anger","disdain","disgust","fear","happy","sadness","surprise"]


class EmotionsDataset(Dataset):
    def __init__(self, root_dir):
        super(EmotionsDataset, self).__init__()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                             transforms.Resize((64, 64))])
        self.emotions = []
        self.images = []
        index = 0
        for file_name in os.listdir(root_dir):
            emotion = file_name.split("_")[0]
            self.emotions.append(np.float32(np.int32(emotion)))
            self.images.append(os.path.join(root_dir, file_name))
            index += 1

    def __len__(self):
        return len(self.images)

    def num_of_sample(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.images[idx]
        img = cv.imread(image_path)
        sample = {"image": self.transform(img), "emotion": self.emotions[idx]}
        return sample


if __name__ == '__main__':
    ds = EmotionsDataset("D:/datasets/emotion_dataset")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample["image"].size(), sample["emotion"])
        if i == 3:
            break
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batch in enumerate(dataloader):
        print(i_batch, sample_batch["image"].size(), sample_batch["emotion"])
        break



