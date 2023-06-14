import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2 as cv

colors = ["white", "gray", "yellow", "red", "green", "blue", "black", "van"]
types = ["car", "bus", "truck", "van"]


class VehicleAttrsDataset(Dataset):
    def __init__(self, root_dir):
        super(VehicleAttrsDataset, self).__init__()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                             transforms.Resize((72, 72))])
        self.vehicle_colors = []
        self.images = []
        self.vehicle_types = []
        index = 0
        for file_name in os.listdir(root_dir):
            color, vehicle_type = file_name.split("_")[:2]
            self.vehicle_types.append(np.float32(types.index(vehicle_type)))
            self.vehicle_colors.append(np.float32(colors.index(color)))
            path = os.path.join(root_dir, file_name)
            self.images.append(path)
            index += 1

    def __len__(self):
        return len(self.images)

    def nums_of_sample(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
        else:
            image_path = self.images[idx]
        img = cv.imread(image_path)
        img = self.transform(img)
        sample = {"image": img, "color": self.vehicle_colors[idx], "type": self.vehicle_types[idx]}
        return sample


if __name__ == '__main__':
    ds = VehicleAttrsDataset("D:/datasets/vehicle_attrs_dataset")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample["image"].size(), sample["color"])
        if i == 3:
            break
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batch in enumerate(dataloader):
        print(i_batch, sample_batch["image"].size(), sample_batch["type"])
        break