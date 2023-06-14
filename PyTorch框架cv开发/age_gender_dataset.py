import os
import cv2 as cv
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

max_age = 116.0


class AgeGenderDataset(Dataset):
    def __init__(self, root_dir):
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Normalize(mean=[0.5,], std=[0.5,]),
        #                                      transforms.Resize((64, 64)),
        #                                      ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        img_files = os.listdir(root_dir)
        nums = len(img_files)
        # 0-116岁
        self.ages = []
        # 0 male  1 female
        self.genders = []
        self.images = []
        index = 0
        for file_name in img_files:
            age_gender_group = file_name.split("_")
            age_ = age_gender_group[0]
            gender_ = age_gender_group[1]
            self.genders.append(np.float32(gender_))
            # 将年龄转化为0-1之间的值
            self.ages.append(np.float32(age_)/ np.float32(max_age))
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
        else:
            image_path = self.images[idx]
        img = cv.imread(image_path)  # BGR
        # h, w, c = img.shape
        img = cv.resize(img, (64, 64))
        img = (np.float32(img) / 255.0 - 0.5) / 0.5
        # HWC to CHW
        img = img.transpose((2, 0, 1))
        sample = {"image": torch.from_numpy(img), "age": self.ages[idx], "gender": self.genders[idx]}
        return sample


if __name__ == '__main__':
    ds = AgeGenderDataset("D:/datasets/UTKFace")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample["image"].size(), sample["age"])
        if i == 3:
            break
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batch in enumerate(dataloader):
        print(i_batch, sample_batch["image"].size(), sample_batch["gender"])
        break


