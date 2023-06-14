import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_vision_detection import transforms
from PIL import Image


class PennFudanDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, "PedMasks"))))

    def __len__(self):
        return len(self.imgs)

    def num_of_samples(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root_dir, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target
if __name__ == '__main__':
    ds = PennFudanDataset("D:/datasets/PennFudanPed")
    for i in range(len(ds)):
        img, target = ds[i]
        print(i, img.size(), target)
        device = torch.device("cuda:0")
        boxes = target["boxes"]
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        targets = [{k: v.to(device) for k, v in t.items()} for t in [target]]
        if i == 3:
            break






