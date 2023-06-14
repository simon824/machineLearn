import torch
from torchvision import models
from torch.utils.data import DataLoader

from pytorch_vision_detection import utils
from pytorch_vision_detection.engine import train_one_epoch
from mask_rcnn_dataset import PennFudanDataset


def main_demo():
    train_gpu = torch.cuda.is_available()
    # 背景 + 行人
    num_classes = 2
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    device = torch.device("cuda:0")
    model.to(device)

    dataset = PennFudanDataset("D:/datasets/PennFudanPed")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=utils.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    num_epochs = 10
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
    torch.save(model.state_dict(), "D:/models/mask_rcnn_model.pt")


if __name__ == '__main__':
    main_demo()

