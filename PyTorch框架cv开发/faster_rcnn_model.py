import torch
from torchvision import models
from pytorch_vision_detection.engine import train_one_epoch
from faster_rcnn_dataset import PetDataset
from pytorch_vision_detection import utils
from torch.utils.data import DataLoader


def train_main():
    train_gpu = torch.cuda.is_available()

    # progress=True, num_classes=num_classes, pretrained_backbone=True
    num_classes = 3
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)
    device = torch.device("cuda:0")
    model.to(device)
    dataset = PetDataset("D:/datasets/pet_data")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=utils.collate_fn)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epoch = 20
    for epoch in range(num_epoch):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
    torch.save(model.state_dict(), "D:/models/faster_rcnn_model.pt")


if __name__ == '__main__':
    train_main()




