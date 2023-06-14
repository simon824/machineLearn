import torch
import numpy as np
from torch import nn
from vehicle_attrs_dataset import VehicleAttrsDataset
from torch.utils.data import DataLoader


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.skip = nn.Sequential()

        # 步长为2的话会下采样，也就是降采样
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        identity = self.skip(x)
        out += identity
        out = nn.functional.relu(out)
        return out


class VehicleAttributeResNet(nn.Module):
    def __init__(self):
        super(VehicleAttributeResNet, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2),

            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2)
        )
        # 全局最大池化
        self.global_max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.color_fc_layers = nn.Sequential(
            nn.Linear(128, 7),
            nn.Sigmoid()
        )
        self.type_fc_layers = nn.Sequential(
            nn.Linear(128, 4)
        )

    def forward(self, x):

        x = self.cnn_layers(x)
        B, C, H, W = x.size()
        out = self.global_max_pooling(x).view(B, -1)
        out_color = self.color_fc_layers(out)
        out_type = self.type_fc_layers(out)
        return out_color, out_type


if __name__ == '__main__':
    model = VehicleAttributeResNet()
    gpu_on = torch.cuda.is_available()
    if gpu_on:
        model.cuda()
    ds = VehicleAttrsDataset("D:/datasets/vehicle_attrs_dataset")
    num_train_samples = ds.nums_of_sample()
    dataloader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)

    num_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    model.train()
    cross_loss = torch.nn.CrossEntropyLoss()
    index = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batch in enumerate(dataloader):
            image_batch, color_batch, type_batch = sample_batch["image"], sample_batch["color"], sample_batch["type"]
            if gpu_on:
                image_batch = image_batch.cuda()
                color_batch = color_batch.cuda()
                type_batch = type_batch.cuda()
            optimizer.zero_grad()

            color_out, type_out = model(image_batch)
            color_batch = color_batch.long()
            type_batch = type_batch.long()

            loss = cross_loss(color_out, color_batch) + cross_loss(type_out, type_batch)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if index % 100 == 0:
                print("step {}\t Training Loss {:.6f}".format(index, loss.item()))
            index += 1
    model.eval()
    torch.save(model, "D:/models/vehicle_attrs_model.pt")