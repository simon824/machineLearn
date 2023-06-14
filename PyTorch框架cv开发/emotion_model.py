import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from emotions_dataset import EmotionsDataset


train_gpu = torch.cuda.is_available()
num_iden = 0


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.skip = nn.Sequential()
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
        if identity.any():
            global num_iden
            num_iden += 1
        out += identity
        out = nn.functional.relu(out)
        return out


class EmotionResNet(nn.Module):
    def __init__(self):
        super(EmotionResNet, self).__init__()
        self.cnn_layers = nn.Sequential(
            ResidualBlock(3, 32, 1),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 2),
            ResidualBlock(256, 8, 1),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        B, C, H, W = x.shape
        out = x.view(B, -1)
        return out


if __name__ == '__main__':
    model = EmotionResNet()
    if train_gpu:
        model.cuda()

    ds = EmotionsDataset("D:/datasets/emotion_dataset")
    num_train_samples = ds.num_of_sample()
    dataloader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)
    num_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()

    cross_loss = nn.CrossEntropyLoss()
    index = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batch in enumerate(dataloader):
            image_batch, emotion_batch = sample_batch["image"], sample_batch["emotion"]
            if train_gpu:
                image_batch = image_batch.cuda()
                emotion_batch = emotion_batch.cuda()
            emotion_out = model(image_batch)
            emotion_batch = emotion_batch.long()
            loss = cross_loss(emotion_out, emotion_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if index % 100 == 0:
                print("step: {}\tTraining Loss {:.6f}".format(index, loss.item()))
            index += 1
        train_loss = train_loss / num_train_samples
        print("Epoch {}\tTraining Loss {:.6f}".format(epoch, train_loss))
    model.eval()
    torch.save(model, "D:/models/emotion_model.pt")
    print(num_iden)


