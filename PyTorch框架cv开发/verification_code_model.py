import torch

from torch import nn
from torch.utils.data import DataLoader
from verification_code_dataset import output_nums
from verification_code_dataset import CapchaDataset


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


class CapchaResNet(nn.Module):
    def __init__(self):
        super(CapchaResNet, self).__init__()
        self.cnn_layers = nn.Sequential(
            # 卷积层 128*32*3
            ResidualBlock(3, 32, 1),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4, output_nums())
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        out = x.view(-1, 4 * 256)
        out = self.fc_layers(out)
        return out


if __name__ == '__main__':
    model = CapchaResNet()
    print(model)
    gpu_on = torch.cuda.is_available()
    if gpu_on:
        model.cuda()
    ds = CapchaDataset("D:/datasets/Verification_Code/samples")
    num_train_samples = ds.num_of_samples()
    bs = 16

    dataloader = DataLoader(ds, batch_size=bs, shuffle=True)

    # 训练模型次数
    num_epochs = 25
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()

    # 损失函数 one-hot编码对应的损失函数
    mul_loss = torch.nn.MultiLabelSoftMarginLoss()
    index = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batch in enumerate(dataloader):
            image_batch, oh_labels = sample_batch['image'], sample_batch['encode']
            if gpu_on:
                image_batch, oh_labels = image_batch.cuda(), oh_labels.cuda()

            optimizer.zero_grad()
            m_label_out = model(image_batch)
            oh_labels = torch.autograd.Variable(oh_labels.float())

            loss = mul_loss(m_label_out, oh_labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if index % 100 == 0:
                print("step: {}\tTraining Loss: {:.6f}".format(index, loss.item()))
            index += 1
        train_loss = train_loss / num_train_samples
        print("Epoch: {}\tTraining Loss: {:.6f}".format(epoch, train_loss))
    model.eval()
    torch.save(model, "D:/models/verification_code_model.pt")


