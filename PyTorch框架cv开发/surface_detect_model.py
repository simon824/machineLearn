import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from surface_detect_dataset import SurfaceDefectDataset

train_gpu = torch.cuda.is_available()


class SurfaceDefectResNet(nn.Module):
    def __init__(self):
        super(SurfaceDefectResNet, self).__init__()
        # pretrained=True下载模型
        self.cnn_layers = models.resnet18(pretrained=False)
        # 手动导入模型
        self.cnn_layers.load_state_dict(torch.load("D:/models/resnet18-f37072fd.pth"))
        num_ftrs = self.cnn_layers.fc.in_features
        self.cnn_layers.fc = torch.nn.Linear(num_ftrs, 6)

    def forward(self, x):
        out = self.cnn_layers(x)
        return out


def demo():
    model = SurfaceDefectResNet()
    print(model)

    if train_gpu:
        model.cuda()

    ds = SurfaceDefectDataset("D:/datasets/enu_surface_defect/train")
    num__train_samples = ds.num_of_samples()
    dataloader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

    num_epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()

    cross_loss = torch.nn.CrossEntropyLoss()
    index = 0
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batch in enumerate(dataloader):
            image_batch, label_batch = sample_batch["image"], sample_batch["defect"]
            if train_gpu:
                image_batch = image_batch.cuda()
                label_batch = label_batch.cuda()
            optimizer.zero_grad()

            label_out = model(image_batch)
            label_batch = label_batch.long()

            loss = cross_loss(label_out, label_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if index % 100 == 0:
                print("step {}, Training Loss {:.6f}".format(index, loss.item()))
            index += 1
        train_loss = train_loss / num__train_samples

        print("Epoch {}, Training Loss {:.6f}".format(epoch, train_loss))
    model.eval()
    torch.save(model.state_dict(), "D:/models/surface_defect_model.pt")


if __name__ == '__main__':
    demo()
