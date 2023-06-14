import torch
import os
import torchvision as tv
from torch.utils.data import DataLoader
from torch import nn

os.environ["CUDA_VISIBLE_DEVICE"] = '1'
transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5,), (0.5,))])

train_ts = tv.datasets.MNIST(root="D:/datasets", train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root="D:/datasets", train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=32, shuffle=True, drop_last=False)


class CNN_MNIST(nn.Module):
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(7*7*32, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.view(-1, 7*7*32)
        out = self.fc_layers(out)
        return out

if __name__ == '__main__':
    model = CNN_MNIST().cuda()
    print("CNN_MNIST STATE_DICT: ")
    for parameter in model.state_dict():
        print(parameter)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for s in range(5):
        print("run in epoch : %d" % s)
        for i, (x_train, y_train) in enumerate(train_dl):
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            y_pred = model.forward(x_train)
            train_loss = loss(y_pred, y_train)
            if (i + 1) % 100 == 0:
                print(i + 1, train_loss.item())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "D:/models/cnn_mnist_model.pt")
    model.eval()

    total = 0
    correct_count = 0
    for test_images, test_labels in test_dl:
        pred_labels = model(test_images.cuda())
        predicted = torch.max(pred_labels, 1)[1]
        correct_count += int((predicted == test_labels.cuda()).sum())
        total += len(test_labels)

    print("准确度: %.2f" % (correct_count / total))
