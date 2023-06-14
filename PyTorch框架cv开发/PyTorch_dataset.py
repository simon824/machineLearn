import torch
import numpy as np
import cv2 as cv
import torchvision as tv
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt


'''
感知器
Sigmoid   ReLU
多层感知器
反向传播算法
    随机/世故梯度下降(online)
        基于单个样本、计算速度快、很容易收敛
    标准梯度下降(batch)
        基于整个数据集、计算速度慢、不容易收敛
    Mini-batch梯度下降
        小批量梯度下降、收敛速度快、算法精度高
常见数据集
    Pascal VOC
    COCO
基础数据集介绍
    torchvision.datasets包
    Mnist/Fashion-Mnist/CIFAR
    ImageNet/Pascal VOC/MS-COCO
    Cityscapes/FakeData
数据集的读取与加载
    torch.utils.data.Dataset的子集
    torch.utils.data.DataLoader加载数据集
模型训练
    超参数设置(批次(batch_size)/学习率(lr))
    优化器选择
    训练epoch/step  每训练batch_size个数据为一个step
# 卷积层输出大小： N=(W-F+2P)/S+1
'''
# ToTensor()把数据转为tensor类型，并且把数据转为0-1之间
# Normalize数据标准化，均值为0，方差为1，先计算出通道数据的均值与方差，然后将每个通道内的每个数据减去均值
# 再除以方差，得到归一化的结果
transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5,), (0.5,))])
# 手写数字识别数据集
train_ts = tv.datasets.MNIST(root="D:/datasets", train=True, download=True, transform=transform)
test_ts = tv.datasets.MNIST(root="D:/datasets", train=False, download=True, transform=transform)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)

# 简单定义神经网络模型
model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.LogSoftmax(dim=1)
)


def train_mnist():
    # 负对数似然损失 主对角线的和 / 主对角线元素的个数
    # NLLLoss  LogSoftmax
    loss_fn = nn.NLLLoss(reduction="mean")
    # 自适应优化
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 训练
    for s in range(5):
        print("run in epoch: %d" % s)
        for i, (x_train, y_train) in enumerate(train_dl):
            x_train = x_train.view(x_train.shape[0], -1)
            y_pred = model(x_train)
            train_loss = loss_fn(y_pred, y_train)
            if (i + 1) % 100 == 0:
                print(i + 1, train_loss.item())
            model.zero_grad()
            train_loss.backward()
            optimizer.step()
    # 预测
    total = 0
    correct_count = 0
    for test_images, test_labels in test_dl:
        for i in range(len(test_labels)):
            image = test_images[i].view(1, 784)
            with torch.no_grad():
                pred_labels = model(image)
            plabels = torch.exp(pred_labels)
            probs = list(plabels.numpy()[0])
            pred_label = probs.index(max(probs))
            true_label = test_labels.numpy()[i]
            if pred_label == true_label:
                correct_count += 1

            total += 1
    print("total acc: %.2f" % (correct_count / total))
    # 保存整个模型   使用torch.load加载
    # 保存推理模型 torch.save(model.state_dict(), "D:/models/nn_mnist_model.pt")
    # 加载推理模型 model.load_state_dict(torch.load())
    # model.eval() 预测/验证时使用，model.train() 训练时使用
    # torch.save(model, "D:/models/nn_mnist_model.pt")
    torch.save(model.state_dict(), "D:/models/nn_mnist_state_dict_model.pt")


if __name__ == '__main__':
    # train_mnist()
    print("Model's state_dict: ")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    model.load_state_dict(torch.load("D:/models/nn_mnist_state_dict_model.pt"))
    model.eval()  # dropout bn
    image = cv.imread("D:/images/9_99.png", cv.IMREAD_GRAYSCALE)
    cv.imshow("input", image)
    # 必须和训练时生成的数据步骤保持一致
    img_f = np.float32(image) / 255.0 - 0.5
    img_f = img_f / 0.5
    img_f = np.reshape(img_f, (1, 784))
    pred_labels = model(torch.from_numpy(img_f))
    # 由于计算过程中使用的是LogSoftmax   log = ln, 这里需要换算回去
    plabels = torch.exp(pred_labels)
    probs = list(plabels.detach().numpy()[0])
    pred_label = probs.index(max(probs))
    print("predict digit number: ", pred_label)
    cv.waitKey(0)
    cv.destroyAllWindows()