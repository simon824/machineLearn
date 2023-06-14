import torch
from cnn_mnist import CNN_MNIST
from emotion_model import EmotionResNet, ResidualBlock
from torchvision import models

import pyttsx3

engine = pyttsx3.init()
engine.say("9")
engine.runAndWait()


def demo():
    # model = EmotionResNet()

    # model.load_state_dict(torch.load("D:/models/emotion_model.pt"))
    # model.eval()
    # print(model)
    num_classes = 3
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes,
                                                     pretrained_backbone=True)
    model.load_state_dict(torch.load("D:/models/faster_rcnn_model.pt"))
    print(model)
    dummy_input1 = torch.randn(1, 3, 400, 600)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.onnx.export(model, (dummy_input1.to(device)), "D:/netron_models/faster_rcnn_model.onnx", verbose=True)


if __name__ == '__main__':
    demo()