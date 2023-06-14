import torch
import cv2 as cv
from torch import nn
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image = cv.imread("D:/images/cat1.jpg")
cv.imshow("input", image)
result = transform(image)

result = result.numpy().transpose(1, 2, 0)
cv.imshow("pre-process", result)
cv.waitKey(0)
cv.destroyAllWindows()
