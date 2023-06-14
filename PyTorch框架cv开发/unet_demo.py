import os

import torch
import cv2 as cv
import numpy as np
from unet_model import UNetModel

def unet_defect_demo():
    cnn_model = torch.load("D:/models/unet_road_model.pt")
    root_dir = "D:/datasets/CrackForest-dataset/test"
    fileNames = os.listdir(root_dir)
    for f in fileNames:
        image = cv.imread(os.path.join(root_dir, f), cv.IMREAD_GRAYSCALE)
        h, w = image.shape
        img = np.float32(image) / 255.0
        img = np.expand_dims(img, 0)
        x_input = torch.from_numpy(img).view(1, 1, h, w)
        probs = cnn_model(x_input.cuda())
        label_out = probs.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
        _, output = label_out.data.max(dim=1)
        output[output > 0] = 255
        predic_ = output.view(h, w).cpu().detach().numpy()
        print(predic_.shape)
        cv.imshow("input", image)
        result = cv.resize(np.uint8(predic_), (w, h))
        contours, h = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bgr_img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        cv.drawContours(bgr_img, contours, -1, (0, 0, 255), -1)
        cv.imshow("output", bgr_img)
        cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    unet_defect_demo()




