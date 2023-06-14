import torch
import numpy as np
import cv2 as cv
from torchvision import models
from torchvision import transforms

model = models.detection.maskrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2, pretrained_backbone=True)
model.load_state_dict(torch.load("D:/models/mask_rcnn_pedestrian_model.pt"))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

train_gpu = torch.cuda.is_available()

if train_gpu:
    model.cuda()

def object_detection_demo():
    frame = cv.imread("D:/images/gaoyy.png")
    cv.imshow("input", frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    blob = transform(frame)
    c, h, w = blob.shape
    print(c, h, w)
    input_x = blob.view(1, c, h, w)
    output = model(input_x.cuda())[0]
    boxes = output["boxes"].cpu().detach().numpy()
    scores = output["scores"].cpu().detach().numpy()
    labels = output["labels"].cpu().detach().numpy()
    masks = output["masks"].cpu().detach().numpy()
    index = 0
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    result_mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        print("score: ", scores[index])
        mask = np.reshape(masks[index], (h, w))
        mask[mask >= 0.5] = 255
        mask[mask < 0.5] = 0
        result_mask = cv.add(result_mask, np.uint8(mask))
        cv.rectangle(frame, (np.int32(x1), np.int32(y1)), (np.int32(x2), np.int32(y2)), (140, 199, 0), 2, 8, 0)
        index += 1
    result = cv.bitwise_and(frame, frame, mask=result_mask)
    cv.imshow("result", result)
    cv.imshow("not result mask", result_mask)

    mm = cv.imread("D:/images/Ped02.png")
    mm = cv.cvtColor(mm, cv.COLOR_BGR2RGB)
    blob = transform(mm)
    mc, mh, mw = blob.shape
    input_x = blob.view(1, mc, mh, mw)
    output = model(input_x.cuda())[0]
    boxes = output["boxes"].cpu().detach().numpy()
    scores = output["scores"].cpu().detach().numpy()
    labels = output["labels"].cpu().detach().numpy()
    masks = output["masks"].cpu().detach().numpy()
    index = 0
    mm = cv.cvtColor(mm, cv.COLOR_RGB2BGR)
    result_mask = np.zeros((mh, mw), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        print("score: ", scores[index])
        mask = np.reshape(masks[index], (mh, mw))
        mask[mask >= 0.5] = 255
        mask[mask < 0.5] = 0
        result_mask = cv.add(result_mask, np.uint8(mask))
        cv.rectangle(mm, (np.int32(x1), np.int32(y1)), (np.int32(x2), np.int32(y2)), (140, 199, 0), 2, 8, 0)
        index += 1
    res = cv.bitwise_and(mm, mm, mask=result_mask)
    cv.imshow("res", res)
    cv.bitwise_not(res, res)
    res = cv.bitwise_and(mm, mm, mask=res)
    cv.imshow("sub_res", res)
    result = cv.bitwise_and(res, res, mask=result_mask)
    cv.imshow("result2", result)
    cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == '__main__':
    object_detection_demo()