import torchvision
import torch
import cv2 as cv
import numpy as np

coco_names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus',
         '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign',
         '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep',
         '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack',
         '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis',
         '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove',
         '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass',
         '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple',
         '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza',
         '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed',
         '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote',
         '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink',
         '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear',
         '89': 'hair drier', '90': 'toothbrush'}
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 使用GPU
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()


def faster_rcnn_image_detection():
    src = cv.imread("D:/images/cars.jpg")
    image = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    blob = transform(image)
    c, h, w = blob.shape
    input_x = blob.view(1, c, h, w)
    output = model(input_x.cuda())[0]
    boxes = output['boxes'].cpu().detach().numpy()
    scores = output['scores'].cpu().detach().numpy()
    labels = output['labels'].cpu().detach().numpy()
    print(boxes.shape, scores.shape, labels.shape)
    index = 0
    for x1, y1, x2, y2 in boxes:
        if scores[index] > 0.5:
            print(x1, y1, x2, y2)
            cv.rectangle(src, (np.int32(x1), np.int32(y1)),
                         (np.int32(x2), np.int32(y2)), (0, 255, 255), 1, 8, 0)
            label_id = labels[index]
            label_txt = coco_names[str(label_id)]
            cv.putText(src, label_txt, (np.int32(x1), np.int32(y1)), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
        index += 1
    cv.imshow("Faster-RCNN Detection Demo", src)
    cv.waitKey(0)
    cv.destroyAllWindows()


def object_detection_video_demo():
    capture = cv.VideoCapture("D:/videos/cars-1900.mp4")
    while True:
        ret, frame = capture.read()
        if ret is not True:
            break
        blob = transform(frame)
        c, h, w = blob.shape
        input_x = blob.view(1, c, h, w)
        output = model(input_x.cuda())[0]
        boxes = output['boxes'].cpu().detach().numpy()
        scores = output['scores'].cpu().detach().numpy()
        labels = output['labels'].cpu().detach().numpy()
        index = 0
        for x1, y1, x2, y2 in boxes:
            if scores[index] > 0.5:
                cv.rectangle(frame, (np.int32(x1), np.int32(y1)),
                             (np.int32(x2), np.int32(y2)), (0, 255, 255), 1, 8, 0)
                label_id = labels[index]
                label_txt = coco_names[str(label_id)]
                cv.putText(frame, label_txt, (np.int32(x1), np.int32(y1)), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 1)
            index += 1
        wk = cv.waitKey(1)
        if wk == 27:
            break
        cv.imshow("video detection Demo", frame)


if __name__ == "__main__":
    faster_rcnn_image_detection()