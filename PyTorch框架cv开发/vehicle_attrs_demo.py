import cv2 as cv
import numpy as np
import torch
from torch import nn
from vehicle_attrs_model import ResidualBlock, VehicleAttributeResNet
from openvino.inference_engine import IECore


color_labels = ["white", "gray", "yellow", "red", "green", "blue", "black"]
type_labels = ["car", "bus", "truck", "van"]

model_dir = "D:/models/"
model_xml = model_dir + "vehicle-detection-adas-0002.xml"
model_bin = model_dir + "vehicle-detection-adas-0002.bin"

ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)

versions = ie.get_versions("CPU")
cnn_model = torch.load("D:/models/vehicle_attrs_model.pt")
input_blob = next(iter(net.input_info))
n, c, h, w = net.input_info[input_blob].input_data.shape

capture = cv.VideoCapture("D:/videos/cars_1900.mp4")
ih = capture.get(cv.CAP_PROP_FRAME_HEIGHT)
iw = capture.get(cv.CAP_PROP_FRAME_WIDTH)

input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

exec_net = ie.load_network(network=net, device_name="CPU")
while True:
    ret, src = capture.read()
    if not ret:
        break
    images = np.ndarray(shape=(n, c, h, w))
    images_hw = []
    sh, sw = src.shape[:-1]
    images_hw.append((sh, sw))
    if (sh, sw) != (h, w):
        image = cv.resize(src, (w, h))
    image = image.transpose((2, 0, 1))
    images[0] = image
    res = exec_net.infer(inputs={input_blob: images})

    # 解析ssd输出内容
    res = res[out_blob]
    license_score = []
    license_boxs = []
    index = 0
    data = res[0][0]
    for number, proposal in enumerate(data):
        if proposal[2] > 0.75:
            sh, sw = images_hw[0]
            label = np.int(proposal[1])
            confidence = proposal[2]
            xmin = np.int(sw * proposal[3])
            ymin = np.int(sh * proposal[4])
            xmax = np.int(sw * proposal[5])
            ymax = np.int(sh * proposal[6])

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax >= sw:
                xmax = sw - 1
            if ymax >= sh:
                ymax = sh - 1
            vehicle_roi = src[ymin:ymax, xmin:xmax]
            img = cv.resize(vehicle_roi, (72, 72))
            img = (np.float32(img) / 255.0 - 0.5) / 0.5
            img = img.transpose((2, 0, 1))
            x_input = torch.from_numpy(img).view(1, 3, 72, 72)
            color_, type_ = cnn_model(x_input.cuda())
            predict_color = torch.max(color_, 1)[1].cpu().detach().numpy()[0]
            predict_type = torch.max(type_, 1)[1].cpu().detach().numpy()[0]
            attrs_txt = "color:%s, type:%s" % (color_labels[predict_color], type_labels[predict_type])
            cv.rectangle(src, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv.putText(src, attrs_txt, (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv.imshow("Vehicle Attributes Recognition Demo", src)
    res_key = cv.waitKey(1)
    if res_key == ord("q"):
        break