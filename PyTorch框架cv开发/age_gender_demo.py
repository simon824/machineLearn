import torch
import numpy as np
import cv2 as cv
import time
from age_gender_cnn_test import MyMultipleTaskNet
from openvino.inference_engine import IECore

model_bin = "D:/netron_models/opencv_face_detector_uint8.pb"
config_text = "D:/netron_models/opencv_face_detector.pbtxt"
genders = ["male", "female"]


def video_age_gender_demo():
    cnn_model = torch.load("D:/models/age_gender_model.pt")
    cap = cv.VideoCapture("D:/images/01.mp4")

    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, c = frame.shape
        blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blobImage)
        cvOut = net.forward()
        # 绘制检测矩形
        for detection in cvOut[0, 0, :, :]:
            score = float(detection[2])
            if score > 0.5:
                left = detection[3] * w
                top = detection[4] * h
                right = detection[5] * w
                bottom = detection[6] * h
                roi = frame[np.int32(top):np.int32(bottom), np.int32(left):np.int32(right), :]
                img = cv.resize(roi, (64, 64))
                img = (np.float32(img) / 255.0 - 0.5) / 0.5
                img = img.transpose((2, 0, 1))
                x_input = torch.from_numpy(img).view(1, 3, 64, 64)
                age, gender = cnn_model(x_input.cuda())
                predict_gender = torch.max(gender, 1)[1].cpu().detach().numpy()[0]
                gender = "Male"
                if predict_gender == 1:
                    gender = "Female"
                predict_age = age.cpu().detach().numpy()*116.0
                print(predict_gender, predict_age)
                # 绘制
                cv.putText(frame, ("gender: %s, age: %s") % (gender, int(predict_age[0][0])), (int(left), int(top)-15), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                c = cv.waitKey(5)
        if c == ord("q"):
            break
        cv.imshow("face detection + landmark", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


def face_age_gender_demo():
    ie = IECore()
    for device in ie.available_devices:
        print(device)
    model_xml = "D:/netron_models/face-detection-0202.xml"
    model_bin = "D:/netron_models/face-detection-0202.bin"
    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)
    cap = cv.VideoCapture("D:/images/01.mp4")
    exec_net = ie.load_network(network=net, device_name="CPU")

    # 加载性别与年龄预测模型
    em_net = ie.read_network(model="D:/netron_models/age_gender_model.onnx")
    em_input_blob = next(iter(em_net.input_info))
    em_it = iter(em_net.outputs)
    em_out_blob1 = next(em_it)
    em_out_blob2 = next(em_it)
    en, ec, eh, ew = em_net.input_info[em_input_blob].input_data.shape
    print(en, ec, eh, ew)

    em_exec_net = ie.load_network(network=em_net, device_name="CPU")
    while True:
        ret, frame = cap.read()
        if ret is not True:
            break
        image = cv.resize(frame, (w, h))
        image = image.transpose(2, 0, 1)
        inf_start = time.time()
        res = exec_net.infer(inputs={input_blob: [image]})
        inf_end = time.time() - inf_start
        # print("infer time(ms)：%.3f"%(inf_end*1000))
        ih, iw, ic = frame.shape
        res = res[out_blob]
        for obj in res[0][0]:
            if obj[2] > 0.75:
                xmin = int(obj[3] * iw)
                ymin = int(obj[4] * ih)
                xmax = int(obj[5] * iw)
                ymax = int(obj[6] * ih)
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax >= iw:
                    xmax = iw - 1
                if ymax >= ih:
                    ymax = ih - 1
                roi = frame[ymin:ymax, xmin:xmax, :]
                roi_img = cv.resize(roi, (ew, eh))
                roi_img = (np.float32(roi_img) / 255.0 - 0.5) / 0.5
                roi_img = roi_img.transpose(2, 0, 1)
                em_res = em_exec_net.infer(inputs={em_input_blob: [roi_img]})
                gender_prob = em_res[em_out_blob1].reshape(1, 2)
                prob_age = em_res[em_out_blob2].reshape(1, 1)[0][0] * 116
                label_index = np.int(np.argmax(gender_prob, 1))
                age = np.int(prob_age)
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                cv.putText(frame, "infer time(ms): %.3f" % (inf_end * 1000), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 0, 255),
                           2, 8)
                cv.putText(frame, genders[label_index] + ', ' + str(age), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (0, 0, 255),
                           2, 8)
        cv.imshow("Face+ age/gender prediction", frame)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # video_age_gender_demo()
    face_age_gender_demo()











