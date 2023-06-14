import cv2 as cv
import numpy as np
import time
from openvino.inference_engine import IECore

emotion_labels = ["neutral","anger","disdain","disgust","fear","happy","sadness","surprise"]


def face_emotion_demo():
    ie = IECore()
    for device in ie.available_devices:
        print(device)
    model_xml = "D:/models/face-detection-0202.xml"
    model_bin = "D:/models/face-detection-0202.bin"

    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(n, c, h, w)

    cap = cv.VideoCapture("D:/videos/Boogie_UP.mp4")
    exec_net = ie.load_network(network=net, device_name="CPU")
    # 加载性别与年龄预测模型
    em_net = ie.read_network(model="D:/models/face_emotions_model.onnx")
    em_input_blob = next(iter(em_net.input_info))
    em_it = iter(em_net.outputs)
    em_out_blob1 = next(em_it)
    en, ec, eh, ew = em_net.input_info[em_input_blob].input_data.shape
    print(en, ec, eh, ew)

    em_exec_net = ie.load_network(network=em_net, device_name="CPU")

    while True:
        ret, frame = cap.read()
        if not ret:
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
                emotion_prob = em_res[em_out_blob1]  # 1x8
                label_index = np.int(np.argmax(emotion_prob, 1))
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)
                cv.putText(frame, "infer time(ms): %.3f" % (inf_end * 1000), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                           (255, 0, 255),
                           2, 8)
                cv.putText(frame, emotion_labels[label_index], (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                           2, 8)
        cv.imshow("Face Emotions Detection Demo", frame)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    face_emotion_demo()





