import torch
import numpy as np
import cv2 as cv

model_bin = "D:"
config_text = "D:"


def image_landmark_demo():
    cnn_model = torch.load("D:/models/model_landmarks.pt")
    image = cv.imread("D:/images/landmark/218.jpg")
    cv.imshow("input", image)
    h, w, c = image.shape
    img = cv.resize(image, (64, 64))
    img = (np.float32(img) / 255.0 - 0.5) / 0.5
    img = img.transpose((2, 0, 1))
    x_input = torch.from_numpy(img).view(1, 3, 64, 64)
    probs = cnn_model(x_input.cuda())
    lm_pts = probs.view(5, 2).cpu().detach().numpy()
    print(lm_pts)
    for x, y in lm_pts:
        print(x, y)
        x1 = x*w
        y1 = y*h
        cv.circle(image, (np.int32(x1), np.int32(y1)), 2, (0, 150, 150), 2, 8, 0)
    cv.imshow("face_landmark", image)
    cv.imwrite("D:/images/landmark/landmark_result.png", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def video_landmark_demo():
    cnn_model = torch.load("D:/models/model_landmarks.pt")
    capture = cv.VideoCapture("D:/images/01.mp4")
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        h, w, c = frame.shape
        blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blobImage)
        cvOut = net.forward()
        # 绘制检测矩形
        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            left = detection[3]*w
            top = detection[4]*h
            right = detection[5]*w
            bottom = detection[6]*h

            roi = frame[np.int32(top):np.int32(bottom), np.int32(left):np.int32(right),:]
            rw = right - left
            rh = bottom - top
            img = cv.resize(roi, (64, 64))
            img = (np.float32(img) / 255.0 - 0.5) / 0.5
            img = img.transpose((2, 0, 1))
            x_input = torch.from_numpy(img).view(1, 3, 64, 64)
            probs = cnn_model(x_input.cuda())
            lm_pts = probs.view(5, 2).cpu().detach().numpy()
            for x, y in lm_pts:
                x1 = x*rw
                y1 = y * rh
                cv.circle(roi, (np.int32(x1), np.int32(y1)), 2, (150, 150, 0), 2, 8, 0)
            cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (150, 0, 150), 2)
            cv.putText(frame, "score:%.2f"%score, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (20, 120, 240), 1)
            c = cv.waitKey(5)
            if c == ord("q"):
                break

            cv.imshow("face detection landmark", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    video_landmark_demo()


