import os
import numpy as np
import cv2 as cv

model_bin = "D:/projects/opencv_tutorial/data/models/face_detector/opencv_face_detector_uint8.pb";
config_text = "D:/projects/opencv_tutorial/data/models/face_detector/opencv_face_detector.pbtxt";


def face_detect(frame, savePath):
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    h, w, c = frame.shape
    blobImage = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False);
    net.setInput(blobImage)
    cvOut = net.forward()
    # 绘制检测矩形
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.5:
            left = detection[3]*w
            top = detection[4]*h
            right = detection[5]*w
            bottom = detection[6]*h

            # roi and detect landmark
            roi = frame[np.int32(top):np.int32(bottom),np.int32(left):np.int32(right),:]
            cv.imwrite(savePath, roi)
            cv.imshow("input", frame)
            cv.waitKey(1)


dest_dir = "D:/facedb/emotion_dataset"
label_dir = "D:/BaiduNetdiskDownload/Emotion_Labels"
image_dir = "D:/BaiduNetdiskDownload/cohn-kanade-images/"
item_dirs = os.listdir(label_dir)
images= []
labels = []
for sub_dir in item_dirs:
    path1 = os.path.join(label_dir, sub_dir)
    if os.path.isdir(path1):
        item_dirs = os.listdir(path1)
        for item in item_dirs:
            path2 = os.path.join(path1, item)
            if os.path.isdir(path2):
                files = os.listdir(path2)
                for f in files:
                    path3 = os.path.join(path2, f)
                    with open(path3) as read_file:
                        for line in read_file:
                            if line is not None:
                                line = line.replace('\n', '').replace('   ','')
                                fileName = f.replace("_emotion.txt", '') + ".png"
                                image_file = image_dir + sub_dir + "/" + item + "/" + fileName
                                print(image_file)
                                src = cv.imread(image_file)
                                fileName = str(np.int32(float(line))) + "_" + fileName
                                save_path = os.path.join(dest_dir, fileName)
                                print(save_path)
                                face_detect(src, save_path)
