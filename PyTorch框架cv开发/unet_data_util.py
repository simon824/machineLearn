import numpy as np
import cv2 as cv
from scipy.io import loadmat
import os
def generate_mask():
    ground_dir = "D:/datasets/CrackForest-dataset/groundTruth"
    seg_dir = "D:/datasets/CrackForest-dataset/seg"
    files = os.listdir(ground_dir)
    sfiles = os.listdir(seg_dir)
    for i in range(len(files)):
        mat_file = os.path.join(ground_dir, files[i])
        seg_file = os.path.join(seg_dir, sfiles[i])
        mask_file_name = sfiles[i].replace(".seg", "")
        # 解析groundTruth
        m = loadmat(mat_file)
        mask = m["groundTruth"][0][0][1]
        # 解析seg
        file = open(seg_file)
        print(mat_file, seg_file)
        line = file.readline()
        while True:
            line = file.readline()
            segInfo = line.split(" ")
            if len(segInfo) == 4:
                c1 = np.int32(segInfo[2])
                c2 = np.int32(segInfo[3])
                seg_num = np.int32(segInfo[0])
                if seg_num == 1:
                    row = np.int32(segInfo[1])
                    for col in range(c1, c2, 1):
                        mask[row, col] = 1
            if not line:
                break
        file.close()
        # 形态学膨胀处理
        mask = np.uint8(mask * 255)
        se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        result = cv.morphologyEx(mask, cv.MORPH_CLOSE, se)
        cv.imwrite("D:/datasets/CrackForest-dataset/mask2/%s.png" %(mask_file_name), result)
if __name__ == '__main__':
    generate_mask()




