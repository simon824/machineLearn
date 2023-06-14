import cv2 as cv
import numpy as np

image = cv.imread("D:/images/Lenna.png")

cv.imshow("input", image)
h, w, c = image.shape
print(h, w, c)

blob = np.transpose(image, (2, 0, 1))
print(blob.shape)

fi = np.float32(image) / 255.0  # 0-1
cv.imshow("fi", fi)

gray = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
cv.imshow("gray", gray)

dst = cv.resize(image, (256, 256))
cv.imshow("zoom out", dst)

r = image[:, :, 0]
g = image[:, :, 1]
b = image[:, :, 2]

box = [200, 100, 200, 200]  # x, y, w, h
roi = image[200:400, 100:300, :]
cv.imshow("roi", roi)

m1 = np.zeros((256, 256, 3), np.uint8)
m1[:, :] = (127, 0, 127)
cv.imshow("m1", m1)
cv.rectangle(image, (200, 200), (400, 400), (127, 127, 0), 3, 8)
cv.imshow("input", image)

cap = cv.VideoCapture("D:/images/lane.avi")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv.imshow("frame", frame)

    c = cv.waitKey(5)
    if c == ord("q"):
        break

cv.waitKey(0)
cv.destroyAllWindows()


