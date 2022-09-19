import cv2
import numpy as np

lower = np.array([15, 100, 20])
upper = np.array([40, 255, 255])

video = cv2.VideoCapture(0)

while True:
    success, img = video.read()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x,y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("Mask", mask)
    cv2.imshow("Webcam", img)
    cv2.imshow('Frame', result)

    cv2.waitKey(1)
