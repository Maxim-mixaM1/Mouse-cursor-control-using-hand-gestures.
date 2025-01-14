import cv2
from numpy.linalg.lapack_lite import xerbla
from ultralytics import YOLO
import mouse
from datetime import datetime
import os
import json
import time

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]
import numpy as np
model = YOLO('C:/PycharmProjects/pythonProject1/runs/detect/Hand_click22/weights/best.pt')
cap = cv2.VideoCapture(0)
ESCAPE = 27
key = 1
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
j = 0
def pr(f):
    r = model.predict(f, device="cpu")[0]
    classes_names = r.names
    classes = r.boxes.cls.cpu().numpy()
    box = r.boxes.xyxy.cpu().numpy().astype(np.int32)
    return [box, classes, classes_names]
xe = 0
ye = 0
dx = 0
dy = 0
o = 0
while(key!=ESCAPE):
    start_time = time.time()
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    if ret == False:
        print("Cant read frame")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (230, 129))

    kl = pr(frame)
    boxes = kl[0]
    classes_names = kl[2]
    classes = kl[1]
    cv2.rectangle(frame, (38, 22), (192, 108), colors[3], 2)
    if len(boxes) > 0:
        centerX = (boxes[0][2]/230*2304 + boxes[0][0]/230*2304) / 2# - (2304 - 1920)
        centerY = (boxes[0][3]/129*1296 + boxes[0][1]/129*1296) / 2# - (1296 - 1080)
        centerX += centerX - 1152
        centerY += centerY - 648
        pos = mouse.get_position()

        x = centerX - pos[0]
        y = centerY - pos[1]
        x += (x * 0.2)
        y += (y * 0.2)

        o = 0
        ye = centerY

        if classes[0] == 0:
            mouse.release(button='left')
            mouse.move(centerX, centerY, absolute=True, duration=0.01)
            xe = centerX
            ye = centerY
        else:
            if o == 0:
                o = 1
                dx = xe - centerX
                dy = ye - centerY
            print(centerX + xe - centerX)
            mouse.move(centerX+100, centerY-100, absolute=True, duration=0.01)
            if not mouse.is_pressed(button='left'):

                mouse.press(button='left')
        grouped_objects = {}
        for class_id, box in zip(classes, boxes): 
            class_name = classes_names [int(class_id)]
            color = colors[int(class_id) % len(colors)]
            if class_name not in grouped_objects:
                grouped_objects[class_name] = []
            grouped_objects[class_name].append(box)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow("framme", frame)
    key = cv2.waitKey(10)
    print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1
cap.release()
cv2.destroyAllWindows()