import time

import cv2
from ultralytics import YOLO
import subprocess
import requests
import json
import random
import base64
from PIL import Image
import threading
from time import sleep
import torch

# model = YOLO('C:\\Users\\MuhametovRD\\PycharmProjects\\YOLO\\runs\\detect\\train2\\weights\\best.pt')
# frame = cv2.imread("C:\\Users\\MuhametovRD\\PycharmProjects\\Handle_detector\\clipboard02.jpg")
# results = model(frame)
# annotated_frame = results[0].plot()
# classes = results[0].boxes.cls.tolist()
# boxes = results[0].boxes.xyxyn.tolist()
# scores = results[0].boxes.conf.tolist()
# rawscores = results[0].boxes.conf.tolist()
# multiscores = results[0].boxes.conf.tolist()
# rawboxes = results[0].boxes.xyxyn.tolist()
# anchors = results[0].boxes.xyxyn.tolist()
#
#
# d =  {'detection_classes': classes, 'raw_detection_boxes': rawboxes, 'raw_detection_scores': rawscores, 'detection_multiclass_scores': multiscores,
#       'detection_boxes': boxes, 'detection_scores': scores, 'detection_anchor_indices': anchors}
# cv2.imwrite("test1.jpeg", annotated_frame)
# Encode the resized annotated frame to base64

# output_dict['detection_classes']
# output_dict['raw_detection_boxes']
# output_dict['raw_detection_scores']
# output_dict['detection_multiclass_scores']
# output_dict['detection_boxes']
# output_dict['detection_scores']
# output_dict['detection_anchor_indices']
# cv2.imshow("YOLOv8 Inference", annotated_frame)


vcap = cv2.VideoCapture("rtsp://user08:Mrd12345678@2.1.3.33:554/1/1")

from ffmpegcv import VideoCaptureStream as VCS
cap2 = VCS(stream_url="rtsp://user08:Mrd12345678@2.1.3.33:554/1/1")
while(1):
    _, frame = cap2.read()
    cv2im = cv2.resize(src=frame, dsize=None, fx=0.5, fy=0.5);
    # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    # img = Image.fromarray(cv2image)
    cv2.imshow("w", cv2im);
    cv2.waitKey(1);
