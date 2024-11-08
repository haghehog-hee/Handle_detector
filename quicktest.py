import tkinter as tk
from cv2 import cvtColor, COLOR_BGR2RGBA
from ffmpegcv import VideoCaptureStream as VCS
from PIL import ImageTk, Image
import cv2 as cv


# window = tk.Tk()
# window.title("Detection")
#
# canvheight = 540
# canvwidth = 960

# label = tk.Label(
#     window, text="", font=("Calibri 15 bold")
# )
# label.pack()
#
# # set window size
# window.geometry("1024x720")
# stream_ip = "rtsp://user07:user07@2.1.3.254:554/38/1"
# # cap2 = VCS("rtsp://user08:Mrd12345678@2.1.3.33:554/1/1")
# f_top = tk.Frame(window)
# f_bot = tk.Frame(window)
# f_top.pack()
# f_bot.pack()
# stream_window = tk.Label(f_bot)
# stream_window.pack(side=tk.BOTTOM)
# cap2 = cv.VideoCapture(stream_ip, cv.CAP_FFMPEG)
# available_backends = [cv.videoio_registry.getBackendName(i) for i in cv.videoio_registry.getBackends()]
# print(available_backends)
# fourcc = cv.VideoWriter_fourcc(*"H265")
# cap2.set(cv.CAP_PROP_FOURCC, fourcc)
# codec = cv.VideoWriter::fourcc('M', 'J', 'P', 'G');
# cap2.set(cv.CAP_PROP_FOURCC , ('H', '2', '6', '4'));

# while(1):
#
#     ret, frame = vcap.read()
#     cv.imshow('VIDEO', frame)
#     cv.waitKey(1)
# def video_stream():
#     global cap2
#     success, frame = cap2.read()
#     # print(frame)
#     # print(type(frame))
#     if success:
#         img = cvtColor(frame, COLOR_BGR2RGBA)
#         img = Image.fromarray(img)
#         img = img.resize((canvwidth, canvheight))
#         imgtk = ImageTk.PhotoImage(image=img)
#         stream_window.imgtk = imgtk
#         stream_window.configure(image=imgtk)
#     else:
#         print("ГООООООЛ")
#         cap2.release()
#         cap2 = cv.VideoCapture(stream_ip)
#     stream_window.after(100, video_stream)
#
# # initiate video stream
# print("0")
# video_stream()
#
# # run the tkinter main loop
# window.mainloop()

# импорт необходимых библиотек
import os
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from detection import remove_overlap, detect_and_count, window_scan_detection
import cv2 as cv
from Affine_transform import Atransform
#
# # пути к модели и меткам
PATH_TO_MODEL_DIR = "C:\\Tensorflow\\models\\research\\object_detection\\interference_graphtest\\saved_model"
PATH_TO_LABELS = "C:\\Tensorflow\\Dataset\\label_map2.pbtxt"
IMAGE_SAVE_PATH = "C:\\Tensorflow\\Dataset\\AffineSide\\test\\"
IMAGE_PATH = "C:\\Tensorflow\\Dataset\\AffineSide\\rotated\\img\\"
# IMAGE_PATH = "C:\\Tensorflow\\Dataset\\AffineSide\\window\\img\\"
IMAGE_PATHS = os.listdir(IMAGE_PATH)
detection_model = tf.saved_model.load(PATH_TO_MODEL_DIR)
# for i in range(300,2000,20):
#     # for j in range(1, 100, 2):
#     #     for k in range(4, 20, 1):
#     #         size = i
#     #         step = j/50
#     #         conf = k/20
size = 500
step = 0.8
conf = 0.7
for PATH in IMAGE_PATHS:
    image_path = IMAGE_PATH + PATH
    print('Running inference for {}... '.format(image_path), end='')
    # загрузка изображения
    #image_np = load_image_into_numpy_array(image_path)
    im = cv.imread(image_path)
    # im = cv.cvtColor(im, cv.COLOR_BGR2RGBA)
    im = window_scan_detection(im, PATH, detection_model, size, False)
    cv.imwrite(IMAGE_SAVE_PATH  + PATH, im)
