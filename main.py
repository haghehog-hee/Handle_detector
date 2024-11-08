import tkinter as tk
from detection import detect_and_count, dumb_detection, detect_and_count_nosplit, yolo_detection, YOLO_segmentation, detect_and_count_YOLO
from cv2 import cvtColor, COLOR_BGR2RGBA, imread
from cv2 import VideoCapture, CAP_FFMPEG, imwrite
from ffmpegcv import VideoCaptureStream as VCS
from PIL import ImageTk, Image
from tkinter import filedialog as fd
from tkinter import simpledialog as sd
from tkinter import messagebox as mb
import numpy as np
from datetime import datetime
import xlrd
from xlutils.copy import copy
from os import environ
import socket
from Affine_transform import AtransformSide
import tensorflow as tf
from ultralytics import YOLO
import cv2

img_path = "clipboard02.jpg"
Frame_flag = False
window = tk.Tk()
window.title("Detection")
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# when detection_flag = False - app shows camera stream in real time
# when the flag = True - app shows image with detection boxes
detection_flag = False
data = None
data2 = None

# Loading and reading config file
Config_path = "config.txt"
config = open(Config_path).read()
config = config.splitlines()
rack_check_treshold =  20
no_rack_treshold =  3
rack_check_flag = 0
no_rack_flag = 0
write_flag = False
use_cpu = config[4]
controller_ip = config[6]
automation = True
img_save_path = config[9]
YOLO_model_path = config[10]
YOLO_model = YOLO(YOLO_model_path)
if use_cpu == "CPU":
    environ["CUDA_VISIBLE_DEVICES"] = "-1"
numbers_save_path = config[2]
filename = "trial.xls"
# label for displaying number of detections
label = tk.Label(
    window, text="", font=("Calibri 15 bold")
)
label.pack()
# set window size
window.geometry("1024x720")

# Camera has two channels Channel_1 - full high resolution, it is used for detection
# Channel_2 - preview low resolution, used in normal "stream" mode
cap2 = VideoCapture(config[3])
capside = VideoCapture(config[7])
model = tf.saved_model.load(config[0])
modelside = YOLO(config[8])

f_top = tk.Frame(window)
f_bot = tk.Frame(window)
f_top.pack()
f_bot.pack()

data = imread(img_path)
data = np.asarray(data)
data2 = imread(img_path)
data2 = np.asarray(data2)

# writes detection results into excel file
def writeXLS(detections, file_path, num_classes):
    bk = xlrd.open_workbook(file_path)
    book = copy(bk)
    sh1 = bk.sheet_by_index(0)
    sh = book.get_sheet(0)
    print("datetime")
    print(datetime.now())
    sh.write(sh1.nrows, 0, str(datetime.now()))
    for x in range(1, num_classes+1):
        if x in detections:
            sh.write(sh1.nrows, x, detections[x])
    book.save(file_path)
    # for y in range(sh.ncols):
    #     col = (sh.col(y))
    #     for x in range(len(col)):
    #
    #         print(record.value)

#Sends request to controller to check does it work in sequence
def readEthernet():
    return True
    global controller_ip
    connected = False
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(1)
    try:
        client.connect((controller_ip, 1285))
        connected = True
    except socket.timeout:
        print("lol")
    if connected:
        message = 0x00FF0A0027020000204D0200.to_bytes(12, 'big')
        client.send(message)
        from_server = client.recv(5050)
    return True

def launch_detection():
    global data
    global data2
    global cap2
    global capside
    global Frame_flag
    global rack_check_flag
    global no_rack_flag
    for i in range(30):
        kek = cap2.grab()
    for i in range(30):
        kek = capside.grab()
    success, frame = cap2.read()
    success2, frame2 = capside.read()
    if success and success2:
        num_handles = yolo_detection(frame, YOLO_model)
        # handle_total = num_handles
        if num_handles >= rack_check_treshold:
            rack_check_flag +=1
            print(rack_check_flag)
            if rack_check_flag == 10:
                frame = cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame2 = cvtColor(frame2, cv2.COLOR_BGR2RGB)
                data, num_handles, num_classes = detect_and_count(frame, model)
                data2, num_handles2 = detect_and_count_YOLO(frame2, modelside, modelside)
                # handle_total = 0
                # for handle in num_handles:
                #     handle_total += num_handles[handle]
                # print("handle total =" + str(handle_total))
                # print("rack check flag =" + str(rack_check_flag))
                # if handle_total > rack_check_treshold:
                #     rack_check_flag += 1
                    # if rack_check_flag == 2:
                numbers = str(num_handles)
                label["text"] = numbers
                _path = numbers_save_path + filename
                writeXLS(num_handles, _path, num_classes)
                writeXLS(num_handles2, _path, num_classes)
                current_time = str(datetime.now()).replace(":", "_")
                imwrite(img_save_path + "\\" + "at" + current_time + "1.jpg", frame)
                imwrite(img_save_path + "\\" + "detected" + current_time + "1.jpg", data)
                imwrite(img_save_path + "\\" + "at" + current_time + "2.jpg", frame2)
                imwrite(img_save_path + "\\" + "detected" + current_time + "2.jpg", data2)
        elif num_handles < rack_check_treshold:
            no_rack_flag += 1
            if no_rack_flag >= no_rack_treshold:
                data = frame
                # data = cvtColor(data, COLOR_BGR2RGBA)
                data2 = frame2
                # data2 = cvtColor(data2, COLOR_BGR2RGBA)
                rack_check_flag = 0
    else:
        cap2.release()
        cap2 = VideoCapture(config[3])
        capside.release()
        capside = VideoCapture(config[7])
        print("STREAM CAP ERROR")

# button that commits single frame detection
def on_click_btn1():
    global detection_flag
    detection_flag = not detection_flag
    if detection_flag:
        launch_detection()


        # numbers = str(num_handles)
        # label["text"] = numbers
        # data = Image.fromarray(data)
        # _path = numbers_save_path + filename
        # writeXLS(num_handles, _path, num_classes)

# automatic mode
def on_click_btn_auto():
    global automation
    if automation:
        btn3.config(text="Авторежим выключен")
        automation = False
    else:
        btn3.config(text="Автоматическое распознавание")
        automation = True

# Open config menu
def on_click_btn2():
    newWindow = tk.Toplevel(window)
    newWindow.title("Настройки")
    newWindow.geometry("500x500")
    f_1 = tk.Frame(newWindow)
    f_2 = tk.Frame(newWindow)
    f_3 = tk.Frame(newWindow)
    f_4 = tk.Frame(newWindow)
    f_1.pack()
    f_2.pack()
    f_3.pack()
    f_4.pack()

# Config menu buttons
    def on_click_conf1():
        config[0] = fd.askdirectory()
        label1["text"] = config[0]

    def on_click_conf2():
        config[1] = fd.askopenfilename(filetypes=[('text Files', '*.pbtxt')])
        label2["text"] = config[1]
    def on_click_conf3():
        config[2] = fd.askdirectory()
        label3["text"] = config[2]
    def on_click_conf4():
        config[3] = sd.askstring(prompt = "new ip", title="camera IP")
        label4["text"] = config[3]
    def on_click_conf5():
        answer = mb.askokcancel(title='Confirmation',
        message='Save new config?',)
        if answer:
            txt = config[0] + "\n" + config[1] + "\n" + config[2] + "\n" + config[3]
            file = open("config.txt", "w")
            file.write(txt)

    label1 = tk.Label(f_1, text="path to model:  " + config[0], font=("Calibri 10"))
    label1.pack()
    btnConf1 = tk.Button(f_1, text="browse", command=on_click_conf1)
    btnConf1.pack()
    label2 = tk.Label(f_2, text="path to label map:  " + config[1], font=("Calibri 10"))
    label2.pack()
    btnConf2 = tk.Button(f_2, text="browse", command=on_click_conf2)
    btnConf2.pack()
    label3 = tk.Label(f_3, text="path to output:  " + config[2], font=("Calibri 10"))
    label3.pack()
    btnConf3 = tk.Button(f_3, text="browse", command=on_click_conf3)
    btnConf3.pack()
    label4 = tk.Label(f_4, text="camera ip:  " + config[3], font=("Calibri 10"))
    label4.pack()
    btnConf4 = tk.Button(f_4, text="Change", command=on_click_conf4)
    btnConf4.pack()
    btnConf5 = tk.Button(f_4, text="Save changes", command=on_click_conf5)
    btnConf5.pack()



btn1 = tk.Button(f_top, text="Подсчет", command=on_click_btn1)
btn1.pack(side=tk.LEFT)

btn2 = tk.Button(f_top, text="Настройки", command=on_click_btn2)
btn2.pack(side=tk.RIGHT)

btn3 = tk.Button(f_top, text="Авторежим выключен", command=on_click_btn_auto)
btn3.pack(side=tk.RIGHT)

stream_window = tk.Label(f_bot)
stream_window2 = tk.Label(f_bot)
stream_window.pack(side=tk.BOTTOM)
stream_window2.pack(side=tk.BOTTOM)

# define canvas dimensions
widthscale = 960/540
canvheight = 700
canvwidth = canvheight * widthscale

# main video stream is an endless loop, updating every 300 ms
# if there's no video stream, it will attempt to run with image from program directory
# if in auto mode, it will wait for signal from controller

def video_stream():
    global cap2
    global capside
    global detection_flag
    im = imread(img_path)
    img = np.asarray(im)
    img2 = img
    if automation and readEthernet():
        launch_detection()
        img = data
        img2 = data2
        # print("auto tick")
        # img = Image.fromarray(im)
    if not cap2.isOpened():
        print("cap not opened")
        cap2.release()
        cap2 = VideoCapture(config[3])
        capside.release()
        capside = VideoCapture(config[7])
        im = Image.open(img_path)
        img = np.asarray(im)
    #     if not detection_flag:
    #         im = Image.open(config[3])
    #         img = np.asarray(im)
    #         img, dict, count = detect_and_count(img)
    #         # img = Image.fromarray(img)
    #         detection_flag = True
    else:
        if rack_check_flag:
            img = data
            img2 = data2
        else:
            success, frame = cap2.read()
            if success:
                # print("show")
                img = cvtColor(frame, COLOR_BGR2RGBA)
            else:
                cap2.release()
                cap2 = VideoCapture(config[3])


        # img = Image.fromarray(cv2image)
    img = Image.fromarray(img)
    img2 = Image.fromarray(img2)
    img = img.resize((int(canvwidth/2), int(canvheight/2)))
    img2 = img2.resize((int(canvwidth/2), int(canvheight/2)))

    imgtk = ImageTk.PhotoImage(image=img)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    stream_window.imgtk = imgtk
    stream_window2.imgtk = imgtk2
    stream_window.configure(image=imgtk)
    stream_window2.configure(image=imgtk2)
    stream_window.after(500, video_stream)

# initiate video stream
video_stream()

# run the tkinter main loop
window.mainloop()

