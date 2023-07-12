import time
import cv2
import mss
import numpy
import tensorflow as tf
from PIL import ImageGrab
import grab_screen
import win32gui
import win32ui
from ctypes import windll
from main import show_inference, run_inference_for_single_image, Affine
from PIL import Image
import win32process as wproc
import win32api as wapi
import pygetwindow as gw
from libs.shape import Shape
from numpy import asarray

# title of our window
title = "FPS benchmark"
# set start time to current time
start_time = time.time()
# displays the frame rate every 2 second
display_time = 2
# Set primarry FPS to 0
fps = 0
# Load mss library as sct
sct = mss.mss()
# Set monitor size to capture to MSS
monitor = {"top": 100, "left": 200, "width": 800, "height": 500}
# Set monitor size to capture


PATH_TO_MODEL_DIR = "C:\\Tensorflow\\models\\research\\object_detection\\interference_graph4\\saved_model"
PATH_TO_LABELS = "C:\\Tensorflow\\Dataset\\label_map.pbtxt"
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR
window_name = 'Smart Client'
app_name='Test app'
# загрузка модели
detection_model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

def screen_recordMSS():
    global fps, start_time
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = catch_window(window_name)
        img = Affine(asarray(img))
        #img = gw.getWindowsWithTitle(window_name)[0]
        #img = sct.grab(monitor)
        #img.activate()
        #img = catch_window(img)
        # to ger real color we do this:

        Imagenp = show_inference(detection_model, img)[0]
        cv2.imshow(app_name, cv2.resize(Imagenp, (1920,1080)))
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
def detect():
    global fps, start_time
    img = catch_window(window_name)
    img = Affine(asarray(img))
    Imagenp = show_inference(detection_model, img)[0]
    return Imagenp

def numbers():
    global fps, start_time
    img = catch_window(window_name)
    img = Affine(asarray(img))
    detections = show_inference(detection_model, img)[1]
    d = dict()
    for i in range(0, detections['detection_classes'].size):
        if (detections['detection_scores'][i] >= 0.2):
            if (d.get(detections['detection_classes'][i]) == None):
                d.setdefault(detections['detection_classes'][i], 1)
            else:
                d[detections['detection_classes'][i]] += 1
    return d

def catch_window(hwnd):
    hwnd = win32gui.FindWindow(None, window_name)
    remote_thread, _ = wproc.GetWindowThreadProcessId(hwnd)
    wproc.AttachThreadInput(wapi.GetCurrentThreadId(), remote_thread, True)
    win32gui.SetActiveWindow(hwnd)
    #hwnd = win32gui.FindWindow(None, window_name)
# Change the line below depending on whether you want the whole window
# or just the client area.
#left, top, right, bot = win32gui.GetClientRect(hwnd)
    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w = right - left
    h = bot - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    hwnd2 = win32gui.FindWindow(None, app_name)
    win32gui.SetActiveWindow(hwnd2)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

    saveDC.SelectObject(saveBitMap)

    # Change the line below depending on whether you want the whole window
    # or just the client area.
    #result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
    result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)
    print(result)
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    return im

#screen_recordMSS()

def list_window_names():
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            print(hex(hwnd), win32gui.GetWindowText(hwnd))
    win32gui.EnumWindows(winEnumHandler, None)

#list_window_names()
