import tkinter as tk
from cv2 import cvtColor, COLOR_BGR2RGBA
from ffmpegcv import VideoCaptureStream as VCS
from PIL import ImageTk, Image



window = tk.Tk()
window.title("Detection")

canvheight = 540
canvwidth = 960

label = tk.Label(
    window, text="", font=("Calibri 15 bold")
)
label.pack()

# set window size
window.geometry("1024x720")

cap2 = VCS("rtsp://user08:Mrd12345678@2.1.11.65:554/1/1")
f_top = tk.Frame(window)
f_bot = tk.Frame(window)
f_top.pack()
f_bot.pack()
stream_window = tk.Label(f_bot)
stream_window.pack(side=tk.BOTTOM)

def video_stream():
    global cap2

    _, frame = cap2.read()
    cv2image = cvtColor(frame, COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img = img.resize((canvwidth, canvheight))
    imgtk = ImageTk.PhotoImage(image=img)
    stream_window.imgtk = imgtk
    stream_window.configure(image=imgtk)
    stream_window.after(300, video_stream)

# initiate video stream
print("0")
video_stream()

# run the tkinter main loop
window.mainloop()

