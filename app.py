import tkinter as tk
from PIL import ImageTk as itk
from PIL import Image
from detection import detect, numbers
import cv2

window = tk.Tk()
window.title("Test app")
flag = False
data = None
label = tk.Label(window, text="Click the Button to update this Text",
font=('Calibri 15 bold'))
label.pack(pady=20)
window.geometry('1000x1000')
def on_click_btn1():
    d = numbers()
    label["text"] = d
def on_click_btn2():
    label["text"] = "You clicked second button"


# Create 1st button to update the label widget
btn1 = tk.Button(window, text="Button1", command=on_click_btn1)
btn1.pack(pady=20)

# Create 2nd button to update the label widget
btn2 = tk.Button(window, text="Button2", command=on_click_btn2)
btn2.pack(pady=40)
canvheight=540
canvwidth=960
canvas = tk.Canvas(window, width=canvwidth, height=canvheight, bg='black')
canvas.pack(anchor="nw", expand=True)
def timer():

    global canvheight
    global canvwidth
    global canvas
    global data
    canvas.delete("all")
    pimage = detect()
    data = Image.fromarray(pimage)
    data = data.resize((canvwidth, canvheight))
    data = itk.PhotoImage(data)
    # data = itk.PhotoImage(file='Clipboard02.jpg')
    canvas.create_image(
        (canvwidth / 2, canvheight / 2),
        image=data
    )
    canvas.after(3000, timer)


canvas.after(1, timer)

window.mainloop()
