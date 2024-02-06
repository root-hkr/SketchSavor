import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import cv2
from tensorflow.keras.models import load_model

model = load_model('fod.h5')
menu_item = {0: "Burger", 1: "Hotdog", 2: "Icecream", 3: "Pizza", 4: "Sandwich"}


def event_function(event):
    x = event.x
    y = event.y

    x1 = x - 12
    y1 = y - 12

    x2 = x + 12
    y2 = y + 12

    canvas.create_oval((x1, y1, x2, y2), fill='black')
    img_draw.ellipse((x1, y1, x2, y2), fill='white')


def save():
    global count

    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(count) + '_original.jpg', img_array)
    img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(count) + '.jpg', img_array)
    count = count + 1


def clear():
    global img, img_draw

    canvas.delete('all')
    img = Image.new('RGB', (500, 500), (0, 0, 0))
    img_draw = ImageDraw.Draw(img)

    label_status.config(text='Draw: NONE')


def predict():
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (28, 28,), interpolation=cv2.INTER_NEAREST)
    img_array = img_array / 255
    img_array = np.reshape(img_array, (1, 28, 28, 1))
    img_array = np.array(img_array)

    result = model.predict([img_array])
    label = np.argmax(result, axis=1)
    label_status.config(text='U want:' + menu_item[label[0]])


count = 0

win = tk.Tk()

canvas = tk.Canvas(win, width=500, height=500, bg='white')
canvas.grid(row=0, column=0, columnspan=4)

button_save = tk.Button(win, text='SAVE', bg='green', fg='white', font='helvetica 20 bold', command=save)
button_save.grid(row=1, column=0)

button_predict = tk.Button(win, text='PREDICT', bg='blue', fg='white', font='helvetica 20 bold', command=predict)
button_predict.grid(row=1, column=1)

button_clear = tk.Button(win, text='CLEAR', bg='yellow', fg='white', font='Helvetica 20 bold', command=clear)
button_clear.grid(row=1, column=2)

button_exit = tk.Button(win, text='EXIT', bg='red', fg='white', font='Helvetica 20 bold', command=win.destroy)
button_exit.grid(row=1, column=3)

label_status = tk.Label(win, text='Draw...', bg='white', font='helvetica 24 bold')
label_status.grid(row=2, column=0, columnspan=4)

canvas.bind('<B1-Motion>', event_function)
img = Image.new('RGB', (500, 500), (0, 0, 0))
img_draw = ImageDraw.Draw(img)

win.mainloop()


