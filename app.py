import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageTk

model = tf.keras.models.load_model("fire_detection_model.h5")

def predict_image():
    file_path = filedialog.askopenfilename()

    if file_path:
        img = load_img(file_path, target_size=(224, 224))

        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)


        prediction = model.predict(img_array)

        print("Prediction:", prediction[0][0])  

        if prediction[0][0] < 0.5:
            result_label.config(text=" Fire Detected", fg="red")
        else:
            result_label.config(text=" No Fire", fg="green")

        img_display = Image.open(file_path)
        img_display = img_display.resize((250, 250))
        img_display = ImageTk.PhotoImage(img_display)

        panel.config(image=img_display)
        panel.image = img_display


root = tk.Tk()
root.title("Fire Detection App")
root.geometry("400x500")

btn = tk.Button(root, text="Upload Image", command=predict_image)
btn.pack(pady=20)

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="", font=("Arial", 20))
result_label.pack(pady=20)

root.mainloop()