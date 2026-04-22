import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model = tf.keras.models.load_model("fire_detection_model.h5")

img_path = r"C:\Users\AboGhada\Desktop\fire_project\Fire_Detection_Data\train\no-Fire\no-Fire1.jpg"

img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

print("Prediction:", prediction[0][0])  

plt.imshow(img)  
plt.axis("off")

if prediction[0][0] < 0.5:
    plt.title(" Fire Detected")
else:
    plt.title(" No Fire")

plt.show()