from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("fire_detection_model.h5")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if file is None:
        return jsonify({"error": "No image received"}), 400

    image = Image.open(file.stream).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        result = " Fire Detected"
    else:
        result = " No Fire"

    print("Prediction:", prediction)

    return jsonify({
        "prediction": float(prediction),
        "result": result
    })


if __name__ == "__main__":
    app.run(debug=True)