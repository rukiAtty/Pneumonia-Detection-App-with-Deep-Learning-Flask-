from flask import Flask, render_template, request
import joblib
import numpy as np
import cv2
import tensorflow as tf
import base64
import os
app = Flask(__name__)
model_file = "pneumonia50.h5"
if os.path.isfile(model_file) :
    model1 = tf.keras.models.load_model(model_file)
else:
    print("Model or encoder file does not exist.")
@app.route('/')
def main():
    return render_template("index.html")
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    img = request.files["image"].read()
    img_array = np.frombuffer(img, np.uint8)
    uploaded_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)    
    resized_pic = cv2.resize(uploaded_image, (150, 150))
    normalized_pic = resized_pic / 255.0
    reshaped_pic = normalized_pic.reshape(1, 150, 150, 3)    
    prediction = model1.predict(reshaped_pic)
    predicted_label = "is effected by pneumonia. Please consult to the doctor." if prediction[0][0] > 0.5 else "normal."
    _, img_encoded = cv2.imencode(".jpg", uploaded_image)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")
    return render_template("index.html", predicted=predicted_label, image=img_base64)
if __name__ == "__main__":
    app.run(port = 3000 ,debug =True)    
    