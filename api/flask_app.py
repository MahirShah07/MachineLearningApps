from flask import Flask, render_template
import numpy as np
from keras.models import load_model
from flask import Flask, request, render_template, make_response
from PIL import Image
import os

app = Flask(__name__)

MODEL_PATH = 'static/models/keras_model.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path)
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    pred = model.predict(data)
    return pred

@app.route("/")
def BrainTumorDetecto_index():
    return render_template('brain_tumor_detector.html')

@app.route("/brain_tumor_detector")
def BrainTumorDetector():
    return render_template('brain_tumor_detector.html')

@app.route('/predict', methods=['GET','POST'])
def BrainTumorDetector_Predict():
    if request.method == 'POST':
        image_file = request.files['file']
        filename = image_file.filename
        filepath = os.path.join('uploads', filename)
        image_file.save(filepath)
        preds = model_predict(filepath, model)
        if preds[0][0]>0.6:
            return render_template('no_tumor.html')
        if preds[0][1]>0.6:
            return render_template('pituitary.html')
        if preds[0][2]>0.6:
            return render_template('meningioma.html')
        if preds[0][3]>0.6:
            return render_template('glioma.html')
        
app.run()