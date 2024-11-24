from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model = load_model('cancer.keras')  # Load your trained Keras model

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('home.html', prediction_text="No file uploaded. Please upload an image.")

    file = request.files['file']
    if file.filename == '':
        return render_template('home.html', prediction_text="No file selected. Please choose an image.")

    # Load and preprocess the image
    image = load_img(file, target_size=(224, 224))  # Adjust target size as per your model's input
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize if required by your model

    # Predict
    prediction = model.predict(image)
    result = "Cancer Found" if prediction[0][0] > 0.5 else "No Cancer Found"

    return render_template('home.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)
