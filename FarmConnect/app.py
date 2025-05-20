from flask import Flask, render_template, request,jsonify, send_from_directory, Markup
import requests
from utils.fertilizer_disc import fertilizer_disc
import numpy as np
import pandas as pd


from PIL import Image
import tensorflow as tf
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

import os
import pickle
import warnings
warnings.filterwarnings('ignore')


disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = ('models\plant_disease_model.pth')

disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction



def weather_fetch(city_name):
   
    api_key = '5aedd3743e1b40f51a97f54ed0d3be32'
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric"  # To get temperature in Celsius
    }
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    # response = requests.get(base_url, params=params)
    x = response.json()
    
    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

app = Flask(__name__,template_folder="template")

@app.route('/')
def index():
    return render_template('index.html')

# Use both @app.route , one for Live server and one for flask.

# @app.route('/template/crop_recom.html')
@app.route('/crop_recom.html')
def crop_recom():
    return render_template('crop_recom.html')

@app.route('/fertilizer.html')
def fert_recom():
    return render_template('fertilizer.html')

@app.route('/disease.html')
def dis_detect():
    return render_template('disease.html')

@app.route('/predict', methods=['POST'])

def predict_crop():
    rf_model = pickle.load(open('models/RandomForest.pkl', 'rb'))
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorus'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph-level'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")

        if weather_fetch(city) is not None:
            temperature, humidity = weather_fetch(city)
            data_my = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = rf_model.predict(data_my)
            final_prediction = prediction[0]
            return jsonify({'prediction': final_prediction})

    return "Try Again"





@app.route('/recommend_fertilizer', methods=['POST'])
def recommend_fert():
    # Retrieve form data
    N = float(request.form['nitrogen'])
    P = float(request.form['phosphorus'])
    K = float(request.form['potassium'])
    crop = request.form['crop']

    df = pd.read_csv('data\\fertilizer.csv')

    nr = df[df['Crop'] == crop]['N'].iloc[0]
    pr = df[df['Crop'] == crop]['P'].iloc[0]
    kr = df[df['Crop'] == crop]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K

    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(fertilizer_disc[key])

    return jsonify({'prediction':response})


@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image found'})

        file = request.files['image']
        if not file:
            return jsonify({'error': 'No file selected'})

        try:
            img = file.read()
            prediction = predict_image(img)
            return jsonify({'result': prediction})

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('disease.html')


if __name__ == '__main__':
    app.run(debug=True)
