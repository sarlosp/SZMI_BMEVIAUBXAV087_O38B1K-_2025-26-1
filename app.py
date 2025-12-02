from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Alap feature-k (10 db)
        pressure = float(data['pressure'])
        maxtemp = float(data['maxtemp'])
        temparature = float(data['temparature'])
        mintemp = float(data['mintemp'])
        dewpoint = float(data['dewpoint'])
        humidity = float(data['humidity'])
        cloud = float(data['cloud'])
        sunshine = float(data['sunshine'])
        winddirection = float(data['winddirection'])
        windspeed = float(data['windspeed'])

        # 🔹 Ugyanaz a feature engineering, mint a train kódban:
        temp_range = maxtemp - mintemp                  # 11. feature
        dew_depression = temparature - dewpoint         # 12. feature
        humidity_sq = humidity ** 2                     # 13. feature

        # 🔹 Pontosan ugyanaz a sorrend, mint a train_model.py-ben!
        features = [
            pressure,
            maxtemp,
            temparature,
            mintemp,
            dewpoint,
            humidity,
            cloud,
            sunshine,
            winddirection,
            windspeed,
            temp_range,
            dew_depression,
            humidity_sq
        ]

        features_array = np.array([features])

        prediction = model.predict(features_array)[0]

        # (1 = Eső, 0 = Nem eső)
        result = "Várható esőzés!" if prediction == 1 else "Nem várható eső."
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
