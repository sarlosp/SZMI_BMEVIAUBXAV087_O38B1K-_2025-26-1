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
        
        
        features = [
            float(data['pressure']),
            float(data['maxtemp']),
            float(data['temparature']),
            float(data['mintemp']),
            float(data['dewpoint']),
            float(data['humidity']),
            float(data['cloud']),
            float(data['sunshine']),
            float(data['winddirection']),
            float(data['windspeed'])
        ]
        
        prediction = model.predict([features])[0]
        
        #(1 = Eső, 0 = Nem eső)
        result = "Várható esőzés!" if prediction == 1 else "Nem várható eső."
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)