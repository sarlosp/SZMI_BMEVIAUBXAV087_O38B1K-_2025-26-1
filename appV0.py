from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. A betanított modell betöltése
# Fontos: a model.pkl fájlnak ugyanott kell lennie, ahol az app.py van!
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. A főoldal megjelenítése
@app.route('/')
def home():
    return render_template('index.html')

# 3. Az előrejelzés végrehajtása (ezt hívja majd a frontend)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Adatok kinyerése a kérésből
        data = request.json
        
        # Az adatok sorrendjének egyeznie kell a tanítással!
        # ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
        #  'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
        
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
        
        # Előrejelzés készítése (2D tömbbé alakítjuk, mert a modell ezt várja)
        prediction = model.predict([features])[0]
        
        # Eredmény visszaküldése (1 = Eső, 0 = Nem eső)
        result = "Várható esőzés!" if prediction == 1 else "Nem várható eső."
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
