import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

print("Adatok betöltése...")
df = pd.read_csv('Rainfall.csv')

df.columns = df.columns.str.strip()

df = df.dropna()

df['rainfall'] = df['rainfall'].apply(lambda x: 1 if x == 'yes' else 0)

feature_cols = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

X = df[feature_cols]
y = df['rainfall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Modell tanítása...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Modell pontossága: {accuracy * 100:.2f}%")

print("Modell mentése...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("A modell sikeresen elmentve a 'model.pkl' fájlba!")