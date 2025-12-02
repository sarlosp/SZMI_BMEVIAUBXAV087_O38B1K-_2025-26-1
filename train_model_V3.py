import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

print("Adatok betöltése...")
df = pd.read_csv("Rainfall.csv")
df.columns = df.columns.str.strip()

# --- Célváltozó előkészítése ---
if 'rainfall' not in df.columns:
    raise ValueError("A 'rainfall' oszlop nem található a Rainfall.csv fájlban!")

df['rainfall'] = df['rainfall'].astype(str).str.lower().str.strip()
df = df[df['rainfall'].isin(['yes', 'no'])]
df['rainfall'] = df['rainfall'].map({'yes': 1, 'no': 0}).astype(int)

# --- Feature engineering ---
# 1) Hőingadozás
df['temp_range'] = df['maxtemp'] - df['mintemp']
# 2) Harmatpont-különbség
df['dew_depression'] = df['temparature'] - df['dewpoint']
# 3) Páratartalom négyzet
df['humidity_sq'] = df['humidity'] ** 2

feature_cols = [
    'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
    'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed',
    'temp_range', 'dew_depression', 'humidity_sq'
]

missing_features = [c for c in feature_cols if c not in df.columns]
if missing_features:
    raise ValueError(f"Hiányzó feature oszlop(ok): {missing_features}")

X = df[feature_cols]

# Numerikus konverzió + hiányzó értékek mediánnal
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())

y = df['rainfall']

print("Tanító- és teszthalmaz szétválasztása...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline: scaler + SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

# Finom hangolás a már jó értékek körül (C≈5, gamma≈0.01)
param_grid = {
    'svc__C': [3, 5, 7, 10, 15],
    'svc__gamma': [0.02, 0.015, 0.01, 0.007, 0.005],
    'svc__kernel': ['rbf']
    # ha szeretnéd kipróbálni:
    # 'svc__class_weight': [None, 'balanced']
}

print("SVM tanítása GridSearch-sel (n_jobs=1)...")
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=1,          # így eltűnnek a loky/joblib hibák
    verbose=1
)

grid.fit(X_train, y_train)

print("Legjobb paraméterek:", grid.best_params_)

best_model = grid.best_estimator_

print("Végső modell kiértékelése...")
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Végső modell pontossága: {acc * 100:.2f}%")

print("\nOsztályozási riport:")
print(classification_report(y_test, y_pred))

print("Konfúziós mátrix:")
print(confusion_matrix(y_test, y_pred))

print("Modell mentése model.pkl fájlba...")
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("A legjobb SVM modell sikeresen elmentve a 'model.pkl' fájlba!")
