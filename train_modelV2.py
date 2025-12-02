import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# 1. Adatok betöltése
print("Adatok betöltése...")
df = pd.read_csv('Rainfall.csv')

# Oszlopnevek tisztítása (esetleges szóközök levágása)
df.columns = df.columns.str.strip()

# 2. Célváltozó (rainfall) előkészítése: 'yes' -> 1, 'no' -> 0
if 'rainfall' not in df.columns:
    raise ValueError("A 'rainfall' oszlop nem található a Rainfall.csv fájlban!")

# kis/nagybetű + szóköz problémák kezelése
df['rainfall'] = df['rainfall'].astype(str).str.strip().str.lower()

# csak az érvényes értékek megtartása
valid_map = {'yes': 1, 'no': 0}
df = df[df['rainfall'].isin(valid_map.keys())]
df['rainfall'] = df['rainfall'].map(valid_map).astype(int)

# 3. Feature oszlopok kiválasztása
feature_cols = [
    'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
    'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'
]

# Ellenőrzés: minden feature létezik-e
missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    raise ValueError(f"Hiányzó feature oszlop(ok) a CSV-ben: {missing_features}")

X = df[feature_cols]
y = df['rainfall']

# 4. Numerikus konverzió + hiányzó értékek kezelése
# Ha valami szövegesen van eltárolva, 'coerce' -> NaN lesz, utána mediánnal töltjük
X = X.apply(pd.to_numeric, errors='coerce')

# Hiányzó értékek kitöltése oszloponkénti mediánnal
X = X.fillna(X.median())

# 5. Train-test split (stratifikált, hogy az osztályarányok megmaradjanak)
print("Tanító- és teszthalmaz szétválasztása...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Alap Random Forest modell tanítása (baseline)
print("Alap Random Forest modell tanítása...")
baseline_model = RandomForestClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=None
)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Random Forest pontosság: {baseline_accuracy * 100:.2f}%")

# 7. Hyperparameter tuning GridSearchCV-vel
print("Hyperparaméter hangolás (GridSearchCV)... Ez eltarthat egy ideig.")

param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [None, 8, 12, 16],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Legjobb hyperparaméterek:", grid_search.best_params_)

# 8. Végső modell kiértékelése
print("Végső modell kiértékelése a teszt halmazon...")
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Végső modell pontossága: {accuracy * 100:.2f}%")

print("\nOsztályozási riport:")
print(classification_report(y_test, y_pred))

print("Konfúziós mátrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Modell mentése
print("Modell mentése model.pkl fájlba...")
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("A legjobb modell sikeresen elmentve a 'model.pkl' fájlba!")
