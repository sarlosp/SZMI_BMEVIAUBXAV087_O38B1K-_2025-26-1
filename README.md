# SZMI_BMEVIAUBXAV087_O38B1K-_2025-26-1
Szoftverfejlesztés MI támogatással BMEVIAUBXAV087 Árvai Péter O38B1K Rainfall Prediction using Machine Learning - Python
# Esőzés Előrejelző Rendszer (Rainfall Prediction AI)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![AI-Assisted](https://img.shields.io/badge/Development-AI--Assisted-green)

Ez a projekt a **BME Szoftverfejlesztés MI támogatással (VIAUBXAV087)** tantárgy keretében készült. A cél egy gépi tanuláson alapuló webalkalmazás fejlesztése volt, kizárólag általános célú nyelvi modellek (LLM) instrukciói alapján.

---

## Projekt Célkitűzés

A fejlesztés során nem hagyományos módon írtuk a kódot, hanem **AI Thought Partner** (ChatGPT-5, Microsoft Copilot) segítségével. A kísérlet célja annak vizsgálata volt, hogy:
1.  A "trendi" chatbotok képesek-e komplex mérnöki feladatok (ML pipeline, Webfejlesztés) önálló megoldására.
2.  Milyen minőségbeli különbség van egy kezdő szintű (V0) és egy AI által optimalizált (V3) megoldás között.
3.  Hogyan viszonyul az AI megoldása a referenciaanyaghoz (GeeksforGeeks).

**Referencia:** [GeeksforGeeks - Rainfall Prediction using Machine Learning](https://www.geeksforgeeks.org/machine-learning/rainfall-prediction-using-machine-learning-python/)

---

## Fájlok és Szerepkörök

A repozitóriumban található fájlok a fejlesztés különböző evolúciós szakaszait reprezentálják.

### Gépi Tanulás (Machine Learning)
| Fájl | Leírás | Státusz |
| :--- | :--- | :--- |
| **`Rainfall.csv`** | A nyers meteorológiai adathalmaz (366 nap mérései). | 📄 Adat |
| **`train_model_V3.py`** | **A végleges modell.** Support Vector Machine (SVC) algoritmust használ, adattisztítással, skálázással (`StandardScaler`) és új változók bevezetésével (Feature Engineering). **Ezt futtasd!** | ✅ Végleges |
| `train_modelV2.py` | A második iteráció. Random Forest algoritmust és GridSearch optimalizációt használ. Jó összehasonlítási alap, de hajlamos a túltanulásra. | ⚠️ Archív |
| `train_model.py` | Az alap (Baseline) verzió. Egyszerű Logisztikus Regresszió a referencia cikk alapján. Alacsonyabb pontosság. | ⚠️ Archív |
| `model.pkl` | A `train_model_V3.py` futtatása után létrejövő bináris fájl. Ez tartalmazza a betanított "agyat", amit a weboldal használ. | ⚙️ Generált |

### Webalkalmazás (Backend & Frontend)
| Fájl | Leírás | Státusz |
| :--- | :--- | :--- |
| **`app.py`** | **A végleges szerver.** Flask alapú backend, amely kezeli az API kéréseket, JSON választ küld, és biztonságosan tölti be a modellt. | ✅ Végleges |
| `appV0.py` | Kezdeti, manuális prototípus. Nincs benne hibakezelés, és nem szabványos választ küld. Demonstrációs célokat szolgál az AI fejlesztés bemutatására. | ⚠️ Archív |
| **`templates/index.html`** | **A végleges felület.** Modern, reszponzív design, JavaScript (Fetch API) alapú aszinkron kommunikációval (nem töltődik újra az oldal). | ✅ Végleges |
| `templates/indexV0.html` | Kezdeti HTML váz. Formázás (CSS) nélküli, egyszerű űrlap. | ⚠️ Archív |

---

## Modell Evolúció (AI Iterációk)

A prediktív modell három fejlesztési fázison ment keresztül az AI javaslatai alapján:

1.  **V1 (Baseline):** Logisztikus Regresszió nyers adatokon.
2.  **V2 (Random Forest):** Hiperparaméter-optimalizáció (`GridSearchCV`) bevezetése.
3.  **V3 (SVC + Feature Engineering):** Meteorológiai származtatott változók (*hőingadozás, harmatpont-különbség*) és adatskálázás bevezetése. **Ez érte el a legmagasabb (~85%) pontosságot.**

---

## Kísérleti Eredmény: A "Deep Research" Határai

A projekt végén kísérletet tettünk a felhasználói felület (UI) "profi termék" szintre emelésére az AI **Deep Research** funkciójával.
* **Hipotézis:** Az AI képes önállóan modern design trendeket kutatni és implementálni.
* **Eredmény:** ❌ **Negatív.** A modell vizuális innováció helyett funkcionális egyszerűsítést hajtott végre, visszatérve a primitív V0 verzió szintjére.
* **Tanulság:** A magas szintű UI/UX megvalósításhoz elengedhetetlen a pontos emberi specifikáció (Human-in-the-loop).
