from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import joblib

# Charger le modèle et le scaler depuis les fichiers .sav
model = joblib.load("logistic_regression_model_v1_5_1_04_08_2025.sav")  # modèle entraîné
scaler = joblib.load("scaler_04_08_2025.sav")   # scaler utilisé à l'entraînement
expected_columns = ["diagonal", "height_left", "height_right", "margin_low", "margin_up", "length"]

app = FastAPI(title="Détection de faux billets")

# ✅ Route racine pour Render (Health Check)
@app.get("/")
def home():
    return {"message": "API de détection de faux billets 🚀 en ligne"}

# Schéma attendu pour JSON
class BilletFeatures(BaseModel):
    diagonal: float
    height_left: float
    height_right: float
    margin_low: float
    margin_up: float
    length: float

@app.post("/predict_json")
async def predict_json(features: BilletFeatures):
    # Convertir en DataFrame
    df = pd.DataFrame([features.dict()])
    
    # Réordonner les colonnes et scaler
    df = df[expected_columns]
    df_scaled = scaler.transform(df)
    
    # Prédiction
    prediction = model.predict(df_scaled)[0]

    # Mapping lisible
    resultat = "Vrai billet" if prediction == 1 else "Faux billet"
    return {"prediction": int(prediction), "resultat": resultat}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    # Lire CSV
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        return {"error": f"Impossible de lire le CSV : {str(e)}"}
    
    # Nettoyage des noms de colonnes
    df.columns = [col.strip().lower() for col in df.columns]

    # Vérifier les colonnes manquantes
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        return {"error": f"Colonnes manquantes dans le CSV : {missing_cols}"}
    
    # Réordonner les colonnes
    df = df[expected_columns]
    
    # Appliquer le scaler
    df_scaled = scaler.transform(df)
    
    # Prédictions
    predictions = model.predict(df_scaled)
    resultats = ["Vrai billet" if p == 1 else "Faux billet" for p in predictions]

    return {"predictions": predictions.tolist(), "resultats": resultats}
