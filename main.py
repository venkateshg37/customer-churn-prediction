from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("../model/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Customer Churn API"}

@app.post("/predict")
def predict(data: dict):
    try:
        features = np.array(list(data.values())).reshape(1, -1)
        prediction = model.predict(features)[0]

        return {
            "prediction": int(prediction),
            "result": "Churn" if prediction == 1 else "No Churn"
        }
    except Exception as e:
        return {"error": str(e)}