from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
import mlflow
import mlflow.sklearn

# Ensure src is in path so we can import features if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

mlflow.set_tracking_uri('data')

class InputText(BaseModel):
    input_texts: str

app = FastAPI()

model = None

def load_model():
    global model
    model_name = os.getenv("MODEL_NAME", "tweet_classifier")
    print(f"Attempting to load model: {model_name}")
    
    try:
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["None"])
        if not versions:
             print(f"No model found for {model_name}.")
             return
        latest_version = versions[0].version
        model_uri = f"models:/{model_name}/{latest_version}"
        print(f"Loading model from {model_uri}...")
        model_type = "sklearn"
        model = mlflow.sklearn.load_model(model_uri)            
    except Exception as e:
        print(f"Error loading model: {e}")

# Load model on startup
load_model()

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):
    global model
    if model is None:
        load_model()
        if model is None:
            return {"error": "Model not available"}

    print(f"Predicting for: {input_data.input_texts}")
    
    try:
        prediction_idx = model.predict([input_data.input_texts])[0]
        labels = {0: "Democrat", 1: "Republican"}
        prediction_label = labels.get(prediction_idx, "Unknown")
        
        return {'prediction': prediction_label}
    except Exception as e:
        return {"error": str(e)}