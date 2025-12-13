from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
import mlflow
import mlflow.pytorch
import torch
from transformers import DistilBertTokenizer

# Ensure src is in path so we can import features if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import custom transformer so pickle can find it
try:
    from features.sentiment import SentimentExtractor
except ImportError:
    print("Warning: Could not import SentimentExtractor. Loading sentiment model might fail.")

mlflow.set_tracking_uri('data')

class InputText(BaseModel):
    input_texts: str

app = FastAPI()

model = None
tokenizer = None
model_type = "sklearn" # or "bert"

def load_model():
    global model, tokenizer, model_type
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
        
        if "bert" in model_name:
            model_type = "bert"
            # Load components dictionary
            components = mlflow.transformers.load_model(model_uri)
            model = components.model
            tokenizer = components.tokenizer
            model.eval() # Set to eval mode
        else:
            model_type = "sklearn"
            model = mlflow.sklearn.load_model(model_uri)
            
        print(f"Model ({model_type}) loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Load model on startup
load_model()

@app.get("/health")
def get_health():
    return {"status": "OK"}

@app.post("/get-prediction/")
def get_prediction(input_data: InputText):
    global model, tokenizer, model_type
    if model is None:
        load_model()
        if model is None:
            return {"error": "Model not available"}

    print(f"Predicting for: {input_data.input_texts}")
    
    try:
        if model_type == "bert":
            inputs = tokenizer(
                input_data.input_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            prediction_idx = torch.argmax(logits, dim=1).item()
        else:
            # Sklearn pipeline
            prediction_idx = model.predict([input_data.input_texts])[0]
        
        # Map prediction to label
        # Assuming 0: Democrat, 1: Republican (alphabetical)
        labels = {0: "Democrat", 1: "Republican"}
        prediction_label = labels.get(prediction_idx, "Unknown")
        
        return {'prediction': prediction_label}
    except Exception as e:
        return {"error": str(e)}