import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sys
import os

# Ensure we can import from src (current directory of this script)
sys.path.append(os.path.dirname(__file__))
# Also add root for other potential needs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from text_loader.loader import DataLoader

def train():
    print("Initializing DataLoader...")
    dl = DataLoader()
    
    print("Loading data...")
    df = dl.data
    
    print("Dropping NaNs...")
    df = df.dropna(subset=['Tweet', 'Party'])
    
    print("Encoding labels...")
    y = dl.label_encoder(df['Party'])
    
    print("Creating Pipeline with Logistic Regression...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=2500, 
            min_df=1, 
            max_df=0.8, 
            preprocessor=dl.clean_text
        )),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    print("Training model...")
    pipeline.fit(df['Tweet'], y)
    
    print("Logging model to MLflow...")
    mlflow.set_tracking_uri('data')
    mlflow.set_experiment("political_parties")
    
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="tweet_classifier_logreg"
        )
        print(f"Model logged to {mlflow.get_tracking_uri()} with run_id {run.info.run_id}")

if __name__ == "__main__":
    train()
