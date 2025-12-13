import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import sys
import os

# Ensure we can import from src (current directory of this script)
sys.path.append(os.path.dirname(__file__))
# Also add root for other potential needs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from text_loader.loader import DataLoader
from features.sentiment import SentimentExtractor

def train():
    print("Initializing DataLoader...")
    dl = DataLoader()
    
    print("Loading data...")
    df = dl.data
    
    print("Dropping NaNs...")
    df = df.dropna(subset=['Tweet', 'Party'])
    
    print("Encoding labels...")
    y = dl.label_encoder(df['Party'])
    
    print("Creating Pipeline with Sentiment Features...")
    
    # FeatureUnion combines features from multiple transformers
    combined_features = FeatureUnion([
        ('tfidf', TfidfVectorizer(
            max_features=2500, 
            min_df=1, 
            max_df=0.8, 
            preprocessor=dl.clean_text
        )),
        ('sentiment', SentimentExtractor())
    ])
    
    pipeline = Pipeline([
        ('features', combined_features),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
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
            registered_model_name="tweet_classifier_sentiment"
        )
        print(f"Model logged to {mlflow.get_tracking_uri()} with run_id {run.info.run_id}")

if __name__ == "__main__":
    train()
