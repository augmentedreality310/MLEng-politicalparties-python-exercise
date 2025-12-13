import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import sys
import os

# Ensure we can import from src (current directory of this script)
sys.path.append(os.path.dirname(__file__))
# Also add root for other potential needs (though maybe not needed now)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from text_loader.loader import DataLoader

def train():
    print("Initializing DataLoader...")
    dl = DataLoader()
    
    print("Loading data...")
    # DataLoader loads data in __init__
    df = dl.data
    
    print("Dropping NaNs...")
    df = df.dropna(subset=['Tweet', 'Party'])
    
    print("Encoding labels...")
    # We need to encode the labels. DataLoader has label_encoder method but it stores it in self.encoder
    # We can use it.
    y = dl.label_encoder(df['Party'])
    
    print("Creating Pipeline...")
    # We use DataLoader.clean_text as the preprocessor for TfidfVectorizer
    # Note: clean_text is an instance method but it doesn't use self state for cleaning,
    # except calling static method remove_characters.
    # However, TfidfVectorizer expects a callable.
    # We can pass dl.clean_text.
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=2500, 
            min_df=1, 
            max_df=0.8, 
            preprocessor=dl.clean_text
        )),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])
    
    print("Training model...")
    # We pass raw text to the pipeline because TfidfVectorizer with preprocessor will handle it.
    pipeline.fit(df['Tweet'], y)
    
    print("Logging model to MLflow...")
    mlflow.set_tracking_uri('data')
    mlflow.set_experiment("political_parties")
    
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="tweet_classifier"
        )
        print(f"Model logged to {mlflow.get_tracking_uri()} with run_id {run.info.run_id}")

if __name__ == "__main__":
    train()
