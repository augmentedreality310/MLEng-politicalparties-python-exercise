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
    
    
    print("Pre-processing tweets (Cleaning)...")
    dl.preprocess_tweets() 
    X = dl.data['Tweet'] 
    
    print("Pre-processing parties (Encoding labels)...")
    y = dl.preprocess_parties()
    
    print("Creating Pipeline with Logistic Regression...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=2500, 
            min_df=1, 
            max_df=0.8, 
            preprocessor=dl.clean_text
        )),
        ('model', LogisticRegression(max_iter=1000))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test data: {accuracy:.4f}")
    print("Logging model to MLflow...")
    mlflow.set_tracking_uri('data')
    mlflow.set_experiment("political_parties")
    
    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="tweet_classifier_logreg"
        )
        print(f"Model logged to {mlflow.get_tracking_uri()} with run_id {run.info.run_id}")

if __name__ == "__main__":
    train()
