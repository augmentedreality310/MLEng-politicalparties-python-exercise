import pandas as pd
import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import sys
import os
import numpy as np

# Ensure we can import from src (current directory of this script)
sys.path.append(os.path.dirname(__file__))
# Also add root for other potential needs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from text_loader.loader import DataLoader

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train():
    print("Initializing DataLoader...")
    dl = DataLoader()
    
    print("Loading data...")
    df = dl.data
    
    print("Dropping NaNs...")
    df = df.dropna(subset=['Tweet', 'Party'])
    
    print("Encoding labels...")
    y = dl.label_encoder(df['Party'])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(df['Tweet'].values, y, test_size=0.2, random_state=42)
    
    print("Initializing Tokenizer and Model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    # Check for device
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    train_dataset = TweetDataset(X_train, y_train, tokenizer)
    val_dataset = TweetDataset(X_val, y_val, tokenizer)
    
    train_loader = TorchDataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=16)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    print("Starting training...")
    mlflow.set_tracking_uri('data')
    mlflow.set_experiment("political_parties_bert")
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("model_type", "DistilBERT")
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("lr", 2e-5)
        
        for epoch in range(1): # Train for 1 epoch for demonstration speed
            model.train()
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: Average Training Loss: {avg_train_loss}")
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            print(f"Epoch {epoch+1}: Validation Loss: {avg_val_loss}, Accuracy: {accuracy}")
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", accuracy, step=epoch)

        print("Logging model to MLflow...")
        # Log the model components
        components = {
            "model": model,
            "tokenizer": tokenizer,
        }
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="model",
            registered_model_name="tweet_classifier_bert"
        )
        print(f"Model logged to {mlflow.get_tracking_uri()} with run_id {run.info.run_id}")

if __name__ == "__main__":
    train()
