import sys
import os
import pandas as pd
import joblib
import torch
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def load_data(file_path):
    """Charge les données avec gestion robuste des formats"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Gestion des lignes vides
            if not line.strip():
                continue
                
            # Séparation texte/label
            parts = line.strip().split('\t', maxsplit=1)
            
            # Cas où il n'y a pas de tabulation
            if len(parts) == 1:
                text, label = parts[0], ''
            else:
                text, label = parts
                
            # Conversion du label en numérique
            # (1 pour 'pos', 0 pour 'neg', -1 pour non spécifié)
            num_label = 1 if label.lower() == 'pos' else (0 if label.lower() == 'neg' else -1)
            data.append((text, num_label))
    
    return pd.DataFrame(data, columns=['text', 'label'])

def evaluate_svm(data):
    """Évaluation SVM (inchangée)"""
    model = joblib.load("models/svm/model.pkl")
    vectorizer = joblib.load("models/svm/vectorizer.pkl")
    
    X = vectorizer.transform(data["text"])
    data["predicted"] = model.predict(X)
    
    print(classification_report(data["label"], data["predicted"], target_names=["neg", "pos"], labels=[0, 1], zero_division=0))

    
    os.makedirs("results/predictions", exist_ok=True)
    data.to_csv("results/predictions/svm_predictions.csv", index=False)

def evaluate_bert(data):
    """Évaluation BERT (inchangée)"""
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("models/bert/model")
    model.eval()
    
    inputs = tokenizer(list(data["text"]), truncation=True, padding=True, return_tensors="pt", max_length=512)
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    loader = DataLoader(dataset, batch_size=16)
    
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Évaluation BERT"):
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).tolist())
    
    data["predicted"] = preds
    print(classification_report(
    data["label"], data["predicted"],
    target_names=['neg', 'pos'],
    labels=[0, 1],
    zero_division=0  # évite erreur si une seule classe présente
    ))


    
    os.makedirs("results/predictions", exist_ok=True)
    data.to_csv("results/predictions/bert_predictions.csv", index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py [svm|bert] input_file.txt")
        sys.exit(1)
    
    model_type = sys.argv[1]
    input_file = sys.argv[2]
    
    data = load_data(input_file)
    
    if model_type == "svm":
        evaluate_svm(data)
    elif model_type == "bert":
        evaluate_bert(data)
    else:
        print("Modèle inconnu. Utilisez 'svm' ou 'bert'.")
        sys.exit(1)
    
    # Sortie standardisée pour l'interface
    last_prediction = data.iloc[-1]["predicted"]
    print(f"PREDICTION_RESULT:{'pos' if last_prediction == 1 else 'neg'}")