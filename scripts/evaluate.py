import os
import sys
import joblib
import torch
import pandas as pd
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', maxsplit=1)
            if len(parts) == 2:
                text, label = parts
                label = 1 if label.lower() == 'pos' else 0
                data.append((text, label))

    # Afficher les 5 premières lignes du DataFrame pour vérifier le contenu
    df = pd.DataFrame(data, columns=['text', 'label'])
    print("Données chargées :")
    print(df.head())  # Affiche les 5 premières lignes du DataFrame
    return df


def evaluate_svm(data):
    model_path = "models/svm/model.pkl"
    vec_path = "models/svm/vectorizer.pkl"
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    X = vectorizer.transform(data["text"])
    y_true = data["label"]
    y_pred = model.predict(X)

    report = classification_report(y_true, y_pred, target_names=["neg", "pos"])
    print(report)

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)

    with open("results/metrics/svm_custom_report.txt", "w") as f:
        f.write(report)

    data["predicted"] = y_pred
    data.to_csv("results/predictions/svm_predictions.csv", index=False)

def evaluate_bert(data):
    model_dir = "models/bert/model"
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(list(data["text"]), truncation=True, padding=True, return_tensors="pt", max_length=512)
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    loader = DataLoader(dataset, batch_size=16)

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Évaluation BERT"):
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).tolist()
            preds.extend(batch_preds)

    y_true = data["label"].tolist()
    y_pred = preds

    report = classification_report(y_true, y_pred, target_names=["neg", "pos"])
    print(report)

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)

    with open("results/metrics/bert_custom_report.txt", "w") as f:
        f.write(report)

    data["predicted"] = y_pred
    data.to_csv("results/predictions/bert_predictions.csv", index=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Utilisation : python evaluate.py [svm|bert] data/samples.txt")
        sys.exit(1)

    model_type = sys.argv[1].lower()
    file_path = sys.argv[2]

    print(f"Chargement des données depuis {file_path}...")
    data = load_data(file_path)

    if model_type == "svm":
        evaluate_svm(data)
    elif model_type == "bert":
        evaluate_bert(data)
    else:
        print("Modèle inconnu. Utilisez 'svm' ou 'bert'.")