import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import signal
import sys
from tqdm import tqdm
import numpy as np
import random

# Configuration
MODEL_NAME = 'distilbert-base-uncased'
BATCH_SIZE = 16
EPOCHS = 2
SAMPLE_FRACTION = 1.0

# Gestion interruption
def signal_handler(sig, frame):
    print("\nArrêt manuel détecté ! Sauvegarde du modèle...")
    model.save_pretrained("models/bert/model_interrupted")
    torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Chargement et échantillonnage
def load_imdb_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', maxsplit=1)
            if len(parts) == 2:
                text, label = parts
                label = 1 if label.lower() == 'pos' else 0
                data.append((text, label))
    sample_size = int(len(data) * SAMPLE_FRACTION)
    data = random.sample(data, sample_size)
    return pd.DataFrame(data, columns=['text', 'label'])

train_path = "data/train.txt"
test_path = "data/test.txt"

print("Chargement des données...")
train_data = load_imdb_data(train_path)
test_data = load_imdb_data(test_path)
print(f"Données chargées : Train={len(train_data)}, Test={len(test_data)}")

# Tokenisation
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
print("Tokenisation...")

def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )

train_encodings = tokenize_texts(train_data['text'])
test_encodings = tokenize_texts(test_data['text'])

# Datasets
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    torch.tensor(train_data['label'].tolist())
)

test_dataset = TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask'],
    torch.tensor(test_data['label'].tolist())
)

# Optimisation matérielle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')

# Modèle
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Optimiseur
optimizer = AdamW(model.parameters(), lr=2e-5)
print(f"\nDébut de l'entraînement (EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE})...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{total_loss/(epoch+1):.4f}"})

# Sauvegarde
os.makedirs("models/bert/model", exist_ok=True)
model.save_pretrained("models/bert/model")
print("Modèle sauvegardé dans models/bert/model")

# Évaluation
model.eval()
y_pred, y_true = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Évaluation"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        y_pred.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Rapport
print("\nRapport de classification :")
print(classification_report(y_true, y_pred, target_names=['neg', 'pos']))

# Sauvegarde du rapport dans un fichier texte
os.makedirs("results/metrics", exist_ok=True)
report = classification_report(y_true, y_pred, target_names=['neg', 'pos'])
with open("results/metrics/bert_report.txt", "w") as f:
    f.write(report)
print("Rapport de classification sauvegardé dans results/metrics/bert_report.txt")


# Matrice de confusion
os.makedirs("results/metrics/confusion_matrices", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(y_true, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=['neg', 'pos'],
    yticklabels=['neg', 'pos']
)
plt.title("Matrice de Confusion (DistilBERT)")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("results/metrics/confusion_matrices/bert.png", dpi=150)
plt.close()

print("Résultats sauvegardés dans results/")
torch.cuda.empty_cache()