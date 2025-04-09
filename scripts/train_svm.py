import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

def load_imdb_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Gestion robuste des tabulations multiples
            parts = line.strip().split('\t', maxsplit=1)
            if len(parts) == 2:
                text, label = parts
                label = 1 if label.lower() == 'pos' else 0
                data.append((text, label))
    return pd.DataFrame(data, columns=['text', 'label'])

# Chemins des fichiers
train_path = "data/train.txt"
test_path = "data/test.txt"

# Chargement des données
try:
    train_data = load_imdb_data(train_path)
    test_data = load_imdb_data(test_path)
    print(f"Nombre d'exemples chargés : Train={len(train_data)}, Test={len(test_data)}")
except Exception as e:
    print(f"Erreur lors du chargement : {e}")
    exit()

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])
y_train = train_data['label']
y_test = test_data['label']

# Entraînement SVM
model = LinearSVC()
model.fit(X_train, y_train)

# Sauvegarde
os.makedirs("models/svm", exist_ok=True)
pickle.dump(model, open("models/svm/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/svm/vectorizer.pkl", "wb"))

# Évaluation
y_pred = model.predict(X_test)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=['neg', 'pos']))

# Création des dossiers de résultats
os.makedirs("results/metrics/confusion_matrices", exist_ok=True)

# Sauvegarde du rapport
with open("results/metrics/svm_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred, target_names=['neg', 'pos']))

# Matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d",
            cmap="Blues",
            xticklabels=['neg', 'pos'],
            yticklabels=['neg', 'pos'])
plt.title("Matrice de Confusion (SVM)")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.savefig("results/metrics/confusion_matrices/svm.png", dpi=300, bbox_inches='tight')
plt.close()

print("Résultats sauvegardés dans results/")