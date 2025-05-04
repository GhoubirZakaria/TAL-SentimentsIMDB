# TAL-SentimentsIMDB

**TAL-SentimentsIMDB** est un projet d’analyse de sentiments sur des critiques de films, utilisant le dataset IMDB.  
Il compare deux modèles d’apprentissage automatique : un modèle **SVM** (Support Vector Machine) rapide et un modèle **DistilBERT** précis.  

Le projet inclut des scripts pour :
- Entraîner les modèles,
- Évaluer des critiques,
- Comparer leurs performances,
- Tester les prédictions via une interface utilisateur interactive.


## Utilisation

#### ➤ Installer le paquet datasets de Hugging Face :
```bash
pip install datasets
```
Ensuite, exécutez download_imdb.py pour télécharger les fichiers train et test en csv, puis les convertir au format .txt.

### 1. Entraînement des Modèles

#### ➤ Entraîner le modèle SVM :
```bash
python scripts/train_svm.py
```

- **Sortie** : `models/model.pkl`  
- **Rapport** : `results/metrics/svm_report.txt`

#### ➤ Entraîner le modèle DistilBERT :
```bash
python scripts/train_bert.py
```

- **Sortie** : `models/model.safetensors`  
- **Rapport** : `results/metrics/bert_report.txt`


### 2. Évaluation d’un Fichier de Critiques

#### ➤ Commande :
```bash
python scripts/evaluate.py <modèle> data/<fichier_texte>
```

- `<modèle>` : `svm` ou `bert`  
- `<fichier_texte>` : Nom du fichier dans `data/` (ex : `reviews.txt` qui respecte le format '(critique)\tab(label)')

#### ➤ Exemple :
```bash
python scripts/evaluate.py svm data/reviews.txt
```

- **Sortie** : Prédictions (positif ou négatif) pour chaque critique, affichées dans la console.


### 3. Comparaison des Performances

#### ➤ Commande :
```bash
python scripts/compare.py
```

- **Entrées** :  
  `results/metrics/bert_report.txt`  
  `results/metrics/svm_report.txt`

- **Sorties** :  
  - `results/model_comparison.png` : Graphique comparatif (accuracy, precision, recall, F1)  
  - `results/comparison_report.txt` : Rapport détaillé des différences

- **Métriques comparées** :
  - Accuracy globale  
  - Precision, Recall, F1 pour la classe positive  
  - Différences absolues entre les modèles


### 4. Lancer l’Interface Utilisateur

#### ➤ Commande :
```bash
python interface/app.py
```

- **Accès** : [http://localhost:5000](http://localhost:5000)

#### ➤ Fonctionnalités :
- Saisie d’une critique dans un champ texte
- Choix du modèle (`SVM` ou `DistilBERT`)
- Choix d’un label original (optionnel)
- Affichage de la prédiction, label original et modèle utilisé
- Bouton **Mode Sombre** pour changer le thème


## Fonctionnement Général

### 🔹 Entraînement :
- `train_svm.py` :
  - Prétraitement (nettoyage, TF-IDF)
  - Entraînement du SVM
  - Sauvegarde dans `models/model.pkl`
- `train_bert.py` :
  - Tokenisation avec DistilBERT tokenizer
  - Entraînement du modèle
  - Sauvegarde dans `models/model.safetensors`

### 🔹 Évaluation :
- `evaluate.py` :
  - Charge un modèle
  - Lit un fichier texte
  - Effectue les prédictions ligne par ligne

### 🔹 Comparaison :
- `compare.py` :
  - Lit les rapports
  - Calcule les écarts entre modèles
  - Génère un graphique + rapport texte

### 🔹 Interface :
- `app.py` : Application Flask (serveur local)
- `index.html` / `style.css` : Design responsive, mode sombre, loader des resultats animé


## Remarques

- Les modèles doivent être entraînés **avant** toute évaluation ou usage via l’interface.
- Les critiques doivent être placées dans `data/`, **une critique par ligne, separant une critique et son label par une tabulation**.
- Si les rapports `bert_report.txt` ou `svm_report.txt` sont absents, exécutez d’abord les scripts d’entraînement.
- **DistilBERT** est plus lent sur CPU. Un GPU est **recommandé**.


## Auteurs

**GHOUBIR Zakaria**  
Projet final pour *"Introduction au traitement automatique des langues"*
