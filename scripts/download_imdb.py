from datasets import load_dataset
import pandas as pd
import os

# Répertoire cible
data_dir = os.path.join("data")
os.makedirs(data_dir, exist_ok=True)

# Téléchargement des données
dataset = load_dataset("imdb")

# Conversion en CSV
train_csv_path = os.path.join(data_dir, "train.csv")
test_csv_path = os.path.join(data_dir, "test.csv")
pd.DataFrame(dataset["train"]).to_csv(train_csv_path, index=False)
pd.DataFrame(dataset["test"]).to_csv(test_csv_path, index=False)

# Conversion en TXT (texte<TAB>label)
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
train_df["label"] = train_df["label"].map({1: "pos", 0: "neg"})
test_df["label"] = test_df["label"].map({1: "pos", 0: "neg"})

train_txt_path = os.path.join(data_dir, "train.txt")
test_txt_path = os.path.join(data_dir, "test.txt")
train_df.to_csv(train_txt_path, sep="\t", columns=["text", "label"], index=False, header=False)
test_df.to_csv(test_txt_path, sep="\t", columns=["text", "label"], index=False, header=False)

print(f"Données prêtes dans {data_dir}")