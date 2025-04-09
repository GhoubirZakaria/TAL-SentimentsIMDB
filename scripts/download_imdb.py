"""from datasets import load_dataset
import pandas as pd

# Téléchargement des données
dataset = load_dataset("imdb")

# Conversion en CSV
pd.DataFrame(dataset["train"]).to_csv("train.csv", index=False)
pd.DataFrame(dataset["test"]).to_csv("test.csv", index=False)

# Conversion en .txt (format "texte<TAB>label")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
train_df["label"] = train_df["label"].map({1: "pos", 0: "neg"})
test_df["label"] = test_df["label"].map({1: "pos", 0: "neg"})

train_df.to_csv("train.txt", sep="\t", columns=["text", "label"], index=False, header=False)
test_df.to_csv("test.txt", sep="\t", columns=["text", "label"], index=False, header=False)

print("Données prêtes dans train.txt et test.txt")"""