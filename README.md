-->pour lancer l'entrainement des deux modèles:
python scripts/train_svm.py
python scripts/train_bert.py

        (apres l'entrainemnet mes modeles sont chargés dans models (model.pkl pour SVM et model.safetensors pour BERT existent))


-->pour evaluer un fichier texte contenant des critiques par un modèle:
python scripts/evaluate.py (le modele svm ou bert) data/(target text file)

-->la Comparaison des resulats sur train et test:
        Charge automatiquement les résultats depuis :

                results/metrics/bert_report.txt

                results/metrics/svm_report.txt

        Génère deux sorties professionnelles :

            Un graphique comparatif (model_comparison.png)

            Un rapport textuel (comparison_report.txt)

        Métriques comparées :

            Accuracy globale

            Precision/Recall/F1 pour la classe positive

            Différences absolues entre les modèles
        L'éxec:
            python scripts/compare.py

