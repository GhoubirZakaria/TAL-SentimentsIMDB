import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

def load_report(file_path):
    """Charge un rapport de classification et extrait les métriques"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extraction des valeurs numériques
    metrics = {
        'accuracy': float(content.split('accuracy')[1].split()[1]),
        'precision_pos': float(content.split('pos')[1].split()[1]),
        'recall_pos': float(content.split('pos')[1].split()[2]),
        'f1_pos': float(content.split('pos')[1].split()[3]),
        'precision_neg': float(content.split('neg')[1].split()[1]),
        'recall_neg': float(content.split('neg')[1].split()[2]),
        'f1_neg': float(content.split('neg')[1].split()[3])
    }
    return metrics

def generate_comparison_plot(bert_metrics, svm_metrics):
    """Génère un graphique comparatif clair"""
    metrics = ['accuracy', 'precision_pos', 'recall_pos', 'f1_pos']
    labels = ['Accuracy', 'Precision (Pos)', 'Recall (Pos)', 'F1-Score (Pos)']
    
    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar([i - width/2 for i in x], 
                   [svm_metrics[m] for m in metrics], 
                   width, label='SVM (TF-IDF)', color='#1f77b4')
    rects2 = ax.bar([i + width/2 for i in x], 
                   [bert_metrics[m] for m in metrics], 
                   width, label='BERT', color='#ff7f0e')

    # Configuration du graphique
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Comparaison des performances entre SVM et BERT', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)

    # Ajout des valeurs sur les barres
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig

def save_comparison_report(bert_metrics, svm_metrics):
    """Génère un rapport textuel détaillé"""
    report = """=== RAPPORT COMPARATIF ===
    
Performance sur le jeu de test IMDB:

1. Scores globaux:
{metrics_table}

2. Détails par modèle:
   • BERT:
     - Accuracy: {bert_acc:.2%}
     - F1-Score Positif: {bert_f1_pos:.2%}
     - F1-Score Negatif: {bert_f1_neg:.2%}

   • SVM:
     - Accuracy: {svm_acc:.2%}
     - F1-Score Positif: {svm_f1_pos:.2%}
     - F1-Score Negatif: {svm_f1_neg:.2%}

3. Analyse:
   • BERT surpasse SVM de {acc_diff:.2%} en accuracy
   • La différence est plus marquée sur le recall positif ({recall_diff:.2%})
""".format(
        metrics_table=pd.DataFrame({
            'Metric': ['Accuracy', 'Precision (Pos)', 'Recall (Pos)', 'F1-Score (Pos)'],
            'BERT': [bert_metrics['accuracy'], bert_metrics['precision_pos'], 
                    bert_metrics['recall_pos'], bert_metrics['f1_pos']],
            'SVM': [svm_metrics['accuracy'], svm_metrics['precision_pos'],
                   svm_metrics['recall_pos'], svm_metrics['f1_pos']]
        }).to_string(index=False),
        bert_acc=bert_metrics['accuracy'],
        bert_f1_pos=bert_metrics['f1_pos'],
        bert_f1_neg=bert_metrics['f1_neg'],
        svm_acc=svm_metrics['accuracy'],
        svm_f1_pos=svm_metrics['f1_pos'],
        svm_f1_neg=svm_metrics['f1_neg'],
        acc_diff=bert_metrics['accuracy'] - svm_metrics['accuracy'],
        recall_diff=bert_metrics['recall_pos'] - svm_metrics['recall_pos']
    )
    
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/comparison_report.txt', 'w') as f:
        f.write(report)

def main():
    # Chargement des rapports
    bert_metrics = load_report('results/metrics/bert_report.txt')
    svm_metrics = load_report('results/metrics/svm_report.txt')
    
    # Génération des visualisations
    fig = generate_comparison_plot(bert_metrics, svm_metrics)
    fig.savefig('results/metrics/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Création du rapport textuel
    save_comparison_report(bert_metrics, svm_metrics)
    
    print("Comparaison terminée. Résultats sauvegardés dans :")
    print("- results/metrics/model_comparison.png")
    print("- results/metrics/comparison_report.txt")

if __name__ == '__main__':
    main()