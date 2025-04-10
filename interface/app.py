from flask import Flask, render_template, request, jsonify
from pathlib import Path
import subprocess
import os
import sys
import logging
from flask_cors import CORS  # Ajout de CORS


# Configuration logging
logging.basicConfig(level=logging.INFO)

# Configuration des chemins
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

app = Flask(__name__,
            template_folder=str(BASE_DIR / 'templates'),
            static_folder=str(BASE_DIR / 'static'))

CORS(app)  # Ajout de CORS pour permettre les requêtes entre différentes origines

def create_input_file(text, true_label):
    try:
        input_path = DATA_DIR / "interface_input.txt"

        # Nettoyage du texte (éviter des sauts de ligne ou tabulations parasites)
        text = text.strip().replace('\n', ' ').replace('\t', ' ')

        label = true_label if true_label != '?' else ''
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(f"{text}\t{label}\n")

        # Vérification format
        with open(input_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if '\t' not in first_line:
                raise Exception("Mauvais format : il manque une tabulation")

        logging.info(f"Fichier d'entrée créé : {input_path}")
        return True

    except Exception as e:
        logging.error(f"Erreur création fichier : {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        model_type = data.get('model', '')
        true_label = data.get('true_label', '?')

        if not text:
            return jsonify({'error': 'Le texte ne peut pas être vide'}), 400
        if model_type not in ['svm', 'bert']:
            return jsonify({'error': 'Modèle invalide'}), 400

        if not create_input_file(text, true_label):
            return jsonify({'error': 'Erreur lors de la création du fichier'}), 500

        script_path = PROJECT_ROOT / "scripts" / "evaluate.py"
        input_path = DATA_DIR / "interface_input.txt"

        cmd = [
            sys.executable,
            str(script_path),
            model_type,
            str(input_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            logging.error("Timeout du script evaluate.py")
            return jsonify({'error': 'Temps d’exécution dépassé'}), 500

        # Logs complets pour debug
        logging.info("----- STDOUT -----")
        logging.info(result.stdout)
        logging.info("----- STDERR -----")
        logging.info(result.stderr)

        output = result.stdout.lower()
        if 'prediction_result:pos' in output:
            prediction = 'pos'
        elif 'prediction_result:neg' in output:
            prediction = 'neg'
        else:
            prediction = 'error'
            logging.error("Sortie inattendue de evaluate.py")
            logging.error(f"stdout: {result.stdout}")
            logging.error(f"stderr: {result.stderr}")

        return jsonify({
            'success': True,
            'prediction': prediction,
            'model_used': model_type,
            'true_label': true_label if true_label != '?' else 'non spécifié'
        })

    except Exception as e:
        logging.exception("Erreur côté serveur Flask")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    required_dirs = [
        BASE_DIR / 'templates',
        BASE_DIR / 'static',
        DATA_DIR
    ]
    for dir_path in required_dirs:
        if not dir_path.exists():
            logging.warning(f"Dossier manquant : {dir_path}")

    app.run(debug=True, port=5000)