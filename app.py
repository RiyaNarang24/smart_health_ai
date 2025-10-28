from flask import Flask, render_template, request, redirect, url_for
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# -----------------------------
# Load Rules from JSON
# -----------------------------
RULES_FILE = "rules.json"
if os.path.exists(RULES_FILE):
    with open(RULES_FILE, "r") as f:
        rules = json.load(f)
else:
    rules = [
        {"if": ["sneezing", "cough", "cold"], "then": "flu"},
        {"if": ["fever", "body pain"], "then": "viral infection"},
        {"if": ["headache", "nausea"], "then": "migraine"},
        {"if": ["rash", "itching"], "then": "allergy"},
        {"if": ["tiredness", "weakness"], "then": "fatigue"}
    ]

# Load CNN Model (for image detection)
MODEL_PATH = "disease_model.h5"
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Helper Functions
# -----------------------------
def save_rules():
    with open(RULES_FILE, "w") as f:
        json.dump(rules, f, indent=4)

def infer_from_rules(facts):
    conclusions = []
    for rule in rules:
        if all(cond in facts for cond in rule["if"]):
            conclusions.append(rule["then"])
    if not conclusions:
        # If no exact match, calculate probabilities based on partial matches
        possible = {}
        for rule in rules:
            match_count = len(set(rule["if"]) & set(facts))
            if match_count > 0:
                prob = (match_count / len(rule["if"])) * 100
                possible[rule["then"]] = round(prob, 2)
        conclusions = {"probabilities": possible}
    return conclusions

def predict_disease_from_image(img_path):
    if not model:
        return "Model not available"
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array)
    class_names = os.listdir('dataset')
    result = class_names[np.argmax(preds)]
    return result

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html', rules=rules, results=None)

@app.route('/add_rule', methods=['POST'])
def add_rule():
    conditions = request.form.get('conditions', '').lower().split(',')
    conclusion = request.form.get('conclusion', '').strip().lower()
    conditions = [c.strip() for c in conditions if c.strip().replace(' ', '').isalpha()]
    if conditions and conclusion.replace(' ', '').isalpha():
        rules.append({"if": conditions, "then": conclusion})
        save_rules()
    return redirect(url_for('index'))

@app.route('/delete_rule/<int:index>')
def delete_rule(index):
    if 0 <= index < len(rules):
        rules.pop(index)
        save_rules()
    return redirect(url_for('index'))

@app.route('/infer', methods=['POST'])
def infer():
    user_facts = request.form.get('facts', '').lower().split(',')
    user_facts = [f.strip() for f in user_facts if f.strip().replace(' ', '').isalpha()]
    conclusions = infer_from_rules(user_facts)
    return render_template('index.html', rules=rules, results={'conclusions': conclusions})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    prediction = predict_disease_from_image(filepath)
    return render_template('index.html', rules=rules, results={'image_result': prediction})

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
