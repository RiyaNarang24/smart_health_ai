# combined_app.py
import streamlit as st
import json
import os
from PIL import Image
import numpy as np

# ---- Image model imports (lazy) ----
MODEL = None
CLASS_NAMES = None

def load_image_model(model_path="disease_model.h5"):
    global MODEL, CLASS_NAMES
    if MODEL is not None:
        return MODEL
    try:
        from tensorflow.keras.models import load_model
    except Exception as e:
        st.error("TensorFlow / Keras not available. Image model won't run.")
        raise
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at {model_path}. Image detection will not work.")
        return None
    MODEL = load_model(model_path)
    # class indices are usually saved in training; assume binary classes if unknown:
    CLASS_NAMES = getattr(MODEL, "class_names", None)
    if CLASS_NAMES is None:
        # fallback to common mapping; change if needed
        CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
    return MODEL

def predict_image(image: Image.Image, model):
    img = image.convert("RGB").resize((128,128))
    arr = np.asarray(img)/255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)
    if preds.shape[-1] > 1:
        # multiclass softmax
        idx = preds.argmax(axis=-1)[0]
        conf = float(preds[0, idx])
        label = CLASS_NAMES[idx] if CLASS_NAMES else str(idx)
    else:
        # single output sigmoid
        conf = float(preds[0][0])
        idx = 1 if conf >= 0.5 else 0
        label = CLASS_NAMES[idx] if CLASS_NAMES else ("PNEUMONIA" if idx==1 else "NORMAL")
        conf = conf if idx==1 else 1.0 - conf
    return label, conf

# ---- Knowledge base functions ----
def load_rules(path="rules.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            rules = json.load(f)
        except Exception:
            rules = []
    return rules

def infer_from_rules(observed_facts, rules):
    observed = set([f.strip().lower() for f in observed_facts if f.strip()])
    conclusions = []
    partial = {}
    for rule in rules:
        conds = [c.strip().lower() for c in rule.get("if", [])]
        then = rule.get("then", "")
        if not conds or not then: continue
        matched = sum(1 for c in conds if c in observed)
        if matched == len(conds):
            conclusions.append(then)
        elif matched > 0:
            p = int(matched/len(conds)*100)
            partial[then] = max(partial.get(then,0), p)
    return list(set(conclusions)), partial

# ---- Streamlit UI ----
st.set_page_config("Smart Health Assistant", layout="centered")
st.title("🤖 Smart Health Assistant — Image + Knowledge-based")

menu = st.sidebar.radio("Choose tool", ["Pneumonia (Image)", "Knowledge-Based Agent"])
rules = load_rules("rules.json")

if menu == "Pneumonia (Image)":
    st.header("Chest X-Ray Pneumonia Detector")
    st.write("Upload a chest x-ray image (jpg, png). Model will predict if pneumonia is present.")
    model = load_image_model("disease_model.h5")
    uploaded = st.file_uploader("Choose a chest X-ray image", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_column_width=True)
        if model is None:
            st.error("Model not loaded. Check disease_model.h5 file or deploy settings.")
        else:
            with st.spinner("Predicting..."):
                label, conf = predict_image(img, model)
            st.success(f"Prediction: **{label}**  — confidence {conf*100:.1f}%")
            # extra: simple explanation
            if label.lower().startswith("pneu"):
                st.info("This prediction suggests possible pneumonia. Ask a doctor for a medical diagnosis.")
    st.write("---")
    st.write("Tip: if your model file is not present upload `disease_model.h5` to repo or use Git LFS.")

elif menu == "Knowledge-Based Agent":
    st.header("Knowledge-Based AI Agent (Propositional rules)")
    st.write("Enter observed symptoms and the rule-based engine will try to conclude conditions.")
    st.write("Example facts: `sneezing, cough, cold`")
    col1, col2 = st.columns([2,1])
    with col1:
        facts_text = st.text_input("Observed facts (comma separated)", "")
        if st.button("Infer"):
            facts = [f.strip().lower() for f in facts_text.split(",") if f.strip()]
            conclusions, partial = infer_from_rules(facts, rules)
            if conclusions:
                st.success("Conclusions: " + ", ".join(conclusions))
            else:
                st.warning("No exact conclusions. Partial matches (probability):")
                if partial:
                    for k,v in partial.items():
                        st.write(f"- {k}: {v}% match")
                else:
                    st.info("No partial matches found; try adding known symptoms or check rules.json content.")
    with col2:
        st.write("Rules (from rules.json):")
        if rules:
            for i,r in enumerate(rules):
                st.write(f"R{i+1}: If {', '.join(r.get('if',[]))} -> {r.get('then','')}")
        else:
            st.info("No rules found. Create a `rules.json` file in the repo with rules list.")
