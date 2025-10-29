import streamlit as st
import json
import os

# ---------- TITLE ----------
st.set_page_config(page_title="Smart Health Assistant", layout="centered")

st.title("🧠 Smart Health Assistant — Image + Knowledge-based")

# Sidebar for navigation
mode = st.sidebar.radio("Choose mode:", ["🧩 Knowledge-Based AI Agent", "🩺 Image-Based Diagnosis"])

# ---------- KNOWLEDGE-BASED AI ----------
if mode == "🧩 Knowledge-Based AI Agent":

    st.markdown("""
        <style>
        .stApp {
            background-color: #f8fbff;
        }
        .stButton>button {
            border-radius: 10px;
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }
        .rule-box {
            background-color: #ffffff;
            padding: 12px 18px;
            border-radius: 10px;
            margin-bottom: 8px;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    st.header("🤖 Smart Health Knowledge Agent")

    st.caption("This mini AI guesses your condition from symptoms. You can teach it new rules too!")

    RULES_FILE = "rules.json"

    # Load rules
    if os.path.exists(RULES_FILE):
        with open(RULES_FILE, "r") as f:
            rules = json.load(f)
    else:
        rules = [
            {"if": ["sneezing", "cough", "cold"], "then": "flu"},
            {"if": ["fever", "body pain"], "then": "viral infection"},
            {"if": ["headache", "nausea"], "then": "migraine"},
            {"if": ["rash", "itching"], "then": "allergy"},
            {"if": ["tiredness", "weakness"], "then": "fatigue"},
        ]

    # Display existing rules
    st.subheader("🧩 Known Rules")
    for i, rule in enumerate(rules):
        cols = st.columns([3, 1])
        cols[0].markdown(
            f'<div class="rule-box">If <b>{", ".join(rule["if"])}</b> → <b>{rule["then"]}</b></div>',
            unsafe_allow_html=True,
        )
        if cols[1].button("❌ Delete", key=f"delete_{i}"):
            del rules[i]
            with open(RULES_FILE, "w") as f:
                json.dump(rules, f, indent=4)
            st.rerun()

    # Add a new rule
    st.subheader("💡 Add a New Rule")
    new_if = st.text_input("If parts (comma separated)")
    new_then = st.text_input("Then part")
    if st.button("Add Rule"):
        if new_if and new_then:
            rules.append({"if": [x.strip() for x in new_if.split(",")], "then": new_then.strip()})
            with open(RULES_FILE, "w") as f:
                json.dump(rules, f, indent=4)
            st.success("✅ Rule added!")
            st.rerun()
        else:
            st.warning("Please fill both fields.")

    # Inference section
    st.subheader("🧠 Ask the AI")
    symptoms_input = st.text_input("Enter symptoms (comma separated)")
    if st.button("Infer"):
        if symptoms_input:
            symptoms = [x.strip().lower() for x in symptoms_input.split(",")]
            conclusions = []
            for rule in rules:
                if all(sym in symptoms for sym in rule["if"]):
                    conclusions.append(rule["then"])
            if conclusions:
                st.success("Possible condition(s): " + ", ".join(conclusions))
            else:
                st.info("No exact match found.")
        else:
            st.warning("Please enter symptoms to infer.")

# ---------- IMAGE-BASED DIAGNOSIS ----------
else:
    st.header("🩺 Image-Based Diagnosis")
    st.caption("Upload a medical image (like X-ray or skin lesion) to analyze with AI.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.info("🔍 (Image analysis model placeholder — connect your model here)")
