import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier

# 📦 Génération du dataset équilibré (simulation contamination)
@st.cache_data
def generate_and_train_model():
    n_samples = 5000
    np.random.seed(42)

    numero_conteneur = ["CT" + str(i).zfill(4) for i in range(1, n_samples + 1)]
    numero_plombage = ["PL" + str(np.random.randint(100000, 999999)) for _ in range(n_samples)]

    bruit_fond_debit = np.random.uniform(30, 90, n_samples)
    bruit_fond_cps = np.random.uniform(0.36, 2, n_samples)

    interieur_debit = bruit_fond_debit + np.random.normal(0, 2, n_samples)
    interieur_cps = bruit_fond_cps + np.random.normal(0, 0.1, n_samples)
    exterieur_debit = bruit_fond_debit + np.random.normal(0, 2, n_samples)
    exterieur_cps = bruit_fond_cps + np.random.normal(0, 0.1, n_samples)
    dose_gamma_debit = np.random.uniform(30, 90, n_samples)
    dose_gamma_cps = np.random.uniform(0.36, 2, n_samples)
    apres_plombage_debit = bruit_fond_debit + np.random.normal(0, 2, n_samples)
    apres_plombage_cps = bruit_fond_cps + np.random.normal(0, 0.1, n_samples)

    threshold = 10.0
    non_contaminated = (
        (dose_gamma_debit <= bruit_fond_debit + threshold) &
        (dose_gamma_debit <= interieur_debit + threshold) &
        (dose_gamma_debit <= exterieur_debit + threshold) &
        (apres_plombage_debit <= bruit_fond_debit + threshold) &
        (apres_plombage_debit <= interieur_debit + threshold) &
        (apres_plombage_debit <= exterieur_debit + threshold)
    )
    contamination = (~non_contaminated).astype(int)

    # Équilibrer
    idx_cont = np.where(contamination == 1)[0]
    idx_non_cont = np.where(contamination == 0)[0]
    n_balanced = min(len(idx_cont), len(idx_non_cont))
    selected = np.concatenate([
        np.random.choice(idx_cont, n_balanced, replace=False),
        np.random.choice(idx_non_cont, n_balanced, replace=False)
    ])
    np.random.shuffle(selected)

    df = pd.DataFrame({
        "bruit_fond_debit_nsvh": bruit_fond_debit[selected],
        "bruit_fond_cps": bruit_fond_cps[selected],
        "interieur_conteneur_debit_nsvh": interieur_debit[selected],
        "interieur_conteneur_cps": interieur_cps[selected],
        "exterieur_conteneur_debit_nsvh": exterieur_debit[selected],
        "exterieur_conteneur_cps": exterieur_cps[selected],
        "apres_plombage_conteneur_debit_nsvh": apres_plombage_debit[selected],
        "apres_plombage_conteneur_cps": apres_plombage_cps[selected],
        "dose_gamma_chargement_debit_nsvh": dose_gamma_debit[selected],
        "dose_gamma_chargement_cps": dose_gamma_cps[selected],
        "contamination": contamination[selected]
    })

    X = df.drop("contamination", axis=1)
    y = df["contamination"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = TabNetClassifier(verbose=0)
    clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy'],
            max_epochs=50,
            patience=10)

    acc = accuracy_score(y_test, clf.predict(X_test))

    # ✅ Sauvegarder modèle et scaler
    clf.save_model("tabnet_model")
    joblib.dump(scaler, "scaler.save")

    return clf, scaler, acc

# 🧠 Entraînement
st.title("☢️ Détection de Contamination - CNSTN")
with st.spinner("🔧 Entraînement du modèle en cours..."):
    model, scaler, accuracy = generate_and_train_model()


# 🧾 Formulaire de prédiction
st.markdown("### 📋 Saisissez des mesures pour tester une prédiction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        bruit_fond_debit = st.number_input("Bruit fond débit (nSv/h)", 0.0, 200.0, 30.0)
        interieur_debit = st.number_input("Intérieur conteneur débit", 0.0, 200.0, 28.0)
        exterieur_debit = st.number_input("Extérieur conteneur débit", 0.0, 200.0, 39.0)
        apres_plombage_debit = st.number_input("Après plombage débit", 0.0, 200.0, 30.0)
        dose_gamma_debit = st.number_input("Dose gamma chargement débit", 0.0, 200.0, 40.0)
    with col2:
        bruit_fond_cps = st.number_input("Bruit fond CPS", 0.0, 10.0, 0.8)
        interieur_cps = st.number_input("Intérieur conteneur CPS", 0.0, 10.0, 0.8)
        exterieur_cps = st.number_input("Extérieur conteneur CPS", 0.0, 10.0, 1.4)
        apres_plombage_cps = st.number_input("Après plombage CPS", 0.0, 10.0, 0.6)
        dose_gamma_cps = st.number_input("Dose gamma chargement CPS", 0.0, 10.0, 0.6)

    submitted = st.form_submit_button("🔎 Prédire")

    if submitted:
        input_data = np.array([[
            bruit_fond_debit, bruit_fond_cps,
            interieur_debit, interieur_cps,
            exterieur_debit, exterieur_cps,
            apres_plombage_debit, apres_plombage_cps,
            dose_gamma_debit, dose_gamma_cps
        ]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probas = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(f"☢️ **Contaminé** ({probas[1]:.2%})")
        else:
            st.success(f"✅ **Non Contaminé** ({probas[0]:.2%})")
