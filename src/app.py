import streamlit as st
import pandas as pd
import numpy as np
import joblib, json

# ---------------------------------------------------------------------------
# 1. Cargar modelo y metadatos ----------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/mejor_modelo_LASmax_71xgboost.pkl")
    feature_order = json.load(open("data/interim/feature_order.json"))
    cats = json.load(open("data/interim/categorical_values.json"))
    return model, feature_order, cats

model, FEATURE_ORDER, CATS = load_artifacts()

# ---------------------------------------------------------------------------
# 2. Interfaz ----------------------------------------------------------------
st.title("Predicción de Nivel de Ruido Aeronáutico (LASmax)")
st.markdown("Completa los datos del evento y obtén una estimación instantánea.")

# Continuas
temp = st.number_input("Temperatura [°C]", value=20.0, step=0.1)
hour = st.slider("Hora (0‑23)", 0, 23, 14)

# Categóricas (usamos listas cargadas)
nmt           = st.selectbox("NMT",            CATS["NMT"])
aircraft_type = st.selectbox("Tipo de Aeronave",  CATS["Aircraft Type"])
airline       = st.selectbox("Aerolínea",      CATS["Airline"])
runway        = st.selectbox("Pista",          CATS["Runway"])
ad            = st.selectbox("Arribo/Despegue",CATS["A/D"])

# ---------------------------------------------------------------------------
# 3. Construir vector de entrada --------------------------------------------
def build_vector():
    base = {
        "Temperature [°C]": temp,
        "Hour": hour,
        "is_night": int(hour >= 22 or hour < 6),
        "NMT": nmt,
        "Aircraft Type": aircraft_type,
        "Airline": airline,
        "Runway": runway,
        "A/D": ad,
    }
    df = pd.DataFrame([base])
    df = pd.get_dummies(
        df,
        columns=["NMT", "Aircraft Type", "Airline", "Runway", "A/D"],
        drop_first=True,
    )

    # Añade columnas faltantes y ordena
    for col in FEATURE_ORDER:
        if col not in df:
            df[col] = 0
    df = df[FEATURE_ORDER]
    return df

# ---------------------------------------------------------------------------
# 4. Predicción --------------------------------------------------------------
if st.button("Predecir"):
    X_pred = build_vector()
    pred = model.predict(X_pred)[0]
    st.success(f"Nivel de ruido estimado: **{pred:.1f} dB LASmax**")
# ---------------------------------------------------------------------------