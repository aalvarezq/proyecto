import json
import joblib
import pandas as pd
import streamlit as st

# 1. Configuración inicial
st.set_page_config(page_title="Predicción de Ruido Aeronáutico (LASmax)", layout="wide")
st.title("Predicción de Nivel de Ruido Aeronáutico (LASmax) en zona Jardines del sur.")
st.markdown("Complete los datos del vuelo para obtener la estimación de ruido.")

# 2. Cargar modelo y mapeos
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_xgb_843_47")  
    
    with open("data/interim/categorical_mappings.json") as f:
        mappings = json.load(f)
        
    with open("data/interim/feature_order.json") as f:
        feature_order = json.load(f)
        
    aerodic = pd.read_csv("data/processed/aerolineas.csv")
    airline_name_to_code = dict(zip(aerodic['Airline (Name)'], aerodic['Airline']))
    
    with open("data/interim/fromto_name_to_code.json") as f:
        fromto_name_to_code = json.load(f)
    
    return model, mappings, feature_order, airline_name_to_code, fromto_name_to_code

model, mappings, FEATURE_ORDER, airline_name_to_code, fromto_name_to_code = load_artifacts()

# 3. Interfaz de usuario
col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Hora (0-23)", 0, 23, 14)
    temp = st.number_input("Temperatura (°C)", value=20, step=1)
    day_of_week = st.selectbox(
        "Día de la semana", 
        ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    )

with col2:
    ad_translation = {"Departure": "Despegue", "Arrival": "Arribo"}
    ad_display = st.selectbox("Arribo/Despegue (A/D)", options=[ad_translation.get(x, x) for x in mappings["A/D"].keys()])
    runway = st.selectbox("Pista", options=list(mappings["Runway"].keys()))
    airline_name = st.selectbox("Aerolínea", options=list(airline_name_to_code.keys()))
    from_to = st.selectbox("Origen/Destino (From/To)", options=list(fromto_name_to_code.keys()))
    aircraft_type = st.selectbox("Tipo de Aeronave", options=list(mappings["Aircraft Type"].keys()))

# 4. Construcción del vector de entrada
def build_vector():
    reverse_ad_translation = {v: k for k, v in ad_translation.items()} #devolver el valor original en inglés antes de mapear

    # Mapeo de días a números
    day_map = {
        "Lunes": 0, "Martes": 1, "Miércoles": 2, 
        "Jueves": 3, "Viernes": 4, "Sábado": 5, "Domingo": 6
    }
    
    # Función para manejar categorías
    def get_mapped_code(category, value):
        if value not in mappings[category]:
            st.error(f"Error: Valor '{value}' no encontrado en {category}.")
            st.stop()
        return mappings[category][value]
    
    vector = {
        "Hour": hour,
        "DayOfWeek": day_map[day_of_week],
        "Temp": temp,
        "A/D": get_mapped_code("A/D", reverse_ad_translation.get(ad_display, ad_display)),
        "Runway": get_mapped_code("Runway", runway),
        "Airline": get_mapped_code("Airline", airline_name_to_code[airline_name]),
        "From/To": get_mapped_code("From/To", fromto_name_to_code[from_to]),
        "Aircraft Type": get_mapped_code("Aircraft Type", aircraft_type),
        "is_night": int(hour >= 22 or hour < 7)
    }
    
    df = pd.DataFrame([vector])
    return df[FEATURE_ORDER], vector["is_night"]

# 5. Predicción y resultados
if st.button("Calcular Nivel de Ruido"):
    X_pred, is_night = build_vector()
    pred = model.predict(X_pred)[0]
    
    pred_ajustado = pred + 10 if is_night else pred
    
    st.success(f"""
    **Resultado:**
    - Nivel de ruido base: **{pred:.1f} dB LASmax**
    - Ajuste nocturno (+10 dB): **{pred_ajustado:.1f} dB LASmax** {"(Aplicado)" if is_night else ""}
    """)
    
    with st.expander("Ver detalles técnicos"):
        st.write("**Variables de entrada:**")
        st.json({
            "Hora": hour,
            "Día de la semana": day_of_week,
            "Temperatura (°C)": temp,
            "A/D": ad_display,
            "Pista": runway,
            "Aerolínea": airline_name,
            "Origen/Destino": from_to,
            "Tipo de Aeronave": aircraft_type,
            "Horario Nocturno": "Sí" if is_night else "No"
        })
        
        st.write("**Vector procesado para el modelo:**")
        st.dataframe(X_pred)

# 6. Notas adicionales
st.markdown("---")
st.markdown("""
**Notas:**
- El modelo predice el nivel máximo de ruido (LASmax) en decibelios (dB).
- Se aplica un ajuste de **+10 dB** para vuelos entre **22:00 y 7:00**.
""")