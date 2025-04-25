import json
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# 1. Configuración inicial
# --------------------------------------------------------------------------
st.set_page_config(page_title="Predicción de Ruido Aeronáutico (LASmax)", layout="wide")
st.title("✈️ Predicción de Nivel de Ruido Aeronáutico (LASmax)")
st.markdown("Complete los datos del vuelo para obtener la estimación de ruido.")

# --------------------------------------------------------------------------
# 2. Cargar modelo y mapeos
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_xgb_843_47")  # Asegúrate de que el archivo exista
    with open("data/interim/categorical_mappings.json") as f:
        mappings = json.load(f)
    with open("data/interim/feature_order.json") as f:
        feature_order = json.load(f)
    return model, mappings, feature_order

model, mappings, FEATURE_ORDER = load_artifacts()

# --------------------------------------------------------------------------
# 3. Interfaz de usuario
# --------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Hora (0-23)", 0, 23, 14)
    temp = st.number_input("Temperatura (°C)", value=20.0, step=0.1)
    day_of_week = st.selectbox("Día de la semana", 
                             ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"])

with col2:
    ad = st.selectbox("Arribo/Despegue (A/D)", options=list(mappings["A/D"].keys()))
    runway = st.selectbox("Pista", options=list(mappings["Runway"].keys()))
    airline = st.selectbox("Aerolínea", options=list(mappings["Airline"].keys()))
    aircraft_type = st.selectbox("Tipo de Aeronave", options=list(mappings["Aircraft Type"].keys()))
    from_to = st.selectbox("Origen/Destino (From/To)", options=list(mappings["From/To"].keys()))

# --------------------------------------------------------------------------
# 4. Construcción del vector de entrada (con validación robusta)
# --------------------------------------------------------------------------
def build_vector():
    # Mapeo de días a números
    day_map = {
        "Lunes": 0, "Martes": 1, "Miércoles": 2, 
        "Jueves": 3, "Viernes": 4, "Sábado": 5, "Domingo": 6
    }
    
    # Función para manejar categorías con coincidencia exacta
    def get_mapped_code(category, value):
        if value not in mappings[category]:
            st.error(f"Error: Valor '{value}' no encontrado en {category}. Valores permitidos: {list(mappings[category].keys())}")
            st.stop()  # Detiene la ejecución si hay un error
        return mappings[category][value]
    
    # Construye el vector
    vector = {
        "Hour": hour,
        "DayOfWeek": day_map[day_of_week],
        "Temp": temp,
        "A/D": get_mapped_code("A/D", ad),
        "Runway": get_mapped_code("Runway", runway),
        "Airline": get_mapped_code("Airline", airline),
        "From/To": get_mapped_code("From/To", from_to),
        "Aircraft Type": get_mapped_code("Aircraft Type", aircraft_type),
        "is_night": int(hour >= 22 or hour < 7)  # 1 si es horario nocturno (22:00-7:00)
    }
    
    # Convierte a DataFrame y asegura el orden correcto
    df = pd.DataFrame([vector])
    return df[FEATURE_ORDER], vector["is_night"]

# --------------------------------------------------------------------------
# 5. Predicción y resultados
# --------------------------------------------------------------------------
if st.button("Calcular Nivel de Ruido"):
    X_pred, is_night = build_vector()
    pred = model.predict(X_pred)[0]
    
    # Ajuste nocturno (+10 dB)
    pred_ajustado = pred + 10 if is_night else pred
    
    # Mostrar resultados
    st.success(f"""
    **Resultado:**
    - Nivel de ruido base: **{pred:.1f} dB LASmax**
    - Ajuste nocturno (+10 dB): **{pred_ajustado:.1f} dB LASmax** {"(Aplicado)" if is_night else ""}
    """)
    
    # Detalles técnicos (opcional)
    with st.expander("🔍 Ver detalles técnicos"):
        st.write("**Variables de entrada:**")
        st.json({
            "Hora": hour,
            "Día de la semana": day_of_week,
            "Temperatura (°C)": temp,
            "A/D": ad,
            "Pista": runway,
            "Aerolínea": airline,
            "Origen/Destino": from_to,
            "Tipo de Aeronave": aircraft_type,
            "Horario Nocturno": "Sí" if is_night else "No"
        })
        
        st.write("**Vector procesado para el modelo:**")
        st.dataframe(X_pred)

# --------------------------------------------------------------------------
# 6. Notas adicionales
# --------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
**Notas:**
- El modelo predice el nivel máximo de ruido (LASmax) en decibelios (dB).
- Se aplica un ajuste de **+10 dB** para vuelos entre **22:00 y 7:00**.
- Los valores categóricos deben coincidir exactamente con los usados en el entrenamiento.
""")