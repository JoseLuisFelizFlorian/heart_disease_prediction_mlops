import streamlit as st
from src import preprocessing, utils

# Importamos los nuevos módulos
from src.tabs import diagnosis, performance, history, batch

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Heart Disease AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NOTA: PARTE IMPORTANTE PARA QUE FUNCIONE EL TAB 3
# INICIO TAB 3

# --- INICIALIZAR ESTADO DE SESIÓN (HISTORIAL) ---
if "history" not in st.session_state:
    st.session_state["history"] = []

# FIN TAB 3

# --- CARGA DE ARTEFACTOS (El Motor) ---
# Usamos la función con caché que fue creada en src/preprocessing.py
artifacts = preprocessing.load_all_artifacts()

if not artifacts:
    st.stop() # Si falla la carga, detenemos la app aquí.

# --- BARRA LATERAL (INPUTS) ---
def sidebar_inputs():
    with st.sidebar:
        
        # Encabezado
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="background-color: #DC2626; color: white; padding: 10px; border-radius: 50%; display: inline-block; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <span style="font-size: 30px;">🫀</span>
                </div>
                <h1 style="font-size: 20px; font-weight: bold; color: #1F2937; margin-top: 10px;">Heart Disease AI APP</h1>
                <span style="background-color: #EFF6FF; color: #2563EB; padding: 4px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; letter-spacing: 1px;">CLINICAL SUPPORT SYSTEM</span>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # --- BOTÓN ALEATORIO REAL ---
        # Al hacer click, llama a la función de utils.py y actualiza los valores
        st.button("🎲 Cargar Caso Aleatorio", 
                  on_click=utils.update_session_state, 
                  use_container_width=True)


        st.markdown("---")

        # Formulario de pacientes

        # --- Grupo 1: Paciente ---
        st.caption("👤 DATOS DEL PACIENTE")
        
        # Agregamos key="Age"
        age = st.slider("Edad", 20, 90, 50, key="Age")
        
        # Agregamos key="Sex"
        sex = st.radio("Sexo", ["M", "F"], horizontal=True, format_func=lambda x: "Masculino" if x == "M" else "Femenino", key="Sex")

        st.markdown("---")

        # --- Grupo 2: Signos Vitales ---
        st.caption("🩺 SIGNOS VITALES")
        c1, c2 = st.columns(2)
        
        # Agregamos key="RestingBP"
        resting_bp = c1.number_input("Presión (BP)", 80, 200, 120, key="RestingBP")
        
        # Agregamos key="Cholesterol"
        cholesterol = c2.number_input("Colesterol", 80, 600, 200, key="Cholesterol")
        
        # Agregamos key="FastingBS"
        fasting_bs = st.selectbox("Glucemia > 120 mg/dl?", [0, 1], format_func=lambda x: "No (Normal)" if x==0 else "Sí (Alta)", key="FastingBS")
        
        # Agregamos key="MaxHR"
        max_hr = st.slider("Frecuencia Cardíaca Máx.", 60, 220, 150, key="MaxHR")

        st.markdown("---")

        # --- Grupo 3: Evaluación Cardíaca ---
        st.caption("💔 EVALUACIÓN CARDÍACA")
        
        # Agregamos key="ChestPainType"
        chest_pain = st.selectbox(
            "Tipo de Dolor (ChestPain)", 
            ["ASY", "NAP", "ATA", "TA"],
            help="ASY: Asintomático | NAP: No Anginoso | ATA: Atípica | TA: Típica",
            key="ChestPainType"
        )
        
        # Agregamos key="ExerciseAngina"
        exercise_angina = st.checkbox("Angina por Ejercicio?", value=False, key="ExerciseAngina")
        ex_angina_val = "Y" if exercise_angina else "N"

        c3, c4 = st.columns(2)
        
        # Agregamos key="Oldpeak"
        oldpeak = c3.number_input("Oldpeak", 0.0, 6.0, 0.0, step=0.1, key="Oldpeak")
        
        # Agregamos key="ST_Slope"
        st_slope = c4.selectbox("Slope", ["Up", "Flat", "Down"], key="ST_Slope")
        
        # Agregamos key="RestingECG"
        resting_ecg = st.selectbox("ECG en Reposo", ["Normal", "ST", "LVH"], key="RestingECG")

        # Botón de Acción Principal
        st.markdown("---")
        analyze_btn = st.button("ANALIZAR RIESGO", type="primary", use_container_width=True)

        # Retornamos los datos en un diccionario limpio
        input_data = {
            'Age': age, 'Sex': sex, 'ChestPainType': chest_pain, 
            'RestingBP': resting_bp, 'Cholesterol': cholesterol, 
            'FastingBS': fasting_bs, 'RestingECG': resting_ecg, 
            'MaxHR': max_hr, 'ExerciseAngina': ex_angina_val, 
            'Oldpeak': oldpeak, 'ST_Slope': st_slope
        }
        return input_data, analyze_btn

# Ejecutamos sidebar y capturamos datos
user_input, is_analyzing = sidebar_inputs()

# --- ÁREA PRINCIPAL (TABS) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Diagnóstico", 
    "📊 Rendimiento de Modelos", 
    "🕰️ Historial de Predicción",
    "📂 Carga Masiva de Datos"
])

# --- LÓGICA DE PESTAÑAS (Ahora modularizada) ---

with tab1:
    diagnosis.render(user_input, artifacts, is_analyzing)

with tab2:
    performance.render(artifacts)

with tab3:
    history.render()

with tab4:
    batch.render(artifacts)