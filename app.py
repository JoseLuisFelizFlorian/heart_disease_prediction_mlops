import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from datetime import datetime
from src import preprocessing, config

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

        # Botón Aleatorio (Visual por ahora)
        if st.button("🎲 Cargar Caso Aleatorio", use_container_width=True):
            st.toast("Funcionalidad de aleatoriedad en construcción...", icon="🚧")

        st.markdown("---")

        # Formulario de pacientes

        # --- Grupo 1: Paciente ---
        st.caption("👤 DATOS DEL PACIENTE")
        age = st.slider("Edad", 20, 90, 50)
        sex = st.radio("Sexo", ["M", "F"], horizontal=True, format_func=lambda x: "Masculino" if x == "M" else "Femenino")

        st.markdown("---")

        # --- Grupo 2: Signos Vitales ---
        st.caption("🩺 SIGNOS VITALES")
        c1, c2 = st.columns(2)
        resting_bp = c1.number_input("Presión (BP)", 80, 200, 120)
        cholesterol = c2.number_input("Colesterol", 80, 600, 200)
        
        fasting_bs = st.selectbox("Glucemia > 120 mg/dl?", [0, 1], format_func=lambda x: "No (Normal)" if x==0 else "Sí (Alta)")
        max_hr = st.slider("Frecuencia Cardíaca Máx.", 60, 220, 150)

        st.markdown("---")

        # --- Grupo 3: Evaluación Cardíaca ---
        st.caption("💔 EVALUACIÓN CARDÍACA")
        chest_pain = st.selectbox(
            "Tipo de Dolor (ChestPain)", 
            ["ASY", "NAP", "ATA", "TA"],
            help="ASY: Asintomático | NAP: No Anginoso | ATA: Atípica | TA: Típica"
        )
        
        exercise_angina = st.checkbox("Angina por Ejercicio?", value=False)
        ex_angina_val = "Y" if exercise_angina else "N"

        c3, c4 = st.columns(2)
        oldpeak = c3.number_input("Oldpeak", 0.0, 6.0, 0.0, step=0.1)
        st_slope = c4.selectbox("Slope", ["Up", "Flat", "Down"])
        
        resting_ecg = st.selectbox("ECG en Reposo", ["Normal", "ST", "LVH"])

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

# --- LÓGICA DE PESTAÑAS (Placeholders) ---


with tab1:
    # Si el usuario aún no presionó el botón
    if not is_analyzing:
        st.markdown(
            """
            <div style="background-color: #F8FAFC; padding: 30px; border-radius: 12px; border: 2px dashed #CBD5E1; text-align: center;">
                <h3 style="color: #64748B; margin: 0; font-size: 20px;">⏱️ Esperando input clínico...</h3>
                <p style="color: #94A3B8; margin-top: 10px;">Configura los 11 parámetros en la barra lateral y presiona <strong style="color: #DC2626">ANALIZAR RIESGO</strong> para iniciar el consenso de IA.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Si el usuario presionó ANALIZAR RIESGO 
    else:
        # Preprocesamiento (Convertir Inputs -> Números Matemáticos)
        with st.spinner("🧠 Procesando datos con los 4 modelos de IA..."):
            # Convertimos el dict del usuario a DataFrame (1 fila)
            user_df = pd.DataFrame([user_input])
            
            # Llamamos a al motor en src/preprocessing.py
            try:
                X_processed = preprocessing.preprocess_input(user_df, artifacts)
            except Exception as e:
                st.error(f"Error en el preprocesamiento: {e}")
                st.stop()

        # Inferencia (Preguntar a los 4 Modelos)
        # Guardaremos los resultados en una lista para calcular el resultado
        model_results = []
        
        for model_name, model in artifacts["models"].items():
            # Predicción dura (0 o 1)
            pred_class = model.predict(X_processed)[0]
            
            # Probabilidad (Confianza del modelo)
            # Nota: Algunos modelos como SVM necesitan probability=True al entrenar. 
            # Si falla, usamos try/except y asumimos 100% o 0%.
            try:
                proba = model.predict_proba(X_processed)[0][1] # Probabilidad de clase 1 (Enfermo)
            except:
                proba = 1.0 if pred_class == 1 else 0.0
            
            model_results.append({
                "Modelo": model_name,
                "Predicción": pred_class,
                "Probabilidad": proba
            })

        # Cálculo del Consenso (Lógica de Negocio)
        # Contamos cuántos modelos dijeron "1" (Enfermo)
        votes_positive = sum(r["Predicción"] for r in model_results)
        total_models = len(model_results)
        avg_probability = np.mean([r["Probabilidad"] for r in model_results])
        
        # Regla de decisión: Si 2 o más modelos dicen enfermo -> ALERTA
        is_high_risk = votes_positive >= 2 


        # NOTA: PARTE IMPORTANTE PARA QUE FUNCIONE EL TAB 3
        # INICIO TAB 3
        
        # Creamos un registro estructurado
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_record = {
            "Fecha": timestamp,
            "Edad": user_input["Age"],
            "Sexo": "M" if user_input["Sex"] == "M" else "F",
            "Colesterol": user_input["Cholesterol"],
            "Diagnóstico": "ALTO RIESGO" if is_high_risk else "Bajo Riesgo",
            "Probabilidad": f"{avg_probability:.1%}",
            "Modelos_Positivos": f"{votes_positive}/4"
        }
        
        # Lo agregamos a la memoria (al principio de la lista para que salga arriba)
        st.session_state["history"].insert(0, new_record)

        # FIN TAB 3


        # Visualización del Resultado Principal (Tarjeta Grande)
        if is_high_risk:
            st.error(f"### 🚨 ALERTA: POSIBLE ENFERMEDAD CARDÍACA")
            st.markdown(
                f"""
                <div style="background-color: #FEF2F2; padding: 20px; border-radius: 10px; border-left: 6px solid #DC2626;">
                    <p style="font-size: 18px; color: #991B1B;">
                        <strong>Consenso Crítico:</strong> {votes_positive} de {total_models} modelos detectan patrones de riesgo.
                    </p>
                    <p style="font-size: 14px; color: #B91C1C;">
                        Probabilidad Promedio Estimada: <strong>{avg_probability:.1%}</strong>
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.success(f"### ✅ RESULTADO: BAJO RIESGO DETECTADO")
            st.markdown(
                f"""
                <div style="background-color: #ECFDF5; padding: 20px; border-radius: 10px; border-left: 6px solid #059669;">
                    <p style="font-size: 18px; color: #065F46;">
                        <strong>Consenso Favorable:</strong> {total_models - votes_positive} de {total_models} modelos indican un estado normal.
                    </p>
                    <p style="font-size: 14px; color: #047857;">
                        Probabilidad Promedio de Enfermedad: <strong>{avg_probability:.1%}</strong>
                    </p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Desglose por Modelo (Las 4 Tarjetas Pequeñas)
        st.subheader("🔍 Segunda Opinión (Detalle por Modelo)")
        
        cols = st.columns(4)
        
        for i, res in enumerate(model_results):
            with cols[i]:
                # Definir color y texto según predicción individual
                if res["Predicción"] == 1:
                    status_color = "red"
                    status_text = "ENFERMO"
                    delta_color = "inverse" # Rojo para malo
                else:
                    status_color = "green"
                    status_text = "SANO"
                    delta_color = "normal" # Verde para bueno
                
                # Usamos st.metric para que se vea profesional
                st.metric(
                    label=res["Modelo"],
                    value=status_text,
                    delta=f"{res['Probabilidad']:.1%} Confianza",
                    delta_color=delta_color
                )

with tab2:
    st.header("🛡️ Auditoría de Rendimiento de los Modelos")
    
    # --- Preparación de Datos ---
    
    # Convertimos el diccionario metrics.json a un DataFrame de Pandas para poder graficarlo
    # Estructura actual: {'Random Forest': {'accuracy': 0.9, ...}, ...}
    metrics_df = pd.DataFrame(artifacts["metrics"]).T.reset_index()
    metrics_df = metrics_df.rename(columns={"index": "Modelo"})
    
    # Damos formato amigable a los nombres (ej: random_forest -> Random Forest)
    metrics_df["Modelo"] = metrics_df["Modelo"].apply(lambda x: x.replace("_", " ").title())

    # --- Controles de Usuario ---
    col_controls, col_graph = st.columns([1, 3])
    
    with col_controls:
        st.markdown("### ⚙️ Configuración")
        st.caption("Selecciona la métrica a evaluar:")
        
        # El usuario elige qué ver. Usamos nombres amigables.
        metric_choice = st.radio(
            "Métrica:",
            options=["recall", "accuracy", "f1_score"],
            format_func=lambda x: {
                "recall": "Recall (Sensibilidad)",
                "accuracy": "Accuracy (Global)",
                "f1_score": "F1-Score (Balance)"
            }[x]
        )
        
        st.markdown("---")
        
        # --- Panel Educativo (Insights Dinámicos) ---
        if metric_choice == "recall":
            st.info(
                """
                **¿Por qué Recall?**
                En medicina, es la métrica MÁS importante. Mide la capacidad de detectar a TODOS los enfermos.
                
                *Un Recall bajo significa que se nos escapan pacientes enfermos (Falsos Negativos).*
                """
            )
        elif metric_choice == "accuracy":
            st.info(
                """
                **¿Qué es Accuracy?**
                Es el porcentaje total de aciertos (tanto sanos como enfermos).
                
                *Es útil para tener una visión general, pero cuidado si los datos están desbalanceados.*
                """
            )
        else:
            st.info(
                """
                **¿Qué es F1-Score?**
                Es el juez imparcial. Combina precisión y sensibilidad.
                
                *Si buscas el modelo más equilibrado y robusto, fíjate en este ganador.*
                """
            )

    with col_graph:
        
        # --- Visualización con Plotly ---

        # Identificamos al mejor modelo para cambiar de color
        best_val = metrics_df[metric_choice].max()
        metrics_df["Color"] = metrics_df[metric_choice].apply(lambda x: "Ganador" if x == best_val else "Otros")
        
        fig = px.bar(
            metrics_df, 
            x="Modelo", 
            y=metric_choice,
            color="Color",
            text_auto='.2%', # Formato porcentaje
            title=f"Comparativa de {metric_choice.upper()} entre Modelos",
            color_discrete_map={"Ganador": "#DC2626", "Otros": "#9CA3AF"}, # Rojo para el mejor, gris para resto
            template="plotly_white"
        )
        
        # Ajustes finos del gráfico
        fig.update_layout(
            yaxis_title="Puntuación (0-1)",
            xaxis_title=None,
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # --- Ficha Técnica (Footer) ---
    st.markdown("---")
    st.caption(f"📅 Datos basados en el entrenamiento del modelo (Test Set). Fuente: {config.METRICS_PATH}")

with tab3:
    st.header("🕰️ Historial de Sesión")
    
    # Verificamos si hay datos en la memoria
    if len(st.session_state["history"]) == 0:
        st.info("No hay registros en esta sesión. Realiza un diagnóstico en la pestaña 1 para ver datos aquí.")
    else:
        # Convertir lista a DataFrame
        history_df = pd.DataFrame(st.session_state["history"])
        
        # Métricas Rápidas (KPIs)
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Pacientes Analizados", len(history_df))
        kpi2.metric("Casos de Alto Riesgo", len(history_df[history_df["Diagnóstico"] == "ALTO RIESGO"]))
        
        # Filtramos último registro
        last_time = history_df.iloc[0]["Fecha"].split(" ")[1]
        kpi3.metric("Último Análisis", last_time)
        
        st.markdown("---")
        
        # Mostrar Tabla con Colores
        def highlight_risk(val):
            color = "#BD0A0A5C" if val == 'ALTO RIESGO' else "#1AE87D5D" # Rojo suave vs Verde suave
            return f'background-color: {color}'

        st.dataframe(
            history_df.style.map(highlight_risk, subset=['Diagnóstico']),
            use_container_width=True,
            hide_index=True
        )
        
        # Botón de Exportación (Descargar CSV)
        csv = history_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Descargar Reporte CSV",
            data=csv,
            file_name="heart_disease_history.csv",
            mime="text/csv",
            type="primary"
        )

with tab4:
    st.header("Procesamiento por Lotes (Batch)")
    st.error("🚧 Aquí irá el área de Drag & Drop para CSV.")