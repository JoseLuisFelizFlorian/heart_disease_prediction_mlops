import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from src import preprocessing

def render(user_input, artifacts, is_analyzing):
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