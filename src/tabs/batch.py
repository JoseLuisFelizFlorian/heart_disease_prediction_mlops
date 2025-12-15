import streamlit as st
import pandas as pd
from src import preprocessing

def render(artifacts):
    st.header("🏭 Procesamiento Masivo de Datos")
    
    # --- GENERADOR DE PLANTILLA ---
    st.markdown("### 1. Descarga la Plantilla")
    st.caption("Usa este archivo CSV como base para cargar tus datos. No cambies los nombres de las columnas.")
    
    # Creamos un dataframe vacío con las columnas correctas para que sirva de ejemplo
    # Usamos las claves del diccionario 'features_names' o un ejemplo manual
    # Para asegurar compatibilidad, definimos las columnas estándar requeridas
    required_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
    template_df = pd.DataFrame(columns=required_columns)
    
    # Convertimos a CSV
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Descargar Plantilla CSV Vacía",
        data=template_csv,
        file_name="plantilla_heart_disease.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # --- ZONA DE CARGA ---
    st.markdown("### 2. Sube tu Archivo")
    uploaded_file = st.file_uploader("Arrastra tu archivo CSV aquí", type=["csv"])
    
    if uploaded_file is not None:
        # Leemos el archivo
        batch_df = pd.read_csv(uploaded_file)
        
        # --- VALIDACIÓN DE ESQUEMA ---
        # Verificamos si faltan columnas
        missing_cols = [col for col in required_columns if col not in batch_df.columns]
        
        if len(missing_cols) > 0:
            st.error(f"❌ Error de Formato: Faltan las siguientes columnas obligatorias: {missing_cols}")
        else:
            st.success(f"✅ Archivo válido: {len(batch_df)} pacientes detectados.")
            st.dataframe(batch_df.head()) # Vista previa
            
            # --- PROCESAMIENTO ---
            if st.button("⚙️ PROCESAR LOTE AHORA", type="primary"):
                with st.spinner("Ejecutando predicciones en paralelo..."):
                    
                    try:
                        # Preprocesamiento Masivo (Reutilizamos la función del backend)
                        X_batch = preprocessing.preprocess_input(batch_df, artifacts)
                        
                        # Inferencia con todos los modelos
                        results_df = batch_df.copy() # Copiamos datos originales
                        
                        # Creamos columnas para cada modelo
                        vote_cols = []
                        for model_name, model in artifacts["models"].items():
                            
                            # Predecimos todo el array de golpe (Vectorizado = Rápido)
                            preds = model.predict(X_batch)
                            col_name = f"Pred_{model_name}"
                            results_df[col_name] = preds
                            vote_cols.append(col_name)
                        
                        # Cálculo de Consenso
                        # Sumamos las predicciones (1=Enfermo, 0=Sano)
                        results_df["Votos_Positivos"] = results_df[vote_cols].sum(axis=1)
                        results_df["Consenso_Final"] = results_df["Votos_Positivos"].apply(
                            lambda x: "ALTO RIESGO" if x >= 2 else "Bajo Riesgo"
                        )
                        
                        # Mostrar Resultados
                        st.markdown("### 🏁 Resultados del Análisis")
                        
                        # Función de color para la tabla
                        def color_risk(val):
                            color = "#BD0A0A5C" if val == 'ALTO RIESGO' else "#1AE87D5D"
                            return f'background-color: {color}'

                        st.dataframe(
                            results_df.style.map(color_risk, subset=['Consenso_Final']),
                            use_container_width=True
                        )
                        
                        # Descarga Final
                        final_csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar Resultados Completos (CSV)",
                            data=final_csv,
                            file_name="heart_disease_results_batch.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Ocurrió un error durante el procesamiento: {e}")