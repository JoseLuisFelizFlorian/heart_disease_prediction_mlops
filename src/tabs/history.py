import streamlit as st
import pandas as pd

def render():
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