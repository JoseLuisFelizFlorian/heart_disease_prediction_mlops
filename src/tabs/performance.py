import streamlit as st
import pandas as pd
import plotly.express as px
from src import config

def render(artifacts):
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