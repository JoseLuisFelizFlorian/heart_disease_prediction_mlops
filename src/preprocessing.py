import pandas as pd
import numpy as np
import pickle
import json
import streamlit as st
from src import config

# --- FUNCIÓN DE CARGA DE MODELOS Y ARTEFACTOS ---

# Carga en caché para mejor la velocidad 
@st.cache_resource
def load_all_artifacts():
    """
    Carga todos los modelos y transformadores una sola vez en memoria.
    Retorna un diccionario con todo listo para usar.
    """
    artifacts = {}
    
    # Cargar Modelos
    artifacts["models"] = {}
    for name, path in config.MODEL_FILES.items():
        try:
            with open(path, "rb") as f:
                artifacts["models"][name] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Error Crítico: No se encontró el modelo {name} en {path}")
            return None

    # Cargar Scaler, Imputer y Features Names y las Metrics
    try:
        with open(config.SCALER_PATH, "rb") as f:
            artifacts["scaler"] = pickle.load(f)
            
        with open(config.IMPUTER_PATH, "rb") as f:
            artifacts["imputer"] = pickle.load(f)
            
        with open(config.FEATURES_PATH, "rb") as f:
            artifacts["features_names"] = pickle.load(f)
            
        with open(config.METRICS_PATH, "r") as f:
            artifacts["metrics"] = json.load(f)
            
    except FileNotFoundError as e:
        st.error(f"Error Crítico: Falta un artefacto de preprocesamiento. {e}")
        return None
        
    return artifacts

# --- MOTOR DE TRANSFORMACIÓN ---

# Pipeline de Inferencia
def preprocess_input(input_df, artifacts):
    """
    Toma el DataFrame crudo del usuario (formulario) y lo transforma
    exactamente como se hizo en el entrenamiento (One-Hot + Imputer + Scaler).
    """
    # --- One-Hot Encoding ---
    # Pandas genera columnas como 'Sex_M', 'ChestPainType_TA', etc.
    df_encoded = pd.get_dummies(input_df)
    
    # --- ALINEACIÓN ---
    # El modelo espera ver TODAS las columnas con las que fue entrenado (artifacts['features_names']).
    # Si el usuario eligió "Male", la columna "Sex_F" no existirá. 
    # 'reindex' crea las columnas faltantes y las rellena con 0.
    expected_columns = artifacts["features_names"]
    df_aligned = df_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # --- Imputación ---
    # Rellenar nulos si hubiese, por seguridad
    imputer = artifacts["imputer"]
    array_imputed = imputer.transform(df_aligned)
    
    # --- Escalado (Normalización)---
    scaler = artifacts["scaler"]
    array_scaled = scaler.transform(array_imputed)
    
    return array_scaled