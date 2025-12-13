import os
from pathlib import Path

# Definir Directorio Raíz del Proyecto
# Busca la carpeta padre de 'src' para situarse en la raíz
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Rutas de Datos
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = DATA_DIR / "05_models"
REPORTS_DIR = DATA_DIR / "06_reporting"

# Rutas de Artefactos
ARTEFACTS_DIR = PROJECT_DIR / "artefacts"

# Archivos Específicos
MODEL_FILES = {
    "Random Forest": MODELS_DIR / "model_random_forest.pkl",
    "Support Vector Machine": MODELS_DIR / "model_support_vector_machine.pkl",  
    "Logistic Regression": MODELS_DIR / "model_logistic_regression.pkl",
    "Decision Tree": MODELS_DIR / "model_decision_tree.pkl"
}

SCALER_PATH = ARTEFACTS_DIR / "scaler.pkl"
IMPUTER_PATH = ARTEFACTS_DIR / "imputer.pkl"
FEATURES_PATH = ARTEFACTS_DIR / "features_names.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"