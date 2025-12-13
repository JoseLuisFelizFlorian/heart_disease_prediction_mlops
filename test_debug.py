from src import config
from src import preprocessing
import pandas as pd

print("--- INICIANDO DEBUG ---")

# Probar Configuración
print(f"Raíz del Proyecto: {config.PROJECT_DIR}")
print(f"Ruta de Modelos: {config.MODELS_DIR}")

# Probar Carga de Artefactos
print("\n Intentando cargar artefactos...")
artifacts = preprocessing.load_all_artifacts()

if artifacts:
    print("¡ÉXITO! Se cargaron los siguientes modelos:")
    print(artifacts["models"].keys())
    
    # Probar Preprocesamiento con un dato falso
    print("\n Probando transformación de datos...")

    # Creamos un DataFrame falso con columnas que sabemos que existen
    # (Solo ponemos Age y Sex para ver si el reindex rellena el resto con 0)
    fake_input = pd.DataFrame({
        'Age': [50],
        'Sex_M': [1],
        'Cholesterol': [200]
    })
    
    try:
        resultado = preprocessing.preprocess_input(fake_input, artifacts)
        print(f"Transformación exitosa. Shape del array: {resultado.shape}")
        print("Valores (primeros 5):", resultado[0][:5])
    except Exception as e:
        print(f"Error en transformación: {e}")

else:
    print("Falló la carga de artefactos.")

print("\n--- FIN DEBUG ---")