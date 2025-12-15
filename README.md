# 🫀 Heart Disease Prediction (MLOps End-to-End) - Clinical Decision Support System (CDSS)

Este proyecto implementa una solución completa de **Machine Learning Operacional (MLOps)** diseñada como una herramienta de soporte al diagnóstico médico. El sistema integra un pipeline de entrenamiento con una aplicación web interactiva y modularizada, priorizando la **Sensibilidad (Recall)** para minimizar los falsos negativos en la detección de patologías cardíacas.

---

## 1. Arquitectura y Componentes Clave

El proyecto se aleja de los scripts monolíticos tradicionales, adoptando una arquitectura de software modular basada en **Cookiecutter Data Science**.

| Componente | Herramienta | Propósito Técnico |
| :--- | :--- | :--- |
| **Pipeline ML** | `scikit-learn`, `pandas` | Entrenamiento comparativo de 4 modelos (`Random Forest`, `Support Vector Machine (SVM)`, `Logistic Regression`, `Decision Tree`) y serialización de artefactos de preprocesamiento. |
| **Frontend / UI** | **Streamlit** | Interfaz clínica interactiva con gestión de estado (`st.session_state`) y diseño modular. |
| **Arquitectura** | **Python Modular** | Desacoplamiento de lógica de negocio (`src/utils.py`) y vistas (`src/tabs/`). |
| **Persistencia** | **Pickle** | Serialización binaria eficiente de modelos y escaladores (`StandardScaler`). |
| **Control de Calidad** | **Git** | Control de versiones siguiendo la convención *Conventional Commits*. |

---

## 2. Stack Tecnológico y Métodos Implementados

El desarrollo se realizó en un entorno local (**Windows 11**) utilizando las siguientes librerías y métodos clave:

### 2.1. Ingeniería de Software y Frontend (Streamlit)
| Librería | Propósito | Implementación Clave |
| :--- | :--- | :--- |
| **`streamlit`** | Framework Web | `st.set_page_config`, `st.tabs` para navegación modular. |
| **Gestión de Estado** | Persistencia de sesión | `st.session_state` para mantener datos al recargar la página e interactuar con widgets. |
| **UX / UI** | Experiencia de Usuario | `st.toast` para notificaciones asíncronas y `st.metric` para KPIs visuales. |
| **`src.utils`** | Lógica Auxiliar | Generador de pacientes aleatorios (`generate_random_patient`) conectado vía *callbacks*. |

### 2.2. Ciencia de Datos y Machine Learning
| Librería | Propósito | Implementación Clave |
| :--- | :--- | :--- |
| **`scikit-learn`** | Modelado | `GridSearchCV`, `Pipeline`, `StandardScaler`, `OneHotEncoder`. |
| **`pickle`** | Serialización | `pickle.dump()`/`load()` para persistir el modelo entrenado y el `features_names`. |
| **`plotly`** | Visualización | `px.bar` y `px.scatter` para la auditoría de rendimiento de modelos. |
| **Métricas** | Evaluación Clínica | Optimización de **Recall** (Sensibilidad) sobre Accuracy. |

---

## 3. Funcionalidades del Sistema (The "App")

La aplicación (`app.py`) actúa como un orquestador que carga módulos independientes situados en `src/tabs/`:

### A. Diagnóstico Individual & Simulación
* **Generador de Casos (Feature Destacada):** Botón "🎲 Cargar Caso Aleatorio" que utiliza `numpy` para simular perfiles clínicos realistas, actualizando automáticamente los widgets mediante `session_state`.
* **Inferencia en Tiempo Real:** Cálculo de riesgo utilizando el modelo seleccionado.

### B. Procesamiento por Lotes (Batch Inference)
* **Carga Masiva:** Permite subir archivos CSV con múltiples pacientes.
* **Vectorización:** El pipeline de predicción utiliza operaciones vectorizadas de Pandas (evitando bucles `for` lentos) para procesar cientos de registros en milisegundos.
* **Exportación:** Generación de reportes descargables en CSV con las predicciones anexadas.

### C. Auditoría de Modelos (Performance Audit)
* **Transparencia:** Dashboard interactivo que compara las métricas (Recall, Accuracy, F1) de los 4 modelos evaluados:
    * *Random Forest*
    * *Support Vector Machine (SVM)*
    * *Logistic Regression*
    * *Decision Tree*

---

## 4. Estructura del Repositorio

El proyecto sigue estrictamente el estándar de la industria para garantizar la reproducibilidad y el orden.

```text
heart_disease_prediction_mlops/
├── api/                      # Código fuente de la API/Backend (si aplica)
├── artefacts/                # Objetos binarios de preprocesamiento (Scaler, Imputer)
├── data/
│   ├── 01_raw/               # Datos originales inmutables
│   ├── 02_interim/           # Datos limpios tras el Data Health Check
│   ├── 03_processed/         # Datos listos para entrenamiento
│   ├── 04_external/          # Fuentes externas
│   ├── 05_models/            # Modelos entrenados (.pkl)
│   └── 06_reporting/         # Métricas (JSON) y Figuras (HTML)
├── docs/                     # Documentación del proyecto
├── notebooks/                # Flujo de trabajo (00_Setup, 01_EDA, 02_Training)
├── references/               # Diccionarios de datos y manuales
├── src/                      # Código fuente modular y scripts auxiliares
├── tests/                    # Tests unitarios para el código
├── .gitignore                # Archivos ignorados por Git
└── README.md                 # Documentación principal
