import numpy as np
import streamlit as st

# Genera datos aleatorios para pruebas
def generate_random_patient():
    """Genera datos clínicos aleatorios para simulación."""
    return {
        'Age': int(np.random.randint(30, 80)),
        'Sex': np.random.choice(['M', 'F']),
        'RestingBP': int(np.random.randint(95, 170)),
        'Cholesterol': int(np.random.randint(150, 400)),
        'FastingBS': int(np.random.choice([0, 1], p=[0.8, 0.2])),
        'MaxHR': int(np.random.randint(80, 200)),
        'ChestPainType': np.random.choice(['ASY', 'NAP', 'ATA', 'TA']),
        'ExerciseAngina': bool(np.random.choice([True, False])),
        'Oldpeak': float(np.round(np.random.uniform(0, 3.5), 1)),
        'ST_Slope': np.random.choice(['Up', 'Flat', 'Down']),
        'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'])
    }


#Generar los datos en memoria
def update_session_state():
    """Función callback para el botón."""
    new_data = generate_random_patient()
    
    # Actualizamos los datos
    for key, value in new_data.items():
        st.session_state[key] = value
        
    # Enviamos la notificación (Toast)
    st.toast("Datos clínicos simulados generados correctamente.", icon="🎲")