import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de la página
st.set_page_config(
    page_title="Triaje Clínico de Dengue",
    page_icon="🦟",
    layout="wide"
)

# 2. Cargar el modelo empaquetado (con caché para velocidad instantánea)
@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_dengue_random_forest.joblib')

artefacto = cargar_modelo()
modelo = artefacto['modelo']
columnas_req = artefacto['columnas']

# 3. Encabezado clínico
st.title("🦟 Sistema de Apoyo al Triaje Clínico de Dengue")
st.markdown(
    "Herramienta predictiva basada en **Random Forest** (`ROC-AUC: 0.996` | `Sensibilidad: 100%` | Nota: El modelo no reemplaza una prueba de diagnóstico). "
    "Complete los datos demográficos, epidemiológicos y sintomáticos del paciente."
)
st.divider()

# 4. Formulario organizado en 3 columnas clínicas
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Perfil y Entorno Epidemiológico")
    
    age = st.number_input(
        "Edad del paciente (años)",
        min_value=1, max_value=110, value=30,
        help="Variable: age | Edad reportada en años."
    )
    
    gender_txt = st.selectbox(
        "Sexo biológico",
        options=["Masculino", "Femenino"],
        help="Variable: gender (Masculino=0, Femenino=1)"
    )
    gender = 0 if gender_txt == "Masculino" else 1
    
    area_txt = st.selectbox(
        "Tipo de zona de residencia",
        options=[
            "Zona Rural",
            "Zona Semiurbana",
            "Zona Urbana"
        ],
        index=2,
        help="Variable: area_type (Rural=0, Semiurbana=1, Urbana=2)"
    )
    mapa_area = {
        "Zona Rural": 0,
        "Zona Semiurbana": 1,
        "Zona Urbana": 2
    }
    area_type = mapa_area[area_txt]
    
    expo_txt = st.radio(
        "¿Casos confirmados en hogar o entorno (últimos 14 días)?",
        options=["No / No está seguro", "Sí, confirmado"],
        help="Variable: household_dengue_exposure_14d (No/Inseguro=0, Sí=1)"
    )
    household_dengue_exposure_14d = 1 if expo_txt == "Sí, confirmado" else 0
    
    agua_txt = st.selectbox(
        "¿Presencia visible de agua estancada cerca de la vivienda?",
        options=[
            "No hay agua estancada",
            "No está seguro",
            "Sí (macetas, neumáticos, drenajes)"
        ],
        index=2,
        help="Variable: stagnant_water (No=0, No seguro=1, Sí=2)"
    )
    mapa_agua = {
        "No hay agua estancada": 0,
        "No está seguro": 1,
        "Sí (macetas, neumáticos, drenajes)": 2
    }
    stagnant_water = mapa_agua[agua_txt]

with col2:
    st.subheader("2. Fiebre y Escalas de Dolor (0–10)")
    
    fever_days = st.number_input(
        "Días totales con fiebre",
        min_value=0, max_value=30, value=4,
        help="Variable: fever_days | Ingrese 0 si no presentó fiebre."
    )
    
    # Permitir ingreso en Celsius y convertir internamente a Fahrenheit para comodidad médica
    temp_c = st.slider(
        "Temperatura corporal máxima registrada (°C)",
        min_value=36.5, max_value=42.0, value=39.2, step=0.1,
        help="Variable: max_temperature_f | Se convierte automáticamente a grados Fahrenheit (°F)."
    )
    max_temperature_f = int(round((temp_c * 9/5) + 32))
    st.caption(f"Equivalente enviado al modelo: **{max_temperature_f} °F**")
    
    headache_severity = st.slider(
        "Intensidad del dolor de cabeza (Cefalea)",
        min_value=0, max_value=10, value=7,
        help="Variable: headache_severity | 0 = Sin dolor, 10 = Dolor insoportable."
    )
    
    retro_orbital_pain = st.slider(
        "Dolor detrás de los ojos (Retroocular)",
        min_value=0, max_value=10, value=6,
        help="Variable: retro_orbital_pain | Síntoma orientador clásico de dengue."
    )
    
    muscle_pain = st.slider(
        "Dolor muscular (Mialgias)",
        min_value=0, max_value=10, value=7,
        help="Variable: muscle_pain | Escala de 0 a 10."
    )
    
    joint_pain = st.slider(
        "Dolor en articulaciones (Artralgias)",
        min_value=0, max_value=10, value=6,
        help="Variable: joint_pain | Escala de 0 a 10."
    )
    
    vomiting_per_day = st.number_input(
        "Episodios promedio de vómito por día",
        min_value=0, max_value=20, value=1,
        help="Variable: vomiting_per_day | El vómito persistente es un signo de alarma OMS."
    )

with col3:
    st.subheader("3. Síntomas Tempranos (Días 1 a 3)")
    st.caption("Marque los síntomas presentados durante los primeros 3 días de enfermedad:")
    
    early_appetite_loss = int(st.checkbox(
        "Pérdida de apetito (Hiporexia / Anorexia)",
        value=True,
        help="Variable: early_appetite_loss"
    ))
    
    early_fatigue = int(st.checkbox(
        "Fatiga o debilidad inusual",
        value=True,
        help="Variable: early_fatigue"
    ))
    
    early_nausea = int(st.checkbox(
        "Náuseas",
        value=True,
        help="Variable: early_nausea"
    ))
    
    early_rash = int(st.checkbox(
        "Erupción cutánea (Exantema / Rash)",
        value=False,
        help="Variable: early_rash"
    ))
    
    early_chills = int(st.checkbox(
        "Escalofríos o temblores",
        value=False,
        help="Variable: early_chills"
    ))
    
    early_diarrhea = int(st.checkbox(
        "Diarrea",
        value=False,
        help="Variable: early_diarrhea"
    ))
    
    st.markdown("---")
    st.markdown("**⚠️ Signos de Alarma Tempranos (OMS):**")
    
    early_abdominal_pain = int(st.checkbox(
        "Dolor abdominal intenso y continuo",
        value=False,
        help="Variable: early_abdominal_pain | Signo de alarma crítico OMS."
    ))
    
    early_bleeding = int(st.checkbox(
        "Sangrado de mucosas (encías o nariz)",
        value=False,
        help="Variable: early_bleeding | Signo de alarma crítico OMS."
    ))

st.divider()

# 5. Botón de predicción y despliegue de resultados
if st.button("🔍 Evaluar Riesgo de Dengue", type="primary", use_container_width=True):
    
    # Construir diccionario con las 20 variables exactas
    datos_paciente = {
        'age': age,
        'gender': gender,
        'area_type': area_type,
        'household_dengue_exposure_14d': household_dengue_exposure_14d,
        'stagnant_water': stagnant_water,
        'fever_days': fever_days,
        'max_temperature_f': max_temperature_f,
        'headache_severity': headache_severity,
        'retro_orbital_pain': retro_orbital_pain,
        'muscle_pain': muscle_pain,
        'joint_pain': joint_pain,
        'vomiting_per_day': vomiting_per_day,
        'early_nausea': early_nausea,
        'early_rash': early_rash,
        'early_bleeding': early_bleeding,
        'early_abdominal_pain': early_abdominal_pain,
        'early_fatigue': early_fatigue,
        'early_appetite_loss': early_appetite_loss,
        'early_chills': early_chills,
        'early_diarrhea': early_diarrhea
    }
    
    # Crear DataFrame asegurando el orden exacto del entrenamiento
    df_input = pd.DataFrame([datos_paciente])[columnas_req]
    
    # Ejecutar predicción
    prediccion = int(modelo.predict(df_input)[0])
    probabilidad = float(modelo.predict_proba(df_input)[0, 1]) * 100
    
    # Mostrar resultados visuales
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if prediccion == 1:
            st.error(f"### 🚨 RESULTADO: SOSPECHA ALTA DE DENGUE (POSITIVO)")
            st.write(f"El modelo estima una probabilidad del **{probabilidad:.1f}%** de infección por dengue.")
        else:
            st.success(f"### ✅ RESULTADO: BAJO RIESGO DE DENGUE (NEGATIVO)")
            st.write(f"El modelo estima una probabilidad de dengue de apenas **{probabilidad:.1f}%** (orientar a diagnóstico diferencial).")
            
        # Alerta clínica automática si hay signos de alarma OMS
        if early_bleeding == 1 or early_abdominal_pain == 1 or vomiting_per_day >= 3:
            st.warning(
                "**⚠️ ALERTA CLÍNICA OMS:** El paciente presenta signos de alarma "
                "(dolor abdominal intenso, sangrado de mucosas o vómito persistente). "
                "Requiere valoración médica prioritaria e hidratación inmediata."
            )
            
    with res_col2:
        st.metric(label="Probabilidad Estimada de Dengue", value=f"{probabilidad:.1f}%")
        st.progress(int(probabilidad))
st.divider()
st.caption("© **Autor: Diego Reyes 2026** | Herramienta de apoyo clínico — No sustituye el diagnóstico médico profesional.")
