# M3EXAM
Diagnostico temprano de dengue
# 🦟 Sistema de Predicción y Triaje Clínico de Dengue con Machine Learning

Proyecto integral de ciencia de datos y despliegue clínico que abarca el análisis exploratorio (EDA), limpieza de datos, entrenamiento de un modelo predictivo basado en **Random Forest** y una aplicación web interactiva en **Streamlit** para apoyar el triaje temprano de infecciones por dengue.

---

## 📋 Descripción del Proyecto

El sistema evalúa el riesgo de infección por dengue a partir de **20 variables demográficas, epidemiológicas y sintomáticas** (incluyendo síntomas tempranos de los días 1 a 3 y escalas de dolor de 0 a 10). La interfaz clínica permite ingresar datos de forma intuitiva, convierte automáticamente la temperatura de grados Celsius (`°C`) a Fahrenheit (`°F`) y genera alertas automáticas basadas en los **signos de alarma de la Organización Mundial de la Salud (OMS)**.

### Rendimiento del Modelo
* **Algoritmo:** Random Forest (`modelo_dengue_random_forest.joblib`)
* **ROC-AUC:** `0.996`
* **Sensibilidad:** `100%`

---

## 📊 Origen de los Datos y Cita Académica

Los datos crudos (`RAW.csv`) provienen de un estudio de encuesta clínica y epidemiológica realizado en Bangladesh, disponible en **Mendeley Data** bajo licencia **CC BY 4.0**:

> **Cita del Dataset:**  
> Naim, Md Niazul Islam; Hasan, Abir (2026), *“Early Detection of Dengue Infection Using Machine Learning: A Survey Dataset from Bangladesh”*, Mendeley Data, V1, DOI: `10.17632/t2h5528tc4.1`  
> 🔗 **Enlace:** [https://data.mendeley.com/datasets/t2h5528tc4/1](https://data.mendeley.com/datasets/t2h5528tc4/1)

---

## 🗂️ Estructura del Repositorio

```text
├── RAW.csv                                 # Dataset original sin procesar (1,018 registros, 23 columnas)
├── analisis_y_modelado.ipynb               # Notebook de Jupyter con EDA, auditoría de calidad, limpieza y entrenamiento
├── modelo_dengue_random_forest.joblib      # Artefacto serializado con el modelo Random Forest y columnas requeridas
├── app.py                                  # Aplicación web interactiva de triaje clínico en Streamlit
└── README.md                               # Documentación principal del proyecto
```

---

## 🧹 Pipeline de Preprocesamiento y Auditoría de Datos

Durante la fase de análisis exploratorio en el cuaderno de Jupyter, se aplicaron las siguientes transformaciones sobre `RAW.csv` para construir el archivo de entrenamiento:

1. **Normalización de texto bilingüe:** Eliminación de etiquetas en bengalí entre paréntesis `(...)` y espacios adicionales mediante expresiones regulares.
2. **Prevención de fuga de datos (*Data Leakage*):**
   * Se descartó `ever_dengue` debido a la ambigüedad temporal sobre si el diagnóstico previo correspondía al episodio actual.
   * Se descartó `district` para evitar sesgos geográficos locales y mejorar la generalización clínica del modelo.
3. **Control de calidad en temperatura (`max_temperature_f`):** Se eliminaron `12` registros con valores nulos (`NaN`) y `2` registros corruptos (`"8"` y `"Munshiganj"`), conservando **1,004 registros válidos** (rango clínico de `99 °F` a `107 °F`).
4. **Distribución de la variable objetivo (`dengue_result`):**
   * **Positivos (`1`):** 626 casos (`62.35%`)
   * **Negativos (`0`):** 378 casos (`37.65%`)
5. **Codificación de variables:**
   * `gender`: Binaria (`Male` = `0`, `Female` = `1`).
   * `area_type`: Ordinal según gradiente de urbanización (`Rural` = `0`, `Semi-urban` = `1`, `Urban` = `2`).
   * `household_dengue_exposure_14d`: Binaria (`No` / `Not sure` = `0`, `Yes` = `1`).
   * `stagnant_water`: Ordinal (`No` = `0`, `Not sure` = `1`, `Yes` = `2`).
   * Síntomas tempranos (días 1 a 3): Binarias (`No` = `0`, `Yes` = `1`).

---

## 📖 Diccionario de Variables del Modelo

| Variable | Descripción Clínica / Epidemiológica | Tipo / Valores |
| :--- | :--- | :--- |
| `age` | Edad reportada por el paciente (años). | Numérica (`14–71`) |
| `gender` | Sexo biológico reportado. | Binaria (`0`: Masculino, `1`: Femenino) |
| `area_type` | Entorno de residencia actual. | Ordinal (`0`: Rural, `1`: Semiurbana, `2`: Urbana) |
| `household_dengue_exposure_14d` | Casos confirmados de dengue en el hogar o entorno (últimos 14 días). | Binaria (`0`: No/Inseguro, `1`: Sí) |
| `stagnant_water` | Presencia visible de agua estancada cerca de la vivienda. | Ordinal (`0`: No, `1`: No seguro, `2`: Sí) |
| `fever_days` | Duración total de la fiebre durante la enfermedad. | Numérica entera (`0–16` días) |
| `max_temperature_f` | Temperatura corporal máxima registrada. | Numérica (`°F`, convertida desde `°C` en la app) |
| `headache_severity` | Intensidad máxima de cefalea (dolor de cabeza). | Ordinal (`0–10`) |
| `retro_orbital_pain` | Intensidad del dolor detrás de los ojos (retroocular). | Ordinal (`0–10`) |
| `muscle_pain` | Intensidad de mialgias (dolor muscular). | Ordinal (`0–10`) |
| `joint_pain` | Intensidad de artralgias (dolor articular). | Ordinal (`0–10`) |
| `vomiting_per_day` | Frecuencia promedio de episodios de vómito por día. | Numérica (`0–6` episodios/día) |
| `early_nausea` | Náuseas en fase temprana (días 1 a 3). | Binaria (`0`: No, `1`: Sí) |
| `early_rash` | Erupción cutánea (exantema) en fase temprana (días 1 a 3). | Binaria (`0`: No, `1`: Sí) |
| `early_bleeding` | Sangrado de mucosas (encías/nariz) en días 1 a 3 (**Signo de alarma OMS**). | Binaria (`0`: No, `1`: Sí) |
| `early_abdominal_pain` | Dolor abdominal intenso en días 1 a 3 (**Signo de alarma OMS**). | Binaria (`0`: No, `1`: Sí) |
| `early_fatigue` | Fatiga o debilidad inusual en fase temprana (días 1 a 3). | Binaria (`0`: No, `1`: Sí) |
| `early_appetite_loss` | Pérdida de apetito (hiporexia/anorexia) en días 1 a 3. | Binaria (`0`: No, `1`: Sí) |
| `early_chills` | Escalofríos o temblores en fase temprana (días 1 a 3). | Binaria (`0`: No, `1`: Sí) |
| `early_diarrhea` | Diarrea en fase temprana (días 1 a 3). | Binaria (`0`: No, `1`: Sí) |
| `dengue_result` | **Variable objetivo (Target):** Resultado diagnóstico de dengue. | Binaria (`0`: Negativo, `1`: Positivo) |

---

## 🚀 Instalación y Ejecución Local

### 1. Clonar el repositorio
```bash
git clone https://github.com
cd triaje-clinico-dengue
```

### 2. Instalar dependencias
Asegúrate de contar con Python 3.9 o superior e instala las librerías necesarias:
```bash
pip install streamlit pandas scikit-learn joblib matplotlib seaborn
```

### 3. Ejecutar la aplicación web
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## ⚠️ Aviso Médico y Limitaciones Metodológicas

* Esta herramienta ha sido desarrollada con fines de investigación y apoyo al triaje clínico inicial; **no sustituye el criterio médico profesional ni las pruebas diagnósticas confirmatorias** (NS1, IgM/IgG o RT-PCR).
* Si el paciente presenta signos de alarma de la OMS (sangrado de mucosas, dolor abdominal intenso o vómito persistente $\ge 3$ episodios/día), la aplicación emite una alerta prioritaria de atención médica e hidratación inmediata independientemente de la probabilidad estimada.

---

## ✒️ Autor

**Diego Reyes** — *2026*
