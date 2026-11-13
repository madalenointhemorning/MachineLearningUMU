#!/usr/bin/env python
# coding: utf-8

# In[115]:


#%pip install mlxtend


# # Preprocesamiento
# 
# Apartado del enunciado: **3.1**, mínimo tres técnicas de preprocesamiento, indicando qué problemática
# resuelve cada una.
# 
# ## Objetivos
# 
# 1. Saber **cómo afecta cada preprocesado a los modelos**, con un ajuste mínimo de hiperparámetros (el
#    ajuste completo se hace en el modelado).
# 2. Saber **si imputar los nulos ayuda o no a Random Forest**, que puede trabajar con NaN directamente.
# 
# ## Estructura del cuaderno
# 
# El cuaderno está pensado para que **añadir un preprocesado nuevo sea solo añadir una línea**:
# 
# 1. **Configuración**: librerías, semilla y umbrales (todos salen de `01_eda.ipynb`).
# 2. **Carga del dataset** y resumen de nulos.
# 3. **Catálogo de pasos**: cada paso es una función `df -> df` que imprime lo que ha hecho.
#    Están agrupados en: eliminar filas, valores imposibles a NaN, incoherencias cruzadas, columnas y pasos
#    opcionales.
# 4. **Registro de preprocesados** (`PREPROCESADOS`): cada preprocesado es una lista de pasos.
# 5. **Codificación, imputación, escalado y modelos**: se hacen dentro de un `Pipeline`, que se ajusta solo
#    con el train de cada partición para evitar fugas de información.
# 6. **Evaluación**: todos los preprocesados se prueban con los mismos modelos y **exactamente las mismas particiones y el mismo test**: el filtrado de filas solo actúa sobre el train de cada partición (sección 5).
# 7. **Resultados**, incluido el experimento de imputación.
# 
# | ID | Pasos que añade | Problemática del EDA que resuelve |
# | :--- | :--- | :--- |
# | `P0_base` | Ninguno (solo codificación y mediana del pipeline) | Ninguna. Es la base de comparación |
# | `P1_limpieza` | Limpieza común | Valores imposibles, incoherencias entre variables, ejercicio con decimales, alcohol negativo, `colesterol_total` corrupto |
# | `P2_diabetes` | P1 + recodificar `diabetes_mellitus` | 597 diabéticos por criterio analítico sin marcar |
# | `P3_pcr` | P2 + eliminar PCR > 10 mg/L | PCR de proceso agudo, que no refleja riesgo crónico |
# | `P4_discretizacion` | P3 + sustituir `edad` por su grupo etario e `indice_masa_corporal` por su categoría de la OMS, ambos como dummies | El EDA propone usar grupos etarios clínicos y discretizar el IMC |
# | `P5_smote` | P4 + SMOTENC (variables categóricas sin binarizar) dentro del `Pipeline` | Desbalanceo de clases |
# | `P6_adasyn` | P4 + ADASYN sobre las numéricas, con las categóricas copiadas de un paciente real, dentro del `Pipeline` | Desbalanceo de clases, con más muestras donde el límite entre clases es más difícil |
# | `P7_rfe` | P4 + selección de variables con RFE por grupos (una variable original se conserva o se elimina entera) dentro del `Pipeline` | Variables poco informativas o redundantes |

# ## 1. Configuración
# 
# Se fija la **semilla** y se reúnen los **umbrales de dominio** en una sola celda. Si se cambia un umbral,
# se cambia aquí y afecta a todos los preprocesados.

# In[116]:


from pathlib import Path
import sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# In[117]:


pd.set_option("display.max_columns", None, "display.width", 200)

SEED = 123
TARGET = "riesgo_cardiovascular"

# Ruta al dataset: funciona desde notebooks/, desde notebooks/workspace/ y desde la raíz del proyecto
RUTAS = ["../data/raw/enfermedad_cardiovascular.csv",
         "../../data/raw/enfermedad_cardiovascular.csv",
         "data/raw/enfermedad_cardiovascular.csv"]
DATA_PATH = next(Path(r) for r in RUTAS if Path(r).exists())

# --- Umbrales del EDA (01_eda.ipynb) ---
EDAD_MINIMA = 18           # el enunciado limita la población a adultos
IMC_MINIMO = 13            # por debajo de 12-13 es biológicamente imposible (mínimo en los datos: 3.6)
EJERCICIO_MAXIMO = 7       # días por semana
HBA1C_MINIMA = 3           # HbA1c < 3 % no es fisiológica
SISTOLICA_MINIMA = 60      # mmHg, ver 01_eda: "Mínimos plausibles de la presión arterial"
DIASTOLICA_MINIMA = 40     # mmHg, ver 01_eda: "Mínimos plausibles de la presión arterial"
PCR_MAXIMA = 10            # mg/L; por encima indica un proceso agudo, no riesgo crónico
HBA1C_DIABETES = 6.5       # % (criterio diagnóstico de diabetes)
GLUCOSA_DIABETES = 126     # mg/dL en ayunas (criterio diagnóstico de diabetes)

# --- Variables categóricas ---
NOMINALES = ["tipo_alimentacion", "antecedentes_familiares_tipo"]
ORDINAL = ["nivel_actividad"]
ORDEN_ACTIVIDAD = ["Sedentario", "Ligeramente activo", "Moderadamente activo", "Muy activo"]
# --- Variables booleanas ---
BOOLEANOS = ['fumador_actual', 'hipertension_previa', 'diabetes_mellitus', 'tratamiento_estatinas', 'uso_antihipertensivos']
# --- Variables numéricas (colesterol_total se conserva aquí: P0 lo usa; P1 en adelante lo elimina del DataFrame) ---
NUMERICAS = ['edad', 'presion_sistolica', 'presion_diastolica', 'colesterol_total',  'colesterol_hdl', 'colesterol_ldl', 'trigliceridos', 'glucosa_ayunas', 'indice_masa_corporal', 'frecuencia_cardiaca_reposo', 'horas_sueno_diarias', 'consumo_alcohol_semanal', 'anos_fumador', 'proteina_c_reactiva', 'acido_urico', 'clearance_creatinina', 'frecuencia_ejercicio_semanal', 'perimetro_cintura', 'hemoglobina_glicosilada']
# --- Variables derivadas de pasos del catálogo ---
DISCRETIZADAS = ["edad_grupo", "imc_grupo"]     # las crea discretizar_edad_imc (P4)


# El pipeline decide qué columnas usa según las que tenga cada DataFrame, en vez de mutar listas globales
# (una lista compartida que se modifica después de crear el ColumnTransformer cambia también ese ColumnTransformer).
def numericas_presentes(X):
    return [c for c in NUMERICAS if c in X.columns]

def booleanos_presentes(X):
    return [c for c in BOOLEANOS if c in X.columns]

def discretas_presentes(X):
    return [c for c in DISCRETIZADAS if c in X.columns]


# ## 2. Carga del dataset
# 
# El CSV se lee sin modificarlo. `resumen_nulos` se reutiliza después para ver cuántos NaN deja cada
# preprocesado.

# In[118]:


def resumen_nulos(df):
    print("Valores nulos:", df.isnull().sum().sum())
    resumen = pd.DataFrame({
        "Nulos": df.isnull().sum(),
        "Porcentaje (%)": (df.isnull().mean() * 100).round(2),
    })
    return resumen[resumen["Nulos"] > 0].sort_values(by="Nulos", ascending=False)


df = pd.read_csv(DATA_PATH)
print(df.shape)
resumen_nulos(df)


# Se separan en un 80% train y un 20% test.

# In[119]:


df_train, df_test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)


# ## 3. Catálogo de pasos
# 
# Cada paso recibe un `DataFrame` y devuelve otro, sin modificar el original. Todos usan dos funciones
# auxiliares para que el mensaje que imprimen tenga siempre el mismo formato.
# 
# Son **reglas fijas** que no aprenden nada de los datos (no calculan medias ni medianas), así que se pueden
# aplicar antes de partir en train y test sin provocar fugas de información.

# In[120]:


def eliminar_filas(df, mascara, motivo):
    antes = len(df)
    df = df[~mascara]
    print(f"{antes - len(df)} filas eliminadas por {motivo}. Hay {len(df)} filas restantes.")
    return df


def poner_nan(df, mascara, columnas, motivo):
    df = df.copy()
    df.loc[mascara, columnas] = np.nan
    print(f"{int(mascara.sum())} valores con {motivo} se han puesto a NaN en {', '.join(columnas)}.")
    return df


# ### 3.1 Eliminar filas
# 
# - **Menores de edad** (4): el enunciado limita la población a adultos. Esto define la **población de estudio**, no un preprocesado: se aplica una sola vez a todos (también a `P0_base`) antes de partir en train y test, así que el test es el mismo para todas las configuraciones.
# - **Más años fumando que edad** (1): el registro entero es incoherente. Como el resto de filtros de filas, **solo se aplica al train** (sección 5).

# In[121]:


def eliminar_menores(df):
    return eliminar_filas(df, df["edad"] < EDAD_MINIMA, "ser menores de edad")


def eliminar_fumador_incoherente(df):
    return eliminar_filas(df, df["anos_fumador"] > df["edad"], "tener más años fumando que edad")


# ### 3.2 Valores imposibles a NaN
# 
# Un valor imposible es un error de registro, pero el resto del paciente puede ser válido. Por eso la celda
# pasa a **NaN** y la fila se conserva; después la imputa el pipeline (sección 5).
# 
# El consumo de alcohol negativo **no** está en esta tabla: no se trata como un valor imposible sino como un posible valor centinela (sección 3.4).
# 
# Las reglas están en una tabla (`REGLAS_RANGO`): para añadir una nueva basta con añadir una línea.
# Los mínimos de la presión arterial están justificados con referencias clínicas en `01_eda.ipynb`.
# 
# **La presión invertida (diastólica >= sistólica) se comprueba antes que los mínimos.** Si fuera al revés,
# un paciente con sistólica 52,2 y diastólica 90,6 perdería primero la sistólica, y la comparación
# `90,6 >= NaN` daría `False`: su diastólica, probablemente errónea, se conservaría. Comprobándola antes,
# las dos presiones pasan a NaN, igual que en el resto de casos invertidos.

# In[122]:


def nan_presion_invertida(df):
    return poner_nan(df, df["presion_diastolica"] >= df["presion_sistolica"],
                     ["presion_sistolica", "presion_diastolica"], "presión diastólica >= sistólica")


# (columna, condición que hace imposible el valor, descripción)
REGLAS_RANGO = [
    ("colesterol_hdl", lambda s: s <= 0, "colesterol_hdl negativo o 0"),
    ("trigliceridos", lambda s: s <= 0, "trigliceridos negativos o 0"),
    ("indice_masa_corporal", lambda s: s < IMC_MINIMO, f"IMC menor que {IMC_MINIMO}"),
    ("frecuencia_ejercicio_semanal", lambda s: s > EJERCICIO_MAXIMO, f"más de {EJERCICIO_MAXIMO} días de ejercicio semanal"),
    ("hemoglobina_glicosilada", lambda s: s < HBA1C_MINIMA, f"hemoglobina glicosilada menor que {HBA1C_MINIMA}%"),
    ("presion_sistolica", lambda s: s < SISTOLICA_MINIMA, f"presión sistólica menor que {SISTOLICA_MINIMA} mmHg"),
    ("presion_diastolica", lambda s: s < DIASTOLICA_MINIMA, f"presión diastólica menor que {DIASTOLICA_MINIMA} mmHg"),
]

def nan_valores_imposibles(df):
    for columna, condicion, motivo in REGLAS_RANGO:
        df = poner_nan(df, condicion(df[columna]), [columna], motivo)
    return df


# ### 3.3 Incoherencias entre variables a NaN
# 
# Cuando dos variables se contradicen no se sabe cuál de las dos está mal, así que **ambas** pasan a NaN.
# La presión invertida es también una incoherencia de este tipo, pero se trata antes, en la sección 3.2.
# 
# **Estas reglas se ejecutan después de recodificar la diabetes** (sección 3.5): una glucosa > 180 mg/dL es un criterio diagnóstico válido por sí solo, y si la regla cruzada la pusiera antes a NaN, el paciente no se recodificaría como diabético (`NaN >= 126` es `False`).
# 
# | Incoherencia | Por qué es un error |
# | :--- | :--- |
# | IMC > 35 y cintura < 70 cm | Obesidad grado II con cintura de persona delgada |
# | IMC < 18,5 y cintura > 110 cm | Bajo peso con cintura de obesidad abdominal |
# | HbA1c < 5,5 % y glucosa > 180 mg/dL | HbA1c normal con glucosa de diabetes descompensada |
# | HbA1c > 9 % y glucosa < 80 mg/dL | HbA1c de diabetes mal controlada con glucosa normal-baja |

# In[123]:


# (condición sobre el DataFrame, columnas que pasan a NaN, descripción)
REGLAS_CRUZADAS = [
    (lambda d: (d["indice_masa_corporal"] > 35) & (d["perimetro_cintura"] < 70),
     ["indice_masa_corporal", "perimetro_cintura"], "IMC mayor que 35 y cintura menor que 70"),
    (lambda d: (d["indice_masa_corporal"] < 18.5) & (d["perimetro_cintura"] > 110),
     ["indice_masa_corporal", "perimetro_cintura"], "IMC menor que 18.5 y cintura mayor que 110"),
    (lambda d: (d["hemoglobina_glicosilada"] < 5.5) & (d["glucosa_ayunas"] > 180),
     ["hemoglobina_glicosilada", "glucosa_ayunas"], "hemoglobina glicosilada menor que 5.5% y glucosa en ayunas mayor que 180"),
    (lambda d: (d["hemoglobina_glicosilada"] > 9) & (d["glucosa_ayunas"] < 80),
     ["hemoglobina_glicosilada", "glucosa_ayunas"], "hemoglobina glicosilada mayor que 9% y glucosa en ayunas menor que 80"),
]


def nan_incoherencias_cruzadas(df):
    for condicion, columnas, motivo in REGLAS_CRUZADAS:
        df = poner_nan(df, condicion(df), columnas, motivo)
    return df


# ### 3.4 Columnas
# 
# **Las variables que cuentan unidades enteras se redondean.** El EDA detectó decimales en variables que
# deberían ser enteras:
# 
# | Variable | Con decimales | Redondeo | Motivo |
# | :--- | :--- | :--- | :--- |
# | `edad` | 2.042 | Hacia abajo (45,8 → 45) | La edad son años cumplidos |
# | `frecuencia_ejercicio_semanal` | 81,5 % | Hacia arriba (6,1 → 7) | Si ha hecho ejercicio algún rato de un día, ese día cuenta |
# | `anos_fumador` | 1.588 | Al más cercano | Años de tabaquismo |
# | `frecuencia_cardiaca_reposo` | 2.014 | Al más cercano | Latidos por minuto |
# | `consumo_alcohol_semanal` | 2.055 | Al más cercano | Unidades de alcohol por semana |
# 
# - El redondeo se hace después de eliminar a los menores y de poner a NaN los imposibles. Como la edad se
#   redondea hacia abajo, nadie que tuviera 18 años o más baja de 18, y tampoco aparece ningún paciente con
#   más años fumando que edad.
# - El ejercicio se redondea después de poner a NaN los valores mayores que 7 (sección 3.2), así que el
#   resultado nunca pasa de 7. Su media sube alrededor de medio día.
# - En el resto, el redondeo no sesga la variable: la media y la correlación con el target apenas cambian
#   (menos de 0,01). Los fumadores actuales con 0 años pasan de 123 a 132, porque quien lleva menos de
#   medio año fumando queda en 0 (el EDA ya consideraba justificable este caso).
# 
# **El alcohol negativo ** 177 filas tienen `consumo_alcohol_semanal < 0`. Este valor se recodifica a 0 para mantener la coherencia clínica.
# 
# **`colesterol_total` se elimina**: el 24,8 % de los pacientes tiene un total menor que HDL + LDL (imposible),
# el 57,76 % no cuadra con la fórmula HDL + LDL + TG/5 y su correlación con el target es muy baja (ver EDA).
# Recalcularla con la fórmula no aportaría información nueva, porque sería una combinación lineal de
# variables que el modelo ya recibe.

# In[124]:


# columna: función de redondeo
REDONDEO = {
    "edad": np.floor,
    "frecuencia_ejercicio_semanal": np.ceil,
    "anos_fumador": np.round,
    "frecuencia_cardiaca_reposo": np.round,
    "consumo_alcohol_semanal": np.round,
}


def redondear_enteros(df):
    df = df.copy()
    for columna, redondeo in REDONDEO.items():
        cambiados = int((df[columna] != redondeo(df[columna])).sum() - df[columna].isna().sum())
        df[columna] = redondeo(df[columna])
        print(f"{cambiados} valores de {columna} se han redondeado ({redondeo.__name__}).")
    return df


def eliminar_colesterol_total(df):
    print("Columna colesterol_total eliminada.")
    return df.drop(columns="colesterol_total")

def alcohol_negativo_a_cero(df):
    df = df.copy()
    negativo = df["consumo_alcohol_semanal"] < 0
    df.loc[negativo, "consumo_alcohol_semanal"] = 0
    print(f"{int(negativo.sum())} valores negativos de consumo_alcohol_semanal se han corregido a 0.")
    return df


# ### 3.5 Pasos opcionales
# 
# - **Recodificar diabetes** (muy recomendable; se ejecuta **antes** de las incoherencias cruzadas): según los criterios diagnósticos, un paciente con
#   HbA1c >= 6,5 % o glucosa en ayunas >= 126 mg/dL es diabético. El EDA encontró 597 pacientes que los
#   cumplen y no están marcados en `diabetes_mellitus`.
# - **Eliminar PCR > 10 mg/L**: una PCR tan alta indica un proceso agudo (infección, inflamación), no riesgo
#   cardiovascular crónico, así que se supone que la prueba no es válida. Los pacientes **sin** PCR medida
#   (NaN) se conservan. Esta eliminación **solo se aplica al train**: en producción los pacientes con PCR > 10 mg/L seguirán acudiendo a consulta, así que el test los conserva (con su PCR original) y todos los preprocesados se evalúan sobre las mismas muestras (sección 5).

# In[125]:


def recodificar_diabetes(df):
    df = df.copy()
    cond = (df["hemoglobina_glicosilada"] >= HBA1C_DIABETES) | (df["glucosa_ayunas"] >= GLUCOSA_DIABETES)
    nuevos = int((cond & (df["diabetes_mellitus"] == 0)).sum())
    df.loc[cond, "diabetes_mellitus"] = 1
    print(f"{int(cond.sum())} pacientes con hemoglobina glicosilada >= {HBA1C_DIABETES}% o glucosa en ayunas "
          f">= {GLUCOSA_DIABETES} mg/dL se han puesto como diabéticos ({nuevos} no lo estaban).")
    return df

# Para eliminar filas durante el entrenamiento (solo en train, no en el conjunto de validación ni test)
def eliminar_pcr_alta(X, y):
    mascara = (X["proteina_c_reactiva"] <= PCR_MAXIMA) | (
        X["proteina_c_reactiva"].isna()
    )
    eliminadas = len(X) - int(mascara.sum())
    restantes = int(mascara.sum())
    print(
        f"{eliminadas} filas eliminadas por PCR > {PCR_MAXIMA} mg/L en train. Hay {restantes} filas restantes."
    )
    return X.loc[mascara].copy(), y.loc[mascara].copy()


# - **Discretizar edad e IMC** (P4): el EDA propone sustituir la edad por **grupos etarios clínicos** y discretizar el IMC según la
#   clasificación de la **OMS**.
# 
#   | Variable nueva | Categorías (código) |
#   | :--- | :--- |
#   | `edad_grupo` | 0: 18-39, 1: 40-54, 2: 55-64, 3: >= 65 años (el último grupo no tiene cota superior) |
#   | `imc_grupo` | 0: bajo peso (< 18,5), 1: normal (18,5-25), 2: sobrepeso (25-30), 3: obesidad I (30-35), 4: obesidad II (35-40), 5: obesidad III (>= 40) |
# 
#   Las dos columnas originales se **sustituyen** por su categoría y, a partir de ahí, se tratan como **categóricas** (dummies), no como números:
# 
#   - Los grupos tienen anchos muy distintos (22, 15, 10 y más de 55 años). Un código entero escalado con `StandardScaler` obligaría a los modelos lineales (`LogisticRegression`, `LinearSVC`) a suponer que cada salto de grupo aumenta el riesgo lo mismo, y no hay motivo clínico para ello.
#   - Los NaN del IMC se imputan con la **moda**, no con la mediana: la mediana de un código puede ser fraccionaria (con un número par de muestras y valores centrales 2 y 3 sería 2,5, una categoría que no existe).
#   - El último grupo de edad llega hasta `np.inf` (igual que el IMC): con una cota rígida como 120, cualquier edad >= 120 se convertiría en NaN.
# 
#   Los cortes son reglas fijas, así que se pueden aplicar antes de partir en train y test.

# In[126]:


EDAD_CORTES = [18, 40, 55, 65, np.inf]              # grupos etarios clínicos del EDA, intervalos [a, b)
IMC_CORTES = [0, 18.5, 25, 30, 35, 40, np.inf]      # clasificación de la OMS, intervalos [a, b)


def discretizar_edad_imc(df):
    df = df.copy()
    df["edad_grupo"] = pd.cut(df["edad"], bins=EDAD_CORTES, labels=False, right=False)
    df["imc_grupo"] = pd.cut(df["indice_masa_corporal"], bins=IMC_CORTES, labels=False, right=False)
    df = df.drop(columns=["edad", "indice_masa_corporal"])
    print(f"edad e indice_masa_corporal se han sustituido por su categoría "
          f"({df['edad_grupo'].nunique()} grupos etarios y {df['imc_grupo'].nunique()} categorías de IMC; "
          f"{int(df['edad_grupo'].isna().sum())} NaN de edad y {int(df['imc_grupo'].isna().sum())} NaN de IMC se mantienen).")
    return df


# ## 4. Registro de preprocesados
# 
# Cada preprocesado es una **lista de pasos** que se aplican en orden. Cada uno añade **un solo cambio** al
# anterior, así que la diferencia de métricas entre dos preprocesados consecutivos se puede atribuir a ese
# cambio.
# 
# ### Cómo añadir un preprocesado nuevo
# 
# 1. Si hace falta, escribir el paso en la sección 3 como una función `df -> df`
#    (o añadir una línea a `REGLAS_RANGO` / `REGLAS_CRUZADAS`).
# 2. Añadir una entrada a `PREPROCESADOS`, por ejemplo:
# 
# ```python
# def mi_paso(df):
#     ...
#     return df
# 
# PREPROCESADOS["P5_mi_paso"] = PREPROCESADOS["P4_discretizacion"] + [mi_paso]
# ```
# 
# 3. Si el cambio está en la codificación, la imputación o el escalado, va en `construir_pipeline`
#    (sección 5).
# 
# El resto del cuaderno (evaluación y resultados) recorre `PREPROCESADOS` y no hay que tocarlo.

# In[127]:


# Orden de la limpieza: la diabetes se recodifica ANTES de las incoherencias cruzadas (ver sección 3.3)
LIMPIEZA_PREVIA = [
    nan_presion_invertida,         # Presiones invertidas antes que los mínimos de presión (sección 3.2)
    nan_valores_imposibles,        # Valores científicamente imposibles (sección 3.2)
    alcohol_negativo_a_cero,       # Alcohol negativo como valor centinela, no como NaN (sección 3.4)
    redondear_enteros,             # Redondeos de variables enteras, después de poner a NaN los imposibles (sección 3.4)
    eliminar_colesterol_total,     # Columna de colesterol total (sección 3.4)
]
INCOHERENCIAS = [nan_incoherencias_cruzadas]  # Incoherencias entre columnas (sección 3.3)

PREPROCESADOS = {
    "P0_base": [],
    "P1_limpieza": LIMPIEZA_PREVIA + INCOHERENCIAS,
    "P2_diabetes": LIMPIEZA_PREVIA + [recodificar_diabetes] + INCOHERENCIAS,  # Recodificar diabetes con HbA1c y glucosa (sección 3.5)
    "P3_pcr": LIMPIEZA_PREVIA + [recodificar_diabetes] + INCOHERENCIAS,  # PCR > 10 mg/L solo en train (sección 3.5)
    "P4_discretizacion": LIMPIEZA_PREVIA + [recodificar_diabetes] + INCOHERENCIAS + [discretizar_edad_imc],  # Edad e IMC por categoría (sección 3.5)
}

# No elimina ninguna fila, solo aplica los pasos de preprocesado que no eliminan filas.
def aplicar_preprocesado(df, pasos):
    for paso in pasos:
        df = paso(df)
    return df


# **Preprocesados que aprenden de los datos (P5, P6 y P7).** El sobremuestreo y la selección de variables no son pasos `df -> df`:
# calculan vecinos o coeficientes a partir de los datos, así que si se aplicaran antes de partir en train y test provocarían fugas de información. Por eso se registran aparte,
# en `PREPROCESADOS_PIPELINE`, y se insertan **dentro del `Pipeline`** de la sección 5, donde se ajustan solo con el train de cada partición.
# Todos parten de los datos de **P4** (`base`), el último preprocesado incremental de arriba.
# 
# - **SMOTENC** (`P5_smote`): genera pacientes sintéticos de las clases minoritarias interpolando entre cada paciente y sus vecinos de su misma clase. Se usa la variante **NC** (nominal-continuous) y **antes del One-Hot**: el `SMOTE` normal, colocado después del `ColumnTransformer`, interpolaría dummies y booleanas y crearía pacientes imposibles (0,35 de dieta mediterránea y 0,45 de occidental a la vez, 0,63 fumadores). SMOTENC interpola solo las numéricas y a las categóricas les da la categoría más frecuente entre los vecinos, así que siempre salen valores válidos. Las numéricas se imputan y escalan **antes** del sampler (con el escalador ajustado al train real) para que las distancias entre vecinos no las domine la variable de mayor escala, y el One-Hot se aplica **después**, sobre las categorías ya remuestreadas. Se consideran categóricas las nominales, `nivel_actividad`, las booleanas y los grupos de edad e IMC.
# - **ADASYN** (`P6_adasyn`): como SMOTE, pero genera más muestras donde la clase minoritaria está más mezclada con las otras. `imblearn` no tiene una versión de ADASYN para datos mixtos, así que `ADASYNMixto` desacopla el remuestreo: ADASYN se aplica **solo a las columnas numéricas** y cada paciente sintético recibe las categorías de su **paciente real más cercano de la misma clase** (medido en las numéricas). Así tampoco aparecen dummies fraccionarias, negativas o mayores que 1.
# - **RFE** (`P7_rfe`): entrena un modelo, elimina lo menos importante y repite. Se hace **por grupos**: las dummies de una misma variable (`antecedentes_familiares_tipo`, `tipo_alimentacion`, `edad_grupo`, `imc_grupo`) se eliminan o se conservan **juntas**. Un RFE sobre dummias sueltas podría quitar `antecedentes_familiares_tipo_Paterno` y dejar `Materno` y `Ambos`, con lo que un paciente con antecedentes paternos se codificaría `[0, 0]`, igual que uno sin antecedentes (la categoría de referencia con `drop="first"`). La importancia de un grupo es la media del valor absoluto de los coeficientes de la regresión logística de sus columnas (en todas las clases), con las variables ya escaladas. Se aplica la **regla del error estándar**: se conserva el **menor número de grupos cuya puntuación media (`f1_macro`, 3 particiones) queda a menos de un error estándar de la mejor** (error estándar = desviación entre particiones / raíz del número de particiones). El `RFECV` normal se queda con el número de variables de mejor puntuación, y en estos datos esa puntuación sube con cada variable que se añade, así que conservaría todas y el preprocesado no haría nada.
# 
# Para añadir otro preprocesado de este tipo basta con añadir una entrada a `PREPROCESADOS_PIPELINE`.

# **Población de estudio y datos preprocesados.** Primero se eliminan los menores de edad, que es la población del enunciado y se aplica igual a todos los preprocesados. Después se aplica cada preprocesado a esa población completa y se guarda el resultado en `DATOS` (los mensajes muestran cuántos valores cambia cada paso.

# In[128]:


print("----------- Población de estudio -----------")
df_train_pob = eliminar_menores(df_train) # El fumador incoherente desaparece ya con la eliminación de menores de edad
df_test_pob = eliminar_menores(df_test)

DATOS = {} # Va a contener los valore para el train de cada preprocesado, no el 100%, que se preprocesa aparte en el df_test_prob
for nombre, pasos in PREPROCESADOS.items():
    print(f"\n----------- {nombre} -----------")
    DATOS[nombre] = aplicar_preprocesado(df_train_pob, pasos)
    print("Dimensiones:", DATOS[nombre].shape)


# **Nulos tras la limpieza común.** Todos estos NaN los imputa el pipeline (sección 5).

# In[129]:


resumen_nulos(DATOS["P1_limpieza"])


# No hay ninguna columna nominal ni booleana que tenga nulos, tanto en P0 como en P1, por lo que no se tiene que aplicar técnicas como imputar los nulos con la moda antes de realizar transformaciones de las variables. Aunque el número de nulos ha crecido, no hay ninguna columna especialmente afectada por la falta de valores para decidir eliminarla del conjunto de datos por pérdida de información relevante. 

# In[130]:


resumen_nulos(DATOS["P4_discretizacion"])


# Sin embargo, ahora que en P4 se ha discretizado la variable del IMC en categorías clínicas según los criterios de la OMS, esta deja de ser numérica y pasa a tratarse como una variable categórica. Por lo tanto, ya no se le aplicaría la imputación por mediana ni el escalado estándar, sino que sus valores nulos pasan a imputarse mediante la moda. No había ni hay nulos para la edad.

# In[131]:


def detectar_baja_varianza(df, freq_ratio_umbral=19, unique_pct_umbral=10):
    resultados = []
    columnas_a_evaluar = df.columns
    columnas_a_evaluar = columnas_a_evaluar.drop('riesgo_cardiovascular')
    for col in columnas_a_evaluar:
        conteo_valores = df[col].value_counts(dropna=True)
        if len(conteo_valores) == 0:
            continue
        # Porcentaje de valores únicos sobre el total de objetos (para comparar con unique_pct_umbral)
        n_unicos = len(conteo_valores)
        n_filas = len(df[col].dropna())
        pct_unicos = (n_unicos / n_filas) * 100
        # Ratio de frecuencia entre el valor más frecuente y el segundo más frecuente (para comparar con freq_ratio_umbral)
        if n_unicos >= 2:
            freq_1 = conteo_valores.iloc[0]
            freq_2 = conteo_valores.iloc[1]
            freq_ratio = freq_1 / freq_2 if freq_2 > 0 else np.inf
        else:
            # Si solo hay 1 valor único (columna 100% constante)
            freq_ratio = np.inf
            
        resultados.append({
            'variable': col,
            'n_unicos': n_unicos,
            'porcentaje_unicos': pct_unicos,
            'ratio_frecuencia': freq_ratio,
            'valor_mas_frecuente': conteo_valores.index[0],
            'freq_valor_mas_frecuente': conteo_valores.iloc[0]
        })
        
    df_metricas = pd.DataFrame(resultados)
    
    # Si el ratio de frecuencia supera un límite establecido y el porcentaje de valores únicos cae por debajo de un límite establecido,
    # podemos considerar la variable como una variable con varianza cercana a cero.
    df_baja_var = df_metricas[
        (df_metricas['ratio_frecuencia'] > freq_ratio_umbral) & 
        (df_metricas['porcentaje_unicos'] < unique_pct_umbral)
    ].reset_index(drop=True)
    
    return df_metricas, df_baja_var


metricas_completas, columnas_baja_varianza = detectar_baja_varianza(DATOS["P1_limpieza"])
print("Métricas de varianza por variable:")
print(metricas_completas)
print("\nVariables con varianza cercana a cero (posibles candidatas a eliminar):")
print(columnas_baja_varianza)


# Aunque claramente la varianza ha cambiado respecto a la ejecución del EDA, especialmente en variables que se han redondeado, que claramente ha provocado la reducción de sus valores únicos, no han aparecido ninguna columna con una varianza cercana a cero para considerar eliminarla.

# La codificación, imputación, escalado y modelos se hacen dentro de un `Pipeline`, que se **ajusta solo con el train** de cada partición. Primero hay que determinar transformaciones categóricas que serán comunes a todos los pipelines del preprocesado, que es el tratamiento de las variables nominales y ordinales, puesto que los modelos no procesan cadenas de texto directamente.
# 
# - **Dummies** (`OneHotEncoder`, `drop="first"`) para `tipo_alimentacion` y `antecedentes_familiares_tipo`, ya que se tratan de variables sin jerarquía. Se usa `handle_unknown="error"`: con `drop="first"`, la opción `"ignore"` convertiría una categoría desconocida en un vector de ceros, indistinguible de la categoría de referencia eliminada. Las categorías son un conjunto cerrado, así que un valor nuevo debe dar error y no pasar desapercibido.
# - **Dummies** también para `edad_grupo` e `imc_grupo` (solo existen desde P4), con imputación por la **moda** antes de codificar.
# - **Ordinal 0-3** (`OrdinalEncoder`) para `nivel_actividad` porque sus categorías tienen un orden natural.
# - Las **booleanas** pasan tal cual.
# 
# Las columnas de cada grupo se eligen **según las que tenga el DataFrame** (`numericas_presentes`, `booleanos_presentes`, `discretas_presentes`), así un mismo pipeline sirve para `P0` (con `colesterol_total`, que es numérica) y para `P1` en adelante (sin esa columna) sin modificar ninguna lista global.

# In[132]:


def grupos_de_columnas(preprocesador, X):
    # Agrupa las columnas de salida del preprocesador por variable original (las dummies juntas)
    nombres = list(clone(preprocesador).fit(X).get_feature_names_out())
    grupos = {}
    for i, nombre in enumerate(nombres):
        clave = nombre
        for c in NOMINALES:
            if nombre.startswith(f"nominales__{c}_"):
                clave = c
        for c in DISCRETIZADAS:
            if nombre.startswith(f"discretas__{c}_"):
                clave = c
        grupos.setdefault(clave, []).append(i)
    return list(grupos.values()), list(grupos.keys())


# En cuanto a los modelos seleccionados para la práctica se deciden 3 modelos base para más adelante seleccionar 2 de ellos:
# 
# * Logistic Regression
# * LinearSVC
# * SVM
# 
# Y 3 ensambles:
# 
# * POR DETERMINAR
# * POR DETERMINAR
# * POR DETERMINAR
# 
# Como se puede observar, algunos de estos modelos son altamente sensibles a la escala numérica y dependen de un escalado estricto (como regresión logística) y no procesan valores nulos de forma nativa. Por lo tanto, para este primer experimento que plantea determinar qué base de preprocesamiento en relación con la limpieza de datos (`P0`, `P1`, `P2`, `P3`) resulta más eficiente, se realizará un pipeline que, además de realizar las transformaciones categóricas, impute valores por la mediana y realice un normalización con un escalado estándar. Como únicamente hay presencia de nulos en variables numerales, no es necesaria una estrategia de imputación como, por ejemplo, la moda en variables nominales o booleanas.
# 
# Es importante mencionar que tanto la mediana como el escalado se deben calcular únicamente con los valores del conjunto de entrenamiento. Al encapsular las transformaciones dentro de este pipeline y usando `pipeline_fold.fit(X_train_fold, y_train_fold)` en la función de evaluación (se verá más adelante) se garantiza que se aplica únicamente sobre los datos de entrenamiento y el conjunto de test haga las predicciones en el `predict()` con dichas transformaciones del train.

# In[133]:


# Lista unificada de todas las columnas categóricas presentes en cada DataFrame
def categoricas_presentes(X):
    return [c for c in (NOMINALES + ORDINAL + booleanos_presentes(X) + discretas_presentes(X)) if c in X.columns]

# 1. IMPUTADOR: Mediana a TODAS las numéricas y Moda a TODAS las categóricas
imputador = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numericas_presentes),
        ("cat", SimpleImputer(strategy="most_frequent"), categoricas_presentes),
    ],
    verbose_feature_names_out=False
).set_output(transform="pandas")

# 2. ESCALADOR DE CONTINUAS (se ejecuta ANTES del sampler para que las distancias de SMOTE sean justas)
escalador_continuas = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numericas_presentes),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
).set_output(transform="pandas")

# 3. CODIFICADOR Y ESCALADOR: Dummies estrictas (0/1) y StandardScaler
codificador_escalador = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numericas_presentes),
        ("nominales", OneHotEncoder(drop="first", handle_unknown="error", sparse_output=False), NOMINALES),
        ("discretas", OneHotEncoder(drop="first", handle_unknown="error", sparse_output=False), discretas_presentes),
        ("ordinal", Pipeline([
            ("encoder", OrdinalEncoder(categories=[ORDEN_ACTIVIDAD])),
            ("scaler", StandardScaler())
        ]), ORDINAL),
        ("booleanos", "passthrough", booleanos_presentes),
    ]
)


# Para facilitar la inserción del modelo con el pipeline de preprocesado, se ha creado la siguiente función, la cual encapsula las etapas del preprocesado y el modelo. También se añade la posibilidad de proporcionar un diccionario de hiperparámetros para determinar la configuración óptima del modelo. Se trata de un grid de validación cruzada con 3 iteraciones y optimización basada en `f1_macro`. Se ha elegido esta métrica para optimizar frente a `recall_alto` (útil en este problema, ya que al ser uno de medicina en el que se mide el riesgo cardiovascular, no detectar un paciente de alto riesgo supondría el error más grave) porque `f1_macro` ofrece un equilibrio entre sensibilidad y precisión para todas las clases por igual obligando al modelo a no disparar las falsas alarmas, como sí ocurriría al usar `recall alto` como métrica, que buscaría clasificar a casi todos los pacientes como de alto riesgo en vez de discriminar entre pacientes, por lo que el modelo no estaría aprendiendo realmente.

# In[ ]:


def construir_pipeline(modelo, param_grid=None, sampler=None, seleccion=None, filtro_pcr=False, imputar=True, X=None):
    pasos = []
    
    # 1. Filtro PCR (solo en train)
    if filtro_pcr:
        pasos.append(("eliminar_pcr", FunctionSampler(func=eliminar_pcr_alta, validate=False)))
        
    # 2. Imputación
    if imputar:
        pasos.append(("imputador", imputador))
        
    # 3. Escalador numérico previo (para que SMOTE no tenga sesgo de escala)
    pasos.append(("escalador_continuas", escalador_continuas))
        
    # 4. Sampler opcional (SMOTENC / RandomOverSampler)
    if sampler is not None:
        pasos.append(("sampler", clone(sampler)))
        
    # 5. Codificador de categóricas a dummies / ordinal
    pasos.append(("codificador_escalador", codificador_escalador))
    
    # 6. Selección de características
    if seleccion is not None:
        if callable(seleccion):
            X_clean = imputador.fit_transform(X) if imputar else X
            X_clean = escalador_continuas.fit_transform(X_clean)
            grupos, _ = grupos_de_columnas(codificador_escalador, X_clean)
            pasos.append(("seleccion", seleccion(grupos)))
        else:
            pasos.append(("seleccion", clone(seleccion)))
        
    # 7. Modelo clasificador
    pasos.append(("modelo", clone(modelo)))
    
    pipeline_completo = ImbPipeline(pasos)
    
    if param_grid is not None:
        return GridSearchCV(estimator=pipeline_completo, param_grid=param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    return pipeline_completo


# - **`StratifiedShuffleSplit`** con 3 iteraciones y 30 % de test. Se estratifica porque las clases están desbalanceadas (la clase 2 tiene menos de la mitad de ejemplos que la 0).
# - Las particiones se generan **una sola vez** sobre la población de estudio (`df_pob`, sin filtrar), así que **todos los preprocesados y modelos se evalúan sobre exactamente las mismas muestras de test**. Los pasos que eliminan filas (`eliminar_fumador_incoherente`, `eliminar_pcr_alta`) se aplican solo al train de cada partición; al test solo se le aplican las transformaciones por celda. Evaluar cada preprocesado sobre un test distinto (o dejar fuera del test a los pacientes difíciles, como los de PCR > 10 mg/L) haría las métricas incomparables y optimistas.
# - **Métricas** de train y de test, para detectar sobreajuste:
#   - *accuracy*;
#   - *F1 macro*, que pesa igual las tres clases;
#   - *recall de alto riesgo* (clase 2), la más crítica clínicamente: un paciente de alto riesgo no detectado es el peor error.
# 
# Las métricas de train se miden sobre el train de cada partición ya filtrado y sin sintéticos (el sobremuestreo solo actúa dentro del `fit`).
# 
# Cada métrica se muestra como **media ± desviación típica** de las 3 iteraciones. La desviación indica cuánto varía el resultado según la partición: si la diferencia entre dos preprocesados es menor que su desviación, no se puede asegurar que uno sea mejor que el otro.
# 
# `gap_f1` es la diferencia de F1 macro entre train y test en cada iteración: cuanto mayor, más sobreajuste.

# In[135]:


N_SPLITS = 3
TEST_SIZE = 0.3

splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SEED)

import contextlib, io

def metricas(y_real, y_pred, conjunto):
    return {
        f"{conjunto}_accuracy": accuracy_score(y_real, y_pred),
        f"{conjunto}_f1_macro": f1_score(y_real, y_pred, average="macro"),
        f"{conjunto}_recall_alto": recall_score(y_real, y_pred, labels=[2], average="macro"),
    }


def evaluar(pipeline, datos, nombre_modelo, nombre_preprocesado, verbose=False):
    metricas_folds = {
        "train_accuracy": [], "train_f1_macro": [], "train_recall_alto": [],
        "val_accuracy": [], "val_f1_macro": [], "val_recall_alto": [],
        "gap_f1_macro": [], "gap_recall_alto": []  # <-- Único añadido aquí
    }
    X = datos.drop(columns=[TARGET])
    y = datos[TARGET].astype(int)
    if verbose:
        print(f"Evaluando modelo {nombre_modelo} con preprocesado {nombre_preprocesado}...")
    for iteracion, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        pipeline_fold = clone(pipeline)
        pipeline_fold.fit(X_train_fold, y_train_fold)
        y_pred_train = pipeline_fold.predict(X_train_fold)
        y_pred_val = pipeline_fold.predict(X_val_fold)
        m_train = metricas(y_train_fold, y_pred_train, "train")
        m_val = metricas(y_val_fold, y_pred_val, "val")
        gap_f1 = m_train['train_f1_macro'] - m_val['val_f1_macro']
        gap_recall = m_train['train_recall_alto'] - m_val['val_recall_alto'] 
        m_fold_extra = {"gap_f1_macro": gap_f1, "gap_recall_alto": gap_recall}
        for k, v in {**m_train, **m_val, **m_fold_extra}.items():
            metricas_folds[k].append(v)
        if verbose:
            print(f"--- Iteración {iteracion} ---")
            if hasattr(pipeline_fold, "best_params_"):
                print(f"  Mejores params -> {pipeline_fold.best_params_}")
            print(f"  F1 Macro    -> Train: {m_train['train_f1_macro']:.4f} | Val: {m_val['val_f1_macro']:.4f} | Gap: {gap_f1:.4f}")
            print(f"  Recall Alto -> Train: {m_train['train_recall_alto']:.4f} | Val: {m_val['val_recall_alto']:.4f} | Gap: {gap_recall:.4f}")
            print(f"  Accuracy    -> Train: {m_train['train_accuracy']:.4f} | Val: {m_val['val_accuracy']:.4f}")
    calculadas = {}
    for metrica, valores in metricas_folds.items():
        media = np.mean(valores)
        std = np.std(valores, ddof=1)
        calculadas[metrica] = f"{media:.4f} ± {std:.4f}"
    resumen_dict = {
        "preprocesado": nombre_preprocesado,
        "modelo": nombre_modelo,
        "train_f1_macro": calculadas["train_f1_macro"],
        "val_f1_macro": calculadas["val_f1_macro"],
        "gap_f1_macro": calculadas["gap_f1_macro"],
        "train_recall_alto": calculadas["train_recall_alto"],
        "val_recall_alto": calculadas["val_recall_alto"],
        "gap_recall_alto": calculadas["gap_recall_alto"],  
        "train_accuracy": calculadas["train_accuracy"],
        "val_accuracy": calculadas["val_accuracy"],
    }
    df_resumen = pd.DataFrame([resumen_dict])
    if verbose:
        print(f"\nResumen global:")
        print(f"  F1 MACRO    -> Train: {resumen_dict['train_f1_macro']} | Val: {resumen_dict['val_f1_macro']} | Gap: {resumen_dict['gap_f1_macro']}")
        print(f"  RECALL ALTO -> Train: {resumen_dict['train_recall_alto']} | Val: {resumen_dict['val_recall_alto']} | Gap: {resumen_dict['gap_recall_alto']}")
        print(f"  ACCURACY    -> Train: {resumen_dict['train_accuracy']} | Val: {resumen_dict['val_accuracy']}")
    return df_resumen


# Antes de comparar preprocesamientos de limpieza, primero se prueban los modelos para hacer un ajuste de hiperparámetos para no obtener resultados con overfitting o underfitting. Además, hay que decidir qué 2 modelos base se van a escoger. Se usa el preprocesamiento de limpieza P0 para hacer las pruebas de ajustes de hiperparámetros. **P0 conserva todas las columnas originales, incluida `colesterol_total`** (como variable numérica): la eliminación de esa columna es exclusiva de P1 en adelante.

# In[136]:


# P0 es la base sin tocar: conserva colesterol_total. La lista NUMERICAS no se modifica nunca; cada DataFrame
# aporta las columnas que tiene (numericas_presentes). Todas las evaluaciones parten de df_pob.
assert "colesterol_total" in DATOS["P0_base"].columns and "colesterol_total" in numericas_presentes(DATOS["P0_base"])
assert "colesterol_total" not in DATOS["P1_limpieza"].columns


# In[137]:


candidatos_modelos = {
    "Logistic Regression": {
        "estimador": LogisticRegression(solver='lbfgs', max_iter=1000, random_state=SEED),
        "params": {
            'modelo__C': [0.01, 0.1, 1.0, 10.0],
            'modelo__class_weight': [None, 'balanced']
        }
    },
    "Linear SVC": {
        "estimador": LinearSVC(dual='auto', max_iter=1000, random_state=SEED),
        "params": {
            'modelo__C': [0.01, 0.1, 1.0, 10.0],
            'modelo__class_weight': [None, 'balanced']
        }
    },
    "SVM (RBF)": {
        "estimador": SVC(kernel='rbf', random_state=SEED),
        "params": {
            'modelo__C': [0.1, 1.0, 10.0],
            'modelo__gamma': ['scale', 'auto'],
            'modelo__class_weight': [None, 'balanced']
        }
    }
}

resultados_seleccion_modelos = []
for nombre_mod, m_cfg in candidatos_modelos.items():
    pipe_eval = construir_pipeline(
        modelo=m_cfg["estimador"],
        param_grid=m_cfg["params"],
    )
    res = evaluar(pipe_eval, DATOS["P0_base"], nombre_mod, "P0_base", verbose=True)
    resultados_seleccion_modelos.append(res)

df_comparativa_modelos = pd.concat(resultados_seleccion_modelos, ignore_index=True)
display(df_comparativa_modelos)


# In[138]:


candidatos_modelos = {
    "Linear SVC": {
        "estimador": LinearSVC(dual='auto', max_iter=1000, random_state=SEED),
        "params": {
            'modelo__C': [0.001, 0.01, 0.1],
            'modelo__class_weight': [None, 'balanced']
        }
    },
    "SVM (RBF)": {
        "estimador": SVC(kernel='rbf', random_state=SEED),
        "params": {
            'modelo__C': [0.001,0.01, 0.1],
            'modelo__gamma': ['scale', 'auto'],
            'modelo__class_weight': [None, 'balanced']
        }
    }
}

resultados_seleccion_modelos = []

for nombre_mod, m_cfg in candidatos_modelos.items():
    pipe_eval = construir_pipeline(
        modelo=m_cfg["estimador"],
        param_grid=m_cfg["params"]
    )
    res = evaluar(pipe_eval, DATOS["P0_base"], nombre_mod, "P0_base", verbose=True)
    resultados_seleccion_modelos.append(res)

df_comparativa_modelos = pd.concat(resultados_seleccion_modelos, ignore_index=True)
display(df_comparativa_modelos)


# In[139]:


candidatos_modelos = {
    "Linear SVC": {
        "estimador": LinearSVC(dual='auto', max_iter=1000, random_state=SEED),
        "params": {
            'modelo__C': [0.001, 0.01, 0.1],
            'modelo__class_weight': [None, 'balanced']
        }
    },
    "Logistic Regression": {
        "estimador": LogisticRegression(solver='lbfgs', max_iter=1000, random_state=SEED),
        "params": {
            'modelo__C': [0.01, 0.1, 1.0, 10.0],
            'modelo__class_weight': [None, 'balanced']
        }
    }
}


# In[140]:


MODELOS = {
    "Logistic Regression": candidatos_modelos["Logistic Regression"],
    "Linear SVC": candidatos_modelos["Linear SVC"] 
}


# In[141]:


df_p4_features = DATOS["P4_discretizacion"].drop(columns=[TARGET])
cols_cat_p4 = [c for c in df_p4_features.columns if c not in numericas_presentes(df_p4_features)]


CONFIG_EXPERIMENTOS = {
    "P0_base": {"datos": DATOS["P0_base"], "sampler": None, "seleccion": None, "filtro_pcr": False},
    "P1_limpieza": {"datos": DATOS["P1_limpieza"], "sampler": None, "seleccion": None, "filtro_pcr": False},
    "P2_diabetes": {"datos": DATOS["P2_diabetes"], "sampler": None, "seleccion": None, "filtro_pcr": False},
    "P3_pcr": {"datos": DATOS["P3_pcr"], "sampler": None, "seleccion": None, "filtro_pcr": True},
    "P4_discretizacion": {"datos": DATOS["P4_discretizacion"], "sampler": None, "seleccion": None, "filtro_pcr": True},
    "P5_smotenc": {
        "datos": DATOS["P4_discretizacion"], 
        "sampler": SMOTENC(categorical_features=cols_cat_p4, random_state=SEED), 
        "seleccion": None, 
        "filtro_pcr": True},
    "P6_ros": {
        "datos": DATOS["P4_discretizacion"], 
        "sampler": RandomOverSampler(random_state=SEED), 
        "seleccion": None, 
        "filtro_pcr": True
        },
    
    # P7: Selección por grupos con SequentialFeatureSelector (mlxtend)
    "P7_sfs_grupos": {
        "datos": DATOS["P4_discretizacion"], 
        "sampler": None, 
        "seleccion": lambda g: SFS(
            estimator=LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED),
            k_features="parsimonious",   # Regla de 1 error estándar: elige el modelo más simple que no pierda rendimiento
            forward=False,               # Backward selection: elimina grupos paso a paso (igual que RFE)
            floating=False,
            scoring="f1_macro",          # Optimiza tu métrica objetivo
            cv=3,                        # Misma validación cruzada
            feature_groups=g,            # Trata cada grupo de dummies como un bloque indivisible
            n_jobs=-1
        ),
        "filtro_pcr": True
    }
}


# In[142]:


resultados_todos = []

for nombre_prep, cfg in CONFIG_EXPERIMENTOS.items():
    df_actual = cfg["datos"]
    X_actual = df_actual.drop(columns=[TARGET])
    
    for nombre_mod, m_cfg in MODELOS.items():
        pipeline = construir_pipeline(
            modelo=m_cfg["estimador"],
            param_grid=m_cfg["params"],
            sampler=cfg["sampler"],
            seleccion=cfg["seleccion"],
            filtro_pcr=cfg["filtro_pcr"],
            imputar=cfg.get("imputar", True),
        )
        
        res = evaluar(pipeline, df_actual, nombre_mod, nombre_prep, verbose=True)
        resultados_todos.append(res)

resultados_preprocesados = pd.concat(resultados_todos, ignore_index=True)
display(resultados_preprocesados)


# In[ ]:


def media_metrica(resultados, preprocesado, metrica):
    sub = resultados[resultados["preprocesado"] == preprocesado]
    return sub[metrica].str.split(" ± ").str[0].astype(float).mean()

resumen_preprocesados = pd.DataFrame({
    "f1_macro_val_medio": {p: media_metrica(resultados_preprocesados, p, "val_f1_macro") for p in resultados_preprocesados["preprocesado"].unique()},
    "recall_alto_val_medio": {p: media_metrica(resultados_preprocesados, p, "val_recall_alto") for p in resultados_preprocesados["preprocesado"].unique()},
    "accuracy_val_medio": {p: media_metrica(resultados_preprocesados, p, "val_accuracy") for p in resultados_preprocesados["preprocesado"].unique()},
}).round(4)

display(resumen_preprocesados)


# Para saber **qué variables elimina RFE y por qué** se ajusta una vez sobre todos los datos de P4. Esta celda es solo informativa: las métricas de arriba se calculan con la selección hecha dentro de cada partición de train.
# Se muestra la puntuación de validación (F1 macro) según el número de **grupos de variables** conservados, la mejor, el umbral de un error estándar y los grupos que se eliminan.

# In[ ]:


# Ver qué variables conserva Lasso sobre P4
X = DATOS["P4_discretizacion"].drop(columns=[TARGET])
y = DATOS["P4_discretizacion"][TARGET].astype(int)

# 1. Transformamos los datos
X_trans = codificador_escalador.fit_transform(imputador.fit_transform(X))
nombres_columnas = codificador_escalador.get_feature_names_out()

# 2. Ajustamos el selector Lasso
selector = clone(CONFIG_EXPERIMENTOS["P7_lasso"]["seleccion"])
selector.fit(X_trans, y)

# 3. Vemos qué columnas se quedan y cuáles se van
mascara = selector.get_support()
print(f"Lasso conserva {mascara.sum()} de {len(mascara)} columnas.")
print("Variables eliminadas por tener peso 0:")
print([nombres_columnas[i] for i, se_queda in enumerate(mascara) if not se_queda])


# ## 5. Evaluación

# In[ ]:


# Se aplican al test las transformaciones del preprocesamiento elegido
pasos_ganador = PREPROCESADOS["P4_discretizacion"] # por ejemplo
df_test_proc = aplicar_preprocesado(df_test_pob, pasos_ganador)

