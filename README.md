# Predicción Multiclase de Riesgo Cardiovascular en Población Adulta

> **Máster Universitario en Inteligencia Artificial — Universidad de Murcia (UMU)**  
> **Asignatura:** Machine Learning | **Curso:** 2026/2027  
> **Miembros del grupo:** Álvaro Arcís Reverte, Alejandro Nicolás Navarro
> **Profesores:** Raquel Martínez España
 
---

## 📋 Descripción del Proyecto

Las enfermedades cardiovasculares representan una de las principales causas de morbimortalidad a nivel global. Este proyecto tiene como objetivo diseñar, entrenar y evaluar modelos predictivos de Machine Learning capaces de clasificar con alta precisión y fiabilidad clínica el nivel de riesgo cardiovascular en una cohorte adulta de **2.254 pacientes** del Hospital Morales Meseguer.

La variable objetivo (`riesgo_cardiovascular`) contempla tres niveles de estratificación:
* **Clase 0:** Bajo Riesgo
* **Clase 1:** Riesgo Moderado
* **Clase 2:** Alto Riesgo

El repositorio recoge el flujo de trabajo completo: análisis exploratorio (EDA), preprocesados incrementales sin fugas de información (limpieza fisiológica, recodificación de la diabetes, filtro de PCR, discretización, SMOTENC, sobremuestreo aleatorio y selección secuencial de variables por grupos), validación cruzada anidada con búsqueda de hiperparámetros, comparación de cinco familias de modelos y evaluación final en un conjunto de test ciego.

---

## 👥 Integrantes del Equipo

| Nombre y Apellidos | Usuario de GitHub | Correo Institucional |
| :--- | :--- | :--- |
| *Álvaro Arcís Reverte* | `@devuelvanlapeseta` | `alvaro.arcisr@um.es` |
| *Alejandro Nicolás Navarro* | `@madalenointhemorning` | `a.nicolasnavarro@um.es` |


## 📂 Estructura del Repositorio

```text
├── data/
│   ├── raw/
│   │   └── enfermedad_cardiovascular.csv          # Dataset original, no se modifica
│   └── processed/
│       └── enfermedad_cardiovascular_preprocesado.csv  # Salida de una versión antigua del preprocesado (no la usan los notebooks actuales)
├── notebooks/
│   ├── 01_eda.ipynb                               # Análisis exploratorio e identificación de problemáticas
│   └── 02_preprocesamiento_y_modelado.ipynb       # Preprocesados P0-P7, ajuste de hiperparámetros por fases, elección del modelo y evaluación en test
├── resultados/
│   ├── figuras/                                   # Figuras de la memoria (EDA, fases 1 y 2, matriz de confusión y curvas ROC de test)
│   ├── tablas/                                    # Tablas LaTeX de la memoria (gaps de las fases 1 y 2, finalistas)
│   └── scripts_memoria/                           # Scripts que generan las figuras y tablas, y los CSV de resultados que usan
├── docs/
│   ├── memoria/
│   │   ├── memoria.tex                            # Memoria técnica en LaTeX (plantilla IEEE)
│   │   └── memoria.pdf                            # Memoria técnica compilada
│   ├── conversacionesIA/                          # Conversaciones con IA exportadas
│   └── decisiones.md                              # Registro de decisiones de diseño y avances
├── Uso_IA.tex / Uso_IA.pdf                        # Informe de uso ético de la IA
├── requirements.txt                               # Dependencias con versiones exactas
└── README.md
```

---

## ⚙️ Reproducción

### 1. Entorno

El proyecto se desarrolló con **Python 3.11**. Las versiones clave son scikit-learn 1.9.1, imbalanced-learn 0.14.2 y mlxtend 0.25.0.

Con conda:

```bash
conda create -n machine_learning python=3.11
conda activate machine_learning
pip install -r requirements.txt
```

O con un entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

`requirements.txt` se generó en Windows e incluye paquetes solo de Windows (`pywin32`) y de Intel MKL (`mkl_fft`, `mkl_random`, `mkl-service`). En Linux o macOS, quita esas líneas antes de instalar.

### 2. Notebooks

Ejecuta los notebooks en orden desde la carpeta `notebooks/`. Leen el dataset con rutas relativas a `data/raw/`.

1. `01_eda.ipynb`: análisis exploratorio.
2. `02_preprocesamiento_y_modelado.ipynb`: preprocesados, ajuste de hiperparámetros con validación anidada (secciones 6.1 a 6.5) y evaluación en test (sección 7). Usa la semilla `SEED = 123`.

El notebook 02 es el más costoso: la selección secuencial de variables (P7) se repite dentro de cada partición de la búsqueda de hiperparámetros.

### 3. Figuras y tablas de la memoria

Los scripts de `resultados/scripts_memoria/` regeneran las figuras y tablas a partir de los CSV de resultados extraídos del notebook 02 (`metricas_fases.csv` y `candidatos.csv`):

```bash
cd resultados/scripts_memoria
python generar_figuras.py        # Figuras de las fases 1 y 2 y de test   -> resultados/figuras/
python tablas_gaps.py            # Tablas de gaps de las fases 1 y 2      -> resultados/tablas/
python tabla_finalistas.py       # Tabla de finalistas                    -> resultados/tablas/
python generar_figuras_eda.py    # Figuras del EDA (lee data/raw/)        -> resultados/scripts_memoria/figuras_eda/
```

---

## 📄 Entregables

- **Memoria técnica:** `docs/memoria/memoria.pdf` (fuente en `docs/memoria/memoria.tex`).
- **Informe de uso de IA:** `Uso_IA.pdf` (fuente en `Uso_IA.tex`).
- **Código:** `notebooks/` y `resultados/scripts_memoria/`.
