# Predicción Multiclase de Riesgo Cardiovascular en Población Adulta

> **Máster Universitario en Inteligencia Artificial — Universidad de Murcia (UMU)**  
> **Asignatura:** Machine Learning | **Curso:** 2026/2027  
> **Miembros del grupo:** Álvaro Arcís Reverte, Alejandro Nicolás Navarro
> **Profesores:** Raquel Martínez España
 
---

## 📋 Descripción del Proyecto

Las enfermedades cardiovasculares representan una de las principales causas de morbimortalidad a nivel global. Este proyecto tiene como objetivo diseñar, entrenar y evaluar modelos predictivos de Machine Learning capaces de clasificar con alta precisión y fiabilidad clínica el nivel de riesgo cardiovascular en una cohorte adulta de **2.254 pacientes**.

La variable objetivo (`riesgo_cardiovascular`) contempla tres niveles de estratificación:
* **Clase 0:** Bajo Riesgo
* **Clase 1:** Riesgo Moderado
* **Clase 2:** Alto Riesgo

El repositorio recoge el flujo de trabajo end-to-end: desde el análisis exploratorio de datos (EDA) y la mitigación de problemáticas de calidad del dato (imputación, outliers, desbalanceo, escalado), hasta el ajuste de hiperparámetros, validación cruzada y comparación estadística de clasificadores base y ensambles.

---

## 👥 Integrantes del Equipo

| Nombre y Apellidos | Usuario de GitHub | Correo Institucional |
| :--- | :--- | :--- |
| *Álvaro Arcís Reverte* | `@devuelvanlapeseta` | `usuario1@um.es` |
| *Alejandro Nicolás Navarro* | `@madalenointhemorning` | `a.nicolasnavarro@um.es` |

---

## 📂 Estructura del Repositorio

El proyecto se organiza bajo una estructura modular orientada a la reproducibilidad científica:

```text
├── data/
│   ├── raw/                        # Dataset original (enfermedad_cardiovascular.csv), no se modifica
│   └── processed/                  # Datasets transformados (si aplica)
├── notebooks/
│   ├── 01_eda.ipynb                # 3.1 Análisis exploratorio e identificación de problemáticas
│   ├── 02_preprocesamiento.ipynb   # 3.1 Técnicas de preprocesamiento (mín. 3)
│   ├── 03_modelado.ipynb           # 3.2/3.3 Modelos base y ensambles, validación e hiperparámetros
│   ├── 04_evaluacion.ipynb         # 3.3/3.4 Métricas, comparativa y conclusiones
│   └── 05_test_estadisticos.ipynb  # 3.5 (Opcional) Test estadísticos
├── src/                            # Código reutilizable importado desde los notebooks
│   ├── config.py                   # Rutas, semilla, columnas
│   ├── data_loader.py              # Carga y comprobaciones de integridad
│   ├── preprocessing.py            # Transformaciones de preprocesamiento
│   ├── models.py                   # Modelos y espacios de hiperparámetros
│   ├── evaluation.py               # Validación y métricas
│   └── statistical_tests.py        # Comparación estadística (opcional)
├── results/
│   ├── figures/                    # Gráficas exportadas para la memoria
│   └── tables/                     # Tablas de resultados (CSV)
├── models/                         # Modelos entrenados serializados (ignorado por git)
├── docs/
│   ├── decisiones.md               # Registro de decisiones de diseño y avances
│   ├── memoria/                    # Memoria técnica en LaTeX (plantilla IEEE, máx. 15 págs.)
│   ├── uso_etico_ia/               # Informe PDF de uso ético de IA
│   └── video.md                    # Enlace al vídeo explicativo (máx. 7 min)
├── Enunciados/                     # Enunciado, instrucciones y plantilla oficial
├── Uso_IA.txt                      # Registro de prompts (base del informe de uso de IA)
├── requirements.txt                # Dependencias con versiones exactas
└── README.md
```

## ⚙️ Reproducción

--- Por hacer ---
