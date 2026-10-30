# Registro de decisiones y avances

Evolución del planteamiento del problema, decisiones de diseño y avances del proyecto
(requisito "Evolución de la Memoria Técnica" de las instrucciones).

| Fecha | Autor | Decisión / avance | Justificación |
| :--- | :--- | :--- | :--- |
| 2026-10-28 | Alejandro Nicolás Navarro | Definición de la estructura de carpetas del repositorio | Organización modular y reproducible |
| 2026-10-29 | Alejandro Nicolás Navarro | Introducción inicial del EDA y descubrimiento de primeras limitaciones | Se realiza un análisis inicial y se encuentran inconsistencias en ciertas variables, como en la del consumo del alcohol, con valores negativos. Este descubrimiento nos permite entender la naturaleza del dataset y nos da un aviso sobre su necesidad de preprocesado |
| 2026-10-29 | Álvaro Arcís Reverte | Inicio del análisis exploratorio (EDA) y revisión progresiva de los primeros resultados | Se inician los primeros análisis de los datos en el notebook de EDA para encontrar anomalías e inconsistencias y se comienza a redactar las primeras hipótesis y puntos a tener en cuenta para el siguiente apartado de preprocesamiento. De momento se ha comprobado que hay 4 columnas con nulos, no hay duplicados y los tipos de datos son correctos. Columnas que deberían tener valores discretos tienen valores continuos, seguramente por cálculo a través de la media aritmética, y de momento no se ha decidido redondear nada para no perder información precisa. Tampoco se ha encontrado a priori ninguna columna innecesaria que no tenga relación con el riesgo cardiovascular |
| 2026-10-30 | Alejandro Nicolás Navarro | Introducción de conclusiones para el análisis univariante y bivariante | Se analizan las gráficas y se explican los resultados de los análisis realizados con gráficas diferentes dependiendo de las variables que tratamos, y un análisis de correlación para las 6 variables más relacionadas con el target |
