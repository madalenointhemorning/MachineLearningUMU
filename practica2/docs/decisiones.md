# Registro de decisiones y avances

Evolución del planteamiento del problema, decisiones de diseño y avances del proyecto
(requisito "Evolución de la Memoria Técnica" de las instrucciones).

| Fecha | Autor | Decisión / avance | Justificación |
|---|---|---|---|
| 2026/11/22 | Alejandro Nicolás Navarro | Crear la carpeta de trabajo | Creacion de la carpeta para empezar la práctica |
| 2026/11/23 | Álvaro Arcís Reverte |  |  |
| 2026/11/24 | Álvaro Arcís Reverte |  |  |
| 2026/11/25 | Alejandro Nicolás Navarro | Decisiones del EDA sobre nulos, generación y serie objetivo diaria | Para analizar nulos se hace por bloques consecutivos, no solo por porcentaje. demanda_real tiene 6 bloques de 17 h, en las horas de más consumo, así que habrá que imputar antes de agregar a diario. demanda_prevista nos podrá servir para rellenar esos huecos, pero no como variable del modelo, porque sería fuga de información. En pvpc hay un hueco en 2021 que parece ser estructural por la tarifa 2.0TD y no se imputa.<br>Respecto a la generación, se elimina gen_fuel_gas (vacía o ±0,001 MWh). Se mantendrán los negativos porque la generación de REE es neta. gen_generación_total es redundante porque es la suma de las demás tecnologías. La generación solo se puede usar con retardo, porque la del día que se predice todavía no se conoce.<br>Para la serie objetivo se tendrá que agregar con la media diaria en MW y no con la suma, que desvía un ±4,2 % los días de cambio de hora. La caída del invierno 2021-2022 se explica por la Navidad. Hay estacionalidad semanal (sábado −10 %, domingo −16 %) y anual (máximos en invierno y verano), y una tendencia decreciente. Los festivos móviles no se recogen con el "día del año".<br>Se añaden los títulos de festivos, descomposición, estacionariedad y autocorrelación, y variables exógenas para un futuro análisis. |

