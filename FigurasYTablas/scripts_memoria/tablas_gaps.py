# Tablas III y IV de la memoria: gaps (entrenamiento - validación) de las fases 1 y 2.
# Datos: metricas_fases.csv (tablas de las celdas 80, 85 y 88 del notebook 02_preprocesamiento_y_modelado.ipynb).
# Escribe las filas LaTeX en la carpeta tablas/, junto a esta carpeta; se pegan dentro del tabular de cada tabla.
# En negrita, los gaps superiores a 0,10.
from pathlib import Path
import pandas as pd

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI.parent / "tablas"
SALIDA.mkdir(exist_ok=True)

df = pd.read_csv(AQUI / "metricas_fases.csv")
for c in ["gap_f1_macro", "gap_recall_alto", "gap_auc"]:
    df[c] = df[c].str.split(" ± ").str[0].astype(float)
MOD = [("Logistic Regression", "Regresión logística"), ("Linear SVC", "Linear SVC"), ("AdaBoost", "AdaBoost"),
       ("Random Forest", "Random Forest"), ("HistGradientBoosting", "HistGradientBoosting")]
MET = [("gap_f1_macro", "F1 macro"), ("gap_recall_alto", "Recall alto"), ("gap_auc", "AUC")]


def tabla(preps):
    lineas = []
    for m, etq in MOD:
        for k, (c, n) in enumerate(MET):
            vals = []
            for p in preps:
                v = df[(df.modelo == m) & (df.preprocesado == p)][c].iloc[0]
                s = f"{v:.3f}".replace(".", "{,}")
                vals.append("\\textbf{" + s + "}" if v > 0.10 else s)
            nombre = etq if k == 0 else ""
            lineas.append(f"{nombre} & {n} & " + " & ".join(vals) + r" \\")
        lineas.append(r"\hline")
    return "\n".join(lineas)


(SALIDA / "tabla_gaps_fase1.tex").write_text(
    tabla(["P0_base", "P1_limpieza", "P2_diabetes", "P3_pcr", "P4_discretizacion"]), encoding="utf-8")
(SALIDA / "tabla_gaps_fase2.tex").write_text(
    tabla(["P2_diabetes", "P5_smotenc", "P6_ros", "P7_sfs_grupos"]), encoding="utf-8")
print("ok")
