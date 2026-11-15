# Tabla V de la memoria: las dos mejores configuraciones de cada familia con gap de F1 <= 0,10.
# Datos: candidatos.csv (tabla de candidatos de la celda 101 del notebook 02_preprocesamiento_y_modelado.ipynb).
# Escribe las filas LaTeX en tablas/tabla_finalistas.tex, junto a esta carpeta.
# En negrita: el mejor valor de cada columna, los gaps de recall superiores a 0,10 y la fila del modelo elegido.
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import pandas as pd

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI.parent / "tablas"
SALIDA.mkdir(exist_ok=True)

COLS = ["val_f1_macro", "gap_f1_macro", "val_recall_alto", "gap_recall_alto", "val_auc", "gap_auc"]
df = pd.read_csv(AQUI / "candidatos.csv")
for c in COLS:
    partes = df[c].str.split(" ± ", expand=True).astype(float)
    df[c], df[c + "_std"] = partes[0], partes[1]

ETQ = {"Logistic Regression": "Regresión logística", "Linear SVC": "Linear SVC", "AdaBoost": "AdaBoost",
       "Random Forest": "Random Forest", "HistGradientBoosting": "HistGradientBoosting"}
CORTO = {"P0_base": "P0", "P1_limpieza": "P1", "P2_diabetes": "P2", "P3_pcr": "P3", "P4_discretizacion": "P4",
         "P5_smotenc": "P5", "P6_ros": "P6", "P7_sfs_grupos": "P7"}

# Dos mejores por familia (F1 de validación) entre las que cumplen el gap; familias ordenadas por su mejor F1
ok = df[df.gap_f1_macro <= 0.10].sort_values("val_f1_macro", ascending=False)
fin = ok.groupby("modelo", sort=False).head(2)
orden = fin.groupby("modelo", sort=False).val_f1_macro.max().sort_values(ascending=False).index
fin = pd.concat([fin[fin.modelo == m] for m in orden])

mejor = {"val_f1_macro": fin.val_f1_macro.max(), "gap_f1_macro": fin.gap_f1_macro.min(),
         "val_recall_alto": fin.val_recall_alto.max(), "val_auc": fin.val_auc.max(), "gap_auc": fin.gap_auc.min()}
ganador = fin.index[0]


def num(v):
    # Redondeo a 3 decimales hacia arriba en los empates (0,8105 -> 0,811), como en la memoria
    return str(Decimal(str(v)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)).replace(".", ",")


def negrita(s, si):
    return "\\textbf{" + s + "}" if si else s


lineas = []
for m in orden:
    for i, r in fin[fin.modelo == m].iterrows():
        nombre = ETQ[m]
        if r.prueba.startswith("fino") and m == "AdaBoost":
            nombre += f" (ajuste fino {r.prueba[4]})"
        celdas = [negrita(nombre, i == ganador), negrita(CORTO[r.preprocesado], i == ganador)]
        for c in COLS:
            s = num(r[c]) + (f" $\\pm$ {num(r[c + '_std'])}" if c in ("val_f1_macro", "val_recall_alto") else "")
            marcar = r[c] > 0.10 if c == "gap_recall_alto" else round(r[c], 3) == round(mejor[c], 3)
            celdas.append(negrita(s, marcar))
        lineas.append(" & ".join(celdas) + r" \\")
    lineas.append(r"\hline")
(SALIDA / "tabla_finalistas.tex").write_text("\n".join(lineas), encoding="utf-8")
print("ok")
