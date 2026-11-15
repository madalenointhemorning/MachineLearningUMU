"""Figuras del EDA para la memoria, legibles al tamaño de impresión.

Reproduce las figuras de la celda 13 de notebooks/01_eda.ipynb con los mismos datos (CSV original, sin limpiar,
igual que el notebook) pero dibujadas al ancho real de la página de IEEEtran a una columna (~7,1 pulgadas), con
letra de 8-10 pt y números en negrita. No modifica el notebook ni el CSV.

Uso (desde pruebas/docunuevo):
    C:/Users/ALEJANDRO/anaconda3/envs/machine_learning/python.exe generar_figuras_eda.py
Salida: figuras_eda/*.png (300 ppp), con los mismos nombres que usa documentacion.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parents[1] / "data" / "raw" / "enfermedad_cardiovascular.csv"
SALIDA = AQUI / "figuras_eda"
SALIDA.mkdir(exist_ok=True)

ANCHO = 7.1  # pulgadas: ancho de texto de IEEEtran a una columna (\textwidth)
DPI = 300

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

TARGET = "riesgo_cardiovascular"
BOOL_COLS = ["fumador_actual", "hipertension_previa", "diabetes_mellitus",
             "tratamiento_estatinas", "uso_antihipertensivos"]
CAT_COLS = ["tipo_alimentacion", "nivel_actividad", "antecedentes_familiares_tipo"]

# Etiquetas cortas y legibles para los ejes
ETIQ = {
    "edad": "Edad",
    "presion_sistolica": "PA sistólica",
    "presion_diastolica": "PA diastólica",
    "colesterol_total": "Colesterol total",
    "colesterol_hdl": "Colesterol HDL",
    "colesterol_ldl": "Colesterol LDL",
    "trigliceridos": "Triglicéridos",
    "glucosa_ayunas": "Glucosa ayunas",
    "indice_masa_corporal": "IMC",
    "frecuencia_cardiaca_reposo": "FC reposo",
    "horas_sueno_diarias": "Horas de sueño",
    "consumo_alcohol_semanal": "Alcohol semanal",
    "anos_fumador": "Años fumador",
    "proteina_c_reactiva": "PCR",
    "acido_urico": "Ácido úrico",
    "clearance_creatinina": "Aclar. creatinina",
    "frecuencia_ejercicio_semanal": "Ejercicio semanal",
    "perimetro_cintura": "Perímetro cintura",
    "hemoglobina_glicosilada": "HbA1c",
    TARGET: "Riesgo CV",
    "fumador_actual": "Fumador actual",
    "hipertension_previa": "Hipertensión previa",
    "diabetes_mellitus": "Diabetes mellitus",
    "tratamiento_estatinas": "Estatinas",
    "uso_antihipertensivos": "Antihipertensivos",
    "tipo_alimentacion": "Tipo de alimentación",
    "nivel_actividad": "Nivel de actividad",
    "antecedentes_familiares_tipo": "Antecedentes familiares",
}
UNIDADES = {
    "consumo_alcohol_semanal": "unidades/semana", "hemoglobina_glicosilada": "%",
    "proteina_c_reactiva": "mg/L", "acido_urico": "mg/dL", "glucosa_ayunas": "mg/dL",
    "frecuencia_ejercicio_semanal": "días/semana", "colesterol_hdl": "mg/dL",
    "indice_masa_corporal": "kg/m²",
}
ORDEN_CAT = {
    "tipo_alimentacion": ["Mediterránea", "Vegetariana", "Baja en carbohidratos", "Occidental"],
    "nivel_actividad": ["Sedentario", "Ligeramente activo", "Moderadamente activo", "Muy activo"],
    "antecedentes_familiares_tipo": ["Ninguno", "Materno", "Paterno", "Ambos"],
}
CLASES = ["Bajo", "Moderado", "Alto"]
# Rampa ordinal del riesgo: se distingue por luminosidad, también en escala de grises
COL_CLASE = ["#9ecae1", "#4a90c2", "#0b3d6e"]

df = pd.read_csv(DATOS)
assert df.shape == (2254, 28)
NUM_COLS = [c for c in df.columns
            if c not in BOOL_COLS + CAT_COLS + [TARGET] and pd.api.types.is_numeric_dtype(df[c])]


def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre)
    plt.close(fig)
    print("guardada", SALIDA / nombre)


def sin_cero(v):
    """0.39 -> .39 y -0.08 -> −.08: números más cortos para que quepan con letra mayor."""
    if abs(v) < 0.005:
        return ".00"
    t = f"{v:.2f}".replace("0.", ".", 1)
    return t.replace("-", "−")


# ---------------------------------------------------------------------------------------------
# 1. Matriz de correlación de Spearman
# ---------------------------------------------------------------------------------------------
corr = df[NUM_COLS + [TARGET]].corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool))
etiquetas = [ETIQ[c] for c in corr.columns]
anot = corr.map(sin_cero) if hasattr(corr, "map") else corr.applymap(sin_cero)

fig, ax = plt.subplots(figsize=(ANCHO, 6.3))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-0.4, vmax=0.4,
            annot=anot, fmt="", annot_kws={"size": 7, "weight": "bold"},
            linewidths=0.6, linecolor="white", square=False,
            cbar_kws={"shrink": 0.6, "pad": 0.01, "label": "ρ de Spearman"}, ax=ax)
ax.set_xticklabels(etiquetas[:-1] + [""], rotation=45, ha="right")
ax.set_yticklabels([""] + etiquetas[1:], rotation=0)
ax.tick_params(length=0)
# Resalta la fila del riesgo cardiovascular
n = len(corr)
ax.add_patch(plt.Rectangle((0, n - 1), n - 1, 1, fill=False, edgecolor="black", lw=1.4))
ax.get_yticklabels()[-1].set_fontweight("bold")
guardar(fig, "eda_spearman.png")

# ---------------------------------------------------------------------------------------------
# 2. Boxplots por clase de las 8 variables más correlacionadas con el riesgo
# ---------------------------------------------------------------------------------------------
r_target = corr.loc[NUM_COLS, TARGET]
top = r_target.abs().sort_values(ascending=False).head(8).index

fig, axes = plt.subplots(2, 4, figsize=(ANCHO, 4.6))
for ax, col in zip(axes.flat, top):
    sns.boxplot(x=TARGET, y=col, data=df, hue=TARGET, palette=COL_CLASE, legend=False, ax=ax,
                width=0.6, linewidth=0.9, fliersize=1.8,
                flierprops={"marker": "o", "markerfacecolor": "none", "markeredgewidth": 0.6},
                medianprops={"color": "#e8a33d", "linewidth": 1.8})
    ax.set_title(ETIQ[col], pad=12)
    ax.text(0.5, 1.0, f"ρ = {r_target[col]:+.2f}".replace("-", "−"), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.5)
    ax.set_xlabel("")
    ax.set_ylabel(UNIDADES.get(col, ""), fontsize=8)
    ax.set_xticks([0, 1, 2], ["Bajo", "Mod.", "Alto"])
    ax.grid(axis="y", lw=0.4, alpha=0.6)
fig.supxlabel("Nivel de riesgo cardiovascular", fontsize=9, y=0.0)
fig.tight_layout(h_pad=1.2, w_pad=0.8)
guardar(fig, "eda_boxplots_top8.png")

# ---------------------------------------------------------------------------------------------
# 3. Prevalencia de comorbilidades y tratamientos por clase (% de pacientes con valor 1)
# ---------------------------------------------------------------------------------------------
prev = df.groupby(TARGET)[BOOL_COLS].mean().T * 100  # filas: variable, columnas: clase
x = np.arange(len(BOOL_COLS))
w = 0.27
fig, ax = plt.subplots(figsize=(ANCHO, 2.6))
for k in range(3):
    barras = ax.bar(x + (k - 1) * w, prev[k], w, color=COL_CLASE[k], label=f"Riesgo {CLASES[k].lower()}",
                    edgecolor="white", linewidth=0.5)
    ax.bar_label(barras, fmt="%.0f", padding=1.5, fontsize=7.5, fontweight="bold")
ax.set_xticks(x, [ETIQ[c] for c in BOOL_COLS])
ax.set_ylabel("% de pacientes")
ax.set_ylim(0, prev.values.max() * 1.18)
ax.grid(axis="y", lw=0.4, alpha=0.6)
ax.set_axisbelow(True)
ax.legend(ncol=3, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.13))
guardar(fig, "eda_comorbilidades.png")

# ---------------------------------------------------------------------------------------------
# 4. Estilo de vida y antecedentes: % de cada categoría dentro de cada nivel de riesgo
# ---------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(ANCHO, 2.9), sharey=True)
for ax, col in zip(axes, CAT_COLS):
    ct = pd.crosstab(df[col], df[TARGET], normalize="columns") * 100
    ct = ct.reindex(ORDEN_CAT[col])
    xc = np.arange(len(ct))
    for k in range(3):
        barras = ax.bar(xc + (k - 1) * w, ct[k], w, color=COL_CLASE[k], label=f"Riesgo {CLASES[k].lower()}",
                        edgecolor="white", linewidth=0.5)
        ax.bar_label(barras, fmt="%.0f", padding=1.2, fontsize=6.8, fontweight="bold")
    cortas = {"Mediterránea": "Medi-\nterránea", "Vegetariana": "Vege-\ntariana",
              "Baja en carbohidratos": "Baja en\ncarbohid.", "Occidental": "Occi-\ndental",
              "Sedentario": "Seden-\ntario", "Ligeramente activo": "Ligeram.\nactivo",
              "Moderadamente activo": "Moderad.\nactivo", "Muy activo": "Muy\nactivo"}
    ax.set_xticks(xc, [cortas.get(c, c) for c in ct.index], fontsize=8)
    ax.set_title(ETIQ[col])
    ax.grid(axis="y", lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("% dentro del nivel de riesgo")
axes[0].set_ylim(0, 68)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(w_pad=0.6)
guardar(fig, "eda_estilo_vida.png")
