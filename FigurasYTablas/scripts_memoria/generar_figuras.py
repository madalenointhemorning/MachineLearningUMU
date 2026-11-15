# Figuras de los apartados 4 y 5 de la memoria. Se guardan en la carpeta figuras/, junto a esta carpeta.
# Datos: salidas del notebook 02_preprocesamiento_y_modelado.ipynb recompilado el 2026-11-15
#   - metricas_fases.csv: tablas de las celdas 80 (fase 1), 85 y 88 (fase 2), media ± desviación de las 3 particiones externas
#   - matriz de confusión y AUC por clase: salidas de las celdas 106 y 107 (evaluación única en test)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI.parent / "figuras"
SALIDA.mkdir(exist_ok=True)

TINTA = "#1f1f1e"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"
AZUL = "#2a78d6"
AZUL_SUAVE = "#d6e6f8"
NARANJA = "#eb6834"
GRIS = "#6f6e6a"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.edgecolor": TINTA_2, "axes.labelcolor": TINTA, "xtick.color": TINTA_2, "ytick.color": TINTA,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})

MODELOS = ["Logistic Regression", "Linear SVC", "AdaBoost", "Random Forest", "HistGradientBoosting"]
ETQ_MODELO = {"Logistic Regression": "Regresión logística", "Linear SVC": "Linear SVC", "AdaBoost": "AdaBoost",
              "Random Forest": "Random Forest", "HistGradientBoosting": "HistGradientBoosting"}
CORTO = {"P0_base": "P0", "P1_limpieza": "P1", "P2_diabetes": "P2", "P3_pcr": "P3", "P4_discretizacion": "P4",
         "P5_smotenc": "P5", "P6_ros": "P6", "P7_sfs_grupos": "P7"}
METRICAS = [("val_f1_macro", "F1 macro"), ("val_recall_alto", "Recall de alto riesgo"), ("val_auc", "AUC ROC macro")]

df = pd.read_csv(AQUI / "metricas_fases.csv")
for c in [c for c in df.columns if c.startswith(("train_", "val_", "gap_"))]:
    partes = df[c].str.split(" ± ", expand=True).astype(float)
    df[c], df[c + "_std"] = partes[0], partes[1]


def forest(preps, referencia, nombre, titulo_ref):
    """Un panel por métrica; filas = modelo x preprocesado; banda = referencia ± desviación."""
    sub = df[df["preprocesado"].isin(preps)]
    n = len(preps)
    hueco = 1.5
    filas = []          # (y, modelo, prep)
    y = 0.0
    centros = {}
    for m in MODELOS:
        ys = []
        for p in preps:
            filas.append((y, m, p)); ys.append(y); y += 1
        centros[m] = (min(ys), max(ys))
        y += hueco
    alto = 0.15 * len(filas) + 0.25 * len(MODELOS) + 0.55
    fig, axes = plt.subplots(1, 3, figsize=(7.16, alto), sharey=True, constrained_layout=True)
    for ax, (met, titulo) in zip(axes, METRICAS):
        for m in MODELOS:
            r = sub[(sub["modelo"] == m) & (sub["preprocesado"] == referencia)].iloc[0]
            y0, y1 = centros[m]
            ax.axvspan(r[met] - r[met + "_std"], r[met] + r[met + "_std"],
                       ymin=0, ymax=1, alpha=0)  # (sin efecto, mantiene límites)
            ax.fill_betweenx([y0 - 0.45, y1 + 0.45], r[met] - r[met + "_std"], r[met] + r[met + "_std"],
                             color=AZUL_SUAVE, lw=0, zorder=0)
            ax.plot([r[met]] * 2, [y0 - 0.45, y1 + 0.45], color=AZUL, lw=0.8, ls=(0, (2, 2)), zorder=1)
        for yy, m, p in filas:
            r = sub[(sub["modelo"] == m) & (sub["preprocesado"] == p)].iloc[0]
            es_ref = p == referencia
            color = AZUL if es_ref else TINTA_2
            ax.errorbar(r[met], yy, xerr=r[met + "_std"], fmt="o", ms=4.2 if es_ref else 3.6,
                        color=color, ecolor=color, elinewidth=0.9, capsize=1.8, zorder=3,
                        mfc=color, mec="white", mew=0.6)
        ax.grid(axis="x", color=REJILLA, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        # El eje empieza y acaba en una marca, así el centro del panel coincide con el centro de la escala
        lo, hi = ax.dataLim.x0, ax.dataLim.x1
        marcas = matplotlib.ticker.MaxNLocator(5).tick_values(lo, hi)
        marcas = marcas[(marcas >= marcas[marcas <= lo].max()) & (marcas <= marcas[marcas >= hi].min())]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(marcas))
        ax.set_xlim(marcas[0], marcas[-1])
        # Título en el punto medio de la escala
        ax.annotate(titulo, xy=((marcas[0] + marcas[-1]) / 2, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, color=TINTA, annotation_clip=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    axes[0].set_yticks([f[0] for f in filas], [CORTO[f[2]] for f in filas], fontsize=8)
    axes[0].set_ylim(y - hueco + 0.4, -2.1)
    # nombre del modelo encima de cada bloque
    for m in MODELOS:
        y0, _ = centros[m]
        axes[0].annotate(ETQ_MODELO[m], xy=(0, y0 - 1.05), xycoords=("axes fraction", "data"),
                         xytext=(-22, 0), textcoords="offset points", ha="left", va="bottom",
                         fontsize=9, fontweight="bold", color=TINTA, annotation_clip=False)
    leyenda = [Line2D([], [], marker="o", ls="", color=AZUL, mec="white", ms=5, label=f"{titulo_ref} (media ± desv.)"),
               Line2D([], [], marker="o", ls="", color=TINTA_2, mec="white", ms=5, label="Resto de preprocesados"),
               Patch(color=AZUL_SUAVE, label=f"Banda ± desviación de {CORTO[referencia]}")]
    fig.legend(handles=leyenda, loc="outside lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.savefig(SALIDA / f"{nombre}.pdf"); fig.savefig(SALIDA / f"{nombre}.png")
    plt.close(fig)


forest(["P0_base", "P1_limpieza", "P2_diabetes", "P3_pcr", "P4_discretizacion"], "P2_diabetes",
       "fase1_metricas", "P2, preprocesado elegido")
forest(["P2_diabetes", "P5_smotenc", "P6_ros", "P7_sfs_grupos"], "P2_diabetes",
       "fase2_metricas", "P2, base de la fase 2")

# ---------------------------------------------------------------- Test: matriz de confusión + AUC por clase
# NOTA: los valores de cm, auc_train, auc_test y gap_nb los ha rellenado a mano un agente (Claude)
# leyendo los resultados del notebook. No se leen automáticamente: si se reejecuta el modelo, hay que
# actualizarlos a mano.
cm =np.array([[172, 46, 8], [44, 63, 28], [2, 19, 69]])
clases = ["Bajo", "Moderado", "Alto"]
pct = cm / cm.sum(axis=1, keepdims=True)
auc_train = {"Bajo": 0.8885, "Moderado": 0.7411, "Alto": 0.9280}
auc_test = {"Bajo": 0.8581, "Moderado": 0.6557, "Alto": 0.9169}
gap_nb = {"Bajo": "0,030", "Moderado": "0,086", "Alto": "0,011"}   # gaps tal como salen en el notebook (celda 106)
AZULES = LinearSegmentedColormap.from_list("azules", ["#f4f8fd", "#9cc3ea", "#2f6fb5", "#163e72"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.75), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1, 1.25]})
ax1.imshow(pct, cmap=AZULES, vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax1.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center", fontsize=9,
                 color="white" if pct[i, j] > 0.5 else TINTA)
ax1.set_xticks(range(3), clases); ax1.set_yticks(range(3), clases)
ax1.set_xlabel("Riesgo predicho"); ax1.set_ylabel("Riesgo real")
ax1.tick_params(length=0)
for s in ax1.spines.values():
    s.set_visible(False)
ax1.set_title("(a) Matriz de confusión en test", loc="left", fontsize=10)

for k, c in enumerate(clases):
    tr, te = auc_train[c], auc_test[c]
    ax2.plot([te, tr], [k, k], color="#b9b8b2", lw=2.2, solid_capstyle="round", zorder=1)
    ax2.plot(tr, k, "o", ms=7, mfc="white", mec=GRIS, mew=1.4, zorder=2)
    ax2.plot(te, k, "o", ms=7, color=NARANJA if c == "Moderado" else AZUL, mec="white", mew=0.8, zorder=3)
    ax2.annotate(f"gap {gap_nb[c]}", xy=(tr, k), xytext=(8, 0), textcoords="offset points",
                 va="center", fontsize=8.5, color=NARANJA if c == "Moderado" else TINTA_2,
                 fontweight="bold" if c == "Moderado" else "normal")
    ax2.annotate(f"{te:.3f}".replace(".", ","), xy=(te, k), xytext=(-7, 0), textcoords="offset points",
                 va="center", ha="right", fontsize=8.5, color=TINTA)
ax2.axvline(0.5, color=TINTA_2, lw=0.8, ls=(0, (3, 2)))
ax2.text(0.508, -0.42, "azar (0,5)", fontsize=8, color=TINTA_2, va="center")
ax2.set_yticks(range(3), clases); ax2.set_ylim(2.6, -0.6)
ax2.set_xlim(0.45, 1.0)
ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
ax2.set_xlabel("AUC uno contra resto")
ax2.tick_params(axis="y", length=0); ax2.spines["left"].set_visible(False)
ax2.grid(axis="x", color=REJILLA, lw=0.6); ax2.set_axisbelow(True)
ax2.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=GRIS, mew=1.4, ms=7, label="Entrenamiento"),
                    Line2D([], [], marker="o", ls="", color=AZUL, mec="white", ms=7, label="Test")],
           loc="center left", bbox_to_anchor=(0.1, 0.27), frameon=False, fontsize=8.5)
ax2.set_title("(b) AUC por clase: entrenamiento frente a test", loc="left", fontsize=10)
fig.savefig(SALIDA / "test_confusion_auc.pdf"); fig.savefig(SALIDA / "test_confusion_auc.png")
plt.close(fig)
print("ok")
