"""
MODULO BENCHMARK - i3 Atlas
=============================
Comparativa sistematica ML Tradicional vs Deep Learning:
  - Metricas de precision (accuracy, F1, precision, recall)
  - Metricas de rendimiento (tiempo, memoria)
  - Graficos comparativos
  - Recomendaciones por tipo de tarea

Autor: Juan Marcelo Gutierrez Miranda | @TodoEconometria

REFERENCIA ACADEMICA:
- Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing
  Supervised Classification Learning Algorithms. Neural Computation, 10(7).
- Raschka, S. (2018). Model Evaluation, Model Selection, and Algorithm
  Selection in Machine Learning. arXiv:1811.12808.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, FIGURES_DIR, COLORS


def load_results():
    """Carga resultados de ML y DL desde JSONs."""
    ml_file = OUTPUT_DIR / "ml_results.json"
    dl_file = OUTPUT_DIR / "dl_results.json"

    ml_results = {}
    dl_results = {}

    if ml_file.exists():
        try:
            with open(str(ml_file), "r", encoding="utf-8") as f:
                ml_results = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [WARN] ml_results.json corrupto: {e}")
    else:
        print("  [WARN] ml_results.json no encontrado.")

    if dl_file.exists():
        try:
            with open(str(dl_file), "r", encoding="utf-8") as f:
                dl_results = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [WARN] dl_results.json corrupto: {e}")
    else:
        print("  [WARN] dl_results.json no encontrado.")

    return ml_results, dl_results


def build_comparison_table(ml_results, dl_results):
    """Construye tabla comparativa unificada."""
    rows = []

    # ── Clasificacion de objetos ──
    if ml_results.get("task1") and ml_results["task1"].get("results"):
        for model_name, metrics in ml_results["task1"]["results"].items():
            rows.append({
                "tarea": "Clasificacion Objetos",
                "enfoque": "ML",
                "modelo": model_name,
                "accuracy": metrics.get("accuracy", 0),
                "f1": metrics.get("f1", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "train_time": metrics.get("train_time", 0),
                "infer_time": metrics.get("infer_time", 0),
                "memory_mb": metrics.get("memory_mb", 0),
            })

    if dl_results.get("task1"):
        t1 = dl_results["task1"]
        rows.append({
            "tarea": "Clasificacion Objetos",
            "enfoque": "DL",
            "modelo": t1.get("method", "DNN"),
            "accuracy": t1.get("accuracy", 0),
            "f1": t1.get("f1", 0),
            "precision": t1.get("precision", 0),
            "recall": t1.get("recall", 0),
            "train_time": t1.get("train_time", 0),
            "infer_time": t1.get("infer_time", 0),
            "memory_mb": 0,
        })

    # ── Deteccion de anomalias ──
    if ml_results.get("task2"):
        t2 = ml_results["task2"]
        rows.append({
            "tarea": "Deteccion Anomalias",
            "enfoque": "ML",
            "modelo": "Isolation Forest",
            "accuracy": 0,
            "f1": 0,
            "precision": 0,
            "recall": 0,
            "train_time": t2.get("train_time", 0),
            "infer_time": 0,
            "memory_mb": t2.get("memory_mb", 0),
            "n_anomalies": t2.get("n_anomalies", 0),
            "iso_detected": t2.get("interstellar_detected"),
        })

    if dl_results.get("task2"):
        t2 = dl_results["task2"]
        rows.append({
            "tarea": "Deteccion Anomalias",
            "enfoque": "DL",
            "modelo": "Autoencoder",
            "accuracy": 0,
            "f1": 0,
            "precision": 0,
            "recall": 0,
            "train_time": t2.get("train_time", 0),
            "infer_time": t2.get("infer_time", 0),
            "memory_mb": 0,
            "n_anomalies": t2.get("n_anomalies", 0),
            "iso_detected": t2.get("interstellar_detected"),
        })

    # ── Clasificacion espectral ──
    if ml_results.get("task3"):
        t3 = ml_results["task3"]
        rows.append({
            "tarea": "Clasificacion Espectral",
            "enfoque": "ML",
            "modelo": t3.get("method", "Random Forest"),
            "accuracy": t3.get("accuracy", 0),
            "f1": t3.get("f1", 0),
            "precision": 0,
            "recall": 0,
            "train_time": t3.get("train_time", 0),
            "infer_time": t3.get("infer_time", 0),
            "memory_mb": t3.get("memory_mb", 0),
        })

    if dl_results.get("task3"):
        t3 = dl_results["task3"]
        rows.append({
            "tarea": "Clasificacion Espectral",
            "enfoque": "DL",
            "modelo": t3.get("method", "1D-CNN"),
            "accuracy": t3.get("accuracy", 0),
            "f1": t3.get("f1", 0),
            "precision": 0,
            "recall": 0,
            "train_time": t3.get("train_time", 0),
            "infer_time": t3.get("infer_time", 0),
            "memory_mb": 0,
        })

    return pd.DataFrame(rows)


def generate_comparison_charts(comparison_df):
    """Genera graficos comparativos ML vs DL."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Barras agrupadas: Accuracy por tarea ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Accuracy comparison
    ax = axes[0, 0]
    clf_df = comparison_df[comparison_df["tarea"] == "Clasificacion Objetos"]
    if len(clf_df) > 0:
        colors_bars = [COLORS["ml"] if e == "ML" else COLORS["dl"] for e in clf_df["enfoque"]]
        bars = ax.bar(clf_df["modelo"], clf_df["accuracy"], color=colors_bars, alpha=0.8)
        ax.set_ylabel("Accuracy")
        ax.set_title("Clasificacion de Objetos - Accuracy", fontweight="bold")
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, clf_df["accuracy"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Training time comparison
    ax = axes[0, 1]
    if len(clf_df) > 0:
        colors_bars = [COLORS["ml"] if e == "ML" else COLORS["dl"] for e in clf_df["enfoque"]]
        bars = ax.bar(clf_df["modelo"], clf_df["train_time"], color=colors_bars, alpha=0.8)
        ax.set_ylabel("Tiempo (segundos)")
        ax.set_title("Clasificacion de Objetos - Tiempo de Entrenamiento", fontweight="bold")
        for bar, val in zip(bars, clf_df["train_time"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.1f}s", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Spectral comparison
    ax = axes[1, 0]
    spec_df = comparison_df[comparison_df["tarea"] == "Clasificacion Espectral"]
    if len(spec_df) > 0:
        colors_bars = [COLORS["ml"] if e == "ML" else COLORS["dl"] for e in spec_df["enfoque"]]
        x_pos = range(len(spec_df))
        bars = ax.bar(x_pos, spec_df["accuracy"], color=colors_bars, alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(spec_df["modelo"], rotation=15)
        ax.set_ylabel("Accuracy")
        ax.set_title("Clasificacion Espectral SDSS - ML vs DL", fontweight="bold")
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, spec_df["accuracy"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Anomaly detection summary
    ax = axes[1, 1]
    anom_df = comparison_df[comparison_df["tarea"] == "Deteccion Anomalias"]
    if len(anom_df) > 0:
        colors_bars = [COLORS["ml"] if e == "ML" else COLORS["dl"] for e in anom_df["enfoque"]]
        bars = ax.bar(anom_df["modelo"], anom_df.get("n_anomalies", 0), color=colors_bars, alpha=0.8)
        ax.set_ylabel("Anomalias Detectadas")
        ax.set_title("Deteccion de Anomalias - ML vs DL", fontweight="bold")
        for bar, row in zip(bars, anom_df.itertuples()):
            detected = getattr(row, "iso_detected", None)
            label = f"{getattr(row, 'n_anomalies', 0)}"
            if detected is not None:
                label += f"\nISO: {'SI' if detected else 'NO'}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    label, ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("i3 Atlas: Benchmark Comparativo ML vs Deep Learning",
                 fontweight="bold", fontsize=16)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "benchmark_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Scatter: Precision vs Tiempo ──
    fig, ax = plt.subplots(figsize=(10, 7))
    clf_all = comparison_df[comparison_df["accuracy"] > 0]
    if len(clf_all) > 0:
        for _, row in clf_all.iterrows():
            color = COLORS["ml"] if row["enfoque"] == "ML" else COLORS["dl"]
            marker = "o" if row["enfoque"] == "ML" else "^"
            ax.scatter(row["train_time"], row["accuracy"],
                       c=color, marker=marker, s=150, edgecolors="black",
                       linewidths=1, zorder=5)
            ax.annotate(row["modelo"],
                        (row["train_time"], row["accuracy"]),
                        fontsize=9, xytext=(8, 8), textcoords="offset points")

        ax.set_xlabel("Tiempo de Entrenamiento (s)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Trade-off: Precision vs Tiempo de Entrenamiento", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Leyenda
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["ml"],
                   markersize=10, label="ML Tradicional"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor=COLORS["dl"],
                   markersize=10, label="Deep Learning"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    fig.savefig(str(FIGURES_DIR / "benchmark_tradeoff.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("      Graficos guardados en output/figuras/")


def print_summary_table(comparison_df):
    """Imprime tabla resumen formateada."""
    print("\n" + "=" * 90)
    print("TABLA COMPARATIVA: ML Tradicional vs Deep Learning")
    print("=" * 90)
    print(f"{'Tarea':<25} {'Enfoque':<5} {'Modelo':<20} {'Accuracy':>9} {'F1':>7} {'T.Train':>8} {'T.Infer':>8}")
    print("-" * 90)
    for _, row in comparison_df.iterrows():
        print(f"{row['tarea']:<25} {row['enfoque']:<5} {row['modelo']:<20} "
              f"{row['accuracy']:>9.4f} {row['f1']:>7.4f} "
              f"{row['train_time']:>7.2f}s {row['infer_time']:>7.4f}s")
    print("=" * 90)


def generate_recommendations(comparison_df):
    """Genera recomendaciones basadas en los resultados."""
    recs = []

    # Clasificacion de objetos
    clf_df = comparison_df[
        (comparison_df["tarea"] == "Clasificacion Objetos") & (comparison_df["accuracy"] > 0)
    ]
    if len(clf_df) > 0:
        best = clf_df.loc[clf_df["accuracy"].idxmax()]
        fastest = clf_df.loc[clf_df["train_time"].idxmin()]
        recs.append({
            "tarea": "Clasificacion de Objetos Astronomicos",
            "mejor_accuracy": f"{best['modelo']} ({best['accuracy']:.4f})",
            "mas_rapido": f"{fastest['modelo']} ({fastest['train_time']:.2f}s)",
            "recomendacion": (
                "Para datasets tabulares con features bien definidas, "
                "ML tradicional (Random Forest/XGBoost) suele ser igual o mejor que DL, "
                "con tiempos de entrenamiento significativamente menores."
            ),
        })

    # Deteccion de anomalias
    anom_df = comparison_df[comparison_df["tarea"] == "Deteccion Anomalias"]
    if len(anom_df) > 0:
        recs.append({
            "tarea": "Deteccion de Anomalias (Interestelares)",
            "recomendacion": (
                "Isolation Forest (ML) es mas rapido y no requiere GPU. "
                "El Autoencoder (DL) puede capturar relaciones no lineales mas complejas "
                "y funciona mejor con datos de alta dimension. "
                "Para datos astronomicos, la combinacion de ambos es ideal."
            ),
        })

    # Espectral
    spec_df = comparison_df[
        (comparison_df["tarea"] == "Clasificacion Espectral") & (comparison_df["accuracy"] > 0)
    ]
    if len(spec_df) > 0:
        best = spec_df.loc[spec_df["accuracy"].idxmax()]
        recs.append({
            "tarea": "Clasificacion Espectral SDSS",
            "mejor_accuracy": f"{best['modelo']} ({best['accuracy']:.4f})",
            "recomendacion": (
                "Con solo 4 features (colores SDSS), ML y DL tienen rendimiento similar. "
                "DL brilla cuando hay cientos/miles de features (espectros completos). "
                "Para features reducidas, Random Forest es mas eficiente."
            ),
        })

    return recs


def run_benchmark():
    """Ejecuta el benchmark comparativo completo."""
    print("=" * 60)
    print("BENCHMARK: ML Tradicional vs Deep Learning")
    print("=" * 60)

    print("\n[1/4] Cargando resultados ...")
    ml_results, dl_results = load_results()

    print("[2/4] Construyendo tabla comparativa ...")
    comparison_df = build_comparison_table(ml_results, dl_results)

    if len(comparison_df) == 0:
        print("  No hay resultados para comparar. Ejecuta ML y DL primero.")
        return None

    print_summary_table(comparison_df)

    print("\n[3/4] Generando graficos comparativos ...")
    generate_comparison_charts(comparison_df)

    print("\n[4/4] Generando recomendaciones ...")
    recommendations = generate_recommendations(comparison_df)

    print("\n" + "=" * 60)
    print("RECOMENDACIONES")
    print("=" * 60)
    for rec in recommendations:
        print(f"\n  {rec['tarea']}:")
        if "mejor_accuracy" in rec:
            print(f"    Mejor accuracy: {rec['mejor_accuracy']}")
        if "mas_rapido" in rec:
            print(f"    Mas rapido: {rec['mas_rapido']}")
        print(f"    -> {rec['recomendacion']}")

    # Guardar todo
    results = {
        "comparison_table": comparison_df.to_dict("records"),
        "recommendations": recommendations,
        "ml_total_time": ml_results.get("total_time", 0),
        "dl_total_time": dl_results.get("total_time", 0),
        "dl_gpu": dl_results.get("gpu", False),
    }

    results_file = OUTPUT_DIR / "benchmark_results.json"
    with open(str(results_file), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Resultados: {results_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_benchmark()
