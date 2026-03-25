"""
MODULO ML TRADICIONAL - i3 Atlas
==================================
Pipeline de Machine Learning clasico (CPU-optimized):
  Tarea 1: Clasificacion supervisada (RF, SVM, XGBoost)
  Tarea 2: Deteccion de anomalias (PCA + K-Means + Isolation Forest)
  Tarea 3: Clasificacion espectral SDSS (Random Forest)

Autor: Juan Marcelo Gutierrez Miranda | @TodoEconometria

REFERENCIA ACADEMICA:
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
  Journal of Machine Learning Research, 12, 2825-2830.
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest.
  IEEE International Conference on Data Mining, 413-422.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
  KDD '16, 785-794.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, silhouette_score,
)

from config import (
    PROCESSED_PARQUET, DATA_CACHE_DIR, FIGURES_DIR, OUTPUT_DIR,
    FEATURES_ORBITAL, FEATURES_PHYSICAL, FEATURES_ENGINEERED,
    ML_PARAMS, COLORS, SPARK_DATA_URI,
)


def _measure_memory():
    """Retorna uso de memoria actual en MB."""
    tracemalloc.start()
    return tracemalloc.get_traced_memory()[1] / 1024 / 1024


# ═══════════════════════════════════════════════════════════════
# TAREA 1: CLASIFICACION SUPERVISADA
# ═══════════════════════════════════════════════════════════════

def task1_classification(spark=None):
    """
    Clasificacion de objetos astronomicos por tipo orbital.
    Compara: Random Forest vs SVM vs XGBoost.
    """
    print("=" * 60)
    print("ML TAREA 1: Clasificacion Supervisada de Objetos")
    print("=" * 60)

    # Cargar datos
    print("\n[1/6] Cargando datos procesados ...")
    df = pd.read_parquet(str(PROCESSED_PARQUET))
    print(f"      Filas: {len(df):,}")

    # Preparar features y target
    print("[2/6] Preparando features ...")
    feature_cols = FEATURES_ORBITAL + FEATURES_PHYSICAL + FEATURES_ENGINEERED
    feature_cols = [f for f in feature_cols if f in df.columns]

    # Filtrar clases con suficientes muestras
    class_counts = df["object_type"].value_counts()
    valid_classes = class_counts[class_counts >= 50].index.tolist()
    df_ml = df[df["object_type"].isin(valid_classes)].copy()
    print(f"      Clases validas (>=50 muestras): {valid_classes}")

    X = df_ml[feature_cols].copy()
    y = df_ml["object_type"].copy()

    # Imputar y escalar
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=ML_PARAMS["test_size"],
        random_state=ML_PARAMS["random_state"],
        stratify=y_encoded,
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    results = {}

    # ── Random Forest ──
    print("\n[3/6] Random Forest ...")
    tracemalloc.start()
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=ML_PARAMS["rf_n_estimators"],
        max_depth=ML_PARAMS["rf_max_depth"],
        random_state=ML_PARAMS["random_state"],
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    train_time_rf = time.time() - t0

    t0 = time.time()
    y_pred_rf = rf.predict(X_test)
    infer_time_rf = time.time() - t0
    mem_rf = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    results["Random Forest"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred_rf, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred_rf, average="weighted", zero_division=0),
        "train_time": train_time_rf,
        "infer_time": infer_time_rf,
        "memory_mb": mem_rf,
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist(),
    }
    print(f"      Accuracy: {results['Random Forest']['accuracy']:.4f}")
    print(f"      F1 (weighted): {results['Random Forest']['f1']:.4f}")
    print(f"      Tiempo train: {train_time_rf:.2f}s | Inferencia: {infer_time_rf:.4f}s")

    # Feature importance
    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"      Top 5 features: {list(importance.head(5).index)}")

    # ── SVM ──
    print("\n[4/6] SVM (RBF kernel) ...")
    # Limitar muestras para SVM si dataset grande
    max_svm = min(20000, len(X_train))
    tracemalloc.start()
    t0 = time.time()
    svm = SVC(
        kernel=ML_PARAMS["svm_kernel"],
        C=ML_PARAMS["svm_C"],
        random_state=ML_PARAMS["random_state"],
    )
    svm.fit(X_train[:max_svm], y_train[:max_svm])
    train_time_svm = time.time() - t0

    t0 = time.time()
    y_pred_svm = svm.predict(X_test)
    infer_time_svm = time.time() - t0
    mem_svm = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    results["SVM"] = {
        "accuracy": accuracy_score(y_test, y_pred_svm),
        "precision": precision_score(y_test, y_pred_svm, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred_svm, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred_svm, average="weighted", zero_division=0),
        "train_time": train_time_svm,
        "infer_time": infer_time_svm,
        "memory_mb": mem_svm,
        "confusion_matrix": confusion_matrix(y_test, y_pred_svm).tolist(),
    }
    print(f"      Accuracy: {results['SVM']['accuracy']:.4f}")
    print(f"      F1 (weighted): {results['SVM']['f1']:.4f}")
    print(f"      Tiempo train: {train_time_svm:.2f}s")

    # ── XGBoost ──
    print("\n[5/6] XGBoost ...")
    try:
        from xgboost import XGBClassifier
        tracemalloc.start()
        t0 = time.time()
        xgb = XGBClassifier(
            n_estimators=ML_PARAMS["xgb_n_estimators"],
            max_depth=ML_PARAMS["xgb_max_depth"],
            learning_rate=ML_PARAMS["xgb_learning_rate"],
            random_state=ML_PARAMS["random_state"],
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        train_time_xgb = time.time() - t0

        t0 = time.time()
        y_pred_xgb = xgb.predict(X_test)
        infer_time_xgb = time.time() - t0
        mem_xgb = tracemalloc.get_traced_memory()[1] / 1024 / 1024
        tracemalloc.stop()

        results["XGBoost"] = {
            "accuracy": accuracy_score(y_test, y_pred_xgb),
            "precision": precision_score(y_test, y_pred_xgb, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred_xgb, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred_xgb, average="weighted", zero_division=0),
            "train_time": train_time_xgb,
            "infer_time": infer_time_xgb,
            "memory_mb": mem_xgb,
            "confusion_matrix": confusion_matrix(y_test, y_pred_xgb).tolist(),
        }
        print(f"      Accuracy: {results['XGBoost']['accuracy']:.4f}")
        print(f"      F1 (weighted): {results['XGBoost']['f1']:.4f}")
        print(f"      Tiempo train: {train_time_xgb:.2f}s")
    except ImportError:
        print("      XGBoost no instalado. Saltando.")

    # Guardar feature importance
    print("\n[6/6] Guardando resultados ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.head(10).plot(kind="barh", ax=ax, color="#3498db")
    ax.set_title("Top 10 Features (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importancia")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(FIGURES_DIR / "ml_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "task": "clasificacion_objetos",
        "classes": le.classes_.tolist(),
        "results": results,
        "feature_importance": importance.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════
# TAREA 2: DETECCION DE ANOMALIAS
# ═══════════════════════════════════════════════════════════════

def task2_anomaly_detection(spark=None):
    """
    Detectar objetos anomalos (candidatos a interestelares).
    PCA 3D + K-Means + Isolation Forest.
    """
    print("\n" + "=" * 60)
    print("ML TAREA 2: Deteccion de Anomalias (Interestelares)")
    print("=" * 60)

    # Cargar datos
    print("\n[1/5] Cargando datos ...")
    df = pd.read_parquet(str(PROCESSED_PARQUET))

    feature_cols = FEATURES_ORBITAL + FEATURES_ENGINEERED
    feature_cols = [f for f in feature_cols if f in df.columns]

    X = df[feature_cols].copy()

    # Imputar y escalar
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # ── PCA 3D ──
    print("\n[2/5] PCA (3 componentes) ...")
    pca = PCA(n_components=ML_PARAMS["pca_components"], random_state=ML_PARAMS["random_state"])
    coords = pca.fit_transform(X_scaled)
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]
    df["pc3"] = coords[:, 2]
    var_exp = pca.explained_variance_ratio_
    print(f"      Varianza explicada: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}, PC3={var_exp[2]:.1%}")
    print(f"      Total: {sum(var_exp):.1%}")

    # ── K-Means ──
    print(f"\n[3/5] K-Means (k={ML_PARAMS['n_clusters']}) ...")
    km = KMeans(
        n_clusters=ML_PARAMS["n_clusters"],
        random_state=ML_PARAMS["random_state"],
        n_init=10,
    )
    df["cluster"] = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, df["cluster"], sample_size=min(10000, len(df)))
    print(f"      Silhouette score: {sil:.4f}")

    # ── Isolation Forest ──
    print(f"\n[4/5] Isolation Forest (contamination={ML_PARAMS['isolation_contamination']}) ...")
    tracemalloc.start()
    t0 = time.time()
    iso_forest = IsolationForest(
        contamination=ML_PARAMS["isolation_contamination"],
        random_state=ML_PARAMS["random_state"],
        n_jobs=-1,
    )
    df["anomaly_score"] = iso_forest.fit_predict(X_scaled)
    df["anomaly_score_raw"] = iso_forest.decision_function(X_scaled)
    train_time = time.time() - t0
    mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    n_anomalies = (df["anomaly_score"] == -1).sum()
    print(f"      Anomalias detectadas: {n_anomalies:,} ({n_anomalies/len(df):.2%})")
    print(f"      Tiempo: {train_time:.2f}s | Memoria: {mem:.1f} MB")

    # Verificar si interestelares son detectados como anomalias
    print("\n[5/5] Verificando deteccion de interestelares ...")
    iso_mask = df["object_type"] == "ISO"
    if iso_mask.any():
        iso_objects = df[iso_mask][["full_name", "anomaly_score", "anomaly_score_raw", "cluster", "pc1", "pc2", "pc3"]]
        print("      Objetos interestelares:")
        for _, row in iso_objects.iterrows():
            detected = "DETECTADO" if row["anomaly_score"] == -1 else "NO detectado"
            print(f"        {row['full_name']}: {detected} (score={row['anomaly_score_raw']:.4f}, cluster={row['cluster']})")
    else:
        print("      No hay objetos interestelares en el dataset.")

    # Guardar grafico PCA 2D
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    for c in range(ML_PARAMS["n_clusters"]):
        mask = df["cluster"] == c
        ax.scatter(df.loc[mask, "pc1"], df.loc[mask, "pc2"],
                   alpha=0.3, s=10, label=f"Cluster {c}")
    # Marcar anomalias
    anom_mask = df["anomaly_score"] == -1
    ax.scatter(df.loc[anom_mask, "pc1"], df.loc[anom_mask, "pc2"],
               c="red", marker="x", s=50, linewidths=1.5, label="Anomalia", zorder=5)
    # Marcar interestelares
    if iso_mask.any():
        ax.scatter(df.loc[iso_mask, "pc1"], df.loc[iso_mask, "pc2"],
                   c="#e91e63", marker="*", s=300, edgecolors="black",
                   linewidths=1.5, label="Interestelar", zorder=10)
        for _, row in df[iso_mask].iterrows():
            ax.annotate(row.get("full_name", "ISO"),
                        (row["pc1"], row["pc2"]),
                        fontsize=9, fontweight="bold", color="#e91e63",
                        xytext=(10, 10), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
    ax.set_title("i3 Atlas: PCA + K-Means + Isolation Forest\nAnomaly Detection", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(str(FIGURES_DIR / "ml_anomaly_detection.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "task": "deteccion_anomalias",
        "method": "Isolation Forest",
        "n_anomalies": int(n_anomalies),
        "silhouette": sil,
        "pca_variance": var_exp.tolist(),
        "train_time": train_time,
        "memory_mb": mem,
        "interstellar_detected": bool(df[iso_mask]["anomaly_score"].eq(-1).all()) if iso_mask.any() else None,
        "pca_coords": json.loads(df[["full_name", "object_type", "pc1", "pc2", "pc3", "cluster", "anomaly_score", "anomaly_score_raw"]].to_json(orient="records")),
    }


# ═══════════════════════════════════════════════════════════════
# TAREA 3: CLASIFICACION ESPECTRAL SDSS
# ═══════════════════════════════════════════════════════════════

def task3_spectral_classification():
    """
    Clasificacion espectral de galaxias SDSS con Random Forest.
    Features: colores u-g, g-r, r-i, i-z.
    """
    print("\n" + "=" * 60)
    print("ML TAREA 3: Clasificacion Espectral (SDSS Galaxy Colors)")
    print("=" * 60)

    # Cargar SDSS
    print("\n[1/4] Cargando SDSS Galaxy Colors ...")
    sdss_file = DATA_CACHE_DIR / "sdss_galaxy_colors.csv"
    if not sdss_file.exists():
        print("      Archivo SDSS no encontrado. Ejecuta data_acquisition.py primero.")
        return None
    df = pd.read_csv(str(sdss_file))
    print(f"      Galaxias: {len(df):,}")

    feature_cols = ["u_g", "g_r", "r_i", "i_z"]
    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df["spectral_class"])
    class_names = sorted(df["spectral_class"].unique())

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=ML_PARAMS["test_size"],
        random_state=ML_PARAMS["random_state"],
        stratify=y,
    )

    # Random Forest
    print("\n[2/4] Entrenando Random Forest ...")
    tracemalloc.start()
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=ML_PARAMS["rf_n_estimators"],
        max_depth=15,
        random_state=ML_PARAMS["random_state"],
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = rf.predict(X_test)
    infer_time = time.time() - t0
    mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"      Accuracy: {acc:.4f}")
    print(f"      F1 (weighted): {f1:.4f}")
    print(f"      Tiempo train: {train_time:.2f}s | Inferencia: {infer_time:.4f}s")

    # Guardar grafico
    print("\n[3/4] Generando visualizacion ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color-color diagram
    ax = axes[0]
    scatter = ax.scatter(df["g_r"], df["r_i"], c=y, cmap="Spectral",
                         alpha=0.3, s=5)
    ax.set_xlabel("g - r")
    ax.set_ylabel("r - i")
    ax.set_title("Diagrama Color-Color SDSS", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Feature importance
    ax = axes[1]
    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    importance.plot(kind="bar", ax=ax, color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"])
    ax.set_title("Feature Importance (SDSS Colors)", fontweight="bold")
    ax.set_ylabel("Importancia")
    ax.grid(True, alpha=0.3)

    fig.suptitle("i3 Atlas: Clasificacion Espectral - ML Tradicional", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "ml_spectral_sdss.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[4/4] Resultados guardados.")

    return {
        "task": "clasificacion_espectral",
        "method": "Random Forest",
        "accuracy": acc,
        "f1": f1,
        "train_time": train_time,
        "infer_time": infer_time,
        "memory_mb": mem,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "class_names": class_names,
        "feature_importance": importance.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════
# ORQUESTADOR ML
# ═══════════════════════════════════════════════════════════════

def run_ml_traditional(spark=None):
    """Ejecuta el pipeline completo de ML tradicional."""
    print("\n" + "#" * 60)
    print("#  ML TRADICIONAL - i3 Atlas (scikit-learn + XGBoost)")
    print("#" * 60)

    t0 = time.time()

    r1 = task1_classification(spark)
    r2 = task2_anomaly_detection(spark)
    r3 = task3_spectral_classification()

    elapsed = time.time() - t0

    # Guardar resultados como JSON
    results = {"task1": r1, "task2": r2, "task3": r3, "total_time": elapsed}
    results_file = OUTPUT_DIR / "ml_results.json"

    # Serializar (convertir numpy a python)
    def _convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return json.loads(obj.to_json(orient="records"))
        if isinstance(obj, pd.Series):
            return json.loads(obj.to_json())
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(str(results_file), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_convert)

    print(f"\n{'#' * 60}")
    print(f"#  ML TRADICIONAL completado en {elapsed:.1f}s")
    print(f"#  Resultados: {results_file}")
    print(f"#  Figuras: {FIGURES_DIR}")
    print(f"{'#' * 60}")

    return results


if __name__ == "__main__":
    run_ml_traditional()
