"""
MODULO DEEP LEARNING - i3 Atlas
=================================
Pipeline DL con TensorFlow/Keras (GPU-accelerated, RTX 4060):
  Tarea 1: DNN Clasificador de objetos astronomicos
  Tarea 2: Autoencoder para deteccion de anomalias
  Tarea 3: 1D-CNN para clasificacion espectral SDSS

GPU Config: RTX 4060 (8GB VRAM), memory_growth=True
Fallback: Funcional en CPU si no hay GPU disponible.

Autor: Juan Marcelo Gutierrez Miranda | @TodoEconometria

REFERENCIA ACADEMICA:
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
- Chollet, F. (2017). Deep Learning with Python. Manning Publications.
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
  ICLR 2014. arXiv:1312.6114.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

from config import (
    PROCESSED_PARQUET, DATA_CACHE_DIR, FIGURES_DIR, OUTPUT_DIR,
    FEATURES_ORBITAL, FEATURES_PHYSICAL, FEATURES_ENGINEERED,
    DL_PARAMS, GPU_CONFIG, SPARK_DATA_URI,
)


def _setup_gpu():
    """Configura GPU con memory_growth para RTX 4060."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, GPU_CONFIG["memory_growth"])
            print(f"  GPU detectada: {gpus[0].name}")
            print(f"  Memory growth: {GPU_CONFIG['memory_growth']}")
            return True
        except RuntimeError as e:
            print(f"  Error configurando GPU: {e}")
            return False
    else:
        print("  No hay GPU disponible. Ejecutando en CPU.")
        return False


def _get_device_info():
    """Retorna info del dispositivo de computo."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"GPU: {gpus[0].name}"
    return "CPU"


# ═══════════════════════════════════════════════════════════════
# TAREA 1: DNN CLASIFICADOR
# ═══════════════════════════════════════════════════════════════

def task1_dnn_classification(spark=None):
    """
    Red neuronal densa para clasificacion de objetos astronomicos.
    Misma tarea que Random Forest para comparacion directa.
    """
    import tensorflow as tf
    from tensorflow import keras

    print("=" * 60)
    print("DL TAREA 1: DNN Clasificador de Objetos Astronomicos")
    print(f"  Device: {_get_device_info()}")
    print("=" * 60)

    # Cargar datos
    print("\n[1/5] Cargando datos procesados ...")
    df = pd.read_parquet(str(PROCESSED_PARQUET))

    feature_cols = FEATURES_ORBITAL + FEATURES_PHYSICAL + FEATURES_ENGINEERED
    feature_cols = [f for f in feature_cols if f in df.columns]

    # Filtrar clases
    class_counts = df["object_type"].value_counts()
    valid_classes = class_counts[class_counts >= 50].index.tolist()
    df_ml = df[df["object_type"].isin(valid_classes)].copy()

    X = df_ml[feature_cols].copy()
    y = df_ml["object_type"].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    n_features = X_scaled.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"      Features: {n_features} | Clases: {n_classes}")

    # Construir modelo DNN
    print("\n[2/5] Construyendo DNN ...")
    model = keras.Sequential(name="i3Atlas_DNN_Classifier")
    model.add(keras.layers.Input(shape=(n_features,)))

    for i, units in enumerate(DL_PARAMS["dnn_layers"]):
        model.add(keras.layers.Dense(units, activation="relu", name=f"dense_{i}"))
        model.add(keras.layers.BatchNormalization(name=f"bn_{i}"))
        model.add(keras.layers.Dropout(DL_PARAMS["dropout_rate"], name=f"dropout_{i}"))

    model.add(keras.layers.Dense(n_classes, activation="softmax", name="output"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DL_PARAMS["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=lambda x: print(f"      {x}"))
    total_params = model.count_params()
    print(f"\n      Total parametros: {total_params:,}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=DL_PARAMS["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=DL_PARAMS["reduce_lr_factor"],
            patience=DL_PARAMS["reduce_lr_patience"],
            verbose=1,
        ),
    ]

    # Entrenar
    print("\n[3/5] Entrenando DNN ...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=DL_PARAMS["epochs"],
        batch_size=DL_PARAMS["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - t0

    # Inferencia
    print("\n[4/5] Evaluando ...")
    t0 = time.time()
    y_pred_prob = model.predict(X_test, batch_size=DL_PARAMS["batch_size"])
    infer_time = time.time() - t0
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"      Accuracy: {acc:.4f}")
    print(f"      F1 (weighted): {f1:.4f}")
    print(f"      Tiempo train: {train_time:.2f}s | Inferencia: {infer_time:.4f}s")
    print(f"      Epochs efectivos: {len(history.history['loss'])}")

    # Guardar loss curves
    print("\n[5/5] Generando visualizaciones ...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax = axes[0]
    ax.plot(history.history["loss"], label="Train Loss", color="#FF5722")
    ax.plot(history.history["val_loss"], label="Val Loss", color="#2196F3", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DNN Training Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(history.history["accuracy"], label="Train Accuracy", color="#FF5722")
    ax.plot(history.history["val_accuracy"], label="Val Accuracy", color="#2196F3", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("DNN Training Accuracy", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("i3 Atlas: DNN Classifier - Training Curves", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "dl_dnn_training.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "task": "clasificacion_objetos",
        "method": "DNN (Keras)",
        "device": _get_device_info(),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "train_time": train_time,
        "infer_time": infer_time,
        "total_params": total_params,
        "epochs_trained": len(history.history["loss"]),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "history": {
            "loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
            "accuracy": [float(x) for x in history.history["accuracy"]],
            "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        },
    }


# ═══════════════════════════════════════════════════════════════
# TAREA 2: AUTOENCODER PARA ANOMALIAS
# ═══════════════════════════════════════════════════════════════

def task2_autoencoder_anomaly(spark=None):
    """
    Autoencoder para deteccion de anomalias.
    Entrena solo con objetos 'normales', detecta interestelares por reconstruction error.
    """
    import tensorflow as tf
    from tensorflow import keras

    print("\n" + "=" * 60)
    print("DL TAREA 2: Autoencoder - Deteccion de Anomalias")
    print(f"  Device: {_get_device_info()}")
    print("=" * 60)

    # Cargar datos
    print("\n[1/5] Cargando datos ...")
    df = pd.read_parquet(str(PROCESSED_PARQUET))

    feature_cols = FEATURES_ORBITAL + FEATURES_ENGINEERED
    feature_cols = [f for f in feature_cols if f in df.columns]
    n_features = len(feature_cols)

    X_all = df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_all), columns=feature_cols, index=X_all.index)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Separar normales vs todo
    normal_mask = df["object_type"].isin(["MBA", "NEO"])
    X_normal = X_scaled[normal_mask]
    print(f"      Total objetos: {len(df):,}")
    print(f"      Objetos normales (train): {len(X_normal):,}")
    print(f"      Features: {n_features}")

    X_train, X_val = train_test_split(X_normal, test_size=0.15, random_state=42)

    # Construir Autoencoder
    print("\n[2/5] Construyendo Autoencoder ...")
    encoder_layers = DL_PARAMS["ae_encoder_layers"]
    latent_dim = DL_PARAMS["ae_latent_dim"]

    # Encoder
    encoder_input = keras.layers.Input(shape=(n_features,), name="encoder_input")
    x = encoder_input
    for i, units in enumerate(encoder_layers):
        x = keras.layers.Dense(units, activation="relu", name=f"enc_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"enc_bn_{i}")(x)
    latent = keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder (simetrico)
    x = latent
    for i, units in enumerate(reversed(encoder_layers)):
        x = keras.layers.Dense(units, activation="relu", name=f"dec_{i}")(x)
        x = keras.layers.BatchNormalization(name=f"dec_bn_{i}")(x)
    decoder_output = keras.layers.Dense(n_features, activation="linear", name="decoder_output")(x)

    autoencoder = keras.Model(encoder_input, decoder_output, name="i3Atlas_Autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DL_PARAMS["learning_rate"]),
        loss="mse",
    )

    total_params = autoencoder.count_params()
    print(f"      Arquitectura: {n_features} -> {encoder_layers} -> {latent_dim} -> {list(reversed(encoder_layers))} -> {n_features}")
    print(f"      Total parametros: {total_params:,}")

    # Entrenar (solo con normales)
    print("\n[3/5] Entrenando Autoencoder (solo objetos normales) ...")
    t0 = time.time()
    history = autoencoder.fit(
        X_train, X_train,  # input = target (reconstruction)
        validation_data=(X_val, X_val),
        epochs=DL_PARAMS["epochs"],
        batch_size=DL_PARAMS["batch_size"],
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=DL_PARAMS["early_stopping_patience"],
                restore_best_weights=True,
            ),
        ],
        verbose=1,
    )
    train_time = time.time() - t0

    # Calcular reconstruction error en TODOS los objetos
    print("\n[4/5] Calculando reconstruction error ...")
    t0 = time.time()
    X_reconstructed = autoencoder.predict(X_scaled, batch_size=DL_PARAMS["batch_size"])
    infer_time = time.time() - t0

    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    df["ae_recon_error"] = reconstruction_error

    # Threshold: percentil 99 de los errores normales
    normal_errors = reconstruction_error[normal_mask]
    threshold = np.percentile(normal_errors, 99)
    df["ae_anomaly"] = (reconstruction_error > threshold).astype(int)

    n_anomalies = df["ae_anomaly"].sum()
    print(f"      Threshold (p99 normales): {threshold:.6f}")
    print(f"      Anomalias detectadas: {n_anomalies:,}")
    print(f"      Tiempo train: {train_time:.2f}s | Inferencia: {infer_time:.4f}s")

    # Verificar interestelares
    print("\n[5/5] Verificando deteccion de interestelares ...")
    iso_mask = df["object_type"] == "ISO"
    if iso_mask.any():
        for _, row in df[iso_mask].iterrows():
            detected = "DETECTADO" if row["ae_anomaly"] == 1 else "NO detectado"
            print(f"      {row.get('full_name', 'ISO')}: {detected} "
                  f"(error={row['ae_recon_error']:.6f}, threshold={threshold:.6f})")

    # Visualizacion
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax = axes[0]
    ax.plot(history.history["loss"], label="Train Loss", color="#FF5722")
    ax.plot(history.history["val_loss"], label="Val Loss", color="#2196F3", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Reconstruction error distribution
    ax = axes[1]
    ax.hist(normal_errors, bins=100, alpha=0.7, label="Normales", color="#2196F3", density=True)
    non_normal_errors = reconstruction_error[~normal_mask]
    if len(non_normal_errors) > 0:
        ax.hist(non_normal_errors, bins=50, alpha=0.7, label="Otros", color="#FF5722", density=True)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold (p99)={threshold:.4f}")
    if iso_mask.any():
        iso_errors = reconstruction_error[iso_mask]
        for err in iso_errors:
            ax.axvline(err, color="#e91e63", linestyle="-", linewidth=2, alpha=0.8)
        ax.plot([], [], color="#e91e63", linewidth=2, label="Interestelares")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribucion de Error de Reconstruccion", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("i3 Atlas: Autoencoder Anomaly Detection", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "dl_autoencoder_anomaly.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "task": "deteccion_anomalias",
        "method": "Autoencoder",
        "device": _get_device_info(),
        "n_anomalies": int(n_anomalies),
        "threshold": float(threshold),
        "train_time": train_time,
        "infer_time": infer_time,
        "total_params": total_params,
        "epochs_trained": len(history.history["loss"]),
        "interstellar_detected": bool(df[iso_mask]["ae_anomaly"].all()) if iso_mask.any() else None,
        "interstellar_errors": df[iso_mask][["full_name", "ae_recon_error", "ae_anomaly"]].to_dict("records") if iso_mask.any() else [],
        "history": {
            "loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
        },
    }


# ═══════════════════════════════════════════════════════════════
# TAREA 3: 1D-CNN CLASIFICACION ESPECTRAL
# ═══════════════════════════════════════════════════════════════

def task3_cnn_spectral():
    """
    1D-CNN para clasificacion espectral de galaxias SDSS.
    Comparacion directa con Random Forest.
    """
    import tensorflow as tf
    from tensorflow import keras

    print("\n" + "=" * 60)
    print("DL TAREA 3: 1D-CNN - Clasificacion Espectral SDSS")
    print(f"  Device: {_get_device_info()}")
    print("=" * 60)

    # Cargar SDSS
    print("\n[1/4] Cargando SDSS Galaxy Colors ...")
    sdss_file = DATA_CACHE_DIR / "sdss_galaxy_colors.csv"
    if not sdss_file.exists():
        print("      Archivo SDSS no encontrado. Ejecuta data_acquisition.py primero.")
        return None
    df = pd.read_csv(str(sdss_file))

    feature_cols = ["u_g", "g_r", "r_i", "i_z"]
    X = df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["spectral_class"])
    n_classes = len(le.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape para Conv1D: (samples, steps, features)
    # Tratamos cada color como un "paso" temporal con 1 feature
    X_cnn = X_scaled.reshape(-1, len(feature_cols), 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_cnn, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"      Input shape: {X_cnn.shape[1:]}")

    # Construir 1D-CNN
    print("\n[2/4] Construyendo 1D-CNN ...")
    model = keras.Sequential([
        keras.layers.Input(shape=(len(feature_cols), 1)),
        keras.layers.Conv1D(DL_PARAMS["cnn_filters"][0], kernel_size=2,
                            activation="relu", padding="same", name="conv1"),
        keras.layers.BatchNormalization(name="bn1"),
        keras.layers.Conv1D(DL_PARAMS["cnn_filters"][1], kernel_size=2,
                            activation="relu", padding="same", name="conv2"),
        keras.layers.BatchNormalization(name="bn2"),
        keras.layers.GlobalAveragePooling1D(name="gap"),
        keras.layers.Dense(64, activation="relu", name="dense1"),
        keras.layers.Dropout(0.3, name="dropout"),
        keras.layers.Dense(n_classes, activation="softmax", name="output"),
    ], name="i3Atlas_1DCNN_Spectral")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DL_PARAMS["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    total_params = model.count_params()
    print(f"      Parametros: {total_params:,}")

    # Entrenar
    print("\n[3/4] Entrenando 1D-CNN ...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=DL_PARAMS["epochs"],
        batch_size=DL_PARAMS["batch_size"],
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=DL_PARAMS["early_stopping_patience"],
                restore_best_weights=True,
            ),
        ],
        verbose=1,
    )
    train_time = time.time() - t0

    # Evaluar
    print("\n[4/4] Evaluando ...")
    t0 = time.time()
    y_pred_prob = model.predict(X_test, batch_size=DL_PARAMS["batch_size"])
    infer_time = time.time() - t0
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(f"      Accuracy: {acc:.4f}")
    print(f"      F1 (weighted): {f1:.4f}")
    print(f"      Tiempo train: {train_time:.2f}s | Inferencia: {infer_time:.4f}s")

    # Guardar
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["accuracy"], label="Train", color="#FF5722")
    ax.plot(history.history["val_accuracy"], label="Val", color="#2196F3", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("1D-CNN Spectral Classification - Training", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(str(FIGURES_DIR / "dl_cnn_spectral.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "task": "clasificacion_espectral",
        "method": "1D-CNN (Keras)",
        "device": _get_device_info(),
        "accuracy": acc,
        "f1": f1,
        "train_time": train_time,
        "infer_time": infer_time,
        "total_params": total_params,
        "epochs_trained": len(history.history["loss"]),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "history": {
            "loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
            "accuracy": [float(x) for x in history.history["accuracy"]],
            "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        },
    }


# ═══════════════════════════════════════════════════════════════
# ORQUESTADOR DL
# ═══════════════════════════════════════════════════════════════

def run_deep_learning(spark=None):
    """Ejecuta el pipeline completo de Deep Learning."""
    print("\n" + "#" * 60)
    print("#  DEEP LEARNING - i3 Atlas (TensorFlow/Keras + GPU)")
    print("#" * 60)

    # Verificar disponibilidad de TensorFlow
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        print(f"  TensorFlow {tf_version} disponible.")
    except ImportError:
        import sys
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"\n  [WARN] TensorFlow no disponible (Python {py_ver}).")
        print(f"  TensorFlow aun no soporta Python {py_ver}.")
        print(f"  Para ejecutar DL, usar el entorno WSL2 (~/tf-gpu-env/) con Python 3.12.")
        print(f"  Generando resultados placeholder para benchmark y dashboard ...\n")

        # Producir resultados placeholder
        results = {
            "task1": {
                "modelo": "DNN (TensorFlow/Keras)",
                "status": "no_ejecutado",
                "nota": f"TensorFlow no soporta Python {py_ver}. Usar entorno WSL2.",
                "accuracy": None, "f1": None, "train_time": None,
            },
            "task2": {
                "modelo": "Autoencoder",
                "status": "no_ejecutado",
                "nota": f"TensorFlow no soporta Python {py_ver}. Usar entorno WSL2.",
            },
            "task3": {
                "modelo": "1D-CNN",
                "status": "no_ejecutado",
                "nota": f"TensorFlow no soporta Python {py_ver}. Usar entorno WSL2.",
                "accuracy": None, "f1": None, "train_time": None,
            },
            "total_time": 0.0,
            "gpu": False,
            "tensorflow_available": False,
            "python_version": py_ver,
        }
        results_file = OUTPUT_DIR / "dl_results.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(str(results_file), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"  Resultados placeholder: {results_file}")
        print(f"{'#' * 60}")
        return results

    # Setup GPU
    has_gpu = _setup_gpu()

    t0 = time.time()

    r1 = task1_dnn_classification(spark)
    r2 = task2_autoencoder_anomaly(spark)
    r3 = task3_cnn_spectral()

    elapsed = time.time() - t0

    # Guardar resultados
    results = {
        "task1": r1, "task2": r2, "task3": r3,
        "total_time": elapsed, "gpu": has_gpu,
        "tensorflow_available": True,
        "tensorflow_version": tf_version,
    }
    results_file = OUTPUT_DIR / "dl_results.json"

    def _convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(str(results_file), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_convert)

    print(f"\n{'#' * 60}")
    print(f"#  DEEP LEARNING completado en {elapsed:.1f}s")
    print(f"#  GPU: {'SI' if has_gpu else 'NO (CPU)'}")
    print(f"#  Resultados: {results_file}")
    print(f"{'#' * 60}")

    return results


if __name__ == "__main__":
    _setup_gpu()
    run_deep_learning()
