"""
CONFIGURACION CENTRAL - i3 Atlas
=================================
Analisis Comparativo: Deep Learning vs Machine Learning
en el Catalogo Astronomico i3 Atlas con Apache Spark.

SETUP
=====
1. APACHE SPARK (Docker):
   docker compose -f docker-compose-spark.yml up -d
   Web UI: localhost:8080 | Master: localhost:7077

2. TENSORFLOW + GPU (optional):
   pip install tensorflow[and-cuda]  # or CPU: pip install tensorflow
   For WSL2 GPU: source ~/tf-gpu-env/bin/activate

3. OUTPUT:
   All results go to ./output/ (auto-created)

Autor: Juan Marcelo Gutierrez Miranda | @TodoEconometria
"""
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# BIG DATA LAB - STORAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent

# Output directory (all results, cache, figures)
OUTPUT_DIR = BASE_DIR / "output"

DATA_CACHE_DIR = OUTPUT_DIR / "datos"
FIGURES_DIR = OUTPUT_DIR / "figuras"
PROCESSED_PARQUET = OUTPUT_DIR / "i3atlas_procesado.parquet"
DASHBOARD_OUTPUT = OUTPUT_DIR / "dashboard_i3atlas_ml_vs_dl.html"

# ═══════════════════════════════════════════════════════════════
# APACHE SPARK - CLUSTER CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SPARK_MASTER_URL = "spark://localhost:7077"
SPARK_APP_NAME = "i3 Atlas - ML vs Deep Learning"
SPARK_CONFIG = {
    "spark.executor.memory": "8g",
    "spark.executor.cores": "4",
    "spark.cores.max": "8",
    "spark.driver.memory": "4g",
    "spark.driver.host": "host.docker.internal",
    "spark.driver.bindAddress": "0.0.0.0",
    "spark.sql.shuffle.partitions": "8",
    "spark.sql.execution.arrow.pyspark.enabled": "false",
}

# Shared volume: Host <-> Docker Spark
# Default: ./spark-data (create if using Spark cluster)
# Docker mapping: ./spark-data <-> /opt/spark-data
SPARK_DATA_PATH = BASE_DIR / "spark-data"
SPARK_DATA_URI = "file:///opt/spark-data"

# ═══════════════════════════════════════════════════════════════
# GPU / DEEP LEARNING (TensorFlow)
# ═══════════════════════════════════════════════════════════════
# Deep Learning works on CPU (slower) or GPU (recommended).
# For GPU: pip install tensorflow[and-cuda]
# The pipeline auto-detects GPU availability.

GPU_CONFIG = {
    "device": "RTX 4060",
    "vram_gb": 8,
    "memory_growth": True,       # tf.config.experimental.set_memory_growth
    "mixed_precision": False,    # fp16 (activar si se necesita mas VRAM)
}

# ═══════════════════════════════════════════════════════════════
# JPL SMALL-BODY DATABASE API
# ═══════════════════════════════════════════════════════════════
JPL_SBDB_API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

# Campos a solicitar de la API
JPL_FIELDS = [
    "spkid", "full_name", "kind", "class", "neo", "pha",
    "e", "a", "i", "om", "w", "q", "ad", "per_y",
    "epoch", "H", "diameter", "albedo", "moid",
]

# Objetos interestelares de interes especial (los 3 confirmados)
INTERSTELLAR_OBJECTS = {
    "1I/2017 U1": "'Oumuamua",      # Descubierto Oct 2017 - forma alargada unica
    "2I/2019 Q4": "Borisov",         # Descubierto Ago 2019 - cometa interestelar
    "3I/2025 K3": "ATLAS",           # Descubierto Jul 2025 - el mas reciente
}

# Datos orbitales de los 3 interestelares (para animaciones cinematograficas)
INTERSTELLAR_ORBITAL_DATA = {
    "1I/'Oumuamua": {
        "e": 1.20113,           # Excentricidad hiperbolica
        "a": -1.2799,           # Semieje mayor (AU, negativo = hiperbolica)
        "q": 0.25534,           # Perihelio (AU) - muy cerca del Sol
        "i": 122.74,            # Inclinacion (grados)
        "v_inf": 26.33,         # Velocidad en infinito (km/s)
        "perihelion_date": "2017-09-09",
        "discovery_date": "2017-10-19",
        "shape": "elongated",   # Forma alargada 10:1
        "size_km": 0.2,         # ~200m de largo
        "origin": "Vega direction",
    },
    "2I/Borisov": {
        "e": 3.3571,            # Muy hiperbolica
        "a": -0.8514,           # AU
        "q": 2.0066,            # Perihelio mas lejano que 'Oumuamua
        "i": 44.05,             # Inclinacion
        "v_inf": 32.2,          # km/s
        "perihelion_date": "2019-12-08",
        "discovery_date": "2019-08-30",
        "shape": "comet",       # Cometa clasico con coma
        "size_km": 1.0,         # ~1km de nucleo
        "origin": "Cassiopeia direction",
    },
    "3I/ATLAS": {
        "e": 1.9,               # Hiperbolica
        "a": -1.4,              # AU (estimado)
        "q": 1.35,              # Perihelio dentro de orbita de Marte
        "i": 50.0,              # Inclinacion (estimado)
        "v_inf": 60.0,          # km/s - 221,000 km/h = 61.4 km/s
        "perihelion_date": "2025-10-30",
        "discovery_date": "2025-07-01",
        "shape": "comet",       # Cometa activo
        "size_km": 2.0,         # 440m - 5.6km, tomamos ~2km
        "origin": "Sagittarius direction (galactic center)",
        "age_billion_years": "3-11",  # Muy antiguo
    },
}

# ═══════════════════════════════════════════════════════════════
# CLASIFICACION DE OBJETOS ASTRONOMICOS
# ═══════════════════════════════════════════════════════════════
OBJECT_CLASSES = {
    "MBA": "Main Belt Asteroid",
    "NEO": "Near-Earth Object",
    "COM": "Comet",
    "TNO": "Trans-Neptunian Object",
    "CEN": "Centaur",
    "ISO": "Interstellar Object",
}

# Mapeo de clases orbitales JPL a nuestras categorias simplificadas
CLASS_MAPPING = {
    # Main Belt
    "MBA": "MBA", "MCA": "MBA", "OMB": "MBA", "IMB": "MBA",
    # Near-Earth
    "ATE": "NEO", "APO": "NEO", "AMO": "NEO", "IEO": "NEO",
    "NEO": "NEO", "NEC": "NEO",
    # Comets
    "COM": "COM", "CTc": "COM", "ETc": "COM", "HTC": "COM",
    "HYP": "COM", "JFc": "COM", "JFC": "COM", "PAR": "COM",
    # Trans-Neptunian
    "TNO": "TNO", "KBO": "TNO", "SDO": "TNO", "CEN": "CEN",
    # Interstellar
    "HYP": "ISO",  # hiperbolicos con e>1.0 se reclasifican
}

# ═══════════════════════════════════════════════════════════════
# FEATURES PARA ML / DL
# ═══════════════════════════════════════════════════════════════
FEATURES_ORBITAL = ["e", "a", "i", "om", "w", "q", "ad", "per_y"]
FEATURES_PHYSICAL = ["diameter", "albedo", "H", "moid"]
FEATURES_ALL = FEATURES_ORBITAL + FEATURES_PHYSICAL
FEATURES_ENGINEERED = ["tisserand_j", "v_inf", "energy_param", "q_over_a"]

# ═══════════════════════════════════════════════════════════════
# PARAMETROS ML TRADICIONAL
# ═══════════════════════════════════════════════════════════════
ML_PARAMS = {
    "test_size": 0.2,
    "random_state": 42,
    "n_clusters": 6,
    "pca_components": 3,
    "isolation_contamination": 0.01,
    "rf_n_estimators": 200,
    "rf_max_depth": 20,
    "svm_kernel": "rbf",
    "svm_C": 10.0,
    "xgb_n_estimators": 200,
    "xgb_max_depth": 8,
    "xgb_learning_rate": 0.1,
}

# ═══════════════════════════════════════════════════════════════
# PARAMETROS DEEP LEARNING
# ═══════════════════════════════════════════════════════════════
DL_PARAMS = {
    "epochs": 50,
    "batch_size": 2048,        # optimizado para RTX 4060
    "learning_rate": 0.001,
    "early_stopping_patience": 8,
    "reduce_lr_patience": 4,
    "reduce_lr_factor": 0.5,
    "dropout_rate": 0.3,
    # DNN Classifier
    "dnn_layers": [256, 128, 64],
    # Autoencoder
    "ae_encoder_layers": [128, 64, 32],
    "ae_latent_dim": 16,
    # 1D-CNN
    "cnn_filters": [64, 32],
    "cnn_kernel_size": 3,
}

# ═══════════════════════════════════════════════════════════════
# SDSS GALAXY COLORS (clasificacion espectral)
# ═══════════════════════════════════════════════════════════════
SDSS_FEATURES = ["u_g", "g_r", "r_i", "i_z"]
SDSS_N_SAMPLES = 50000

# ═══════════════════════════════════════════════════════════════
# COLORES PARA VISUALIZACION
# ═══════════════════════════════════════════════════════════════
COLORS = {
    "MBA": "#3498db",
    "NEO": "#e74c3c",
    "COM": "#2ecc71",
    "TNO": "#9b59b6",
    "CEN": "#f39c12",
    "ISO": "#e91e63",
    "ml": "#2196F3",
    "dl": "#FF5722",
    "benchmark": "#4CAF50",
}
