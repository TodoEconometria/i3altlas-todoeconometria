"""
ORQUESTADOR PRINCIPAL - i3 Atlas
==================================
Pipeline completo: Data Acquisition -> Spark ETL -> ML -> DL -> Benchmark -> Dashboard

Analisis Comparativo: Deep Learning vs Machine Learning
en el Catalogo Astronomico i3 Atlas con Apache Spark.

Ejecutar:
    python main.py

Prerequisitos:
    - Cluster Spark activo (docker-compose up)
    - pip install -r requirements_i3atlas.txt
    - GPU RTX disponible para DL (opcional, funciona en CPU)

Autor: Juan Marcelo Gutierrez Miranda | @TodoEconometria
"""
import sys
import os
import subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from pyspark.sql import SparkSession

from config import (
    SPARK_MASTER_URL, SPARK_APP_NAME, SPARK_CONFIG,
    OUTPUT_DIR, FIGURES_DIR, DASHBOARD_OUTPUT,
)


def create_spark_session():
    """Crea SparkSession conectada al cluster Docker Spark."""
    print("  Conectando a cluster Spark ...")
    builder = SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .master(SPARK_MASTER_URL)

    for key, value in SPARK_CONFIG.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    sc = spark.sparkContext
    print(f"    SparkSession activa: {sc.master}")
    print(f"    App ID: {sc.applicationId}")
    print(f"    Cores max: {sc.getConf().get('spark.cores.max', 'auto')}")
    print(f"    Executor memory: {sc.getConf().get('spark.executor.memory', 'default')}")
    print(f"    Driver memory: {sc.getConf().get('spark.driver.memory', 'default')}")
    return spark


def create_spark_local():
    """Fallback: SparkSession local si el cluster no esta disponible."""
    print("  Iniciando Spark en modo local ...")
    spark = SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print(f"    SparkSession local activa")
    return spark


def main():
    start_total = time.time()

    print("\n" + "#" * 70)
    print("#")
    print("#   i3 ATLAS: Deep Learning vs Machine Learning")
    print("#   Analisis Comparativo sobre Datos Astronomicos con Spark")
    print("#")
    print("#   Profesor: Juan Marcelo Gutierrez Miranda")
    print("#   @TodoEconometria")
    print("#")
    print("#" * 70)

    # Crear directorios
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Intentar cluster, fallback a local
    try:
        spark = create_spark_session()
        spark_mode = "CLUSTER"
    except Exception as e:
        print(f"\n  [WARN] Cluster Spark no disponible: {e}")
        print("  Usando modo local ...")
        spark = create_spark_local()
        spark_mode = "LOCAL"

    try:
        # ═══════════════════════════════════════════════════════════
        # PASO 1/6: ADQUISICION DE DATOS
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print(">>> PASO 1/7: Adquisicion de Datos (APIs Astronomicas)")
        print("=" * 70)
        from data_acquisition import run_data_acquisition
        try:
            run_data_acquisition()
        except Exception as e:
            print(f"  Error en adquisicion de datos: {e}")
            print("  Intentando continuar con datos existentes ...")

        # ═══════════════════════════════════════════════════════════
        # PASO 2/6: ETL CON SPARK
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print(f">>> PASO 2/7: ETL con Spark ({spark_mode})")
        print("=" * 70)
        from spark_etl import run_spark_etl
        try:
            run_spark_etl(spark)
        except Exception as e:
            print(f"  Error en Spark ETL: {e}")
            import traceback
            traceback.print_exc()

        # ═══════════════════════════════════════════════════════════
        # PASO 3/6: ML TRADICIONAL
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print(">>> PASO 3/7: Machine Learning Tradicional (scikit-learn)")
        print("=" * 70)
        from ml_traditional import run_ml_traditional
        try:
            ml_results = run_ml_traditional(spark)
        except Exception as e:
            print(f"  Error en ML (no critico): {e}")
            import traceback
            traceback.print_exc()

        # ═══════════════════════════════════════════════════════════
        # PASO 4/7: DEEP LEARNING
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print(">>> PASO 4/7: Deep Learning (TensorFlow/Keras + GPU)")
        print("=" * 70)

        # Detectar si TensorFlow esta disponible localmente
        tf_available = False
        try:
            import tensorflow
            tf_available = True
        except ImportError:
            pass

        if tf_available:
            # Ejecutar localmente (TF disponible en este Python)
            from deep_learning_pipeline import run_deep_learning
            try:
                dl_results = run_deep_learning(spark)
            except Exception as e:
                print(f"  Error en DL (no critico): {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  TensorFlow no disponible en este Python.")
            print("  Instala con: pip install tensorflow")
            print("  O para GPU: pip install tensorflow[and-cuda]")
            print("  Saltando paso DL ...")

        # ═══════════════════════════════════════════════════════════
        # PASO 5/6: BENCHMARK COMPARATIVO
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print(">>> PASO 5/7: Benchmark Comparativo ML vs DL")
        print("=" * 70)
        from benchmark_comparison import run_benchmark
        try:
            run_benchmark()
        except Exception as e:
            print(f"  Error en benchmark (no critico): {e}")

        # ═══════════════════════════════════════════════════════════
        # PASO 6/6: DASHBOARD HTML
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print(">>> PASO 6/7: Exportar Dashboard HTML Interactivo")
        print("=" * 70)
        from export_dashboard_html import run_export_dashboard
        try:
            dashboard_path = run_export_dashboard()
        except Exception as e:
            print(f"  Error en dashboard (no critico): {e}")
            dashboard_path = "N/A"

        # ═══════════════════════════════════════════════════════════
        # PASO 7/7: ANIMACIONES GIF
        # ═══════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print(">>> PASO 7/7: Animaciones GIF (Trayectorias Interestelares)")
        print("=" * 70)
        from animation_trajectories import run_animations
        try:
            gifs = run_animations()
        except Exception as e:
            print(f"  Error en animaciones (no critico): {e}")
            gifs = []

    finally:
        spark.stop()
        print("\n  SparkSession cerrada.")

    # ═══════════════════════════════════════════════════════════
    # RESUMEN FINAL
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - start_total
    print("\n" + "#" * 70)
    print("#")
    print(f"#   i3 ATLAS - PIPELINE COMPLETADO en {elapsed:.1f} segundos")
    print("#")
    print(f"#   Motor:     Apache Spark ({spark_mode})")
    print(f"#   Datos:     {OUTPUT_DIR / 'datos'}")
    print(f"#   Parquet:   {OUTPUT_DIR / 'i3atlas_procesado.parquet'}")
    print(f"#   Figuras:   {FIGURES_DIR}")
    print(f"#   ML:        {OUTPUT_DIR / 'ml_results.json'}")
    print(f"#   DL:        {OUTPUT_DIR / 'dl_results.json'}")
    print(f"#   Benchmark: {OUTPUT_DIR / 'benchmark_results.json'}")
    print(f"#   Dashboard: {dashboard_path}")
    print(f"#   GIFs:      {FIGURES_DIR}")
    print("#")
    print("#   @TodoEconometria | Prof. Juan Marcelo Gutierrez Miranda")
    print("#")
    print("#" * 70)


if __name__ == "__main__":
    main()
