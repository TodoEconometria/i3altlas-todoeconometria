"""
MODULO ETL CON SPARK - i3 Atlas
=================================
Pipeline ETL distribuido sobre cluster Spark Docker:
  1) Carga CSVs desde volumen compartido
  2) Limpieza de nulos y anomalos
  3) Feature engineering (Tisserand, v_inf, energy)
  4) Clasificacion por tipo orbital
  5) Export a Parquet particionado

Ejecutar standalone:
    python spark_etl.py

Autor: Juan Marcelo Gutierrez Miranda | @TodoEconometria

REFERENCIA ACADEMICA:
- Zaharia, M., Xin, R. S., Wendell, P., et al. (2016). Apache Spark: A Unified
  Engine for Big Data Processing. Communications of the ACM, 59(11), 56-65.
- Murray, C. D., & Dermott, S. F. (1999). Solar System Dynamics.
  Cambridge University Press.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType,
)

from config import (
    SPARK_DATA_URI, SPARK_DATA_PATH, PROCESSED_PARQUET,
    CLASS_MAPPING, FEATURES_ORBITAL, FEATURES_PHYSICAL,
    FEATURES_ENGINEERED, OUTPUT_DIR,
)

# Semieje mayor de Jupiter (AU) para parametro de Tisserand
A_JUPITER = 5.2044


def run_spark_etl(spark):
    """
    Pipeline ETL completo sobre datos astronomicos con Spark.
    Usa Spark DataFrame API (sin UDFs Python) para rendimiento optimo.
    """
    print("=" * 60)
    print("SPARK ETL: Procesamiento de datos i3 Atlas")
    print("=" * 60)

    # ───────────────────────────────────────────────────────────
    # [1/5] Carga desde volumen compartido
    # ───────────────────────────────────────────────────────────
    print("\n[1/5] Cargando CSV desde volumen Spark ...")
    csv_uri = f"{SPARK_DATA_URI}/jpl_small_bodies.csv"

    df = spark.read.csv(
        csv_uri,
        header=True,
        inferSchema=True,
        nanValue="",
        nullValue="",
    )
    df.cache()
    total_raw = df.count()
    print(f"      Filas cargadas: {total_raw:,}")
    print(f"      Columnas: {df.columns}")

    # ───────────────────────────────────────────────────────────
    # [2/5] Limpieza de datos
    # ───────────────────────────────────────────────────────────
    print("\n[2/5] Limpieza de datos ...")

    # Convertir columnas numericas (por si inferSchema fallo)
    numeric_cols = FEATURES_ORBITAL + FEATURES_PHYSICAL
    for col_name in numeric_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

    # Filtrar filas con al menos excentricidad y semieje mayor
    df = df.filter(F.col("e").isNotNull() & F.col("a").isNotNull())

    # Limpiar rangos fisicos
    # Excentricidad: [0, inf) - permitir >1 para hiperbolicas
    df = df.filter(F.col("e") >= 0)

    # Inclinacion: [0, 180]
    df = df.withColumn("i", F.when(F.col("i") > 180, 180.0).otherwise(F.col("i")))
    df = df.withColumn("i", F.when(F.col("i") < 0, 0.0).otherwise(F.col("i")))

    # Magnitud absoluta H: rango razonable [-5, 35]
    df = df.withColumn("H",
        F.when((F.col("H") < -5) | (F.col("H") > 35), F.lit(None))
        .otherwise(F.col("H"))
    )

    # Diametro: positivo
    df = df.withColumn("diameter",
        F.when(F.col("diameter") <= 0, F.lit(None))
        .otherwise(F.col("diameter"))
    )

    cleaned = df.count()
    dropped = total_raw - cleaned
    print(f"      Filas tras limpieza: {cleaned:,} (eliminadas: {dropped:,})")

    # ───────────────────────────────────────────────────────────
    # [3/5] Feature engineering
    # ───────────────────────────────────────────────────────────
    print("\n[3/5] Feature engineering (Spark DataFrame API) ...")

    # Parametro de Tisserand respecto a Jupiter
    # T_J = (a_J / a) + 2 * cos(i) * sqrt((a / a_J) * (1 - e^2))
    df = df.withColumn("tisserand_j",
        F.lit(A_JUPITER) / F.col("a")
        + 2.0 * F.cos(F.radians(F.col("i")))
        * F.sqrt(
            F.abs(F.col("a") / F.lit(A_JUPITER))
            * (1.0 - F.col("e") * F.col("e"))
        )
    )

    # Velocidad en el infinito (v_inf) para objetos hiperbolicos
    # v_inf^2 = -mu/a  (para a negativo en hiperbolicas)
    # Simplificado: v_inf = sqrt(|1/a|) * 29.78 km/s (unidades solares)
    df = df.withColumn("v_inf",
        F.when(F.col("e") > 1.0,
            F.sqrt(F.abs(1.0 / F.col("a"))) * 29.78
        ).otherwise(F.lit(0.0))
    )

    # Parametro de energia orbital: -1/(2a) (negativo=ligado, positivo=no ligado)
    df = df.withColumn("energy_param",
        F.lit(-1.0) / (2.0 * F.col("a"))
    )

    # Ratio perihelio/semieje: q/a (cercano a 1 = circular, cercano a 0 = elongada)
    df = df.withColumn("q_over_a",
        F.when(F.col("a") != 0,
            F.abs(F.col("q") / F.col("a"))
        ).otherwise(F.lit(None))
    )

    print(f"      Features creadas: {FEATURES_ENGINEERED}")

    # ───────────────────────────────────────────────────────────
    # [4/5] Clasificacion por tipo orbital
    # ───────────────────────────────────────────────────────────
    print("\n[4/5] Clasificando objetos por tipo orbital ...")

    # Primero usar la clase de JPL mapeada
    class_expr = F.col("class")
    for jpl_class, our_class in CLASS_MAPPING.items():
        class_expr = F.when(F.col("class") == jpl_class, F.lit(our_class)).otherwise(class_expr)

    df = df.withColumn("object_type", class_expr)

    # Reclasificar hiperbolicos con e > 1.0 como interestelares
    df = df.withColumn("object_type",
        F.when(F.col("e") > 1.0, F.lit("ISO"))
        .otherwise(F.col("object_type"))
    )

    # Si kind = "iso", forzar ISO
    if "kind" in df.columns:
        df = df.withColumn("object_type",
            F.when(F.col("kind") == "iso", F.lit("ISO"))
            .otherwise(F.col("object_type"))
        )

    # Estadisticas por tipo
    print("      Distribucion por tipo:")
    type_counts = df.groupBy("object_type").count().orderBy(F.desc("count")).collect()
    for row in type_counts:
        print(f"        {row['object_type']}: {row['count']:,}")

    # ───────────────────────────────────────────────────────────
    # [5/5] Guardar Parquet
    # ───────────────────────────────────────────────────────────
    print("\n[5/5] Guardando Parquet procesado ...")

    import pandas as pd
    import numpy as np
    df_pandas = None

    # Estrategia 1: Recoger Spark DataFrame con toPandas()
    try:
        print("      Recolectando desde Spark con toPandas() ...")
        df_pandas = df.toPandas()
        print(f"      toPandas() exitoso: {len(df_pandas):,} filas")
    except Exception as e:
        print(f"      toPandas() fallo: {e}")

    df.unpersist()

    # Estrategia 2: Fallback con pandas puro desde CSV
    if df_pandas is None or len(df_pandas) == 0:
        csv_path = SPARK_DATA_PATH / "jpl_small_bodies.csv"
        if csv_path.exists():
            print("      Fallback: procesando CSV con pandas ...")
            df_pandas = pd.read_csv(str(csv_path))
            numeric_cols = FEATURES_ORBITAL + FEATURES_PHYSICAL
            for col in numeric_cols:
                if col in df_pandas.columns:
                    df_pandas[col] = pd.to_numeric(df_pandas[col], errors="coerce")
            df_pandas = df_pandas.dropna(subset=["e", "a"])
            df_pandas = df_pandas[df_pandas["e"] >= 0]
            # Inclinacion: clip a [0, 180]
            if "i" in df_pandas.columns:
                df_pandas["i"] = df_pandas["i"].clip(0, 180)
            # Magnitud H: rango [-5, 35]
            if "H" in df_pandas.columns:
                df_pandas.loc[(df_pandas["H"] < -5) | (df_pandas["H"] > 35), "H"] = np.nan
            # Diametro: positivo
            if "diameter" in df_pandas.columns:
                df_pandas.loc[df_pandas["diameter"] <= 0, "diameter"] = np.nan
            # Tisserand
            a_j = A_JUPITER
            df_pandas["tisserand_j"] = (
                a_j / df_pandas["a"]
                + 2 * np.cos(np.radians(df_pandas["i"].fillna(0)))
                * np.sqrt(np.abs(df_pandas["a"] / a_j) * (1 - df_pandas["e"]**2))
            )
            df_pandas["v_inf"] = np.where(
                df_pandas["e"] > 1.0,
                np.sqrt(np.abs(1.0 / df_pandas["a"])) * 29.78,
                0.0
            )
            df_pandas["energy_param"] = -1.0 / (2.0 * df_pandas["a"])
            df_pandas["q_over_a"] = np.where(
                df_pandas["a"] != 0,
                np.abs(df_pandas["q"] / df_pandas["a"]),
                np.nan
            )
            # Clasificar
            df_pandas["object_type"] = df_pandas["class"].map(CLASS_MAPPING).fillna(df_pandas["class"])
            df_pandas.loc[df_pandas["e"] > 1.0, "object_type"] = "ISO"
            if "kind" in df_pandas.columns:
                df_pandas.loc[df_pandas["kind"] == "iso", "object_type"] = "ISO"
            print(f"      Fallback completado: {len(df_pandas):,} filas")
        else:
            print("      ERROR: No se encontro CSV en el volumen Spark.")
            return pd.DataFrame()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_pandas.to_parquet(str(PROCESSED_PARQUET), index=False, engine="pyarrow")
    print(f"      Parquet local: {PROCESSED_PARQUET}")
    print(f"      Filas: {len(df_pandas):,} | Columnas: {len(df_pandas.columns)}")

    print("\n" + "=" * 60)
    print("SPARK ETL completado.")
    print("=" * 60)

    return df_pandas


if __name__ == "__main__":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("i3 Atlas ETL Test") \
        .master("local[*]") \
        .getOrCreate()
    run_spark_etl(spark)
    spark.stop()
