"""
MODULO DE ADQUISICION DE DATOS - i3 Atlas
==========================================
Descarga datos astronomicos reales desde APIs publicas:
  A) JPL Small-Body Database  -> Elementos orbitales + propiedades fisicas
  B) JPL Horizons             -> Efemerides (trayectoria interestelares)
  C) SDSS Galaxy Colors       -> Clasificacion espectral (astroML o sintetico)

Ejecutar standalone:
    python data_acquisition.py

Autor: Juan Marcelo Gutierrez Miranda | @TodoEconometria

REFERENCIA ACADEMICA:
- Tonry, J. L., et al. (2018). ATLAS: A High-cadence All-sky Survey System.
  Publications of the Astronomical Society of the Pacific, 130(988), 064505.
- Park, R. S., et al. (2021). JPL Small-Body Database. NASA/JPL.
  https://ssd.jpl.nasa.gov/
- Alam, S., et al. (2015). The Eleventh and Twelfth Data Releases of the
  Sloan Digital Sky Survey. The Astrophysical Journal Supplement Series, 219(1), 12.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time
import requests
import numpy as np
import pandas as pd

from config import (
    DATA_CACHE_DIR, JPL_SBDB_API, JPL_FIELDS,
    INTERSTELLAR_OBJECTS, SDSS_N_SAMPLES, SPARK_DATA_PATH,
)


# ═══════════════════════════════════════════════════════════════
# A) JPL SMALL-BODY DATABASE
# ═══════════════════════════════════════════════════════════════

def fetch_jpl_small_bodies(max_objects=50000):
    """
    Descarga asteroides y cometas del JPL Small-Body Database API.
    Campos: elementos orbitales (e, a, i, ...) + fisicos (H, diameter, albedo).
    """
    cache_file = DATA_CACHE_DIR / "jpl_small_bodies.csv"
    if cache_file.exists():
        print(f"  [CACHE] Cargando desde {cache_file.name}")
        return pd.read_csv(str(cache_file))

    print("  Consultando JPL Small-Body Database API ...")
    print(f"  URL: {JPL_SBDB_API}")
    print(f"  Campos: {JPL_FIELDS}")

    # Construir query: asteroides con diametro conocido + todos los cometas
    params = {
        "fields": ",".join(JPL_FIELDS),
        "limit": str(max_objects),
        "sb-kind": "a",  # asteroids first
        "full-prec": "false",
    }

    # Query 1: Asteroides con diametro
    print("\n  [1/3] Descargando asteroides (con diametro conocido) ...")
    params_ast = {**params, "sb-kind": "a"}
    try:
        resp = requests.get(JPL_SBDB_API, params=params_ast, timeout=120)
        resp.raise_for_status()
        data_ast = resp.json()
        fields_ast = data_ast.get("fields", [])
        rows_ast = data_ast.get("data", [])
        df_ast = pd.DataFrame(rows_ast, columns=fields_ast)
        print(f"      Asteroides recibidos: {len(df_ast):,}")
    except Exception as e:
        print(f"      Error descargando asteroides: {e}")
        print("      Generando dataset sintetico como fallback ...")
        return _generate_synthetic_small_bodies(max_objects)

    # Query 2: Cometas
    print("  [2/3] Descargando cometas ...")
    params_com = {**params, "sb-kind": "c", "limit": "5000"}
    try:
        resp = requests.get(JPL_SBDB_API, params=params_com, timeout=60)
        resp.raise_for_status()
        data_com = resp.json()
        fields_com = data_com.get("fields", [])
        rows_com = data_com.get("data", [])
        df_com = pd.DataFrame(rows_com, columns=fields_com)
        print(f"      Cometas recibidos: {len(df_com):,}")
    except Exception as e:
        print(f"      Error descargando cometas: {e}")
        df_com = pd.DataFrame(columns=fields_ast)

    # Combinar
    df = pd.concat([df_ast, df_com], ignore_index=True)

    # Convertir columnas numericas
    numeric_cols = ["e", "a", "i", "om", "w", "q", "ad", "per_y",
                    "epoch", "H", "diameter", "albedo", "moid"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Agregar objetos interestelares manualmente (no estan en queries normales)
    print("  [3/3] Agregando objetos interestelares ...")
    iso_rows = _get_interstellar_objects()
    if iso_rows is not None and len(iso_rows) > 0:
        df = pd.concat([df, iso_rows], ignore_index=True)
        print(f"      Interestelares agregados: {len(iso_rows)}")

    # Guardar cache
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(cache_file), index=False)
    print(f"\n  Guardado: {cache_file} ({len(df):,} objetos)")

    return df


def _get_interstellar_objects():
    """
    Datos de 'Oumuamua (1I) y Borisov (2I) - elementos orbitales conocidos.
    Fuente: JPL Small-Body Database Browser.
    """
    iso_data = [
        {
            "spkid": "3788040",
            "full_name": "1I/'Oumuamua (A/2017 U1)",
            "kind": "iso",
            "class": "HYP",
            "neo": "N",
            "pha": "N",
            "e": 1.20113,        # hiperbolica
            "a": -1.2799,        # negativo (hiperbolica)
            "i": 122.7417,       # alta inclinacion
            "om": 24.5969,
            "w": 241.8105,
            "q": 0.25525,        # perihelio muy cercano al Sol
            "ad": None,          # no aplica (hiperbolica)
            "per_y": None,       # no tiene periodo
            "epoch": 2458080.5,
            "H": 22.08,
            "diameter": 0.200,   # estimado ~200m (elongado)
            "albedo": 0.1,       # estimado
            "moid": 0.0958,
        },
        {
            "spkid": "3788041",
            "full_name": "2I/Borisov (C/2019 Q4)",
            "kind": "iso",
            "class": "HYP",
            "neo": "N",
            "pha": "N",
            "e": 3.3571,         # altamente hiperbolica
            "a": -0.8514,        # negativo
            "i": 44.0529,
            "om": 308.1481,
            "w": 209.1246,
            "q": 2.0066,         # perihelio mas lejano que Oumuamua
            "ad": None,
            "per_y": None,
            "epoch": 2458820.5,
            "H": 13.6,
            "diameter": 1.0,     # estimado ~1km
            "albedo": 0.04,      # tipico de cometa
            "moid": 1.9432,
        },
    ]
    return pd.DataFrame(iso_data)


def _generate_synthetic_small_bodies(n=50000):
    """
    Fallback: genera dataset sintetico con estructura realista.
    Distribucion de elementos orbitales basada en estadisticas reales.
    """
    print("\n  Generando dataset sintetico (fallback) ...")
    rng = np.random.default_rng(42)

    records = []
    class_dist = {"MBA": 0.70, "NEO": 0.12, "COM": 0.08, "TNO": 0.08, "CEN": 0.02}

    for cls, frac in class_dist.items():
        n_cls = int(n * frac)
        for j in range(n_cls):
            if cls == "MBA":
                e = rng.beta(2, 8) * 0.4
                a = rng.uniform(2.0, 3.5)
                i = rng.exponential(8)
                H = rng.normal(15, 2)
                diameter = max(0.01, rng.lognormal(0, 1.5))
            elif cls == "NEO":
                e = rng.beta(2, 5) * 0.8
                a = rng.uniform(0.5, 2.0)
                i = rng.exponential(12)
                H = rng.normal(22, 3)
                diameter = max(0.001, rng.lognormal(-2, 1.5))
            elif cls == "COM":
                e = rng.uniform(0.4, 0.999)
                a = rng.uniform(3, 100)
                i = rng.uniform(0, 180)
                H = rng.normal(10, 4)
                diameter = max(0.1, rng.lognormal(0.5, 1))
            elif cls == "TNO":
                e = rng.beta(2, 8) * 0.5
                a = rng.uniform(30, 100)
                i = rng.exponential(10)
                H = rng.normal(7, 2)
                diameter = max(10, rng.lognormal(4, 1))
            else:  # CEN
                e = rng.uniform(0.1, 0.7)
                a = rng.uniform(5.2, 30)
                i = rng.exponential(15)
                H = rng.normal(9, 2)
                diameter = max(1, rng.lognormal(2, 1))

            q = a * (1 - e)
            ad = a * (1 + e)
            per_y = a ** 1.5  # Kepler
            om = rng.uniform(0, 360)
            w = rng.uniform(0, 360)
            albedo = max(0.01, rng.beta(2, 10))
            moid = max(0, rng.exponential(0.5))

            records.append({
                "spkid": str(1000000 + len(records)),
                "full_name": f"Synthetic_{cls}_{j:05d}",
                "kind": "a" if cls != "COM" else "c",
                "class": cls,
                "neo": "Y" if cls == "NEO" else "N",
                "pha": "Y" if (cls == "NEO" and moid < 0.05) else "N",
                "e": round(e, 6),
                "a": round(a, 6),
                "i": round(min(i, 180), 4),
                "om": round(om, 4),
                "w": round(w, 4),
                "q": round(q, 6),
                "ad": round(ad, 6),
                "per_y": round(per_y, 4),
                "epoch": 2460000.5,
                "H": round(H, 2),
                "diameter": round(diameter, 4),
                "albedo": round(albedo, 4),
                "moid": round(moid, 4),
            })

    df = pd.DataFrame(records)

    # Agregar interestelares
    iso = _get_interstellar_objects()
    df = pd.concat([df, iso], ignore_index=True)

    cache_file = DATA_CACHE_DIR / "jpl_small_bodies.csv"
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(cache_file), index=False)
    print(f"  Sintetico guardado: {cache_file} ({len(df):,} objetos)")
    return df


# ═══════════════════════════════════════════════════════════════
# B) JPL HORIZONS - EFEMERIDES INTERESTELARES
# ═══════════════════════════════════════════════════════════════

def fetch_interstellar_ephemeris():
    """
    Descarga efemerides de 'Oumuamua y Borisov desde JPL Horizons.
    Trayectoria post-perihelio: posicion, velocidad, distancia heliocentrica.
    """
    cache_file = DATA_CACHE_DIR / "interstellar_ephemeris.csv"
    if cache_file.exists():
        print(f"  [CACHE] Cargando efemerides desde {cache_file.name}")
        return pd.read_csv(str(cache_file), parse_dates=["datetime"])

    print("  Intentando astroquery.jplhorizons ...")

    try:
        from astroquery.jplhorizons import Horizons

        all_eph = []

        # 'Oumuamua: observado Oct 2017 - Ene 2018
        print("    Descargando efemerides de 'Oumuamua ...")
        obj_oum = Horizons(
            id="'Oumuamua",
            location="500@10",  # heliocentrico
            epochs={"start": "2017-09-01", "stop": "2018-06-01", "step": "1d"},
        )
        eph_oum = obj_oum.ephemerides()
        df_oum = eph_oum.to_pandas()
        df_oum["object"] = "1I/'Oumuamua"
        all_eph.append(df_oum)

        # Borisov: observado Ago 2019 - Mar 2020
        print("    Descargando efemerides de 2I/Borisov ...")
        obj_bor = Horizons(
            id="2I/Borisov",
            location="500@10",
            epochs={"start": "2019-08-01", "stop": "2020-06-01", "step": "1d"},
        )
        eph_bor = obj_bor.ephemerides()
        df_bor = eph_bor.to_pandas()
        df_bor["object"] = "2I/Borisov"
        all_eph.append(df_bor)

        df = pd.concat(all_eph, ignore_index=True)

        # Simplificar columnas
        keep_cols = ["object", "datetime_str", "RA", "DEC", "delta", "r", "V", "alpha"]
        available = [c for c in keep_cols if c in df.columns]
        df = df[available].copy()
        if "datetime_str" in df.columns:
            df.rename(columns={"datetime_str": "datetime"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])

        DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(cache_file), index=False)
        print(f"    Guardado: {cache_file} ({len(df)} registros)")
        return df

    except ImportError:
        print("    astroquery no disponible. Generando efemerides sinteticas ...")
        return _generate_synthetic_ephemeris()
    except Exception as e:
        print(f"    Error en JPL Horizons: {e}")
        print("    Generando efemerides sinteticas como fallback ...")
        return _generate_synthetic_ephemeris()


def _generate_synthetic_ephemeris():
    """Genera efemerides sinteticas de trayectorias hiperbolicas."""
    rng = np.random.default_rng(42)
    records = []

    # 'Oumuamua: perihelio 2017-09-09, q=0.255 AU
    dates_oum = pd.date_range("2017-09-01", "2018-06-01", freq="D")
    t0_oum = pd.Timestamp("2017-09-09")
    for dt in dates_oum:
        days = (dt - t0_oum).days
        r = 0.255 + 0.015 * abs(days) + 0.0001 * days ** 2  # hiperbolica
        v = 87.7 - 0.05 * abs(days)  # km/s, desacelerando
        records.append({
            "object": "1I/'Oumuamua",
            "datetime": dt,
            "r": round(r, 4),
            "delta": round(r * 0.8 + rng.normal(0, 0.05), 4),
            "V": round(26 - 2.5 * np.log10(max(0.01, 1 / r ** 2)), 2),
            "RA": round((45 + days * 0.3) % 360, 4),
            "DEC": round(20 + days * 0.15, 4),
            "alpha": round(max(0, 90 - abs(days) * 0.5), 2),
            "velocity_kms": round(v, 2),
        })

    # Borisov: perihelio 2019-12-08, q=2.007 AU
    dates_bor = pd.date_range("2019-08-01", "2020-06-01", freq="D")
    t0_bor = pd.Timestamp("2019-12-08")
    for dt in dates_bor:
        days = (dt - t0_bor).days
        r = 2.007 + 0.005 * abs(days) + 0.00005 * days ** 2
        v = 32.2 - 0.01 * abs(days)
        records.append({
            "object": "2I/Borisov",
            "datetime": dt,
            "r": round(r, 4),
            "delta": round(r * 0.7 + rng.normal(0, 0.1), 4),
            "V": round(18 - 2.5 * np.log10(max(0.01, 1 / r ** 2)), 2),
            "RA": round((220 + days * 0.2) % 360, 4),
            "DEC": round(-30 + days * 0.1, 4),
            "alpha": round(max(0, 60 - abs(days) * 0.3), 2),
            "velocity_kms": round(v, 2),
        })

    df = pd.DataFrame(records)
    cache_file = DATA_CACHE_DIR / "interstellar_ephemeris.csv"
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(cache_file), index=False)
    print(f"    Sintetico guardado: {cache_file} ({len(df)} registros)")
    return df


# ═══════════════════════════════════════════════════════════════
# C) SDSS GALAXY COLORS
# ═══════════════════════════════════════════════════════════════

def fetch_sdss_galaxy_colors(n_samples=None):
    """
    Carga colores de galaxias SDSS para clasificacion espectral.
    Intenta astroML primero, genera sintetico como fallback.
    """
    if n_samples is None:
        n_samples = SDSS_N_SAMPLES

    cache_file = DATA_CACHE_DIR / "sdss_galaxy_colors.csv"
    if cache_file.exists():
        print(f"  [CACHE] Cargando SDSS desde {cache_file.name}")
        df = pd.read_csv(str(cache_file))
        if len(df) >= n_samples:
            return df.head(n_samples)

    print(f"  Descargando SDSS Galaxy Colors ({n_samples:,} muestras) ...")

    try:
        from astroML.datasets import fetch_sdss_specgals
        data = fetch_sdss_specgals()
        df = pd.DataFrame({
            "u_g": data["u"] - data["g"],
            "g_r": data["g"] - data["r"],
            "r_i": data["r"] - data["i"],
            "i_z": data["i"] - data["z"],
            "redshift": data["redshift"],
        })
        # Clasificar por redshift (proxy de tipo espectral)
        df["spectral_class"] = pd.cut(
            df["redshift"],
            bins=[0, 0.05, 0.1, 0.2, 0.5, 10],
            labels=["nearby", "low_z", "mid_z", "high_z", "distant"],
        ).astype(str)
        df = df.head(n_samples)
        print(f"    astroML: {len(df):,} galaxias cargadas")

    except (ImportError, Exception) as e:
        print(f"    astroML no disponible ({e}). Generando SDSS sintetico ...")
        df = _generate_synthetic_sdss(n_samples)

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(cache_file), index=False)
    print(f"    Guardado: {cache_file}")
    return df


def _generate_synthetic_sdss(n=50000):
    """Genera colores SDSS sinteticos con estructura realista."""
    rng = np.random.default_rng(42)

    classes = ["nearby", "low_z", "mid_z", "high_z", "distant"]
    fracs = [0.15, 0.30, 0.30, 0.20, 0.05]

    # Parametros por clase (media y std de colores)
    class_params = {
        "nearby":  {"u_g": (1.4, 0.3), "g_r": (0.6, 0.15), "r_i": (0.3, 0.1), "i_z": (0.15, 0.08), "z": (0.02, 0.01)},
        "low_z":   {"u_g": (1.6, 0.4), "g_r": (0.7, 0.2),  "r_i": (0.35, 0.12), "i_z": (0.2, 0.1), "z": (0.07, 0.02)},
        "mid_z":   {"u_g": (1.8, 0.5), "g_r": (0.8, 0.25), "r_i": (0.4, 0.15), "i_z": (0.25, 0.12), "z": (0.15, 0.03)},
        "high_z":  {"u_g": (2.0, 0.6), "g_r": (0.9, 0.3),  "r_i": (0.45, 0.2), "i_z": (0.3, 0.15), "z": (0.35, 0.08)},
        "distant": {"u_g": (2.5, 0.8), "g_r": (1.1, 0.4),  "r_i": (0.55, 0.25), "i_z": (0.4, 0.2), "z": (0.6, 0.15)},
    }

    records = []
    for cls, frac in zip(classes, fracs):
        n_cls = int(n * frac)
        p = class_params[cls]
        for _ in range(n_cls):
            records.append({
                "u_g": rng.normal(*p["u_g"]),
                "g_r": rng.normal(*p["g_r"]),
                "r_i": rng.normal(*p["r_i"]),
                "i_z": rng.normal(*p["i_z"]),
                "redshift": max(0.001, rng.normal(*p["z"])),
                "spectral_class": cls,
            })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════
# COPIAR A VOLUMEN SPARK
# ═══════════════════════════════════════════════════════════════

def copy_to_spark_volume():
    """Copia CSVs descargados al volumen compartido Spark/Docker."""
    import shutil

    print("\n  Copiando datos al volumen Spark ...")
    SPARK_DATA_PATH.mkdir(parents=True, exist_ok=True)

    for csv_file in DATA_CACHE_DIR.glob("*.csv"):
        dest = SPARK_DATA_PATH / csv_file.name
        shutil.copy2(str(csv_file), str(dest))
        print(f"    {csv_file.name} -> {dest}")

    print("  Datos listos para Spark.")


# ═══════════════════════════════════════════════════════════════
# ORQUESTADOR
# ═══════════════════════════════════════════════════════════════

def run_data_acquisition():
    """Ejecuta la adquisicion completa de datos."""
    print("=" * 60)
    print("DATA ACQUISITION: i3 Atlas - Datos Astronomicos Reales")
    print("=" * 60)

    # A) JPL Small-Body Database
    print("\n[1/4] JPL Small-Body Database (asteroides + cometas) ...")
    df_bodies = fetch_jpl_small_bodies()
    print(f"      Total objetos: {len(df_bodies):,}")
    print(f"      Columnas: {list(df_bodies.columns)}")

    # B) Efemerides interestelares
    print("\n[2/4] JPL Horizons (efemerides interestelares) ...")
    df_eph = fetch_interstellar_ephemeris()
    print(f"      Total registros: {len(df_eph):,}")

    # C) SDSS Galaxy Colors
    print("\n[3/4] SDSS Galaxy Colors (clasificacion espectral) ...")
    df_sdss = fetch_sdss_galaxy_colors()
    print(f"      Total galaxias: {len(df_sdss):,}")

    # D) Copiar al volumen Spark
    print("\n[4/4] Copiando al volumen Spark ...")
    copy_to_spark_volume()

    print("\n" + "=" * 60)
    print("DATA ACQUISITION completada.")
    print(f"  Objetos astronomicos: {len(df_bodies):,}")
    print(f"  Efemerides interestelares: {len(df_eph):,}")
    print(f"  Galaxias SDSS: {len(df_sdss):,}")
    print(f"  Cache: {DATA_CACHE_DIR}")
    print(f"  Spark volume: {SPARK_DATA_PATH}")
    print("=" * 60)

    return df_bodies, df_eph, df_sdss


if __name__ == "__main__":
    run_data_acquisition()
