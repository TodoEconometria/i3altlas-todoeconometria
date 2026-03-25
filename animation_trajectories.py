"""
MODULO ANIMACION - i3 Atlas
=============================
Genera animaciones GIF de las trayectorias de objetos interestelares:
  1) Trayectoria 3D de 'Oumuamua pasando por el Sistema Solar
  2) Comparativa 'Oumuamua vs Borisov (orbitas superpuestas)
  3) Distribucion orbital animada (objetos normales vs interestelares)

Usa matplotlib.animation para generar GIFs de alta calidad.
Dependencia: pip install pillow (para GIF export)

Ejecutar standalone:
    python animation_trajectories.py

Autor: Juan Marcelo Gutierrez Miranda | @TodoEconometria

REFERENCIA ACADEMICA:
- Meech, K. J., et al. (2017). A brief visit from a red and extremely elongated
  interstellar asteroid. Nature, 552(7685), 378-381.
- Micheli, M., et al. (2018). Non-gravitational acceleration in the trajectory
  of 1I/2017 U1 ('Oumuamua). Nature, 559(7713), 223-226.
- Guzik, P., et al. (2020). Initial characterization of interstellar comet
  2I/Borisov. Nature Astronomy, 4(1), 53-57.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from config import DATA_CACHE_DIR, FIGURES_DIR, PROCESSED_PARQUET


# ═══════════════════════════════════════════════════════════════
# PARAMETROS ORBITALES REALES
# ═══════════════════════════════════════════════════════════════

# 'Oumuamua: orbita hiperbolica, e=1.201, q=0.255 AU, i=122.7 deg
OUMUAMUA = {
    "name": "1I/'Oumuamua",
    "e": 1.20113,
    "q": 0.25525,  # AU
    "i": np.radians(122.7417),
    "om": np.radians(24.5969),   # longitud nodo ascendente
    "w": np.radians(241.8105),   # argumento perihelio
    "color": "#e91e63",
    "perihelion_date": "2017-09-09",
}

# Borisov: e=3.357, q=2.007 AU, i=44.05 deg
BORISOV = {
    "name": "2I/Borisov",
    "e": 3.3571,
    "q": 2.0066,
    "i": np.radians(44.0529),
    "om": np.radians(308.1481),
    "w": np.radians(209.1246),
    "color": "#2196F3",
    "perihelion_date": "2019-12-08",
}

# Planetas (orbitas circulares simplificadas, distancia en AU)
PLANETS = [
    {"name": "Mercury", "a": 0.387, "color": "#8c8c8c", "size": 3},
    {"name": "Venus",   "a": 0.723, "color": "#e6c35c", "size": 4},
    {"name": "Earth",   "a": 1.000, "color": "#4a90d9", "size": 4},
    {"name": "Mars",    "a": 1.524, "color": "#c1440e", "size": 3},
    {"name": "Jupiter", "a": 5.204, "color": "#c88b3a", "size": 8},
]


def compute_hyperbolic_orbit(obj, true_anomaly_range=(-2.5, 2.5), n_points=500):
    """
    Calcula posiciones 3D de una orbita hiperbolica.
    Para hiperbolicas: r = q(1+e) / (1 + e*cos(v)), v = anomalia verdadera
    """
    e = obj["e"]
    q = obj["q"]
    inc = obj["i"]
    om = obj["om"]
    w = obj["w"]

    # Semieje mayor (negativo para hiperbolicas)
    a = q / (e - 1)

    # Anomalia verdadera: limitada por asintota: cos(v_max) = -1/e
    v_max = np.arccos(-1.0 / e) * 0.95  # 95% del limite
    v_range = np.linspace(-v_max, v_max, n_points)

    # Radio
    r = q * (1 + e) / (1 + e * np.cos(v_range))

    # Coordenadas en el plano orbital
    x_orb = r * np.cos(v_range)
    y_orb = r * np.sin(v_range)

    # Rotacion 3D: w (argumento perihelio), i (inclinacion), om (nodo)
    # Matriz de rotacion completa
    cos_w, sin_w = np.cos(w), np.sin(w)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    cos_om, sin_om = np.cos(om), np.sin(om)

    x = (cos_om * cos_w - sin_om * sin_w * cos_i) * x_orb + \
        (-cos_om * sin_w - sin_om * cos_w * cos_i) * y_orb
    y = (sin_om * cos_w + cos_om * sin_w * cos_i) * x_orb + \
        (-sin_om * sin_w + cos_om * cos_w * cos_i) * y_orb
    z = (sin_w * sin_i) * x_orb + (cos_w * sin_i) * y_orb

    return x, y, z, v_range


def compute_planet_orbit(a, n_points=200):
    """Orbita circular de un planeta en el plano ecliptico."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = a * np.cos(theta)
    y = a * np.sin(theta)
    z = np.zeros_like(theta)
    return x, y, z


# ═══════════════════════════════════════════════════════════════
# ANIMACION 1: TRAYECTORIA 3D DE 'OUMUAMUA
# ═══════════════════════════════════════════════════════════════

def create_oumuamua_trajectory_gif(fps=20, duration_s=8):
    """
    GIF animado: 'Oumuamua atravesando el Sistema Solar interior.
    Vista 3D con orbitas planetarias y trayectoria hiperbolica.
    """
    print("  Generando animacion: Trayectoria de 'Oumuamua ...")

    n_frames = fps * duration_s
    x_oum, y_oum, z_oum, v_oum = compute_hyperbolic_orbit(OUMUAMUA, n_points=n_frames)

    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    # Estilo oscuro espacial
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-4, 4)
    ax.set_xlabel("X (AU)", color="white", fontsize=8)
    ax.set_ylabel("Y (AU)", color="white", fontsize=8)
    ax.set_zlabel("Z (AU)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")
    ax.grid(True, alpha=0.15, color="gray")

    # Sol
    ax.scatter([0], [0], [0], c="yellow", s=200, marker="o",
               edgecolors="orange", linewidths=1, zorder=10)
    ax.text(0.2, 0.2, 0.2, "Sol", color="yellow", fontsize=8)

    # Orbitas planetarias
    for planet in PLANETS:
        xp, yp, zp = compute_planet_orbit(planet["a"])
        ax.plot(xp, yp, zp, color=planet["color"], alpha=0.3, linewidth=0.5)
        # Posicion del planeta (fija para simplificar)
        ax.scatter([planet["a"]], [0], [0], c=planet["color"],
                   s=planet["size"] * 5, zorder=5)
        if planet["a"] <= 2:
            ax.text(planet["a"] + 0.1, 0.1, 0.1, planet["name"],
                    color=planet["color"], fontsize=6, alpha=0.7)

    # Trayectoria completa (tenue)
    ax.plot(x_oum, y_oum, z_oum, color=OUMUAMUA["color"], alpha=0.15,
            linewidth=1, linestyle="--")

    # Elementos animados
    trail, = ax.plot([], [], [], color=OUMUAMUA["color"], linewidth=2, alpha=0.8)
    point = ax.scatter([], [], [], c=OUMUAMUA["color"], s=80, marker="D",
                       edgecolors="white", linewidths=1, zorder=15)
    title_text = ax.set_title("", color="white", fontsize=12, fontweight="bold", pad=20)

    def init():
        trail.set_data_3d([], [], [])
        point._offsets3d = ([], [], [])
        return trail, point

    def update(frame):
        # Trail (ultimos 30 puntos)
        start = max(0, frame - 30)
        trail.set_data_3d(x_oum[start:frame+1], y_oum[start:frame+1], z_oum[start:frame+1])

        # Punto actual
        point._offsets3d = ([x_oum[frame]], [y_oum[frame]], [z_oum[frame]])

        # Distancia al Sol
        r = np.sqrt(x_oum[frame]**2 + y_oum[frame]**2 + z_oum[frame]**2)
        progress = frame / n_frames * 100

        title_text.set_text(
            f"1I/'Oumuamua - Trayectoria Hiperbolica\n"
            f"Distancia al Sol: {r:.2f} AU | Progreso: {progress:.0f}%"
        )

        # Rotar vista suavemente
        ax.view_init(elev=25 + 10 * np.sin(frame / n_frames * np.pi),
                      azim=45 + frame * 0.5)

        return trail, point

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, interval=1000 // fps, blit=False,
    )

    output_path = FIGURES_DIR / "oumuamua_trajectory_3d.gif"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"    Guardando GIF ({n_frames} frames, {fps} fps) ...")
    anim.save(str(output_path), writer="pillow", fps=fps, dpi=80)
    plt.close(fig)
    print(f"    Guardado: {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
# ANIMACION 2: COMPARATIVA 'OUMUAMUA vs BORISOV
# ═══════════════════════════════════════════════════════════════

def create_comparison_gif(fps=15, duration_s=10):
    """
    GIF animado: Ambos objetos interestelares simultaneamente.
    Vista cenital (plano ecliptico) + vista lateral.
    """
    print("  Generando animacion: Comparativa 'Oumuamua vs Borisov ...")

    n_frames = fps * duration_s

    x_oum, y_oum, z_oum, _ = compute_hyperbolic_orbit(OUMUAMUA, n_points=n_frames)
    x_bor, y_bor, z_bor, _ = compute_hyperbolic_orbit(BORISOV, n_points=n_frames)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="black")

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=7)
        ax.grid(True, alpha=0.15, color="gray")

    ax1, ax2 = axes

    # Vista cenital (X-Y)
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)
    ax1.set_xlabel("X (AU)", color="white", fontsize=9)
    ax1.set_ylabel("Y (AU)", color="white", fontsize=9)
    ax1.set_title("Vista Cenital (Plano Ecliptico)", color="white", fontsize=11, fontweight="bold")
    ax1.set_aspect("equal")

    # Vista lateral (X-Z)
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel("X (AU)", color="white", fontsize=9)
    ax2.set_ylabel("Z (AU)", color="white", fontsize=9)
    ax2.set_title("Vista Lateral (Fuera del Plano)", color="white", fontsize=11, fontweight="bold")

    # Sol
    for ax in axes:
        ax.plot(0, 0, "o", color="yellow", markersize=12, markeredgecolor="orange", zorder=10)

    # Orbitas planetarias
    for planet in PLANETS:
        theta = np.linspace(0, 2 * np.pi, 100)
        ax1.plot(planet["a"] * np.cos(theta), planet["a"] * np.sin(theta),
                 color=planet["color"], alpha=0.2, linewidth=0.5)
        ax2.plot(planet["a"] * np.cos(theta), np.zeros(100),
                 color=planet["color"], alpha=0.2, linewidth=0.5)

    # Trayectorias completas (tenues)
    ax1.plot(x_oum, y_oum, color=OUMUAMUA["color"], alpha=0.1, linewidth=0.8, linestyle="--")
    ax1.plot(x_bor, y_bor, color=BORISOV["color"], alpha=0.1, linewidth=0.8, linestyle="--")
    ax2.plot(x_oum, z_oum, color=OUMUAMUA["color"], alpha=0.1, linewidth=0.8, linestyle="--")
    ax2.plot(x_bor, z_bor, color=BORISOV["color"], alpha=0.1, linewidth=0.8, linestyle="--")

    # Elementos animados
    trail_oum_xy, = ax1.plot([], [], color=OUMUAMUA["color"], linewidth=2.5, alpha=0.9)
    trail_bor_xy, = ax1.plot([], [], color=BORISOV["color"], linewidth=2.5, alpha=0.9)
    point_oum_xy, = ax1.plot([], [], "D", color=OUMUAMUA["color"], markersize=8,
                              markeredgecolor="white", markeredgewidth=1, zorder=15)
    point_bor_xy, = ax1.plot([], [], "D", color=BORISOV["color"], markersize=8,
                              markeredgecolor="white", markeredgewidth=1, zorder=15)

    trail_oum_xz, = ax2.plot([], [], color=OUMUAMUA["color"], linewidth=2.5, alpha=0.9)
    trail_bor_xz, = ax2.plot([], [], color=BORISOV["color"], linewidth=2.5, alpha=0.9)
    point_oum_xz, = ax2.plot([], [], "D", color=OUMUAMUA["color"], markersize=8,
                              markeredgecolor="white", markeredgewidth=1, zorder=15)
    point_bor_xz, = ax2.plot([], [], "D", color=BORISOV["color"], markersize=8,
                              markeredgecolor="white", markeredgewidth=1, zorder=15)

    # Leyenda
    ax1.plot([], [], "D-", color=OUMUAMUA["color"], label="1I/'Oumuamua (e=1.20)")
    ax1.plot([], [], "D-", color=BORISOV["color"], label="2I/Borisov (e=3.36)")
    ax1.legend(loc="upper left", facecolor="black", edgecolor="gray",
               labelcolor="white", fontsize=9)

    info_text = fig.text(0.5, 0.02, "", ha="center", color="white", fontsize=10,
                          fontfamily="monospace")

    def init():
        for line in [trail_oum_xy, trail_bor_xy, trail_oum_xz, trail_bor_xz]:
            line.set_data([], [])
        for pt in [point_oum_xy, point_bor_xy, point_oum_xz, point_bor_xz]:
            pt.set_data([], [])
        return []

    def update(frame):
        trail_len = 40
        start = max(0, frame - trail_len)

        # Trails
        trail_oum_xy.set_data(x_oum[start:frame+1], y_oum[start:frame+1])
        trail_bor_xy.set_data(x_bor[start:frame+1], y_bor[start:frame+1])
        trail_oum_xz.set_data(x_oum[start:frame+1], z_oum[start:frame+1])
        trail_bor_xz.set_data(x_bor[start:frame+1], z_bor[start:frame+1])

        # Points
        point_oum_xy.set_data([x_oum[frame]], [y_oum[frame]])
        point_bor_xy.set_data([x_bor[frame]], [y_bor[frame]])
        point_oum_xz.set_data([x_oum[frame]], [z_oum[frame]])
        point_bor_xz.set_data([x_bor[frame]], [z_bor[frame]])

        # Info
        r_oum = np.sqrt(x_oum[frame]**2 + y_oum[frame]**2 + z_oum[frame]**2)
        r_bor = np.sqrt(x_bor[frame]**2 + y_bor[frame]**2 + z_bor[frame]**2)
        info_text.set_text(
            f"'Oumuamua: r={r_oum:.2f} AU  |  "
            f"Borisov: r={r_bor:.2f} AU  |  "
            f"Frame {frame+1}/{n_frames}"
        )

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, interval=1000 // fps, blit=False,
    )

    output_path = FIGURES_DIR / "interstellar_comparison.gif"
    print(f"    Guardando GIF ({n_frames} frames, {fps} fps) ...")
    anim.save(str(output_path), writer="pillow", fps=fps, dpi=80)
    plt.close(fig)
    print(f"    Guardado: {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
# ANIMACION 3: ESPACIO ORBITAL ANIMADO
# ═══════════════════════════════════════════════════════════════

def create_orbital_space_gif(fps=12, duration_s=8):
    """
    GIF animado: Rotacion 3D del espacio orbital (e vs a vs i).
    Muestra donde viven los interestelares vs objetos normales.
    """
    print("  Generando animacion: Espacio Orbital 3D ...")

    n_frames = fps * duration_s

    # Cargar datos procesados
    if not PROCESSED_PARQUET.exists():
        print("    Parquet no encontrado. Generando datos de ejemplo ...")
        rng = np.random.default_rng(42)
        n = 5000
        df = pd.DataFrame({
            "e": np.concatenate([rng.beta(2, 8, n - 2) * 0.4, [1.201, 3.357]]),
            "a": np.concatenate([rng.uniform(1, 5, n - 2), [-1.28, -0.85]]),
            "i": np.concatenate([rng.exponential(10, n - 2), [122.7, 44.1]]),
            "object_type": ["MBA"] * (n - 2) + ["ISO", "ISO"],
            "full_name": [f"Obj_{j}" for j in range(n - 2)] + ["1I/'Oumuamua", "2I/Borisov"],
        })
    else:
        df = pd.read_parquet(str(PROCESSED_PARQUET))

    # Sample para rendimiento
    if len(df) > 8000:
        normal = df[df["object_type"] != "ISO"].sample(n=7000, random_state=42)
        iso = df[df["object_type"] == "ISO"]
        df_plot = pd.concat([normal, iso])
    else:
        df_plot = df

    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")

    # Estilo
    ax.tick_params(colors="white", labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")
    ax.grid(True, alpha=0.1, color="gray")

    # Objetos normales
    normal = df_plot[df_plot["object_type"] != "ISO"]
    iso_obj = df_plot[df_plot["object_type"] == "ISO"]

    # Color por tipo
    type_colors = {"MBA": "#3498db", "NEO": "#e74c3c", "COM": "#2ecc71",
                   "TNO": "#9b59b6", "CEN": "#f39c12"}

    for obj_type, color in type_colors.items():
        mask = normal["object_type"] == obj_type
        if mask.any():
            subset = normal[mask]
            ax.scatter(subset["e"], subset["a"].clip(-10, 100),
                       subset["i"].clip(0, 180),
                       c=color, s=3, alpha=0.3, label=obj_type)

    # Interestelares grandes
    if len(iso_obj) > 0:
        ax.scatter(iso_obj["e"], iso_obj["a"].clip(-10, 10),
                   iso_obj["i"].clip(0, 180),
                   c="#e91e63", s=200, marker="*",
                   edgecolors="white", linewidths=1.5, zorder=10,
                   label="INTERESTELAR")
        for _, row in iso_obj.iterrows():
            ax.text(row["e"], min(row["a"], 10), min(row["i"], 180),
                    f"  {row.get('full_name', 'ISO')}",
                    color="#e91e63", fontsize=8, fontweight="bold")

    ax.set_xlabel("Excentricidad (e)", color="white", fontsize=9)
    ax.set_ylabel("Semieje Mayor (AU)", color="white", fontsize=9)
    ax.set_zlabel("Inclinacion (deg)", color="white", fontsize=9)
    ax.legend(loc="upper left", facecolor="black", edgecolor="gray",
              labelcolor="white", fontsize=7, markerscale=2)

    title = ax.set_title("i3 Atlas: Espacio Orbital\nObjetos Interestelares vs Sistema Solar",
                          color="white", fontsize=13, fontweight="bold", pad=20)

    def update(frame):
        ax.view_init(elev=20 + 15 * np.sin(frame / n_frames * 2 * np.pi),
                      azim=frame * (360 / n_frames))
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False,
    )

    output_path = FIGURES_DIR / "orbital_space_rotation.gif"
    print(f"    Guardando GIF ({n_frames} frames, {fps} fps) ...")
    anim.save(str(output_path), writer="pillow", fps=fps, dpi=80)
    plt.close(fig)
    print(f"    Guardado: {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
# ORQUESTADOR
# ═══════════════════════════════════════════════════════════════

def run_animations():
    """Genera todas las animaciones GIF."""
    print("=" * 60)
    print("ANIMACIONES GIF - i3 Atlas")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    gifs = []

    print("\n[1/3] Trayectoria 3D de 'Oumuamua ...")
    try:
        gifs.append(create_oumuamua_trajectory_gif())
    except Exception as e:
        print(f"    Error: {e}")

    print("\n[2/3] Comparativa 'Oumuamua vs Borisov ...")
    try:
        gifs.append(create_comparison_gif())
    except Exception as e:
        print(f"    Error: {e}")

    print("\n[3/3] Espacio Orbital 3D Rotacion ...")
    try:
        gifs.append(create_orbital_space_gif())
    except Exception as e:
        print(f"    Error: {e}")

    print("\n" + "=" * 60)
    print(f"ANIMACIONES completadas: {len(gifs)} GIFs")
    for g in gifs:
        print(f"  - {g}")
    print("=" * 60)

    return gifs


if __name__ == "__main__":
    run_animations()
