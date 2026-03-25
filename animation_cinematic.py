"""
ANIMACIONES CINEMATOGRAFICAS - i3 Atlas
=========================================
Visualizaciones estilo astronomo/NASA de los 3 objetos interestelares:
  - 1I/'Oumuamua (2017) - forma alargada unica
  - 2I/Borisov (2019) - cometa interestelar
  - 3I/ATLAS (2025) - el visitante mas reciente

Incluye:
  - Fondo estrellado realista (Milky Way simulation)
  - Trayectorias hiperbolicas reales
  - Orbitas planetarias de referencia
  - Estelas y efectos visuales
  - Comparativa de los 3 objetos

Autor: Juan Marcelo Gutierrez Miranda | @TodoEconometria

REFERENCIA ACADEMICA:
- Meech, K. J., et al. (2017). A brief visit from a red and extremely elongated
  interstellar asteroid. Nature, 552(7685), 378-381.
- Guzik, P., et al. (2020). Initial characterization of interstellar comet
  2I/Borisov. Nature Astronomy, 4, 53-57.
- NASA (2025). 3I/ATLAS: Third confirmed interstellar object.
  science.nasa.gov/solar-system/comets/3i-atlas/
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Ellipse
from matplotlib.collections import PathCollection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

from config import FIGURES_DIR, INTERSTELLAR_ORBITAL_DATA, OUTPUT_DIR


# ═══════════════════════════════════════════════════════════════
# CONSTANTES ASTRONOMICAS
# ═══════════════════════════════════════════════════════════════

# Orbitas planetarias (semieje mayor en AU)
PLANET_ORBITS = {
    "Mercury": 0.387,
    "Venus": 0.723,
    "Earth": 1.0,
    "Mars": 1.524,
    "Jupiter": 5.203,
}

# Colores para objetos interestelares
ISO_COLORS = {
    "1I/'Oumuamua": "#FF6B6B",   # Rojo coral - objeto rocoso
    "2I/Borisov": "#4ECDC4",      # Turquesa - cometa con coma
    "3I/ATLAS": "#FFE66D",        # Amarillo dorado - el nuevo
}


# ═══════════════════════════════════════════════════════════════
# GENERACION DE FONDO ESTRELLADO
# ═══════════════════════════════════════════════════════════════

def generate_starfield(ax, n_stars=2000, seed=42):
    """Genera un campo de estrellas realista para fondo."""
    np.random.seed(seed)

    # Estrellas con diferentes brillos (magnitudes)
    x = np.random.uniform(-15, 15, n_stars)
    y = np.random.uniform(-15, 15, n_stars)
    z = np.random.uniform(-15, 15, n_stars)

    # Tamaños basados en "magnitud" (estrellas brillantes son raras)
    magnitudes = np.random.exponential(0.3, n_stars)
    sizes = np.clip(magnitudes * 2, 0.1, 4)

    # Colores estelares (de azul a rojo segun tipo espectral)
    colors = []
    for _ in range(n_stars):
        r = np.random.random()
        if r < 0.1:  # Estrellas azules (O, B) - raras
            colors.append("#A0C4FF")
        elif r < 0.4:  # Estrellas blancas (A, F)
            colors.append("#FFFFFF")
        elif r < 0.7:  # Estrellas amarillas (G, K)
            colors.append("#FFFACD")
        else:  # Estrellas rojas (M) - comunes
            colors.append("#FFB6A3")

    ax.scatter(x, y, z, s=sizes, c=colors, alpha=0.8, depthshade=False)


def generate_starfield_2d(ax, n_stars=3000, seed=42):
    """Genera campo de estrellas para graficos 2D."""
    np.random.seed(seed)

    x = np.random.uniform(ax.get_xlim()[0], ax.get_xlim()[1], n_stars)
    y = np.random.uniform(ax.get_ylim()[0], ax.get_ylim()[1], n_stars)

    sizes = np.random.exponential(0.5, n_stars)
    sizes = np.clip(sizes, 0.1, 3)

    alphas = np.random.uniform(0.3, 1.0, n_stars)

    ax.scatter(x, y, s=sizes, c='white', alpha=0.6, zorder=0)


# ═══════════════════════════════════════════════════════════════
# TRAYECTORIAS HIPERBOLICAS
# ═══════════════════════════════════════════════════════════════

def compute_hyperbolic_trajectory(e, q, i, n_points=500, t_range=(-2, 2)):
    """
    Calcula trayectoria hiperbolica para un objeto interestelar.

    e: excentricidad (>1 para hiperbolica)
    q: distancia perihelio (AU)
    i: inclinacion (grados)
    """
    # Parametro semilatusrectum
    a = q / (e - 1)  # semieje mayor (negativo para hiperbolica)
    p = a * (e**2 - 1)

    # Anomalia verdadera: de -limite a +limite
    # Para hiperbolica: |nu| < arccos(-1/e)
    nu_max = np.arccos(-1/e) - 0.1  # Un poco menos del limite
    nu = np.linspace(-nu_max, nu_max, n_points)

    # Ecuacion de la orbita en coordenadas polares
    r = p / (1 + e * np.cos(nu))

    # Convertir a cartesianas (plano orbital)
    x_orbit = r * np.cos(nu)
    y_orbit = r * np.sin(nu)
    z_orbit = np.zeros_like(x_orbit)

    # Rotar por inclinacion
    i_rad = np.radians(i)
    x = x_orbit
    y = y_orbit * np.cos(i_rad) - z_orbit * np.sin(i_rad)
    z = y_orbit * np.sin(i_rad) + z_orbit * np.cos(i_rad)

    return x, y, z


def draw_planet_orbits(ax, planets=None, is_3d=True):
    """Dibuja orbitas planetarias como referencia."""
    if planets is None:
        planets = ["Earth", "Mars", "Jupiter"]

    theta = np.linspace(0, 2*np.pi, 100)

    planet_colors = {
        "Mercury": "#B5B5B5",
        "Venus": "#E6C229",
        "Earth": "#4A90D9",
        "Mars": "#D9534F",
        "Jupiter": "#D9A441",
    }

    for planet in planets:
        if planet not in PLANET_ORBITS:
            continue
        r = PLANET_ORBITS[planet]
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        color = planet_colors.get(planet, "gray")

        if is_3d:
            z = np.zeros_like(x)
            ax.plot(x, y, z, color=color, alpha=0.3, linewidth=1, linestyle='--')
        else:
            ax.plot(x, y, color=color, alpha=0.3, linewidth=1, linestyle='--')


# ═══════════════════════════════════════════════════════════════
# ANIMACION 1: COMPARATIVA DE LOS 3 INTERESTELARES
# ═══════════════════════════════════════════════════════════════

def create_three_interstellar_comparison(output_path=None, fps=15, duration=12):
    """
    Animacion cinematografica comparando los 3 objetos interestelares
    atravesando el Sistema Solar.
    """
    if output_path is None:
        output_path = FIGURES_DIR / "three_interstellar_cinematic.gif"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("  Generando animacion: Los 3 Visitantes Interestelares ...")

    fig = plt.figure(figsize=(16, 10), facecolor='black')

    # Layout: 3D grande arriba, 3 paneles pequeños abajo
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # Panel principal 3D
    ax_main = fig.add_subplot(gs[0, :], projection='3d', facecolor='black')

    # Paneles individuales 2D
    ax1 = fig.add_subplot(gs[1, 0], facecolor='black')
    ax2 = fig.add_subplot(gs[1, 1], facecolor='black')
    ax3 = fig.add_subplot(gs[1, 2], facecolor='black')

    # Configurar panel principal
    ax_main.set_xlim(-8, 8)
    ax_main.set_ylim(-8, 8)
    ax_main.set_zlim(-4, 4)
    ax_main.set_facecolor('black')
    ax_main.xaxis.pane.fill = False
    ax_main.yaxis.pane.fill = False
    ax_main.zaxis.pane.fill = False
    ax_main.grid(False)
    ax_main.set_axis_off()

    # Generar estrellas de fondo
    generate_starfield(ax_main, n_stars=1500)

    # Sol en el centro
    ax_main.scatter([0], [0], [0], s=200, c='yellow', marker='o',
                    edgecolors='orange', linewidths=2, zorder=10)

    # Orbitas planetarias
    draw_planet_orbits(ax_main, ["Earth", "Mars", "Jupiter"], is_3d=True)

    # Calcular trayectorias de los 3 objetos
    trajectories = {}
    for name, data in INTERSTELLAR_ORBITAL_DATA.items():
        x, y, z = compute_hyperbolic_trajectory(
            e=data["e"],
            q=data["q"],
            i=data["i"],
            n_points=300
        )
        trajectories[name] = (x, y, z)

    # Dibujar trayectorias completas (tenues)
    for name, (x, y, z) in trajectories.items():
        color = ISO_COLORS.get(name, "white")
        ax_main.plot(x, y, z, color=color, alpha=0.2, linewidth=1)

    # Objetos animados (puntos que se moveran)
    objects = {}
    trails = {}
    for name in trajectories.keys():
        color = ISO_COLORS.get(name, "white")
        obj, = ax_main.plot([], [], [], 'o', color=color, markersize=10,
                           markeredgecolor='white', markeredgewidth=1)
        trail, = ax_main.plot([], [], [], color=color, alpha=0.6, linewidth=2)
        objects[name] = obj
        trails[name] = trail

    # Configurar paneles individuales
    for ax, name in [(ax1, "1I/'Oumuamua"), (ax2, "2I/Borisov"), (ax3, "3I/ATLAS")]:
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.tick_params(colors='white', labelsize=8)
        ax.set_title(name, color=ISO_COLORS.get(name, "white"), fontsize=12, fontweight='bold')

        # Orbitas en 2D
        draw_planet_orbits(ax, ["Earth", "Mars"], is_3d=False)

        # Sol
        ax.scatter([0], [0], s=100, c='yellow', marker='o', zorder=10)

        # Trayectoria
        if name in trajectories:
            x, y, z = trajectories[name]
            ax.plot(x, y, color=ISO_COLORS.get(name, "white"), alpha=0.3, linewidth=1)

    # Objetos en paneles 2D
    obj_2d = {
        "1I/'Oumuamua": ax1.plot([], [], 'o', color=ISO_COLORS["1I/'Oumuamua"], markersize=8)[0],
        "2I/Borisov": ax2.plot([], [], 'o', color=ISO_COLORS["2I/Borisov"], markersize=8)[0],
        "3I/ATLAS": ax3.plot([], [], 'o', color=ISO_COLORS["3I/ATLAS"], markersize=8)[0],
    }

    # Titulo principal
    title = fig.suptitle("Los 3 Visitantes Interestelares del Sistema Solar\n" +
                         "1I/'Oumuamua (2017) | 2I/Borisov (2019) | 3I/ATLAS (2025)",
                         color='white', fontsize=14, fontweight='bold', y=0.98)

    # Info panel
    info_text = ax_main.text2D(0.02, 0.98, "", transform=ax_main.transAxes,
                               color='white', fontsize=9, verticalalignment='top',
                               fontfamily='monospace')

    n_frames = fps * duration

    def init():
        for obj in objects.values():
            obj.set_data([], [])
            obj.set_3d_properties([])
        for trail in trails.values():
            trail.set_data([], [])
            trail.set_3d_properties([])
        for obj in obj_2d.values():
            obj.set_data([], [])
        info_text.set_text("")
        return list(objects.values()) + list(trails.values()) + list(obj_2d.values()) + [info_text]

    def animate(frame):
        progress = frame / n_frames

        for name, (x, y, z) in trajectories.items():
            # Posicion actual
            idx = int(progress * (len(x) - 1))

            # Objeto principal 3D
            objects[name].set_data([x[idx]], [y[idx]])
            objects[name].set_3d_properties([z[idx]])

            # Trail (ultimos 30 puntos)
            trail_start = max(0, idx - 30)
            trails[name].set_data(x[trail_start:idx+1], y[trail_start:idx+1])
            trails[name].set_3d_properties(z[trail_start:idx+1])

            # Objeto 2D
            if name in obj_2d:
                obj_2d[name].set_data([x[idx]], [y[idx]])

        # Rotar vista
        ax_main.view_init(elev=25, azim=45 + progress * 180)

        # Actualizar info
        data = INTERSTELLAR_ORBITAL_DATA.get("3I/ATLAS", {})
        info = (
            f"Velocidad 3I/ATLAS: {data.get('v_inf', 60):.0f} km/s\n"
            f"Perihelio: {data.get('q', 1.35):.2f} AU\n"
            f"Origen: {data.get('origin', 'Desconocido')[:25]}"
        )
        info_text.set_text(info)

        return list(objects.values()) + list(trails.values()) + list(obj_2d.values()) + [info_text]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=1000/fps, blit=False)

    print(f"    Guardando GIF ({n_frames} frames, {fps} fps) ...")
    anim.save(str(output_path), writer='pillow', fps=fps, dpi=100)
    plt.close(fig)

    print(f"    Guardado: {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
# ANIMACION 2: VIAJE DE 3I/ATLAS (CINEMATOGRAFICO)
# ═══════════════════════════════════════════════════════════════

def create_atlas_journey(output_path=None, fps=20, duration=10):
    """
    Animacion cinematografica del viaje de 3I/ATLAS a traves del Sistema Solar.
    Estilo NASA/documental.
    """
    if output_path is None:
        output_path = FIGURES_DIR / "3i_atlas_journey_cinematic.gif"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("  Generando animacion: Viaje de 3I/ATLAS ...")

    fig = plt.figure(figsize=(14, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Configurar ejes
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-3, 3)
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_axis_off()

    # Fondo estrellado denso
    generate_starfield(ax, n_stars=2500, seed=2025)

    # Sol con brillo
    ax.scatter([0], [0], [0], s=300, c='#FFFF00', marker='o',
               edgecolors='#FFA500', linewidths=3, zorder=10, alpha=0.9)
    # Halo solar
    for r, alpha in [(0.3, 0.3), (0.5, 0.2), (0.7, 0.1)]:
        theta = np.linspace(0, 2*np.pi, 50)
        x_halo = r * np.cos(theta)
        y_halo = r * np.sin(theta)
        ax.plot(x_halo, y_halo, np.zeros_like(x_halo),
                color='orange', alpha=alpha, linewidth=2)

    # Orbitas planetarias con labels
    for planet, r in [("Earth", 1.0), ("Mars", 1.524), ("Jupiter", 5.203)]:
        theta = np.linspace(0, 2*np.pi, 100)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        color = {"Earth": "#4A90D9", "Mars": "#D9534F", "Jupiter": "#D9A441"}[planet]
        ax.plot(x, y, np.zeros_like(x), color=color, alpha=0.4, linewidth=1.5, linestyle='--')
        # Posicion del planeta (simplificada)
        ax.scatter([r], [0], [0], s=40, c=color, marker='o', alpha=0.8)

    # Trayectoria de 3I/ATLAS
    data = INTERSTELLAR_ORBITAL_DATA["3I/ATLAS"]
    x, y, z = compute_hyperbolic_trajectory(
        e=data["e"],
        q=data["q"],
        i=data["i"],
        n_points=400
    )

    # Trayectoria completa (muy tenue)
    ax.plot(x, y, z, color='#FFE66D', alpha=0.15, linewidth=1)

    # Objeto animado
    comet, = ax.plot([], [], [], 'o', color='#FFE66D', markersize=12,
                     markeredgecolor='white', markeredgewidth=2)

    # Estela del cometa (coma simulada)
    trail, = ax.plot([], [], [], color='#FFE66D', alpha=0.7, linewidth=3)
    tail, = ax.plot([], [], [], color='#87CEEB', alpha=0.4, linewidth=6)  # Cola azulada

    # Textos
    title = fig.suptitle("3I/ATLAS - Tercer Visitante Interestelar\n" +
                        "Descubierto: Julio 2025 | Origen: Centro Galactico (Sagitario)",
                        color='white', fontsize=14, fontweight='bold', y=0.95)

    # Panel de informacion
    info_box = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                         color='white', fontsize=10, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Creditos
    credits = ax.text2D(0.98, 0.02, "@TodoEconometria | Prof. Juan Marcelo Gutierrez Miranda",
                        transform=ax.transAxes, color='gray', fontsize=8,
                        horizontalalignment='right', verticalalignment='bottom')

    n_frames = fps * duration

    def init():
        comet.set_data([], [])
        comet.set_3d_properties([])
        trail.set_data([], [])
        trail.set_3d_properties([])
        tail.set_data([], [])
        tail.set_3d_properties([])
        info_box.set_text("")
        return [comet, trail, tail, info_box]

    def animate(frame):
        progress = frame / n_frames
        idx = int(progress * (len(x) - 1))

        # Posicion del cometa
        comet.set_data([x[idx]], [y[idx]])
        comet.set_3d_properties([z[idx]])

        # Estela (ultimos 50 puntos)
        trail_start = max(0, idx - 50)
        trail.set_data(x[trail_start:idx+1], y[trail_start:idx+1])
        trail.set_3d_properties(z[trail_start:idx+1])

        # Cola (mas larga, direccion opuesta al Sol)
        if idx > 0:
            # Vector desde Sol hacia cometa
            dx, dy, dz = x[idx], y[idx], z[idx]
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist > 0:
                # Cola apunta lejos del Sol
                tail_len = 1.5  # longitud de cola
                tail_x = [x[idx], x[idx] + (dx/dist) * tail_len]
                tail_y = [y[idx], y[idx] + (dy/dist) * tail_len]
                tail_z = [z[idx], z[idx] + (dz/dist) * tail_len]
                tail.set_data(tail_x, tail_y)
                tail.set_3d_properties(tail_z)

        # Calcular distancia al Sol
        dist_au = np.sqrt(x[idx]**2 + y[idx]**2 + z[idx]**2)

        # Determinar fase del viaje
        if progress < 0.4:
            phase = "Aproximacion"
        elif progress < 0.6:
            phase = "PERIHELIO"
        else:
            phase = "Salida"

        # Informacion actualizada
        info = (
            f"Fase: {phase}\n"
            f"Distancia al Sol: {dist_au:.2f} AU\n"
            f"Velocidad: ~{data['v_inf']:.0f} km/s\n"
            f"Excentricidad: {data['e']:.2f}\n"
            f"Edad estimada: {data['age_billion_years']} Gyr"
        )
        info_box.set_text(info)

        # Rotar camara suavemente
        ax.view_init(elev=20 + 10*np.sin(progress * np.pi),
                     azim=30 + progress * 270)

        return [comet, trail, tail, info_box]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=1000/fps, blit=False)

    print(f"    Guardando GIF ({n_frames} frames, {fps} fps) ...")
    anim.save(str(output_path), writer='pillow', fps=fps, dpi=100)
    plt.close(fig)

    print(f"    Guardado: {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
# ANIMACION 3: FORMA ALARGADA DE 'OUMUAMUA
# ═══════════════════════════════════════════════════════════════

def create_oumuamua_shape_rotation(output_path=None, fps=15, duration=8):
    """
    Animacion mostrando la forma alargada unica de 'Oumuamua rotando.
    Basado en la estimacion 10:1 de su forma.
    """
    if output_path is None:
        output_path = FIGURES_DIR / "oumuamua_shape_rotation.gif"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("  Generando animacion: Forma de 'Oumuamua rotando ...")

    fig = plt.figure(figsize=(12, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Configurar
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_facecolor('black')
    ax.set_axis_off()

    # Fondo estrellado
    generate_starfield(ax, n_stars=1000)

    # Crear forma de 'Oumuamua (elipsoide alargado 10:1:1)
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 30)

    # Dimensiones: ~200m largo, ~20m ancho
    a, b, c = 1.0, 0.1, 0.1  # Proporcion 10:1:1

    x_shape = a * np.outer(np.cos(u), np.sin(v))
    y_shape = b * np.outer(np.sin(u), np.sin(v))
    z_shape = c * np.outer(np.ones(np.size(u)), np.cos(v))

    # Color rojizo como observado
    surface = ax.plot_surface(x_shape, y_shape, z_shape,
                              color='#8B4513', alpha=0.9,
                              shade=True, lightsource=matplotlib.colors.LightSource(azdeg=45, altdeg=45))

    # Titulo
    title = fig.suptitle("1I/'Oumuamua - Forma Alargada Unica (10:1)\n" +
                        "Primer objeto interestelar confirmado | Oct 2017",
                        color='white', fontsize=14, fontweight='bold', y=0.95)

    # Info
    info = ax.text2D(0.02, 0.95,
                     "Tamano: ~200m x 20m x 20m\n"
                     "Rotacion: 7.3 horas\n"
                     "Color: Rojizo (organicos)\n"
                     "Origen: Direccion de Vega",
                     transform=ax.transAxes, color='white', fontsize=10,
                     verticalalignment='top', fontfamily='monospace')

    n_frames = fps * duration

    def animate(frame):
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_facecolor('black')
        ax.set_axis_off()

        # Re-generar estrellas (para que se vean con la rotacion)
        generate_starfield(ax, n_stars=800, seed=42)

        # Angulo de rotacion
        angle = (frame / n_frames) * 360 * 2  # 2 rotaciones completas
        angle_rad = np.radians(angle)

        # Rotar el objeto (tumbling motion como observado)
        # Rotacion alrededor del eje Y (principal)
        x_rot = x_shape * np.cos(angle_rad) + z_shape * np.sin(angle_rad)
        y_rot = y_shape
        z_rot = -x_shape * np.sin(angle_rad) + z_shape * np.cos(angle_rad)

        # Tambien rotar un poco en otro eje para efecto de tumbling
        tilt = np.radians(23 * np.sin(angle_rad * 0.5))
        x_final = x_rot
        y_final = y_rot * np.cos(tilt) - z_rot * np.sin(tilt)
        z_final = y_rot * np.sin(tilt) + z_rot * np.cos(tilt)

        ax.plot_surface(x_final, y_final, z_final,
                       color='#8B4513', alpha=0.9, shade=True,
                       lightsource=matplotlib.colors.LightSource(azdeg=45, altdeg=45))

        ax.view_init(elev=20, azim=30)

        return []

    anim = animation.FuncAnimation(fig, animate, frames=n_frames,
                                   interval=1000/fps, blit=False)

    print(f"    Guardando GIF ({n_frames} frames, {fps} fps) ...")
    anim.save(str(output_path), writer='pillow', fps=fps, dpi=100)
    plt.close(fig)

    print(f"    Guardado: {output_path}")
    return str(output_path)


# ═══════════════════════════════════════════════════════════════
# ORQUESTADOR
# ═══════════════════════════════════════════════════════════════

def run_cinematic_animations():
    """Genera todas las animaciones cinematograficas."""
    print("=" * 60)
    print("ANIMACIONES CINEMATOGRAFICAS - i3 Atlas")
    print("Los 3 Visitantes Interestelares del Sistema Solar")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gifs = []

    # 1. Comparativa de los 3 interestelares
    print("\n[1/3] Comparativa: 3 Interestelares ...")
    try:
        path = create_three_interstellar_comparison()
        gifs.append(path)
    except Exception as e:
        print(f"    Error: {e}")

    # 2. Viaje de 3I/ATLAS
    print("\n[2/3] Viaje de 3I/ATLAS ...")
    try:
        path = create_atlas_journey()
        gifs.append(path)
    except Exception as e:
        print(f"    Error: {e}")

    # 3. Forma de 'Oumuamua
    print("\n[3/3] Forma de 'Oumuamua ...")
    try:
        path = create_oumuamua_shape_rotation()
        gifs.append(path)
    except Exception as e:
        print(f"    Error: {e}")

    print("\n" + "=" * 60)
    print(f"ANIMACIONES CINEMATOGRAFICAS completadas: {len(gifs)} GIFs")
    for g in gifs:
        print(f"  - {g}")
    print("=" * 60)

    return gifs


if __name__ == "__main__":
    run_cinematic_animations()
