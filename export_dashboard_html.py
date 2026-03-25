"""
DASHBOARD HTML - i3 Atlas
==========================
Dashboard interactivo autocontenido con:
- Filtros para explorar sin superposicion
- Animaciones cinematograficas embebidas (GIFs)
- Seccion destacada para los 3 objetos interestelares
- Comparativa ML vs DL completa

ECOSISTEMA:
- Storage: output/ (primario)
- GPU: RTX 4060 via WSL2 + TensorFlow 2.20
- Cluster: Spark 3.5.4 (Docker)

Autor: Juan Marcelo Gutierrez Miranda | @TodoEconometria

REFERENCIAS:
- Meech, K. J., et al. (2017). A brief visit from a red and extremely elongated
  interstellar asteroid. Nature, 552, 378-381.
- Jewitt, D., et al. (2019). Initial characterization of interstellar comet
  2I/Borisov. ApJ, 886, L29.
- Tonry, J. L., et al. (2018). ATLAS: A High-cadence All-sky Survey System.
  PASP, 130(988), 064505.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from config import (
    PROCESSED_PARQUET, DATA_CACHE_DIR, OUTPUT_DIR, FIGURES_DIR, DASHBOARD_OUTPUT,
    COLORS, OBJECT_CLASSES, INTERSTELLAR_OBJECTS, INTERSTELLAR_ORBITAL_DATA,
)


# =====================================================================
# CARGA DE DATOS
# =====================================================================

def load_all_data():
    """Carga todos los datos y resultados necesarios."""
    data = {}

    if PROCESSED_PARQUET.exists():
        data["bodies"] = pd.read_parquet(str(PROCESSED_PARQUET))
    else:
        print("  [WARN] Parquet no encontrado.")
        data["bodies"] = pd.DataFrame()

    eph_file = DATA_CACHE_DIR / "interstellar_ephemeris.csv"
    if eph_file.exists():
        data["ephemeris"] = pd.read_csv(str(eph_file), parse_dates=["datetime"])
    else:
        data["ephemeris"] = pd.DataFrame()

    for name in ["ml_results", "dl_results", "benchmark_results"]:
        fpath = OUTPUT_DIR / f"{name}.json"
        if fpath.exists():
            try:
                with open(str(fpath), "r", encoding="utf-8") as f:
                    data[name] = json.load(f)
            except json.JSONDecodeError:
                data[name] = {}
        else:
            data[name] = {}

    return data


def load_gif_as_base64(gif_path):
    """Carga un GIF como base64 para embeber en HTML."""
    if Path(gif_path).exists():
        with open(gif_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


# =====================================================================
# PESTANA 1: LOS 3 INTERESTELARES - FOCO PRINCIPAL
# =====================================================================

def fig_three_interstellar(data):
    """Visualizacion detallada de los 3 objetos interestelares."""

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "scatter3d", "colspan": 2}, None],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
        subplot_titles=[
            "Trayectorias Hiperbolicas de los 3 Interestelares",
            "Velocidad en el Infinito (km/s)",
            "Distancia al Perihelio (AU)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        row_heights=[0.6, 0.4],
    )

    colors_iso = {
        "1I/'Oumuamua": "#FF6B6B",
        "2I/Borisov": "#4ECDC4",
        "3I/ATLAS": "#FFE66D",
    }

    theta = np.linspace(-np.pi/2, np.pi/2, 100)

    for name, orbital in INTERSTELLAR_ORBITAL_DATA.items():
        e = orbital["e"]
        a = abs(orbital["a"])
        i_rad = np.radians(orbital["i"])

        r = a * (e**2 - 1) / (1 + e * np.cos(theta))
        r = np.clip(r, 0, 50)

        x = r * np.cos(theta)
        y = r * np.sin(theta) * np.cos(i_rad)
        z = r * np.sin(theta) * np.sin(i_rad)

        color = colors_iso.get(name, "#888")

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=color, width=4),
            name=name,
            hovertemplate=f"<b>{name}</b><br>e={e:.2f}<br>q={orbital['q']:.2f} AU<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter3d(
            x=[orbital["q"]], y=[0], z=[0],
            mode="markers",
            marker=dict(size=10, color=color, symbol="diamond"),
            name=f"{name} perihelio",
            showlegend=False,
        ), row=1, col=1)

    # Sol
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers",
        marker=dict(size=15, color="#FFD700", symbol="circle"),
        name="Sol",
    ), row=1, col=1)

    # Velocidades
    names = list(INTERSTELLAR_ORBITAL_DATA.keys())
    v_inf = [INTERSTELLAR_ORBITAL_DATA[n]["v_inf"] for n in names]
    bar_colors = [colors_iso.get(n, "#888") for n in names]

    fig.add_trace(go.Bar(
        x=names, y=v_inf,
        marker_color=bar_colors,
        text=[f"{v:.1f}" for v in v_inf],
        textposition="outside",
        showlegend=False,
    ), row=2, col=1)

    # Scatter perihelio vs excentricidad
    q_vals = [INTERSTELLAR_ORBITAL_DATA[n]["q"] for n in names]
    e_vals = [INTERSTELLAR_ORBITAL_DATA[n]["e"] for n in names]

    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=[q_vals[i]], y=[e_vals[i]],
            mode="markers+text",
            marker=dict(size=20, color=bar_colors[i], symbol="star"),
            text=[name.split("/")[0]],
            textposition="top center",
            name=name,
            showlegend=False,
        ), row=2, col=2)

    fig.update_xaxes(title_text="Perihelio (AU)", row=2, col=2)
    fig.update_yaxes(title_text="Excentricidad", row=2, col=2)

    fig.update_layout(
        height=800,
        template="plotly_dark",
        title=dict(
            text="<b>Los 3 Visitantes Interestelares del Sistema Solar</b>",
            font=dict(size=20, color="#FFE66D"),
        ),
        scene=dict(
            xaxis=dict(title="X (AU)", range=[-20, 20]),
            yaxis=dict(title="Y (AU)", range=[-20, 20]),
            zaxis=dict(title="Z (AU)", range=[-20, 20]),
            bgcolor="#0a0a0a",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


# =====================================================================
# PESTANA 2: EXPLORACION 3D CON FILTROS
# =====================================================================

def fig_exploration_filtered(data):
    """Scatter 3D con filtros por tipo de objeto."""
    df = data["bodies"]
    if len(df) == 0:
        return go.Figure().update_layout(title="Sin datos", template="plotly_dark")

    fig = go.Figure()
    obj_types = df["object_type"].unique().tolist()

    for otype in obj_types:
        subset = df[df["object_type"] == otype]
        sample = subset.sample(n=min(3000, len(subset)), random_state=42) if len(subset) > 3000 else subset

        fig.add_trace(go.Scatter3d(
            x=sample["e"], y=sample["a"], z=sample["i"],
            mode="markers",
            marker=dict(size=4, color=COLORS.get(otype, "#888888"), opacity=0.6),
            name=f"{otype} ({len(subset):,})",
            text=sample["full_name"],
            hovertemplate="<b>%{text}</b><br>e=%{x:.3f}<br>a=%{y:.2f} AU<br>i=%{z:.1f}<extra></extra>",
            visible=True,
        ))

    # Interestelares siempre visibles y destacados
    iso = df[df["object_type"] == "ISO"]
    if len(iso) > 0:
        fig.add_trace(go.Scatter3d(
            x=iso["e"], y=iso["a"], z=iso["i"],
            mode="markers+text",
            marker=dict(size=15, color="#e91e63", symbol="diamond",
                        line=dict(width=3, color="white")),
            text=iso["full_name"],
            textposition="top center",
            textfont=dict(size=12, color="white"),
            name="INTERESTELARES",
            hovertemplate="<b>%{text}</b><br>INTERESTELAR<extra></extra>",
        ))

    # Botones de filtro
    buttons = []
    vis_iso = [False] * len(obj_types) + [True]
    buttons.append(dict(label="Solo Interestelares", method="update", args=[{"visible": vis_iso}]))

    vis_all = [True] * (len(obj_types) + 1)
    buttons.append(dict(label="Mostrar Todo", method="update", args=[{"visible": vis_all}]))

    for i, otype in enumerate(obj_types):
        vis = [False] * len(obj_types) + [True]
        vis[i] = True
        buttons.append(dict(label=otype, method="update", args=[{"visible": vis}]))

    fig.update_layout(
        updatemenus=[dict(
            type="dropdown", direction="down",
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            buttons=buttons,
            bgcolor="#1e1e1e", font=dict(color="white"), bordercolor="#444",
        )],
        height=700,
        template="plotly_dark",
        title=dict(text="Espacio Orbital 3D - Usa el filtro para explorar", font=dict(size=18)),
        scene=dict(
            xaxis=dict(title="Excentricidad (e)", gridcolor="#333"),
            yaxis=dict(title="Semieje Mayor (AU)", gridcolor="#333"),
            zaxis=dict(title="Inclinacion", gridcolor="#333"),
            bgcolor="#0d1117",
        ),
        paper_bgcolor="#0d1117",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(30,30,30,0.8)"),
    )
    return fig


# =====================================================================
# PESTANA 3: ML CLASIFICACION
# =====================================================================

def fig_ml_classification(data):
    """Resultados de clasificacion ML."""
    ml = data.get("ml_results", {})
    task1 = ml.get("task1", {})
    results = task1.get("results", {})

    if not results:
        return go.Figure().update_layout(
            title="ML: Sin resultados de clasificacion",
            template="plotly_dark", paper_bgcolor="#0d1117",
        )

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Metricas por Modelo", "Tiempo de Entrenamiento"])

    models = list(results.keys())
    for i, metric in enumerate(["accuracy", "f1", "precision", "recall"]):
        values = [results[m].get(metric, 0) for m in models]
        colors_met = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
        fig.add_trace(go.Bar(
            name=metric.capitalize(), x=models, y=values,
            marker_color=colors_met[i],
        ), row=1, col=1)

    train_times = [results[m].get("train_time", 0) for m in models]
    fig.add_trace(go.Bar(
        name="Train Time", x=models, y=train_times,
        marker_color="#9b59b6",
        text=[f"{t:.2f}s" for t in train_times],
        textposition="outside",
    ), row=1, col=2)

    fig.update_layout(
        height=500, template="plotly_dark",
        title="ML Tradicional: Clasificacion de Objetos Astronomicos",
        barmode="group", paper_bgcolor="#0d1117",
    )
    return fig


# =====================================================================
# PESTANA 4: DL CLASIFICACION
# =====================================================================

def fig_dl_classification(data):
    """Resultados DL: loss curves + metricas."""
    dl = data.get("dl_results", {})
    task1 = dl.get("task1", {})

    if not task1:
        return go.Figure().update_layout(
            title="DL: Sin resultados", template="plotly_dark", paper_bgcolor="#0d1117",
        )

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Training Curves (DNN)", "Accuracy: ML vs DL"])

    history = task1.get("history", {})
    if history.get("loss"):
        epochs = list(range(1, len(history["loss"]) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history["loss"],
            name="Train Loss", line=dict(color="#FF5722"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epochs, y=history.get("val_loss", []),
            name="Val Loss", line=dict(color="#2196F3", dash="dash"),
        ), row=1, col=1)

    ml = data.get("ml_results", {})
    ml_results = ml.get("task1", {}).get("results", {})

    all_models = {m: {"acc": r.get("accuracy", 0), "tipo": "ML"} for m, r in ml_results.items()}
    all_models["DNN"] = {"acc": task1.get("accuracy") or 0, "tipo": "DL"}

    names = list(all_models.keys())
    accs = [all_models[m]["acc"] for m in names]
    colors = [COLORS["ml"] if all_models[m]["tipo"] == "ML" else COLORS["dl"] for m in names]

    fig.add_trace(go.Bar(
        x=names, y=accs, marker_color=colors,
        text=[f"{a:.3f}" for a in accs],
        textposition="outside", showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        height=500, template="plotly_dark",
        title=f"Deep Learning: DNN ({task1.get('device', 'CPU')}) | Accuracy={task1.get('accuracy', 0):.4f}",
        paper_bgcolor="#0d1117",
    )
    return fig


# =====================================================================
# PESTANA 5: ANOMALIAS ML
# =====================================================================

def fig_anomaly_ml(data):
    """PCA 3D + Isolation Forest."""
    ml = data.get("ml_results", {})
    task2 = ml.get("task2", {})
    pca_data = task2.get("pca_coords", [])

    if not pca_data:
        return go.Figure().update_layout(
            title="ML Anomalias: Sin datos PCA",
            template="plotly_dark", paper_bgcolor="#0d1117",
        )

    df_pca = pd.DataFrame(pca_data)

    fig = px.scatter_3d(
        df_pca, x="pc1", y="pc2", z="pc3",
        color="object_type",
        hover_name="full_name",
        color_discrete_map=COLORS,
        opacity=0.4,
        title="Isolation Forest: Deteccion de Anomalias (PCA 3D)",
    )

    iso = df_pca[df_pca["object_type"] == "ISO"]
    if len(iso) > 0:
        fig.add_trace(go.Scatter3d(
            x=iso["pc1"], y=iso["pc2"], z=iso["pc3"],
            mode="markers+text",
            marker=dict(size=15, color="#e91e63", symbol="diamond"),
            text=iso["full_name"],
            textposition="top center",
            name="INTERESTELARES",
        ))

    fig.update_layout(height=650, template="plotly_dark", paper_bgcolor="#0d1117")
    return fig


# =====================================================================
# PESTANA 6: ANOMALIAS DL
# =====================================================================

def fig_anomaly_dl(data):
    """Autoencoder reconstruction error."""
    dl = data.get("dl_results", {})
    task2 = dl.get("task2", {})

    if not task2:
        return go.Figure().update_layout(
            title="DL Anomalias: Sin resultados",
            template="plotly_dark", paper_bgcolor="#0d1117",
        )

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Autoencoder Training Loss", "Interestelares Detectados"])

    history = task2.get("history", {})
    if history.get("loss"):
        epochs = list(range(1, len(history["loss"]) + 1))
        fig.add_trace(go.Scatter(
            x=epochs, y=history["loss"],
            name="Train MSE", line=dict(color="#FF5722"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=epochs, y=history.get("val_loss", []),
            name="Val MSE", line=dict(color="#2196F3", dash="dash"),
        ), row=1, col=1)

    iso_errors = task2.get("interstellar_errors", [])
    threshold = task2.get("threshold", 0)

    if iso_errors:
        names = [e.get("full_name", "?") for e in iso_errors]
        errors = [e.get("ae_recon_error", 0) for e in iso_errors]
        detected = [e.get("ae_anomaly", 0) for e in iso_errors]
        bar_colors = ["#e91e63" if d else "#666" for d in detected]

        fig.add_trace(go.Bar(
            x=names, y=errors, marker_color=bar_colors,
            text=["SI DETECTADO" if d else "no" for d in detected],
            textposition="outside", showlegend=False,
        ), row=1, col=2)

        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Umbral={threshold:.4f}", row=1, col=2)

    fig.update_layout(
        height=500, template="plotly_dark",
        title=f"Autoencoder: Deteccion de Anomalias | {task2.get('n_anomalies', 0)} detectadas",
        paper_bgcolor="#0d1117",
    )
    return fig


# =====================================================================
# PESTANA 7: BENCHMARK
# =====================================================================

def fig_benchmark(data):
    """Benchmark global ML vs DL."""
    bench = data.get("benchmark_results", {})
    table = bench.get("comparison_table", [])

    if not table:
        return go.Figure().update_layout(
            title="Benchmark: Sin datos",
            template="plotly_dark", paper_bgcolor="#0d1117",
        )

    df = pd.DataFrame(table)
    clf = df[df["accuracy"] > 0]

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Precision vs Tiempo", "Tiempo Total"])

    if len(clf) > 0:
        for _, row in clf.iterrows():
            color = COLORS["ml"] if row["enfoque"] == "ML" else COLORS["dl"]
            symbol = "circle" if row["enfoque"] == "ML" else "triangle-up"
            fig.add_trace(go.Scatter(
                x=[row["train_time"]], y=[row["accuracy"]],
                mode="markers+text",
                marker=dict(size=18, color=color, symbol=symbol),
                text=[row["modelo"]], textposition="top center",
                name=row["modelo"], showlegend=False,
            ), row=1, col=1)

    ml_time = bench.get("ml_total_time", 0)
    dl_time = bench.get("dl_total_time", 0)
    fig.add_trace(go.Bar(
        x=["ML Total", "DL Total"],
        y=[ml_time, dl_time],
        marker_color=[COLORS["ml"], COLORS["dl"]],
        text=[f"{ml_time:.1f}s", f"{dl_time:.1f}s"],
        textposition="outside", showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        height=500, template="plotly_dark",
        title=f"Benchmark: ML vs DL | GPU: {'SI' if bench.get('dl_gpu') else 'NO'}",
        paper_bgcolor="#0d1117",
    )
    return fig


# =====================================================================
# HTML BUILDER CON GIFS EMBEBIDOS
# =====================================================================

def build_html(figures, gifs_base64):
    """Construir HTML con pestanas, filtros y GIFs embebidos."""

    tab_names = [
        "Interestelares",
        "Exploracion 3D",
        "Animaciones",
        "ML: Clasificacion",
        "DL: Clasificacion",
        "Anomalias ML",
        "Anomalias DL",
        "Benchmark",
    ]

    tabs_html = ""
    divs_html = ""

    # Figuras (excluye Animaciones que es especial)
    fig_indices = [0, 1, 3, 4, 5, 6, 7]  # Skip index 2 (Animaciones)
    for tab_idx, fig_idx in enumerate(fig_indices):
        if tab_idx >= len(figures):
            break
        name = tab_names[tab_idx if tab_idx < 2 else tab_idx + 1]
        real_idx = tab_idx if tab_idx < 2 else tab_idx + 1
        active_class = "active" if real_idx == 0 else ""
        display = "block" if real_idx == 0 else "none"
        fig_html = figures[tab_idx].to_html(full_html=False, include_plotlyjs=(real_idx == 0))
        tabs_html += f'<button class="tab-btn {active_class}" onclick="showTab({real_idx})">{name}</button>\n'
        divs_html += f'<div class="tab-content" id="tab-{real_idx}" style="display:{display}">{fig_html}</div>\n'

    # Pestana de Animaciones (index 2)
    tabs_html = tabs_html.replace(
        '<button class="tab-btn " onclick="showTab(3)">',
        '<button class="tab-btn " onclick="showTab(2)">Animaciones</button>\n<button class="tab-btn " onclick="showTab(3)">'
    )

    animations_html = '<div class="tab-content" id="tab-2" style="display:none">'
    animations_html += '<div class="animations-grid">'

    gif_info = [
        ("three_interstellar", "Los 3 Visitantes Interestelares",
         "Comparativa de las trayectorias hiperbolicas de 1I/'Oumuamua, 2I/Borisov y 3I/ATLAS"),
        ("3i_atlas_journey", "El Viaje de 3I/ATLAS",
         "El objeto interestelar mas reciente atravesando el Sistema Solar a 220,000 km/h"),
        ("oumuamua_shape", "'Oumuamua: Forma Unica",
         "Rotacion del misterioso objeto con proporcion 10:1"),
    ]

    for key, title, desc in gif_info:
        b64 = gifs_base64.get(key, "")
        if b64:
            animations_html += f'''
            <div class="gif-card">
                <h3>{title}</h3>
                <img src="data:image/gif;base64,{b64}" alt="{title}">
                <p>{desc}</p>
            </div>
            '''
        else:
            animations_html += f'''
            <div class="gif-card">
                <h3>{title}</h3>
                <div class="gif-placeholder">GIF no disponible - ejecutar animation_cinematic.py</div>
                <p>{desc}</p>
            </div>
            '''

    animations_html += '</div></div>'

    # Insertar animaciones en la posicion correcta
    insert_pos = divs_html.find('<div class="tab-content" id="tab-3"')
    if insert_pos > 0:
        divs_html = divs_html[:insert_pos] + animations_html + divs_html[insert_pos:]

    # Info de interestelares para el header
    iso_info = ""
    for name, orbital in INTERSTELLAR_ORBITAL_DATA.items():
        iso_info += f'''
        <div class="iso-card">
            <h4>{name}</h4>
            <p><strong>Descubierto:</strong> {orbital["discovery_date"]}</p>
            <p><strong>Perihelio:</strong> {orbital["q"]:.2f} AU</p>
            <p><strong>Velocidad:</strong> {orbital["v_inf"]:.1f} km/s</p>
            <p><strong>Origen:</strong> {orbital["origin"]}</p>
        </div>
        '''

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>i3 Atlas - Los Visitantes Interestelares</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', -apple-system, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
        }}
        .header {{
            background: linear-gradient(135deg, #0d1b2a 0%, #1b0a2e 50%, #2d1b4a 100%);
            color: white;
            padding: 2.5rem 2rem;
            text-align: center;
            border-bottom: 3px solid #e91e63;
            position: relative;
            overflow: hidden;
        }}
        .header::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="30" r="1" fill="white" opacity="0.3"/><circle cx="60" cy="10" r="0.5" fill="white" opacity="0.5"/><circle cx="80" cy="50" r="1.5" fill="white" opacity="0.2"/><circle cx="10" cy="70" r="0.8" fill="white" opacity="0.4"/><circle cx="90" cy="80" r="1" fill="white" opacity="0.3"/></svg>');
            opacity: 0.5;
        }}
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #FFE66D, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            position: relative;
            z-index: 1;
        }}
        .header p {{ opacity: 0.9; font-size: 1rem; position: relative; z-index: 1; }}
        .iso-summary {{
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
            position: relative;
            z-index: 1;
        }}
        .iso-card {{
            background: rgba(255,255,255,0.1);
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            min-width: 220px;
            backdrop-filter: blur(10px);
        }}
        .iso-card h4 {{ color: #FFE66D; margin-bottom: 0.5rem; font-size: 1.1rem; }}
        .iso-card p {{ font-size: 0.82rem; margin: 0.2rem 0; opacity: 0.9; }}
        .tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            padding: 1rem 2rem 0;
            background: #161b22;
            border-bottom: 2px solid #30363d;
        }}
        .tab-btn {{
            padding: 0.8rem 1.5rem;
            border: 1px solid #30363d;
            background: #21262d;
            color: #8b949e;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 600;
            border-radius: 10px 10px 0 0;
            transition: all 0.2s;
        }}
        .tab-btn:hover {{ background: #30363d; color: #c9d1d9; transform: translateY(-2px); }}
        .tab-btn.active {{
            background: linear-gradient(135deg, #e91e63, #ff5722);
            color: white;
            border-color: #e91e63;
        }}
        .content {{ padding: 1rem 2rem 2rem; }}
        .animations-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            padding: 1rem;
        }}
        .gif-card {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
        }}
        .gif-card h3 {{ color: #FFE66D; margin-bottom: 1rem; font-size: 1.2rem; }}
        .gif-card img {{ max-width: 100%; border-radius: 8px; border: 2px solid #30363d; }}
        .gif-card p {{ margin-top: 1rem; font-size: 0.9rem; color: #8b949e; }}
        .gif-placeholder {{
            background: #0d1117;
            padding: 4rem 2rem;
            border-radius: 8px;
            color: #666;
        }}
        .intro {{
            background: linear-gradient(135deg, #161b22, #1a1f2e);
            padding: 1.5rem 2rem;
            margin: 1rem 2rem 0;
            border-radius: 12px;
            border: 1px solid #30363d;
            font-size: 0.95rem;
            line-height: 1.7;
        }}
        .intro strong {{ color: #e91e63; }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #8b949e;
            font-size: 0.85rem;
            border-top: 1px solid #30363d;
            margin-top: 2rem;
            background: #161b22;
        }}
        .footer a {{ color: #58a6ff; text-decoration: none; }}
        .ref {{
            background: #161b22;
            padding: 1.5rem 2rem;
            margin: 1rem 2rem;
            border-radius: 12px;
            border: 1px solid #30363d;
            font-size: 0.85rem;
            line-height: 1.8;
        }}
        .ref strong {{ color: #bc8cff; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>i3 Atlas: Los Visitantes Interestelares</h1>
        <p>Analisis con Machine Learning y Deep Learning de los 3 objetos que han visitado nuestro Sistema Solar</p>
        <p style="margin-top:0.5rem; font-size:0.85rem;">
            Prof. Juan Marcelo Gutierrez Miranda - @TodoEconometria
        </p>
        <div class="iso-summary">
            {iso_info}
        </div>
    </div>

    <div class="intro">
        <strong>Proyecto i3 Atlas:</strong> Comparativa directa entre <strong>Machine Learning</strong> (Random Forest, SVM, XGBoost, Isolation Forest)
        y <strong>Deep Learning</strong> (DNN, Autoencoder, 1D-CNN) aplicado al catalogo astronomico JPL Small-Body Database.
        Procesado con <strong>Apache Spark</strong> (cluster Docker) y Deep Learning acelerado con <strong>GPU RTX 4060</strong> via WSL2.
        <br><br>
        <strong>Los 3 Interestelares:</strong> 1I/'Oumuamua (2017), 2I/Borisov (2019) y 3I/ATLAS (2025) son los unicos objetos confirmados
        que provienen de fuera de nuestro Sistema Solar. Sus trayectorias hiperbolicas (e > 1) los distinguen de cualquier otro objeto conocido.
    </div>

    <div class="tabs">
        {tabs_html}
    </div>
    <div class="content">
        {divs_html}
    </div>

    <div class="ref">
        <strong>Curso:</strong> Big Data con Python - De Cero a Produccion<br>
        <strong>Profesor:</strong> Juan Marcelo Gutierrez Miranda | @TodoEconometria<br>
        <strong>Referencias academicas:</strong><br>
        - Meech, K. J., et al. (2017). A brief visit from a red and extremely elongated interstellar asteroid. Nature, 552, 378-381.<br>
        - Jewitt, D., et al. (2019). Initial characterization of interstellar comet 2I/Borisov. ApJ, 886, L29.<br>
        - Tonry, J. L., et al. (2018). ATLAS: A High-cadence All-sky Survey System. PASP, 130(988).<br>
        - Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.<br>
        - Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
    </div>

    <div class="footer">
        <p>i3 Atlas - Big Data con Python | <a href="https://github.com/TodoEconometria/i3altlas-todoeconometria">@TodoEconometria</a></p>
        <p>Dashboard generado con Plotly | Animaciones con Matplotlib | Datos: JPL + SDSS</p>
        <p style="margin-top:0.5rem; font-size:0.75rem; color:#666;">
            Storage: output | GPU: RTX 4060 via WSL2
        </p>
    </div>

    <script>
    function showTab(idx) {{
        document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-' + idx).style.display = 'block';
        document.querySelectorAll('.tab-btn')[idx].classList.add('active');
        window.dispatchEvent(new Event('resize'));
    }}
    </script>
</body>
</html>"""
    return html


# =====================================================================
# MAIN
# =====================================================================

def run_export_dashboard():
    """Genera el dashboard HTML completo con GIFs embebidos."""
    print("=" * 65)
    print("EXPORTANDO DASHBOARD HTML - i3 Atlas")
    print("=" * 65)
    print(f"Storage: {OUTPUT_DIR}")
    print(f"GIFs: {FIGURES_DIR}")

    data = load_all_data()

    # Cargar GIFs
    print("\n[1/8] Cargando animaciones GIF ...")
    gifs = {}
    gif_files = {
        "three_interstellar": FIGURES_DIR / "three_interstellar_cinematic.gif",
        "3i_atlas_journey": FIGURES_DIR / "3i_atlas_journey_cinematic.gif",
        "oumuamua_shape": FIGURES_DIR / "oumuamua_shape_rotation.gif",
    }
    for key, path in gif_files.items():
        b64 = load_gif_as_base64(path)
        if b64:
            print(f"    [OK] {path.name} ({len(b64)//1024} KB)")
            gifs[key] = b64
        else:
            print(f"    [--] {path.name} no encontrado")

    print("[2/8] Los 3 Interestelares ...")
    f1 = fig_three_interstellar(data)

    print("[3/8] Exploracion 3D con filtros ...")
    f2 = fig_exploration_filtered(data)

    print("[4/8] ML: Clasificacion ...")
    f3 = fig_ml_classification(data)

    print("[5/8] DL: Clasificacion ...")
    f4 = fig_dl_classification(data)

    print("[6/8] Anomalias ML ...")
    f5 = fig_anomaly_ml(data)

    print("[7/8] Anomalias DL ...")
    f6 = fig_anomaly_dl(data)

    print("[8/8] Benchmark Global ...")
    f7 = fig_benchmark(data)

    print("\nConstruyendo HTML con GIFs embebidos ...")
    figures = [f1, f2, f3, f4, f5, f6, f7]
    html = build_html(figures, gifs)

    DASHBOARD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_OUTPUT.write_text(html, encoding="utf-8")

    size_kb = DASHBOARD_OUTPUT.stat().st_size / 1024
    size_mb = size_kb / 1024

    print(f"\n{'='*65}")
    print(f"[OK] Dashboard guardado: {DASHBOARD_OUTPUT}")
    print(f"     Tamano: {size_mb:.1f} MB ({size_kb:.0f} KB)")
    print(f"     GIFs embebidos: {len(gifs)}")
    print(f"     Pestanas: 8 (incluye Animaciones)")
    print(f"{'='*65}")

    return str(DASHBOARD_OUTPUT)


if __name__ == "__main__":
    run_export_dashboard()
