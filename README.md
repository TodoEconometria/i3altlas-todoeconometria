# i3 Atlas: Deep Learning vs Machine Learning

## Analisis Comparativo sobre Datos Astronomicos Reales con Apache Spark

Ejercicio avanzado sin entrega - Comparativa directa entre enfoques de Deep Learning
y Machine Learning tradicional para el analisis del catalogo astronomico i3 Atlas,
utilizando un entorno distribuido Spark para procesamiento Big Data y visualizaciones
interactivas.

## Tabla Comparativa

| Aspecto | Deep Learning | ML Tradicional |
|---------|---------------|----------------|
| **Procesamiento** | End-to-end, redes neuronales | Extraccion manual de features |
| **Precision** | Alta para tareas complejas | Limitada por features manuales |
| **Requisitos** | GPU, mas datos, mas tiempo | CPU, menos recursos |
| **Interpretabilidad** | Caja negra (limitada) | Alta (feature importance) |
| **Escalabilidad** | Escala con datos y compute | Limitada por algoritmo |
| **Caso ideal** | Deteccion de anomalias complejas | Clasificacion con features claras |

## Arquitectura

```

├── config.py                    # Configuracion central (Spark, GPU, parametros)
├── main.py                      # Orquestador: ejecuta todo el pipeline
├── data_acquisition.py          # Descarga datos reales (JPL, SDSS)
├── spark_etl.py                 # ETL distribuido con Apache Spark
├── ml_traditional.py            # ML: Random Forest, SVM, XGBoost, Isolation Forest
├── deep_learning_pipeline.py    # DL: DNN, Autoencoder, 1D-CNN (TensorFlow/Keras)
├── benchmark_comparison.py      # Metricas comparativas ML vs DL
├── export_dashboard_html.py     # Dashboard Plotly HTML (8 pestanas)
├── requirements_i3atlas.txt     # Dependencias
├── README.md                    # Este archivo
└── output/
    ├── datos/                   # Cache de datos descargados
    ├── figuras/                 # Graficos generados
    ├── i3atlas_procesado.parquet
    ├── ml_results.json
    ├── dl_results.json
    ├── benchmark_results.json
    └── dashboard_i3atlas_ml_vs_dl.html
```

## Fuentes de Datos Reales

| Fuente | Datos | API |
|--------|-------|-----|
| **JPL Small-Body Database** | 50,000+ asteroides/cometas: elementos orbitales + propiedades fisicas | `ssd-api.jpl.nasa.gov` |
| **JPL Horizons** | Efemerides de 'Oumuamua y Borisov (trayectoria post-perihelio) | `astroquery.jplhorizons` |
| **SDSS Galaxy Colors** | 50,000 galaxias: colores u-g, g-r, r-i, i-z + redshift | `astroML.datasets` |

## Pipeline (6 pasos)

```
PASO 1/6: Adquisicion de Datos      -> APIs JPL + SDSS
PASO 2/6: ETL con Spark             -> Limpieza + Feature Engineering
PASO 3/6: ML Tradicional            -> RF, SVM, XGBoost, Isolation Forest
PASO 4/6: Deep Learning             -> DNN, Autoencoder, 1D-CNN (GPU)
PASO 5/6: Benchmark Comparativo     -> Metricas, graficos, recomendaciones
PASO 6/6: Dashboard HTML            -> 8 pestanas Plotly interactivas
```

## Ejecucion

### Prerequisitos

1. Cluster Spark activo:
   ```bash
   cd docker
   docker compose -f docker-compose-spark.yml up -d
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements_i3atlas.txt
   ```

3. (Opcional) Para GPU - en WSL2:
   ```bash
   source ~/tf-gpu-env/bin/activate
   ```

### Ejecutar pipeline completo

```bash
python main.py
```

### Ejecutar modulos individuales

```bash
python data_acquisition.py     # Solo descargar datos
python spark_etl.py            # Solo ETL
python ml_traditional.py       # Solo ML
python deep_learning_pipeline.py  # Solo DL
python benchmark_comparison.py # Solo benchmark
python export_dashboard_html.py   # Solo dashboard
```

## Dashboard (8 pestanas)

| Pestana | Contenido |
|---------|-----------|
| **Exploracion 3D** | Scatter 3D de orbitas (e vs a vs i), objetos interestelares destacados |
| **ML: Clasificacion** | Metricas RF, SVM, XGBoost + feature importance |
| **DL: Clasificacion** | Loss curves DNN + comparativa directa con ML |
| **Anomalias ML** | PCA 3D + Isolation Forest, interestelares marcados |
| **Anomalias DL** | Autoencoder reconstruction error, deteccion de anomalias |
| **Espectral** | RF vs 1D-CNN en clasificacion espectral SDSS |
| **Benchmark** | Graficos comparativos globales: precision vs tiempo |
| **i3 Atlas** | Trayectorias post-perihelio de objetos interestelares |

## Entorno Tecnico

| Componente | Version |
|------------|---------|
| Apache Spark | 3.5.4 (cluster Docker: master + 2 workers) |
| TensorFlow | 2.20.0 (CUDA 12.x bundled) |
| GPU | NVIDIA RTX 4060 (8 GB VRAM) |
| scikit-learn | 1.2+ |
| XGBoost | 1.7+ |
| Python | 3.9+ |
| RAM | 64 GB |
| CPU | Intel i9 |

## Feature Engineering Astronomico

| Feature | Formula | Significado |
|---------|---------|-------------|
| `tisserand_j` | T_J = a_J/a + 2cos(i)sqrt(a/a_J * (1-e^2)) | Parametro de Tisserand respecto a Jupiter |
| `v_inf` | sqrt(\|1/a\|) * 29.78 km/s | Velocidad en el infinito (hiperbolicas) |
| `energy_param` | -1/(2a) | Energia orbital especifica |
| `q_over_a` | q/a | Ratio perihelio/semieje (circularidad) |

---

**Curso:** Big Data con Python - De Cero a Produccion
**Profesor:** Juan Marcelo Gutierrez Miranda | @TodoEconometria
**Metodologia:** Ejercicios progresivos con datos reales y herramientas profesionales

**Referencias academicas:**
- Tonry, J. L., et al. (2018). ATLAS: A High-cadence All-sky Survey System. Publications of the Astronomical Society of the Pacific, 130(988), 064505.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
- Zaharia, M., Xin, R. S., Wendell, P., et al. (2016). Apache Spark: A Unified Engine for Big Data Processing. Communications of the ACM, 59(11), 56-65.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of KDD '16, 785-794.
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. IEEE International Conference on Data Mining, 413-422.
- Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR 2014.
- Murray, C. D., & Dermott, S. F. (1999). Solar System Dynamics. Cambridge University Press.
