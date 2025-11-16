# 📂 Estructura del Proyecto

## Organización de Carpetas

```
04-forecasting-demanda-urgente-manufactura/
│
├── 📁 code/                          # Scripts de análisis
│   ├── config.py                     # Configuración global
│   ├── 00_setup_datos_m5.py         # Fase 0: Setup dataset M5
│   └── 01_deteccion_urgencias_predecibles.py  # Fase 1: Detección urgencias
│
├── 📁 data/                          # Datos del proyecto
│   ├── processed/                    # Datos procesados
│   │   ├── sales_weekly.csv         # Ventas semanales agregadas
│   │   └── sales_components.csv     # Componentes de la serie temporal
│   └── simulated/                   # Datos con urgencias detectadas
│       └── urgencias_weekly.csv     # Dataset final con urgencias
│
├── 📁 results/                       # Resultados del análisis
│   └── figures/                     # Visualizaciones
│       ├── 00_serie_temporal_m5_top5.png
│       ├── 01_descomposicion_temporal.png
│       ├── 01_deteccion_urgencias.png
│       ├── 01_patrones_temporales_urgencias.png
│       └── 01_distribucion_urgente_vs_normal.png
│
├── README.md                         # Documentación principal
├── ESTRUCTURA.md                     # Este archivo
└── requirements.txt                  # Dependencias Python
```

## Flujo de Trabajo

### 📍 Fase 0: Setup Dataset M5
**Script:** `00_setup_datos_m5.py`

**Input (externo - NO incluido):**
- `data/raw/sales_train_evaluation.csv` (60 MB)
- `data/raw/calendar.csv` (1 MB)
- `data/raw/sell_prices.csv` (145 MB)

**Output:**
- `data/processed/sales_weekly.csv` - Ventas semanales M5
- `results/figures/00_serie_temporal_m5.png`

**Descripción:**
Procesa el dataset M5 de Kaggle:
- Carga 30,490 productos × 1,941 días
- Transforma formato ancho → largo
- Agrega a nivel semanal
- Merge con calendario y precios

**Descarga M5:**
https://www.kaggle.com/c/m5-forecasting-accuracy/data

---

### 📍 Fase 1: Detección de Urgencias Predecibles
**Script:** `01_deteccion_urgencias_predecibles.py`

**Input:** `data/processed/sales_weekly.csv`

**Output:**
- `data/simulated/urgencias_weekly.csv` - Dataset con urgencias detectadas
- `data/simulated/products_predictability_ranking.csv` - Ranking de productos
- `results/figures/01_descomposicion_temporal.png`
- `results/figures/01_deteccion_urgencias.png`
- `results/figures/01_patrones_temporales_urgencias.png`
- `results/figures/01_distribucion_urgente_vs_normal.png`

**Descripción:**
Detecta urgencias POR PRODUCTO usando dos criterios:
- **Criterio A:** Top 15% de ventas en ventana móvil de 12 semanas
- **Criterio B:** Crecimiento >12% vs semana anterior
- **Híbrido:** A OR B

**Proceso:**
1. Carga datos multi-producto de Fase 0
2. Por cada producto: detecta urgencias con criterios A y B
3. Calcula "score de predictibilidad" por producto
4. Genera ranking de productos más predecibles
5. Selecciona TOP 25 productos para análisis profundo

**Métricas de predictibilidad:**
- Concentración temporal (Chi-cuadrado)
- Estacionalidad (amplitud estacional)
- Correlación con calendario
- Proporción de urgencias detectadas

---

### 📍 Fase 2: Feature Engineering

**Script:** `02_feature_engineering.py`

**Input:**
- `data/simulated/urgencias_weekly.csv` - Dataset con urgencias detectadas
- `data/simulated/products_predictability_ranking.csv` - Ranking de productos

**Output:**
- `data/simulated/features_weekly.csv` - Dataset con features para TOP 25
- `data/simulated/feature_list.json` - Lista de features por tipo
- `results/figures/02_correlation_matrix.png`
- `results/figures/02_feature_distributions.png`
- `results/figures/02_time_series_features.png`

**Descripción:**
Crea variables predictivas temporales para los TOP 25 productos más predecibles.

**Features creados:**
1. **Lags:** ventas en t-1, t-2, t-4, t-52 (año anterior)
2. **Rolling stats:** media, std, min, max en ventanas 4, 12, 52 semanas
3. **Tendencia:** índice temporal, diferencias, aceleración
4. **Estacionales:** mes (1-12), trimestre (1-4), semana del mes (1-5) one-hot
5. **Urgencias pasadas:** lags de urgencias, conteos en ventanas móviles
6. **Ratios:** ratio respecto medias, distancias min/max, coeficiente de variación

**Proceso:**
1. Carga urgencias detectadas de Fase 1
2. Filtra TOP 25 productos del ranking
3. Para cada producto: crea ~80 features temporales
4. Analiza correlaciones con target `is_urgent`
5. Genera visualizaciones de features clave
6. Guarda dataset final listo para modelización

---

### 📍 Fase 3: Modelización

**Script:** `03_modelizacion.py`

**Input:**
- `data/simulated/features_weekly.csv` - Dataset con features
- `data/simulated/feature_list.json` - Lista de features

**Output:**
- `data/simulated/model_results.csv` - Resultados por modelo y producto
- `data/simulated/best_models.csv` - Mejor modelo por producto
- `models/*.pkl` - Modelos entrenados guardados
- `results/figures/03_regression_comparison.png`
- `results/figures/03_classification_comparison.png`

**Descripción:**
Entrena y compara múltiples modelos de Machine Learning para:
1. **Regresión:** Predecir ventas futuras (RMSE, MAE, MAPE)
2. **Clasificación:** Predecir urgencias futuras (Precision, Recall, F1, AUC)

**Modelos evaluados:**
- **Random Forest:** Ensemble de árboles de decisión
- **XGBoost:** Gradient boosting optimizado

**Proceso:**
1. Train/Val/Test split temporal (70/15/15) - sin data leakage
2. Entrenar cada modelo en TOP 25 productos
3. Evaluar en validation set
4. Guardar modelos entrenados (.pkl)
5. Comparar métricas entre modelos
6. Seleccionar mejor modelo por producto

---

### 📍 Próximas Fases

#### Fase 4: Validación
- Evaluación en test set
- Análisis de errores y feature importance
- Métricas de clasificación (urgente vs normal)
- Predicciones futuras

#### Fase 5: Valor Operativo
- ROI de predicción
- Costos evitados
- Métricas de negocio

---

## Convenciones de Nombres

### Scripts
- Formato: `{número}_{nombre_descriptivo}.py`
- Número corresponde a la fase (00, 01, 02, ...)
- Snake_case para nombres

### Datos
- `sales_*.csv` - Datos de ventas
- `urgencias_*.csv` - Datos con urgencias
- `*_components.csv` - Componentes descompuestos

### Figuras
- Formato: `{fase}_{nombre_descriptivo}.png`
- Fase corresponde al script que la genera
- Ejemplo: `01_deteccion_urgencias.png` (generada por script 01)

---

## Cómo Ejecutar

### 1. Descargar Dataset M5 de Kaggle
Descarga los archivos de https://www.kaggle.com/c/m5-forecasting-accuracy/data

Coloca en `data/raw/`:
- `sales_train_evaluation.csv` (~60 MB)
- `calendar.csv` (~1 MB)
- `sell_prices.csv` (~145 MB)

### 2. Procesar Dataset M5
```bash
cd 04-forecasting-demanda-urgente-manufactura
python code/00_setup_datos_m5.py
```

**Output:**
- `data/processed/sales_weekly.csv` - Ventas semanales multi-producto
- `data/processed/products_list.csv` - Lista de productos
- `results/figures/00_serie_temporal_m5_top5.png`

### 3. Detectar Urgencias y Rankear Productos
```bash
python code/01_deteccion_urgencias_predecibles.py
```

**Output:**
- `data/simulated/urgencias_weekly.csv` - Urgencias por producto
- `data/simulated/products_predictability_ranking.csv` - Ranking TOP 25
- 4 figuras de análisis

### 4. Verificar Outputs
```bash
ls data/processed/           # Datos procesados M5
ls data/simulated/           # Urgencias y ranking
ls results/figures/          # Visualizaciones
```

---

## Estado Actual

✅ **Completado:**
- Fase 0A: Setup M5 (script listo, requiere descarga externa)
- Fase 0B: Generación de datos sintéticos
- Fase 1: Detección de urgencias predecibles

⏳ **Pendiente:**
- Fase 2: Feature Engineering
- Fase 3: Modelización
- Fase 4: Validación
- Fase 5: Valor Operativo

---

## Notas Importantes

⚠️ **Reproducibilidad:**
- Todos los scripts usan `RANDOM_SEED = 42`
- Los resultados son determinísticos

⚠️ **Data Leakage:**
- No usar información del futuro
- Validación estrictamente temporal
- Features solo con datos de t-1

⚠️ **Datos Sintéticos:**
- Los datos NO son reales, son generados
- Diseñados para demostrar concepto
- Patrones controlados para validación
