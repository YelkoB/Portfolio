# 📂 Estructura del Proyecto

## Organización de Carpetas

```
04-forecasting-demanda-urgente-manufactura/
│
├── 📁 code/                          # Scripts de análisis
│   ├── config.py                     # Configuración global
│   ├── 00_setup_datos_m5.py         # Fase 0A: Setup dataset M5
│   ├── 01_generar_datos_sinteticos.py    # Fase 0B: Alternativa sintética
│   └── 02_deteccion_urgencias_predecibles.py  # Fase 1: Detección urgencias
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
│       ├── 00_serie_temporal_m5.png  # (si se usa M5)
│       ├── 02_descomposicion_temporal.png
│       ├── 02_deteccion_urgencias.png
│       ├── 02_patrones_temporales_urgencias.png
│       └── 02_distribucion_urgente_vs_normal.png
│
├── README.md                         # Documentación principal
├── ESTRUCTURA.md                     # Este archivo
└── requirements.txt                  # Dependencias Python
```

## Flujo de Trabajo

### ⚙️ DOS OPCIONES DE SETUP

El proyecto soporta DOS fuentes de datos:

**OPCIÓN A: Dataset M5 (Recomendado)**
- Dataset real de Walmart (Kaggle)
- 30K productos, 1,941 días, 10 tiendas
- Ejecutar: `00_setup_datos_m5.py`
- Requiere descarga previa de Kaggle

**OPCIÓN B: Datos Sintéticos (Alternativa)**
- Datos generados con patrones predecibles
- 278 semanas, tendencia + estacionalidad
- Ejecutar: `01_generar_datos_sinteticos.py`
- No requiere descarga externa

⚠️ **Importante:** Ejecuta SOLO UNA de las dos opciones. Ambas generan `sales_weekly.csv`.

---

### 📍 Fase 0A: Setup Dataset M5 (Opción A)
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

### 📍 Fase 0B: Generación Datos Sintéticos (Opción B)
**Script:** `01_generar_datos_sinteticos.py`

**Input:** Ninguno (genera desde cero)

**Output:**
- `data/processed/sales_weekly.csv` - 278 semanas sintéticas
- `data/processed/sales_components.csv` - Componentes descompuestos

**Descripción:**
Genera datos sintéticos con patrones predecibles:
- Tendencia creciente (118% en 5 años)
- Estacionalidad anual (picos verano/navidad)
- Estacionalidad mensual (fin de mes)
- Picos predecibles controlados
- Ruido aleatorio

---

### 📍 Fase 1: Detección de Urgencias Predecibles
**Script:** `02_deteccion_urgencias_predecibles.py`

**Input:** `data/processed/sales_weekly.csv`

**Output:**
- `data/simulated/urgencias_weekly.csv` - Dataset con urgencias detectadas
- `results/figures/02_descomposicion_temporal.png`
- `results/figures/02_deteccion_urgencias.png`
- `results/figures/02_patrones_temporales_urgencias.png`
- `results/figures/02_distribucion_urgente_vs_normal.png`

**Descripción:**
Detecta urgencias usando dos criterios:
- **Criterio A:** Top 15% de ventas en ventana móvil de 12 semanas
- **Criterio B:** Crecimiento >12% vs semana anterior
- **Híbrido:** A OR B

**Resultados:**
- 83 urgencias detectadas (29.9%)
- Patrones estacionales confirmados (p < 0.0001)
- Concentración en Mar/Abr/May

---

### 📍 Próximas Fases

#### Fase 2: Feature Engineering
- Crear lags (1, 2, 4, 52 semanas)
- Rolling stats (media, std, min, max)
- Features estacionales (mes, trimestre, semana del año)
- Features de tendencia

#### Fase 3: Modelización
- ARIMA/SARIMA
- Prophet
- Random Forest
- XGBoost
- Comparación de métricas

#### Fase 4: Validación
- Train/Val/Test split temporal
- Validación sin data leakage
- Métricas de clasificación (urgente vs normal)

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

### OPCIÓN A: Con Dataset M5

#### 1. Descargar M5 de Kaggle
Descarga y coloca en `data/raw/`:
- sales_train_evaluation.csv
- calendar.csv
- sell_prices.csv

#### 2. Procesar M5
```bash
cd 04-forecasting-demanda-urgente-manufactura
python code/00_setup_datos_m5.py
```

#### 3. Detectar Urgencias
```bash
python code/02_deteccion_urgencias_predecibles.py
```

---

### OPCIÓN B: Con Datos Sintéticos

#### 1. Generar Datos
```bash
cd 04-forecasting-demanda-urgente-manufactura
python code/01_generar_datos_sinteticos.py
```

#### 2. Detectar Urgencias
```bash
python code/02_deteccion_urgencias_predecibles.py
```

---

### Verificar Outputs
```bash
ls data/processed/           # Ver datos generados
ls data/simulated/           # Ver urgencias detectadas
ls results/figures/          # Ver visualizaciones
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
