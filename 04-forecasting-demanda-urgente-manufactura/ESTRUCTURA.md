# 📂 Estructura del Proyecto

## Organización de Carpetas

```
04-forecasting-demanda-urgente-manufactura/
│
├── 📁 code/                          # Scripts de análisis
│   ├── config.py                     # Configuración global
│   ├── 00_generar_datos_sinteticos.py    # Fase 0: Generación de datos
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

### 📍 Fase 0: Generación de Datos
**Script:** `00_generar_datos_sinteticos.py`

**Input:** Ninguno (genera datos desde cero)

**Output:**
- `data/processed/sales_weekly.csv` - 278 semanas de datos
- `data/processed/sales_components.csv` - Componentes (tendencia, estacionalidad, etc.)

**Descripción:**
Genera datos sintéticos de ventas con:
- Tendencia creciente (118% en 5 años)
- Estacionalidad anual (picos en verano/navidad)
- Estacionalidad mensual (fin de mes)
- Picos predecibles en meses/semanas específicas
- Ruido aleatorio

---

### 📍 Fase 1: Detección de Urgencias Predecibles
**Script:** `01_deteccion_urgencias_predecibles.py`

**Input:** `data/processed/sales_weekly.csv`

**Output:**
- `data/simulated/urgencias_weekly.csv` - Dataset con urgencias detectadas
- 4 visualizaciones en `results/figures/`

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

### 1. Generar Datos
```bash
cd 04-forecasting-demanda-urgente-manufactura
python code/00_generar_datos_sinteticos.py
```

### 2. Detectar Urgencias
```bash
python code/01_deteccion_urgencias_predecibles.py
```

### 3. Verificar Outputs
```bash
ls data/processed/           # Ver datos generados
ls data/simulated/           # Ver urgencias detectadas
ls results/figures/          # Ver visualizaciones
```

---

## Estado Actual

✅ **Completado:**
- Fase 0: Generación de datos
- Fase 1: Detección de urgencias

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
