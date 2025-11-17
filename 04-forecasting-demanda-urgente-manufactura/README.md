# ⚙️ Forecasting de Demanda Urgente en Manufactura

## 📢 Actualizaciones Recientes (2025-11-16)

### ✨ Mejoras Implementadas:

1. **🔧 SMAPE en lugar de MAPE**
   - Problema resuelto: MAPE explotaba a 138-154% con valores cercanos a 0
   - Implementado SMAPE (Symmetric MAPE) robusto a divisiones por cero
   - Más info: `code/03_modelizacion.py`

2. **📈 Entrenamiento en TODOS los productos M5**
   - Antes: Solo TOP 25 productos
   - Ahora: TOP_N = None → Entrena en los **2,862 productos completos**
   - Permite identificar qué productos son más predecibles
   - Config en: `code/02_feature_engineering.py` línea 117

3. **🎄 Features de Holidays USA**
   - 6 features adicionales: Thanksgiving, Christmas, Black Friday, Summer, etc.
   - Captura patrones estacionales de retail USA
   - Implementado en: `code/02_feature_engineering.py`

4. **🎯 Resultados con M5 Real**
   - **AUC 0.675** con RandomForest ✅ (supera umbral 0.60)
   - 33% mejor que con datos sintéticos (0.507 → 0.675)
   - Ver: `EXPERIMENTO.md` para análisis completo

### 📁 Archivos Eliminados:
- ~~`00_generar_multiproducto.py`~~ (ya no necesario con M5 real)

---

## 🎯 Objetivo
Desarrollar un sistema predictivo multi-modelo para anticipar pedidos urgentes en entorno manufacturero, optimizando capacidad productiva mediante forecasting de series temporales y reduciendo variabilidad operativa.

## 📂 Estructura del Proyecto

### Datos
- **[📁 data/raw/](./data/raw/)** - Datos originales de ventas
- **[📁 data/processed/](./data/processed/)** - Datos procesados y agregados
- **[📁 data/simulated/](./data/simulated/)** - Urgencias sintéticas simuladas

### Análisis
- **[📓 code/](./code/)** - Scripts de análisis por fase:
  - `config.py` - Configuración global del proyecto
  - `00_setup_datos_m5.py` - Carga y procesamiento dataset M5 (Kaggle)
  - `01_deteccion_urgencias_predecibles.py` - Detección y validación de urgencias
  - *(Próximas fases: feature engineering, modelización, validación)*

### Resultados
- **[📊 results/](./results/)** - Outputs y métricas de modelos
- **[📈 results/figures/](./results/figures/)** - Visualizaciones
- **[💼 output/](./output/)** - Memoria ejecutiva y reportes

### 📦 Manejo de Datos
> ⚠️ **Nota importante:** Los archivos CSV de datos no están incluidos en el repositorio por su tamaño.

**Los datos se mantienen localmente en:**
- `data/raw/*.csv` - Datos originales (excluidos de Git)
- `data/processed/*.csv` - Datos procesados (excluidos de Git)
- `data/simulated/*.csv` - Urgencias sintéticas (excluidos de Git)

**Configuración:**
- Los archivos CSV están incluidos en `.gitignore` para evitar problemas de tamaño
- El código espera encontrar los archivos en las carpetas correspondientes localmente
- La documentación del esquema de datos está en [data/README.md](./data/README.md)
- Para reproducir el análisis, coloca tus archivos CSV en `data/raw/`

## 🔬 Metodología

### Concepto Clave
**"Urgencias Predecibles"**: El comprador percibe ciertos pedidos como urgencias impredecibles, pero en realidad siguen patrones estacionales y de tendencia que SÍ son predecibles mediante análisis temporal.

### Criterios de Detección de Urgencias
- **Criterio A (Percentil Móvil)**: Top 15% de ventas en ventana de 12 semanas
- **Criterio B (Crecimiento Acelerado)**: Crecimiento >12% vs semana anterior
- **Criterio Híbrido**: Combinación de ambos (A OR B)

### Fases de Desarrollo
| Fase | Script | Objetivo | Estado |
|------|--------|----------|--------|
| **0. Setup M5** | `00_setup_datos_m5.py` | Carga y procesamiento dataset M5 (Kaggle) | ✅ Completado |
| **1. Detección + EDA** | `01_deteccion_urgencias_predecibles.py` | Detectar urgencias predecibles y validar patrones | ✅ Completado |
| **2. Feature Engineering** | `02_feature_engineering.py` | Crear variables predictivas temporales | ✅ Completado |
| **3. Modelización** | `03_modelizacion.py` | Comparación multi-modelo (RF, XGBoost) | ✅ Completado |
| **4. Validación** | `04_validacion.py` | Evaluación en test set y feature importance | ✅ Completado |
| **5. Valor Operativo** | `05_valor_operativo.py` | Cuantificación de ROI y métricas de negocio | ✅ Completado |

### Modelos Evaluados
- **ARIMA/SARIMA** - Baseline estadístico para series temporales
- **Prophet** - Detección automática de estacionalidad y tendencias
- **Negative Binomial GLM** - Modelización de counts con sobredispersión
- **Random Forest** - Ensemble con features temporales
- **XGBoost** - Gradient boosting optimizado

### Métricas de Evaluación
**Accuracy:**
- RMSE, MAE - Error absoluto
- MAPE - Error porcentual
- R² - Varianza explicada

**Negocio:**
- Precision/Recall en urgencias
- Cost-sensitive metrics
- Impacto en capacidad productiva

## 💡 Configuración del Proyecto

### Variables Clave
```python
# Detección de urgencias predecibles
URGENCY_PERCENTILE = 85  # Top 15% en ventana móvil
PERCENTILE_WINDOW = 12  # 12 semanas (~3 meses)
URGENCY_GROWTH_THRESHOLD = 0.12  # 12% crecimiento semanal
USE_HYBRID_CRITERIA = True  # Criterio A OR B

# Modelización
TRAIN_RATIO = 0.80
FORECAST_HORIZON = [1, 2, 4]  # semanas
AGGREGATION = 'weekly'
RANDOM_SEED = 42
```

### Principios de Ejecución
- ✅ Reproducibilidad total (seeds fijados)
- ✅ Validación sin data leakage
- ✅ Documentación inline rigurosa
- ✅ Enfoque orientado a valor de negocio

## 🛠️ Tecnologías
`Python` • `Pandas` • `Scikit-learn` • `Statsmodels` • `Prophet` • `XGBoost` • `Matplotlib` • `Seaborn`

## 📈 Principales Hallazgos

### Fase 1: Detección de Urgencias Predecibles ✅
✅ **Urgencias detectadas: 83 semanas (29.9%)**
- Solo Criterio A (percentil): 76 semanas
- Solo Criterio B (crecimiento): 2 semanas
- Ambos criterios: 5 semanas

✅ **Patrones estacionales confirmados:**
- Concentración en meses: Marzo, Abril, Mayo (primavera)
- Concentración en semanas: 3ra y 4ta semana del mes
- Test Chi-cuadrado: p < 0.0001 (NO distribución uniforme)

✅ **Conclusión validada:**
Las urgencias percibidas como impredecibles SÍ muestran patrones predecibles mediante análisis temporal

## 🔗 Aplicaciones Potenciales
- **Manufactura:** Optimización de capacidad y planificación de producción
- **Supply Chain:** Gestión proactiva de inventario y recursos
- **Operaciones:** Reducción de variabilidad y costos de urgencias
- **Analytics:** Framework replicable para forecasting multi-modelo

## 📚 Referencias
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice*
- Taylor, S. J., & Letham, B. (2018). *Forecasting at scale (Prophet)*
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*
