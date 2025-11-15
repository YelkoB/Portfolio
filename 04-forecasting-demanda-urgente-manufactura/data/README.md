# 📁 Estructura de Datos

## Organización de Carpetas

### `/raw/`
**Datos originales sin procesar**
- `ventas_historicas.csv` - Datos de ventas originales
- Fuente: [Especificar origen de datos]
- Período: [Especificar rango temporal]
- No modificar estos archivos

### `/processed/`
**Datos procesados y limpios**
- `ventas_weekly.csv` - Ventas agregadas semanalmente
- `ventas_with_features.csv` - Dataset con features temporales
- `train_set.csv` - Conjunto de entrenamiento (80%)
- `val_set.csv` - Conjunto de validación (10%)
- `test_set.csv` - Conjunto de prueba (10%)

### `/simulated/`
**Urgencias predecibles detectadas**
- `urgencias_weekly.csv` - Dataset con urgencias detectadas (criterios A y B)
- Configuración:
  - Criterio A: Percentil 85 móvil (ventana 12 semanas)
  - Criterio B: Crecimiento >12% semanal
  - Híbrido: A OR B

## Esquema de Datos

### Dataset Procesado (weekly)
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `week_id` | int | Identificador de semana (YYWW) |
| `week_start` | datetime | Fecha inicio de semana |
| `week_num` | int | Número de semana secuencial (0-277) |
| `year` | int | Año |
| `month` | int | Mes (1-12) |
| `quarter` | int | Trimestre (1-4) |
| `week_of_year` | int | Semana del año (1-52) |
| `week_of_month` | int | Semana del mes (1-5) |
| `total_sales` | int | Ventas totales semanales (unidades) |
| `total_revenue` | float | Revenue total ($) |
| `avg_price` | float | Precio promedio unitario |

### Dataset con Urgencias (simulated/urgencias_weekly.csv)
| Campo | Tipo | Descripción |
|-------|------|-------------|
| *(todos los campos anteriores)* | - | Campos base del dataset procesado |
| `percentile_threshold` | float | Umbral percentil móvil (P85 en 12w) |
| `growth_rate` | float | Tasa de crecimiento semanal |
| `urgent_criterio_a` | int | Urgencia detectada por Criterio A (0/1) |
| `urgent_criterio_b` | int | Urgencia detectada por Criterio B (0/1) |
| `is_urgent` | int | Urgencia final (A OR B) (0/1) |

### Features Temporales
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `lag_1` | float | Ventas semana anterior |
| `lag_2` | float | Ventas hace 2 semanas |
| `lag_4` | float | Ventas hace 4 semanas |
| `ma_4` | float | Media móvil 4 semanas |
| `trend` | float | Tendencia lineal |
| `seasonality` | float | Componente estacional |

## Validación de Datos

### Checks Implementados
- ✅ No hay valores nulos en campos críticos
- ✅ Fechas son consecutivas sin gaps
- ✅ Valores numéricos están en rangos esperados
- ✅ Splits temporales sin data leakage

### Estadísticas Descriptivas

**Dataset generado (sales_weekly.csv):**
- Total semanas: 278 (~5.3 años)
- Período: 2011-01-01 a 2016-04-23
- Ventas promedio: 291,233 unidades/semana
- Desviación estándar: 70,292 unidades
- Coef. variación: 24.1%

**Urgencias detectadas (urgencias_weekly.csv):**
- Total urgencias: 83 semanas (29.9%)
- Solo Criterio A: 76 semanas
- Solo Criterio B: 2 semanas
- Ambos criterios: 5 semanas

```python
# Cargar datos procesados
import pandas as pd
df = pd.read_csv('simulated/urgencias_weekly.csv', parse_dates=['week_start'])
print(f"Urgencias: {df['is_urgent'].sum()} ({df['is_urgent'].mean()*100:.1f}%)")
```

## Notas Importantes

⚠️ **Data Leakage Prevention:**
- Nunca usar datos futuros para features
- Features solo con información disponible en t-1
- Validación estrictamente temporal

⚠️ **Reproducibilidad:**
- Todos los procesos con `random_seed=42`
- Documentar cada transformación en notebooks
- Mantener raw/ sin modificaciones
