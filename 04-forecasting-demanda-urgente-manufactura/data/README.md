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
**Urgencias sintéticas generadas**
- `urgencias_synthetic.csv` - Urgencias simuladas con ground truth
- `urgencias_features.csv` - Urgencias con features para modelado
- Configuración: threshold=1.5, proportion=0.30

## Esquema de Datos

### Ventas Históricas (raw)
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `date` | datetime | Fecha de la transacción |
| `product_id` | string | Identificador del producto |
| `quantity` | int | Cantidad vendida |
| `is_urgent` | bool | Flag de pedido urgente |
| `lead_time` | int | Días de entrega |

### Dataset Procesado (weekly)
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `week_num` | int | Número de semana (índice) |
| `week_start` | datetime | Fecha inicio de semana |
| `total_sales` | float | Ventas totales semanales |
| `urgent_sales` | float | Ventas urgentes semanales |
| `urgent_ratio` | float | Proporción de urgencias |

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
> 🔄 *Se actualizará después de Fase 1*

```python
# Cargar datos procesados
import pandas as pd
df = pd.read_csv('processed/ventas_weekly.csv', parse_dates=['week_start'])
df.info()
df.describe()
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
