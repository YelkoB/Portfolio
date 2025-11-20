# Estrategia de Granularidad Dual

## Objetivo

Entrenar modelos en dos niveles de granularidad para maximizar la robustez y precisión:

1. **GRANULAR** (producto-tienda): `FOODS_3_764_CA_3`
   - Máxima precisión cuando hay datos suficientes
   - Captura patrones específicos de cada tienda

2. **AGREGADO** (producto-base): `FOODS_3_764`
   - Más robusto con datos limitados
   - Consolida patrones de múltiples tiendas

## Workflow Modificado

### Script 00 ✅ (COMPLETADO)
- Genera `sales_weekly.csv` (granular)
- Genera `sales_weekly_aggregated.csv` (agregado)
- Genera `products_list.csv` y `products_base_list.csv`

### Script 01 (EN PROGRESO)
**Opción A - Conservadora**:
- Procesa solo datos granulares (compatibilidad)
- TODO posterior: Añadir procesamiento agregado

**Opción B - Completa** (RECOMENDADA):
- Procesa ambos niveles en paralelo
- Genera outputs con sufijos:
  - `urgencias_weekly_granular.csv`
  - `urgencias_weekly_aggregated.csv`
  - `ranking_granular.csv`
  - `ranking_aggregated.csv`

### Script 02 (Feature Engineering)
- Calcular features para ambos niveles
- Outputs:
  - `features_weekly_granular.csv`
  - `features_weekly_aggregated.csv`

### Script 03 (Modelización)
- Entrenar modelos para ambos niveles
- Outputs:
  - `models_granular/` (directorio con modelos por product_id)
  - `models_aggregated/` (directorio con modelos por product_base)

### Script 04 (Validación)
- Validar modelos en ambos niveles
- Outputs:
  - `validation_metrics_granular.csv`
  - `validation_metrics_aggregated.csv`
  - `test_predictions_granular.csv`
  - `test_predictions_aggregated.csv`

### Script 05 (Análisis y Selección) ⭐
**LÓGICA INTELIGENTE DE SELECCIÓN**:

Para cada `product_base`:
1. Obtener métricas de todos los `product_id` granulares asociados
2. Obtener métricas del modelo agregado
3. Aplicar criterios de decisión:

```python
def select_best_model(product_base, granular_models, aggregated_model):
    """
    Selecciona el mejor enfoque para un producto base.

    Returns:
        - 'granular' + list of product_ids → usar modelos granulares
        - 'aggregated' → usar modelo agregado
        - 'reject' → rechazar producto (ni granular ni agregado sirven)
    """

    # Criterio 1: ¿Hay suficientes datos granulares?
    granular_data_sufficient = all(
        model.n_train >= MIN_DATA_THRESHOLD
        for model in granular_models
    )

    # Criterio 2: ¿Performance granular es buena?
    granular_performance_good = any(
        model.f1 >= TIER1_THRESHOLD
        for model in granular_models
    )

    # Criterio 3: ¿Agregado supera promedio granular?
    avg_granular_f1 = mean([m.f1 for m in granular_models])
    aggregated_better = aggregated_model.f1 > avg_granular_f1 + 0.1  # 10% mejor

    # DECISIÓN
    if granular_performance_good and granular_data_sufficient:
        return 'granular', [m.product_id for m in granular_models if m.f1 >= TIER2_THRESHOLD]
    elif aggregated_better:
        return 'aggregated', product_base
    else:
        return 'reject', None
```

**Outputs**:
- `product_selection_strategy.csv`:
  ```
  product_base,strategy,product_ids,f1_selected,tier
  FOODS_3_764,granular,"FOODS_3_764_CA_3,FOODS_3_764_TX_1",0.62,1
  FOODS_3_123,aggregated,FOODS_3_123,0.48,2
  HOBBIES_1_001,granular,HOBBIES_1_001_WI_2,0.71,1
  ```

- `products_filtered.csv` (para ROI):
  - Tier 1+2 con estrategia óptima por producto

### Script 06 (Valor Operativo)
- Usa `product_selection_strategy.csv` para cargar modelos correctos
- Calcula ROI considerando granularidad óptima

## Ventajas

1. **Robustez**: Datos insuficientes → usar agregado
2. **Precisión**: Datos suficientes → usar granular
3. **Flexibilidad**: Diferentes productos pueden usar diferentes estrategias
4. **Escalabilidad**: Fácil añadir nuevos productos

## Casos de Uso

### Caso 1: Producto popular en muchas tiendas
- `FOODS_3_764` vendido en 10 tiendas
- Cada tienda tiene 200+ semanas de datos
- **Decisión**: Usar modelos **granulares** (10 modelos, uno por tienda)
- **Razón**: Suficientes datos, máxima precisión

### Caso 2: Producto nuevo en pocas tiendas
- `HOBBIES_2_999` vendido en 2 tiendas
- Cada tienda tiene 50 semanas de datos
- **Decisión**: Usar modelo **agregado** (1 modelo consolidado)
- **Razón**: Pocos datos por tienda, agregación mejora robustez

### Caso 3: Producto con patrones mixtos
- `HOUSEHOLD_1_500` vendido en 8 tiendas
- 3 tiendas con datos suficientes y buen F1 (0.65)
- 5 tiendas con datos insuficientes
- **Decisión**: Híbrido
  - 3 tiendas con buen modelo → usar **granular**
  - 5 tiendas sin buen modelo → usar **agregado**
- **Razón**: Aprovechar lo mejor de ambos mundos

## Implementación

**Fase 1** (Actual):
- ✅ Script 00 genera ambos datasets

**Fase 2** (Next):
- Scripts 01-04 procesan ambos niveles en paralelo
- Guardan outputs separados

**Fase 3** (Final):
- Script 05 implementa lógica de selección inteligente
- Script 06 usa modelos óptimos

## Archivos Modificados

- [x] `00_setup_datos_m5.py` - Genera ambos datasets
- [ ] `01_deteccion_urgencias_predecibles.py` - Procesar ambos niveles
- [ ] `02_feature_engineering.py` - Features para ambos niveles
- [ ] `03_modelizacion.py` - Entrenar ambos niveles
- [ ] `04_validacion.py` - Validar ambos niveles
- [ ] `05_analisis_por_producto.py` - Selección inteligente
- [ ] `06_valor_operativo.py` - Usar modelos óptimos
