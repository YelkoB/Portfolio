"""
05. Análisis por Producto - Selección de Nivel Óptimo
=======================================================

OBJETIVO:
Decidir qué nivel de granularidad usar para cada producto:
- GRANULAR (producto-tienda): mejor para productos con alta variabilidad entre tiendas
- AGREGADO (producto-base): mejor para productos con comportamiento homogéneo

ESTRATEGIA:
1. Comparar métricas entre granular y agregado por producto base
2. Seleccionar nivel óptimo basándose en F1 (clasificación) y RMSE (regresión)
3. Generar mapping producto → nivel óptimo
4. Análisis de cuándo usar cada nivel

INPUT:
- data/simulated/validation_metrics_granular.csv
- data/simulated/validation_metrics_aggregated.csv
- data/simulated/test_predictions_granular.csv
- data/simulated/test_predictions_aggregated.csv

OUTPUT:
- data/simulated/product_level_selection.csv - Nivel óptimo por producto
- data/simulated/level_comparison_by_product.csv - Comparación detallada
- results/figures/05_*.png - Visualizaciones
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Importar configuración
from config import (
    DATA_SIMULATED, FIGURES, PROJECT_ROOT,
    FIGSIZE_STANDARD, FIGSIZE_WIDE,
    COLORS, RANDOM_SEED
)

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

print("="*80)
print("ANÁLISIS POR PRODUCTO - SELECCIÓN DE NIVEL ÓPTIMO")
print("="*80)
print()

# ============================================================================
# 1. CARGAR DATOS DE VALIDACIÓN
# ============================================================================
print("1. CARGANDO MÉTRICAS DE VALIDACIÓN")
print("-" * 80)

# Métricas GRANULAR
metrics_granular = pd.read_csv(DATA_SIMULATED / 'validation_metrics_granular.csv')
print(f"✅ Métricas GRANULAR: {metrics_granular.shape}")
print(f"   Productos únicos: {metrics_granular['product_id'].nunique()}")

# Métricas AGREGADO
metrics_aggregated = pd.read_csv(DATA_SIMULATED / 'validation_metrics_aggregated.csv')
print(f"✅ Métricas AGREGADO: {metrics_aggregated.shape}")
print(f"   Productos únicos: {metrics_aggregated['product_base'].nunique()}")

print()

# ============================================================================
# 2. EXTRAER PRODUCT_BASE DE PRODUCT_ID GRANULAR
# ============================================================================
print("2. EXTRAYENDO PRODUCT_BASE DE PRODUCT_ID GRANULAR")
print("-" * 80)

# product_id formato: FOODS_3_764_CA_3
# product_base formato: FOODS_3_764 (quitar últimos 2 componentes)

def extract_product_base(product_id):
    """Extraer product_base desde product_id granular"""
    parts = product_id.rsplit('_', 2)  # Split desde el final
    return parts[0]

metrics_granular['product_base'] = metrics_granular['product_id'].apply(extract_product_base)

print(f"✅ Product_base extraído de product_id granular")
print(f"   Ejemplo: {metrics_granular['product_id'].iloc[0]} → {metrics_granular['product_base'].iloc[0]}")
print()

# ============================================================================
# 3. AGREGAR MÉTRICAS GRANULARES POR PRODUCT_BASE
# ============================================================================
print("3. AGREGANDO MÉTRICAS GRANULARES POR PRODUCT_BASE")
print("-" * 80)

# Separar regresión y clasificación
granular_reg = metrics_granular[metrics_granular['task'] == 'regression'].copy()
granular_clf = metrics_granular[metrics_granular['task'] == 'classification'].copy()

# Agregar a nivel de product_base (promedio de todas las tiendas)
granular_reg_agg = granular_reg.groupby('product_base').agg({
    'rmse': 'mean',
    'mae': 'mean',
    'mape': 'mean',
    'product_id': 'count'  # Número de tiendas
}).rename(columns={'product_id': 'num_stores'})

granular_clf_agg = granular_clf.groupby('product_base').agg({
    'precision': 'mean',
    'recall': 'mean',
    'f1': 'mean',
    'auc': 'mean'
})

print(f"✅ REGRESIÓN granular agregada: {len(granular_reg_agg)} productos base")
print(f"   Promedio tiendas por producto: {granular_reg_agg['num_stores'].mean():.1f}")
print()
print(f"✅ CLASIFICACIÓN granular agregada: {len(granular_clf_agg)} productos base")
print()

# ============================================================================
# 4. PREPARAR MÉTRICAS AGREGADAS
# ============================================================================
print("4. PREPARANDO MÉTRICAS AGREGADAS")
print("-" * 80)

aggregated_reg = metrics_aggregated[metrics_aggregated['task'] == 'regression'].copy()
aggregated_clf = metrics_aggregated[metrics_aggregated['task'] == 'classification'].copy()

aggregated_reg = aggregated_reg.set_index('product_base')[['rmse', 'mae', 'mape']]
aggregated_clf = aggregated_clf.set_index('product_base')[['precision', 'recall', 'f1', 'auc']]

print(f"✅ REGRESIÓN agregada: {len(aggregated_reg)} productos base")
print(f"✅ CLASIFICACIÓN agregada: {len(aggregated_clf)} productos base")
print()

# ============================================================================
# 5. COMPARACIÓN POR PRODUCTO BASE
# ============================================================================
print("5. COMPARACIÓN GRANULAR vs AGREGADO POR PRODUCTO")
print("-" * 80)

# Merge regresión
comparison_reg = pd.DataFrame({
    'rmse_granular': granular_reg_agg['rmse'],
    'rmse_aggregated': aggregated_reg['rmse'],
    'mae_granular': granular_reg_agg['mae'],
    'mae_aggregated': aggregated_reg['mae'],
    'mape_granular': granular_reg_agg['mape'],
    'mape_aggregated': aggregated_reg['mape'],
    'num_stores': granular_reg_agg['num_stores']
})

# Merge clasificación
comparison_clf = pd.DataFrame({
    'f1_granular': granular_clf_agg['f1'],
    'f1_aggregated': aggregated_clf['f1'],
    'precision_granular': granular_clf_agg['precision'],
    'precision_aggregated': aggregated_clf['precision'],
    'recall_granular': granular_clf_agg['recall'],
    'recall_aggregated': aggregated_clf['recall'],
    'auc_granular': granular_clf_agg['auc'],
    'auc_aggregated': aggregated_clf['auc']
})

# Merge ambos
comparison = comparison_reg.join(comparison_clf, how='outer')

print(f"✅ Comparación creada: {len(comparison)} productos base")
print()

# ============================================================================
# 6. SELECCIONAR NIVEL ÓPTIMO POR PRODUCTO
# ============================================================================
print("6. SELECCIONANDO NIVEL ÓPTIMO POR PRODUCTO")
print("-" * 80)

selection = []

for product_base in comparison.index:
    row = comparison.loc[product_base]

    # Decisión REGRESIÓN: menor RMSE
    best_regression = 'granular' if row['rmse_granular'] <= row['rmse_aggregated'] else 'aggregated'
    rmse_diff = abs(row['rmse_granular'] - row['rmse_aggregated'])
    rmse_improvement = (rmse_diff / max(row['rmse_granular'], row['rmse_aggregated'])) * 100

    # Decisión CLASIFICACIÓN: mayor F1
    best_classification = 'granular' if row['f1_granular'] >= row['f1_aggregated'] else 'aggregated'
    f1_diff = abs(row['f1_granular'] - row['f1_aggregated'])
    f1_improvement = (f1_diff / max(row['f1_granular'], row['f1_aggregated'], 0.001)) * 100

    # Decisión FINAL: Priorizar clasificación (más importante para urgencias)
    # pero considerar regresión si la diferencia en F1 es mínima
    if f1_diff < 0.05:  # Diferencia F1 < 0.05 → usar mejor regresión
        best_overall = best_regression
        reason = 'regression_priority'
    else:
        best_overall = best_classification
        reason = 'classification_priority'

    selection.append({
        'product_base': product_base,
        'best_overall': best_overall,
        'best_regression': best_regression,
        'best_classification': best_classification,
        'reason': reason,
        'num_stores': row['num_stores'],
        'rmse_granular': row['rmse_granular'],
        'rmse_aggregated': row['rmse_aggregated'],
        'rmse_improvement_pct': rmse_improvement,
        'f1_granular': row['f1_granular'],
        'f1_aggregated': row['f1_aggregated'],
        'f1_improvement_pct': f1_improvement
    })

df_selection = pd.DataFrame(selection)

print(f"✅ Selección completada para {len(df_selection)} productos")
print()

# Estadísticas
print("DISTRIBUCIÓN DE SELECCIÓN:")
print(df_selection['best_overall'].value_counts())
print()

print("RAZONES DE SELECCIÓN:")
print(df_selection['reason'].value_counts())
print()

# ============================================================================
# 7. ANÁLISIS DE RESULTADOS
# ============================================================================
print("7. ANÁLISIS DE RESULTADOS")
print("-" * 80)

# ¿Cuántos productos se benefician de cada nivel?
n_granular = (df_selection['best_overall'] == 'granular').sum()
n_aggregated = (df_selection['best_overall'] == 'aggregated').sum()
pct_granular = (n_granular / len(df_selection)) * 100
pct_aggregated = (n_aggregated / len(df_selection)) * 100

print(f"NIVEL ÓPTIMO:")
print(f"  • GRANULAR:  {n_granular:4d} productos ({pct_granular:5.1f}%)")
print(f"  • AGREGADO:  {n_aggregated:4d} productos ({pct_aggregated:5.1f}%)")
print()

# Mejora promedio
print(f"MEJORA PROMEDIO (usando nivel óptimo):")
print(f"  • RMSE: {df_selection['rmse_improvement_pct'].mean():.2f}%")
print(f"  • F1:   {df_selection['f1_improvement_pct'].mean():.2f}%")
print()

# Relación con número de tiendas
df_granular_sel = df_selection[df_selection['best_overall'] == 'granular']
df_aggregated_sel = df_selection[df_selection['best_overall'] == 'aggregated']

print(f"ANÁLISIS POR NÚMERO DE TIENDAS:")
if len(df_granular_sel) > 0:
    print(f"  • Productos GRANULAR - tiendas promedio: {df_granular_sel['num_stores'].mean():.1f}")
if len(df_aggregated_sel) > 0:
    print(f"  • Productos AGREGADO - tiendas promedio: {df_aggregated_sel['num_stores'].mean():.1f}")
print()

# ============================================================================
# 8. GUARDAR RESULTADOS
# ============================================================================
print("8. GUARDANDO RESULTADOS")
print("-" * 80)

# Selección final
selection_file = DATA_SIMULATED / 'product_level_selection.csv'
df_selection.to_csv(selection_file, index=False)
print(f"✓ Selección guardada: {selection_file}")

# Comparación completa
comparison_file = DATA_SIMULATED / 'level_comparison_by_product.csv'
comparison.to_csv(comparison_file)
print(f"✓ Comparación guardada: {comparison_file}")
print()

# ============================================================================
# 9. VISUALIZACIONES
# ============================================================================
print("9. VISUALIZACIONES")
print("-" * 80)

# A. Distribución de selección
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Pie chart
df_selection['best_overall'].value_counts().plot(
    kind='pie', ax=axes[0], autopct='%1.1f%%',
    colors=[COLORS['primary'], COLORS['secondary']],
    labels=['Granular', 'Agregado']
)
axes[0].set_title('Distribución de Nivel Óptimo por Producto', fontsize=12, fontweight='bold')
axes[0].set_ylabel('')

# Bar chart por razón
reason_counts = df_selection.groupby(['reason', 'best_overall']).size().unstack(fill_value=0)
reason_counts.plot(kind='bar', ax=axes[1], color=[COLORS['primary'], COLORS['secondary']])
axes[1].set_title('Razón de Selección', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Razón')
axes[1].set_ylabel('Número de Productos')
axes[1].legend(title='Nivel Seleccionado', labels=['Agregado', 'Granular'])
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES / '05_level_selection_distribution.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '05_level_selection_distribution.png'}")
plt.close()

# B. RMSE Granular vs Agregado
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot coloreado por selección
colors_map = df_selection['best_overall'].map({'granular': COLORS['primary'], 'aggregated': COLORS['secondary']})
ax.scatter(comparison['rmse_granular'], comparison['rmse_aggregated'],
          c=colors_map, alpha=0.6, s=30)

# Línea diagonal (empate)
max_rmse = max(comparison['rmse_granular'].max(), comparison['rmse_aggregated'].max())
ax.plot([0, max_rmse], [0, max_rmse], 'k--', linewidth=2, alpha=0.5, label='Empate')

ax.set_xlabel('RMSE Granular', fontsize=11)
ax.set_ylabel('RMSE Agregado', fontsize=11)
ax.set_title('Comparación RMSE: Granular vs Agregado', fontsize=12, fontweight='bold')
ax.legend(['Empate', 'Granular óptimo', 'Agregado óptimo'])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / '05_rmse_comparison.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '05_rmse_comparison.png'}")
plt.close()

# C. F1 Granular vs Agregado
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(comparison['f1_granular'], comparison['f1_aggregated'],
          c=colors_map, alpha=0.6, s=30)

# Línea diagonal (empate)
max_f1 = max(comparison['f1_granular'].max(), comparison['f1_aggregated'].max())
ax.plot([0, max_f1], [0, max_f1], 'k--', linewidth=2, alpha=0.5, label='Empate')

ax.set_xlabel('F1 Granular', fontsize=11)
ax.set_ylabel('F1 Agregado', fontsize=11)
ax.set_title('Comparación F1: Granular vs Agregado', fontsize=12, fontweight='bold')
ax.legend(['Empate', 'Granular óptimo', 'Agregado óptimo'])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / '05_f1_comparison.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '05_f1_comparison.png'}")
plt.close()

# D. Mejora por número de tiendas
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# RMSE improvement vs num_stores
axes[0].scatter(df_selection['num_stores'], df_selection['rmse_improvement_pct'],
               c=colors_map, alpha=0.6, s=30)
axes[0].set_xlabel('Número de Tiendas', fontsize=11)
axes[0].set_ylabel('Mejora RMSE (%)', fontsize=11)
axes[0].set_title('Mejora RMSE vs Número de Tiendas', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# F1 improvement vs num_stores
axes[1].scatter(df_selection['num_stores'], df_selection['f1_improvement_pct'],
               c=colors_map, alpha=0.6, s=30)
axes[1].set_xlabel('Número de Tiendas', fontsize=11)
axes[1].set_ylabel('Mejora F1 (%)', fontsize=11)
axes[1].set_title('Mejora F1 vs Número de Tiendas', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / '05_improvement_vs_stores.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '05_improvement_vs_stores.png'}")
plt.close()

print()

# ============================================================================
# 10. RESUMEN EJECUTIVO
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO - SELECCIÓN DE NIVEL ÓPTIMO")
print("="*80)
print()

print(f"📊 PRODUCTOS ANALIZADOS: {len(df_selection)}")
print()

print(f"🎯 SELECCIÓN ÓPTIMA:")
print(f"  • GRANULAR (producto-tienda):  {n_granular:4d} productos ({pct_granular:5.1f}%)")
print(f"  • AGREGADO (producto-base):    {n_aggregated:4d} productos ({pct_aggregated:5.1f}%)")
print()

print(f"📈 MEJORA PROMEDIO (vs usar siempre el mismo nivel):")
print(f"  • RMSE: {df_selection['rmse_improvement_pct'].mean():.2f}%")
print(f"  • F1:   {df_selection['f1_improvement_pct'].mean():.2f}%")
print()

print(f"🏪 ANÁLISIS POR TIENDAS:")
if len(df_granular_sel) > 0:
    print(f"  • Productos con nivel GRANULAR óptimo:")
    print(f"    - Promedio tiendas: {df_granular_sel['num_stores'].mean():.1f}")
    print(f"    - RMSE promedio: {df_granular_sel['rmse_granular'].mean():.2f}")
    print(f"    - F1 promedio: {df_granular_sel['f1_granular'].mean():.3f}")
    print()
if len(df_aggregated_sel) > 0:
    print(f"  • Productos con nivel AGREGADO óptimo:")
    print(f"    - Promedio tiendas: {df_aggregated_sel['num_stores'].mean():.1f}")
    print(f"    - RMSE promedio: {df_aggregated_sel['rmse_aggregated'].mean():.2f}")
    print(f"    - F1 promedio: {df_aggregated_sel['f1_aggregated'].mean():.3f}")
    print()

print(f"📁 OUTPUTS GENERADOS:")
print(f"  • product_level_selection.csv - Nivel óptimo por producto")
print(f"  • level_comparison_by_product.csv - Comparación detallada")
print(f"  • 05_level_selection_distribution.png")
print(f"  • 05_rmse_comparison.png")
print(f"  • 05_f1_comparison.png")
print(f"  • 05_improvement_vs_stores.png")
print()

print("="*80)
print("✓ ANÁLISIS POR PRODUCTO COMPLETADO")
print("="*80)
print()

print("CONCLUSIÓN:")
print(f"  ✓ Sistema de dual granularidad implementado")
print(f"  ✓ {pct_aggregated:.1f}% de productos se benefician de nivel agregado")
print(f"  ✓ {pct_granular:.1f}% de productos requieren nivel granular")
print(f"  ✓ Decisión personalizada por producto maximiza rendimiento")
print()

print("PRÓXIMO PASO:")
print(f"  → Script 06: Valor Operativo (06_valor_operativo.py)")
print()
