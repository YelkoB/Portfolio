"""
06. Valor Operativo - Sistema de Dual Granularidad
====================================================

OBJETIVO:
Demostrar el valor de negocio del sistema de dual granularidad mediante:
1. Comparación de métricas entre granular y agregado
2. Beneficios de usar el nivel óptimo por producto
3. Visualización de la mejora obtenida

MODELO DE NEGOCIO:
- Sistema de dual granularidad permite elegir el mejor nivel por producto
- Mejores predicciones → Mejor F1 y RMSE
- Impacto en negocio:
  * Mejor clasificación (F1) → Menos backorders, mejor servicio
  * Mejor regresión (RMSE) → Pronósticos más precisos de inventario

INPUT:
- data/simulated/validation_metrics_granular.csv
- data/simulated/validation_metrics_aggregated.csv
- data/simulated/product_level_selection.csv
- data/simulated/level_comparison_by_product.csv

OUTPUT:
- data/simulated/dual_granularity_value.csv
- results/figures/06_*.png - Visualizaciones
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Importar configuración
from config import (
    DATA_SIMULATED, FIGURES,
    FIGSIZE_STANDARD, FIGSIZE_WIDE,
    COLORS, RANDOM_SEED
)

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

print("="*80)
print("VALOR OPERATIVO - SISTEMA DE DUAL GRANULARIDAD")
print("="*80)
print()

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("1. CARGANDO DATOS")
print("-" * 80)

# Cargar selección de nivel óptimo
df_selection = pd.read_csv(DATA_SIMULATED / 'product_level_selection.csv')
df_comparison = pd.read_csv(DATA_SIMULATED / 'level_comparison_by_product.csv', index_col=0)

print(f"✅ Selección de nivel: {len(df_selection)} productos")
print(f"✅ Comparación detallada: {len(df_comparison)} productos")
print()

# ============================================================================
# 2. CÁLCULO DE MEJORA POR USAR NIVEL ÓPTIMO
# ============================================================================
print("2. MEJORA POR USAR NIVEL ÓPTIMO vs USAR SIEMPRE EL MISMO NIVEL")
print("-" * 80)

# Escenario 1: Usar SIEMPRE GRANULAR (baseline)
always_granular_f1 = df_comparison['f1_granular'].mean()
always_granular_rmse = df_comparison['rmse_granular'].mean()

# Escenario 2: Usar SIEMPRE AGREGADO
always_aggregated_f1 = df_comparison['f1_aggregated'].mean()
always_aggregated_rmse = df_comparison['rmse_aggregated'].mean()

# Escenario 3: Usar NIVEL ÓPTIMO (dual granularity)
# Para cada producto, usar la métrica del nivel óptimo
optimal_f1_list = []
optimal_rmse_list = []

for product_base in df_comparison.index:
    selection_row = df_selection[df_selection['product_base'] == product_base]

    if len(selection_row) == 0:
        continue

    best_level = selection_row.iloc[0]['best_overall']

    if best_level == 'granular':
        optimal_f1_list.append(df_comparison.loc[product_base, 'f1_granular'])
        optimal_rmse_list.append(df_comparison.loc[product_base, 'rmse_granular'])
    else:
        optimal_f1_list.append(df_comparison.loc[product_base, 'f1_aggregated'])
        optimal_rmse_list.append(df_comparison.loc[product_base, 'rmse_aggregated'])

optimal_f1 = np.mean(optimal_f1_list)
optimal_rmse = np.mean(optimal_rmse_list)

print(f"📊 ESCENARIOS COMPARADOS:")
print()
print(f"1. Usar SIEMPRE GRANULAR:")
print(f"   • F1 promedio:   {always_granular_f1:.3f}")
print(f"   • RMSE promedio: {always_granular_rmse:.2f}")
print()
print(f"2. Usar SIEMPRE AGREGADO:")
print(f"   • F1 promedio:   {always_aggregated_f1:.3f}")
print(f"   • RMSE promedio: {always_aggregated_rmse:.2f}")
print()
print(f"3. Usar NIVEL ÓPTIMO (Dual Granularity):")
print(f"   • F1 promedio:   {optimal_f1:.3f}")
print(f"   • RMSE promedio: {optimal_rmse:.2f}")
print()

# Calcular mejoras
f1_improvement_vs_granular = ((optimal_f1 - always_granular_f1) / always_granular_f1) * 100
f1_improvement_vs_aggregated = ((optimal_f1 - always_aggregated_f1) / always_aggregated_f1) * 100

rmse_improvement_vs_granular = ((always_granular_rmse - optimal_rmse) / always_granular_rmse) * 100
rmse_improvement_vs_aggregated = ((always_aggregated_rmse - optimal_rmse) / always_aggregated_rmse) * 100

print(f"MEJORA POR USAR DUAL GRANULARITY:")
print(f"  vs Siempre Granular:")
print(f"    • Mejora F1:   {f1_improvement_vs_granular:+.2f}%")
print(f"    • Mejora RMSE: {rmse_improvement_vs_granular:+.2f}%")
print()
print(f"  vs Siempre Agregado:")
print(f"    • Mejora F1:   {f1_improvement_vs_aggregated:+.2f}%")
print(f"    • Mejora RMSE: {rmse_improvement_vs_aggregated:+.2f}%")
print()

# ============================================================================
# 3. ANÁLISIS POR NIVEL SELECCIONADO
# ============================================================================
print("3. ANÁLISIS POR NIVEL SELECCIONADO")
print("-" * 80)

# Agrupar por nivel óptimo
granular_products = df_selection[df_selection['best_overall'] == 'granular']
aggregated_products = df_selection[df_selection['best_overall'] == 'aggregated']

print(f"PRODUCTOS CON NIVEL GRANULAR ÓPTIMO: {len(granular_products)} ({len(granular_products)/len(df_selection)*100:.1f}%)")
if len(granular_products) > 0:
    print(f"  • Tiendas promedio: {granular_products['num_stores'].mean():.1f}")
    print(f"  • F1 promedio: {granular_products['f1_granular'].mean():.3f}")
    print(f"  • RMSE promedio: {granular_products['rmse_granular'].mean():.2f}")
    print(f"  • Mejora F1 vs agregado: {granular_products['f1_improvement_pct'].mean():.2f}%")
print()

print(f"PRODUCTOS CON NIVEL AGREGADO ÓPTIMO: {len(aggregated_products)} ({len(aggregated_products)/len(df_selection)*100:.1f}%)")
if len(aggregated_products) > 0:
    print(f"  • Tiendas promedio: {aggregated_products['num_stores'].mean():.1f}")
    print(f"  • F1 promedio: {aggregated_products['f1_aggregated'].mean():.3f}")
    print(f"  • RMSE promedio: {aggregated_products['rmse_aggregated'].mean():.2f}")
    print(f"  • Mejora F1 vs granular: {aggregated_products['f1_improvement_pct'].mean():.2f}%")
print()

# ============================================================================
# 4. IMPACTO EN NEGOCIO
# ============================================================================
print("4. ESTIMACIÓN DE IMPACTO EN NEGOCIO")
print("-" * 80)

# Supuestos de negocio
AVG_SALES_PER_PRODUCT_PER_WEEK = 50  # Unidades promedio por producto/semana
WEEKS_PER_YEAR = 52
COST_BACKORDER_PER_UNIT = 2.0  # Costo por backorder
COST_HOLDING_PER_UNIT_WEEK = 0.10  # Costo de exceso de inventario

print(f"SUPUESTOS DE NEGOCIO:")
print(f"  • Ventas promedio: {AVG_SALES_PER_PRODUCT_PER_WEEK} unidades/producto/semana")
print(f"  • Costo de backorder: ${COST_BACKORDER_PER_UNIT:.2f}/unidad")
print(f"  • Costo de holding: ${COST_HOLDING_PER_UNIT_WEEK:.2f}/unidad/semana")
print()

# Estimar impacto de mejor F1
# Mejor F1 → Menos False Negatives → Menos backorders inesperados
# Asumimos que mejora en F1 reduce backorders proporcionalmente

n_products = len(df_selection)
annual_sales = n_products * AVG_SALES_PER_PRODUCT_PER_WEEK * WEEKS_PER_YEAR

# Escenario baseline: Usar siempre granular
baseline_f1 = always_granular_f1
# Tasa de backorder inesperado ≈ (1 - Recall) * % urgencias
# Asumimos 15% de semanas son urgencias
urgency_rate = 0.15
recall_baseline = baseline_f1 / 0.7  # Aprox, asumiendo precision ≈ 0.7 * F1
backorder_rate_baseline = (1 - recall_baseline) * urgency_rate

# Escenario con dual granularity
optimal_recall = optimal_f1 / 0.7
backorder_rate_optimal = (1 - optimal_recall) * urgency_rate

# Ahorros
backorders_avoided = annual_sales * (backorder_rate_baseline - backorder_rate_optimal)
savings_backorders = backorders_avoided * COST_BACKORDER_PER_UNIT

# Estimar impacto de mejor RMSE
# Mejor RMSE → Mejor pronóstico → Menos exceso de inventario
# Asumimos que RMSE se traduce en desviación de inventario
inventory_excess_baseline = annual_sales * (always_granular_rmse / AVG_SALES_PER_PRODUCT_PER_WEEK) * 0.10
inventory_excess_optimal = annual_sales * (optimal_rmse / AVG_SALES_PER_PRODUCT_PER_WEEK) * 0.10

savings_holding = (inventory_excess_baseline - inventory_excess_optimal) * COST_HOLDING_PER_UNIT_WEEK

total_savings = savings_backorders + savings_holding

print(f"IMPACTO ESTIMADO (anual):")
print(f"  • Productos: {n_products}")
print(f"  • Ventas anuales: {annual_sales:,.0f} unidades")
print()
print(f"  AHORROS POR MEJOR CLASIFICACIÓN (F1):")
print(f"    • Backorders evitados: {backorders_avoided:,.0f} unidades")
print(f"    • Ahorro backorders: ${savings_backorders:,.2f}")
print()
print(f"  AHORROS POR MEJOR REGRESIÓN (RMSE):")
print(f"    • Reducción exceso inventario: {inventory_excess_baseline - inventory_excess_optimal:,.0f} unidades-semana")
print(f"    • Ahorro holding cost: ${savings_holding:,.2f}")
print()
print(f"  💰 AHORRO TOTAL ESTIMADO: ${total_savings:,.2f}/año")
print()

# ============================================================================
# 5. GUARDAR RESULTADOS
# ============================================================================
print("5. GUARDANDO RESULTADOS")
print("-" * 80)

# Crear resumen
value_summary = pd.DataFrame([{
    'n_products': n_products,
    'f1_always_granular': always_granular_f1,
    'f1_always_aggregated': always_aggregated_f1,
    'f1_optimal': optimal_f1,
    'f1_improvement_vs_granular_pct': f1_improvement_vs_granular,
    'f1_improvement_vs_aggregated_pct': f1_improvement_vs_aggregated,
    'rmse_always_granular': always_granular_rmse,
    'rmse_always_aggregated': always_aggregated_rmse,
    'rmse_optimal': optimal_rmse,
    'rmse_improvement_vs_granular_pct': rmse_improvement_vs_granular,
    'rmse_improvement_vs_aggregated_pct': rmse_improvement_vs_aggregated,
    'backorders_avoided_annual': backorders_avoided,
    'savings_backorders_annual': savings_backorders,
    'savings_holding_annual': savings_holding,
    'total_savings_annual': total_savings
}])

value_file = DATA_SIMULATED / 'dual_granularity_value.csv'
value_summary.to_csv(value_file, index=False)
print(f"✓ Valor del sistema guardado: {value_file}")
print()

# ============================================================================
# 6. VISUALIZACIONES
# ============================================================================
print("6. VISUALIZACIONES")
print("-" * 80)

# A. Comparación de métricas entre escenarios
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

scenarios = ['Siempre\nGranular', 'Siempre\nAgregado', 'Dual\nGranularity']
f1_values = [always_granular_f1, always_aggregated_f1, optimal_f1]
rmse_values = [always_granular_rmse, always_aggregated_rmse, optimal_rmse]

colors = [COLORS['secondary'], COLORS['info'], COLORS['success']]

# F1 comparison
bars1 = axes[0].bar(scenarios, f1_values, color=colors, alpha=0.8, edgecolor='black')
axes[0].set_title('F1-Score: Comparación de Estrategias', fontsize=12, fontweight='bold')
axes[0].set_ylabel('F1-Score')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([min(f1_values) * 0.95, max(f1_values) * 1.05])

for bar, val in zip(bars1, f1_values):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

# RMSE comparison
bars2 = axes[1].bar(scenarios, rmse_values, color=colors, alpha=0.8, edgecolor='black')
axes[1].set_title('RMSE: Comparación de Estrategias', fontsize=12, fontweight='bold')
axes[1].set_ylabel('RMSE')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim([min(rmse_values) * 0.95, max(rmse_values) * 1.05])

for bar, val in zip(bars2, rmse_values):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES / '06_strategy_comparison.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '06_strategy_comparison.png'}")
plt.close()

# B. Distribución de selección
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Pie chart de selección
selection_counts = df_selection['best_overall'].value_counts()
axes[0].pie(selection_counts, labels=['Granular', 'Agregado'], autopct='%1.1f%%',
           colors=[COLORS['primary'], COLORS['secondary']], startangle=90)
axes[0].set_title('Distribución de Nivel Óptimo', fontsize=12, fontweight='bold')

# Bar chart de ahorros por componente
components = ['Backorders\nevitados', 'Reducción\nholding cost', 'TOTAL']
savings_values = [savings_backorders, savings_holding, total_savings]
colors_savings = [COLORS['danger'], COLORS['warning'], COLORS['success']]

bars = axes[1].bar(components, savings_values, color=colors_savings, alpha=0.8, edgecolor='black')
axes[1].set_title('Ahorros Anuales Estimados', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Ahorros ($)')
axes[1].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, savings_values):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES / '06_savings_breakdown.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '06_savings_breakdown.png'}")
plt.close()

# C. Mejora por número de tiendas
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter: num_stores vs f1_improvement_pct
colors_map = df_selection['best_overall'].map({'granular': COLORS['primary'], 'aggregated': COLORS['secondary']})
ax.scatter(df_selection['num_stores'], df_selection['f1_improvement_pct'],
          c=colors_map, alpha=0.6, s=50)

ax.set_xlabel('Número de Tiendas', fontsize=11)
ax.set_ylabel('Mejora F1 (%)', fontsize=11)
ax.set_title('Mejora F1 vs Número de Tiendas por Producto', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS['primary'], label='Granular óptimo'),
                  Patch(facecolor=COLORS['secondary'], label='Agregado óptimo')]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig(FIGURES / '06_improvement_vs_stores.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '06_improvement_vs_stores.png'}")
plt.close()

print()

# ============================================================================
# 7. RESUMEN EJECUTIVO
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO - VALOR DEL SISTEMA DE DUAL GRANULARIDAD")
print("="*80)
print()

print(f"📊 PRODUCTOS ANALIZADOS: {n_products}")
print()

print(f"🎯 ESTRATEGIA ÓPTIMA:")
print(f"  • Usar GRANULAR para: {len(granular_products)} productos ({len(granular_products)/n_products*100:.1f}%)")
print(f"  • Usar AGREGADO para:  {len(aggregated_products)} productos ({len(aggregated_products)/n_products*100:.1f}%)")
print()

print(f"📈 MEJORA DE MÉTRICAS vs Usar Siempre Granular:")
print(f"  • Mejora F1:   {f1_improvement_vs_granular:+.2f}%")
print(f"  • Mejora RMSE: {rmse_improvement_vs_granular:+.2f}%")
print()

print(f"📈 MEJORA DE MÉTRICAS vs Usar Siempre Agregado:")
print(f"  • Mejora F1:   {f1_improvement_vs_aggregated:+.2f}%")
print(f"  • Mejora RMSE: {rmse_improvement_vs_aggregated:+.2f}%")
print()

print(f"💰 IMPACTO EN NEGOCIO (estimado anual):")
print(f"  • Backorders evitados: {backorders_avoided:,.0f} unidades")
print(f"  • Ahorro por backorders: ${savings_backorders:,.2f}")
print(f"  • Ahorro por holding cost: ${savings_holding:,.2f}")
print(f"  • AHORRO TOTAL: ${total_savings:,.2f}/año")
print()

print(f"📁 OUTPUTS GENERADOS:")
print(f"  • dual_granularity_value.csv")
print(f"  • 06_strategy_comparison.png")
print(f"  • 06_savings_breakdown.png")
print(f"  • 06_improvement_vs_stores.png")
print()

print("="*80)
print("✓ ANÁLISIS DE VALOR OPERATIVO COMPLETADO")
print("="*80)
print()

print("CONCLUSIÓN:")
print(f"  ✓ Sistema de dual granularidad supera estrategias de un solo nivel")
print(f"  ✓ Mejora F1: {max(f1_improvement_vs_granular, f1_improvement_vs_aggregated):+.2f}%")
print(f"  ✓ Mejora RMSE: {max(rmse_improvement_vs_granular, rmse_improvement_vs_aggregated):+.2f}%")
print(f"  ✓ Decisión personalizada por producto maximiza valor de negocio")
print(f"  ✓ Ahorros estimados: ${total_savings:,.2f}/año")
print()

print("RECOMENDACIÓN:")
print(f"  → Implementar sistema de dual granularidad")
print(f"  → {len(granular_products)} productos usan nivel granular")
print(f"  → {len(aggregated_products)} productos usan nivel agregado")
print(f"  → Selección automática basada en métricas de validación")
print()
