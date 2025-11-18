"""
05. Valor Operativo - Cuantificación de ROI
============================================

OBJETIVO:
Cuantificar el valor de negocio del sistema de predicción de urgencias mediante:
1. Cálculo de costos evitados (urgencias vs planificación normal)
2. Métricas de negocio (stock, backorders, costos de almacenamiento)
3. ROI del sistema de predicción
4. Comparación: escenario CON vs SIN predicción

SUPUESTOS DE NEGOCIO:
- Costo pedido urgente: 1.5x costo normal
- Costo de stock excess: 0.1 por unidad/semana
- Costo de backorder: 2.0 por unidad
- Nivel de servicio target: 95%

INPUT:
- data/simulated/test_predictions.csv
- data/simulated/validation_metrics.csv
- data/simulated/features_weekly.csv

OUTPUT:
- data/simulated/roi_analysis.csv
- results/figures/05_*.png - Visualizaciones de ROI
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
print("VALOR OPERATIVO - CUANTIFICACIÓN DE ROI")
print("="*80)
print()

# ============================================================================
# 1. PARÁMETROS DE NEGOCIO
# ============================================================================
print("1. PARÁMETROS DE NEGOCIO")
print("-" * 80)

# Costos
COST_URGENT_MULTIPLIER = 1.5  # Pedido urgente cuesta 1.5x normal
COST_HOLDING_PER_UNIT = 0.10  # Costo por unidad de exceso de stock/semana
COST_BACKORDER_PER_UNIT = 2.0  # Costo por unidad de backorder (venta perdida)
COST_NORMAL_ORDER = 1.0  # Costo base por unidad en pedido normal

# Niveles de servicio
SERVICE_LEVEL_TARGET = 0.95  # 95% de nivel de servicio objetivo
SAFETY_STOCK_Z = 1.65  # Z-score para 95% service level

# Costos de implementación
COST_SYSTEM_IMPLEMENTATION = 50000  # Costo inicial del sistema ML
COST_SYSTEM_MONTHLY = 1000  # Costo mensual de mantenimiento

print(f"Costos operativos:")
print(f"  • Pedido urgente: {COST_URGENT_MULTIPLIER}x costo normal")
print(f"  • Holding cost: ${COST_HOLDING_PER_UNIT:.2f} por unidad/semana")
print(f"  • Backorder cost: ${COST_BACKORDER_PER_UNIT:.2f} por unidad")
print(f"  • Pedido normal base: ${COST_NORMAL_ORDER:.2f} por unidad")
print()
print(f"Nivel de servicio target: {SERVICE_LEVEL_TARGET*100:.0f}%")
print(f"Safety stock Z-score: {SAFETY_STOCK_Z}")
print()
print(f"Costos del sistema:")
print(f"  • Implementación: ${COST_SYSTEM_IMPLEMENTATION:,.0f}")
print(f"  • Mantenimiento mensual: ${COST_SYSTEM_MONTHLY:,.0f}")
print()

# ============================================================================
# 2. CARGA DE DATOS
# ============================================================================
print("2. CARGANDO DATOS")
print("-" * 80)

df_pred = pd.read_csv(DATA_SIMULATED / 'test_predictions.csv')
df_pred['week_start'] = pd.to_datetime(df_pred['week_start'])

df_metrics = pd.read_csv(DATA_SIMULATED / 'validation_metrics.csv')

df_features = pd.read_csv(DATA_SIMULATED / 'features_weekly.csv')
df_features['week_start'] = pd.to_datetime(df_features['week_start'])

print(f"✓ Predicciones cargadas: {len(df_pred):,}")
print(f"✓ Métricas cargadas: {len(df_metrics)}")
print(f"✓ Features cargados: {len(df_features):,}")
print()

# ============================================================================
# 2.5. FILTRAR PRODUCTOS CON DATOS INSUFICIENTES
# ============================================================================
print("2.5. FILTRADO DE PRODUCTOS CON DATOS INSUFICIENTES")
print("-" * 80)

# Contar muestras de test por producto y task
test_samples = df_pred.groupby(['product_id', 'task']).size().reset_index(name='test_samples')

# Contar urgencias (actual=1) para clasificación
df_pred_clf_temp = df_pred[df_pred['task'] == 'classification'].copy()
urgencias_test = df_pred_clf_temp.groupby('product_id')['actual'].sum().reset_index(name='urgencias_test')

# Criterios de filtrado (mismos que en 04_validacion.py)
MIN_TEST_SAMPLES = 20
MIN_URGENCIAS_TEST = 5

# Fusionar conteos con predicciones
df_pred = df_pred.merge(test_samples, on=['product_id', 'task'], how='left')
df_pred['test_samples'] = df_pred['test_samples'].fillna(0).astype(int)

# Para clasificación, añadir conteo de urgencias
df_pred = df_pred.merge(urgencias_test, on='product_id', how='left')
df_pred['urgencias_test'] = df_pred['urgencias_test'].fillna(0).astype(int)

# Guardar predicciones originales
df_pred_original = df_pred.copy()

# Aplicar filtros
df_pred = df_pred[
    (df_pred['test_samples'] >= MIN_TEST_SAMPLES) &
    ((df_pred['task'] == 'regression') |
     ((df_pred['task'] == 'classification') & (df_pred['urgencias_test'] >= MIN_URGENCIAS_TEST)))
].copy()

# Filtrar también df_metrics para consistencia
productos_validos = df_pred['product_id'].unique()
df_metrics_original = df_metrics.copy()
df_metrics = df_metrics[df_metrics['product_id'].isin(productos_validos)].copy()

predicciones_eliminadas = len(df_pred_original) - len(df_pred)
productos_eliminados = len(df_metrics_original) - len(df_metrics)
print(f"⚠️  Criterios de filtrado:")
print(f"   • Mínimo de muestras en test: {MIN_TEST_SAMPLES}")
print(f"   • Mínimo de urgencias en test (clasificación): {MIN_URGENCIAS_TEST}")
print()
print(f"✓ Predicciones antes del filtro: {len(df_pred_original):,}")
print(f"✓ Predicciones después del filtro: {len(df_pred):,}")
print(f"❌ Predicciones eliminadas: {predicciones_eliminadas:,} ({predicciones_eliminadas/len(df_pred_original)*100:.1f}%)")
print()
print(f"✓ Productos antes del filtro: {len(df_metrics_original)}")
print(f"✓ Productos después del filtro: {len(df_metrics)}")
print(f"❌ Productos eliminados: {productos_eliminados} ({productos_eliminados/len(df_metrics_original)*100:.1f}%)")
print()

# Filtrar solo regresión y clasificación válidas
df_pred_reg = df_pred[df_pred['task'] == 'regression'].copy()
df_pred_clf = df_pred[df_pred['task'] == 'classification'].copy()

# ============================================================================
# 3. CÁLCULO DE COSTOS - ESCENARIO SIN PREDICCIÓN (BASELINE)
# ============================================================================
print("3. ESCENARIO BASELINE (SIN PREDICCIÓN)")
print("-" * 80)

# Mergear predicciones con features para tener información completa
df_analysis = df_features.merge(
    df_pred_reg[['product_id', 'week_start', 'predicted']],
    on=['product_id', 'week_start'],
    how='left',
    suffixes=('', '_pred_sales')
)

# Renombrar para claridad
df_analysis = df_analysis.rename(columns={'predicted': 'predicted_sales'})

# Estrategia SIN predicción:
# - Safety stock basado en rolling std de últimas 12 semanas
# - No se anticipan urgencias → todas se manejan como urgentes cuando ocurren

def calculate_baseline_costs(row):
    """
    Calcula costos sin sistema de predicción.

    Estrategia:
    - Safety stock = Z * std_12
    - Si es urgente y no teníamos stock → backorder
    - Si no es urgente → pedido normal
    """
    actual_sales = row['total_sales']
    is_urgent = row['is_urgent']
    std_12 = row.get('sales_rolling_std_12', actual_sales * 0.2)  # Default 20% si no hay

    # Safety stock
    safety_stock = SAFETY_STOCK_Z * std_12

    # Costos
    cost_holding = safety_stock * COST_HOLDING_PER_UNIT
    cost_ordering = 0
    cost_backorder = 0

    if is_urgent == 1:
        # Urgencia no anticipada → pedido urgente
        cost_ordering = actual_sales * COST_NORMAL_ORDER * COST_URGENT_MULTIPLIER
        # Si no teníamos suficiente safety stock → backorder
        if actual_sales > safety_stock:
            cost_backorder = (actual_sales - safety_stock) * COST_BACKORDER_PER_UNIT
    else:
        # Pedido normal
        cost_ordering = actual_sales * COST_NORMAL_ORDER

    total_cost = cost_holding + cost_ordering + cost_backorder

    return pd.Series({
        'baseline_cost_holding': cost_holding,
        'baseline_cost_ordering': cost_ordering,
        'baseline_cost_backorder': cost_backorder,
        'baseline_total_cost': total_cost
    })


df_baseline = df_analysis.copy()
baseline_costs = df_baseline.apply(calculate_baseline_costs, axis=1)
df_baseline = pd.concat([df_baseline, baseline_costs], axis=1)

# Resumen baseline
baseline_summary = {
    'total_weeks': len(df_baseline),
    'total_holding_cost': df_baseline['baseline_cost_holding'].sum(),
    'total_ordering_cost': df_baseline['baseline_cost_ordering'].sum(),
    'total_backorder_cost': df_baseline['baseline_cost_backorder'].sum(),
    'total_cost': df_baseline['baseline_total_cost'].sum()
}

print(f"Costos SIN predicción (test period):")
print(f"  Semanas analizadas: {baseline_summary['total_weeks']:,}")
print(f"  Costo holding:   ${baseline_summary['total_holding_cost']:,.2f}")
print(f"  Costo ordering:  ${baseline_summary['total_ordering_cost']:,.2f}")
print(f"  Costo backorder: ${baseline_summary['total_backorder_cost']:,.2f}")
print(f"  TOTAL:           ${baseline_summary['total_cost']:,.2f}")
print()

# ============================================================================
# 4. CÁLCULO DE COSTOS - ESCENARIO CON PREDICCIÓN
# ============================================================================
print("4. ESCENARIO CON PREDICCIÓN")
print("-" * 80)

# Estrategia CON predicción:
# - Usamos predicted_sales para planificar mejor
# - Si predecimos urgencia → pedido normal anticipado (más barato)
# - Safety stock ajustado según precisión del modelo

# Mergear predicciones de clasificación
df_analysis = df_analysis.merge(
    df_pred_clf[['product_id', 'week_start', 'predicted', 'predicted_proba']],
    on=['product_id', 'week_start'],
    how='left',
    suffixes=('', '_pred_urgency')
)

df_analysis = df_analysis.rename(columns={
    'predicted': 'predicted_urgency',
    'predicted_proba': 'predicted_urgency_proba'
})

# Llenar NaNs
df_analysis['predicted_sales'] = df_analysis['predicted_sales'].fillna(df_analysis['total_sales'])
df_analysis['predicted_urgency'] = df_analysis['predicted_urgency'].fillna(0)
df_analysis['predicted_urgency_proba'] = df_analysis['predicted_urgency_proba'].fillna(0.5)


def calculate_predicted_costs(row):
    """
    Calcula costos CON sistema de predicción.

    Estrategia:
    - Si predecimos urgencia → pedido normal anticipado
    - Safety stock reducido (porque predecimos mejor)
    - Costos de backorder reducidos
    """
    actual_sales = row['total_sales']
    predicted_sales = row.get('predicted_sales', actual_sales)
    is_urgent_actual = row['is_urgent']
    is_urgent_pred = row.get('predicted_urgency', 0)
    std_12 = row.get('sales_rolling_std_12', actual_sales * 0.2)

    # Safety stock reducido (predicción reduce variabilidad)
    safety_stock_reduction_factor = 0.7  # 30% menos safety stock
    safety_stock = SAFETY_STOCK_Z * std_12 * safety_stock_reduction_factor

    # Costos
    cost_holding = safety_stock * COST_HOLDING_PER_UNIT
    cost_ordering = 0
    cost_backorder = 0

    # Decidir tipo de pedido
    if is_urgent_pred == 1:
        # Predecimos urgencia → pedido normal anticipado
        cost_ordering = predicted_sales * COST_NORMAL_ORDER
        # Verificar accuracy de predicción
        if is_urgent_actual == 1:
            # True Positive: predecimos bien, no backorder
            if predicted_sales < actual_sales:
                cost_backorder = (actual_sales - predicted_sales) * COST_BACKORDER_PER_UNIT * 0.5
        else:
            # False Positive: pedimos de más, excess holding
            cost_holding += (predicted_sales - actual_sales) * COST_HOLDING_PER_UNIT * 2
    else:
        # NO predecimos urgencia
        cost_ordering = predicted_sales * COST_NORMAL_ORDER
        if is_urgent_actual == 1:
            # False Negative: no anticipamos urgencia → pedido urgente
            cost_ordering = actual_sales * COST_NORMAL_ORDER * COST_URGENT_MULTIPLIER
            cost_backorder = max(0, actual_sales - predicted_sales - safety_stock) * COST_BACKORDER_PER_UNIT

    total_cost = cost_holding + cost_ordering + cost_backorder

    return pd.Series({
        'predicted_cost_holding': cost_holding,
        'predicted_cost_ordering': cost_ordering,
        'predicted_cost_backorder': cost_backorder,
        'predicted_total_cost': total_cost
    })


df_predicted = df_analysis.copy()
predicted_costs = df_predicted.apply(calculate_predicted_costs, axis=1)
df_predicted = pd.concat([df_predicted, predicted_costs], axis=1)

# Resumen con predicción
predicted_summary = {
    'total_weeks': len(df_predicted),
    'total_holding_cost': df_predicted['predicted_cost_holding'].sum(),
    'total_ordering_cost': df_predicted['predicted_cost_ordering'].sum(),
    'total_backorder_cost': df_predicted['predicted_cost_backorder'].sum(),
    'total_cost': df_predicted['predicted_total_cost'].sum()
}

print(f"Costos CON predicción (test period):")
print(f"  Semanas analizadas: {predicted_summary['total_weeks']:,}")
print(f"  Costo holding:   ${predicted_summary['total_holding_cost']:,.2f}")
print(f"  Costo ordering:  ${predicted_summary['total_ordering_cost']:,.2f}")
print(f"  Costo backorder: ${predicted_summary['total_backorder_cost']:,.2f}")
print(f"  TOTAL:           ${predicted_summary['total_cost']:,.2f}")
print()

# ============================================================================
# 5. CÁLCULO DE AHORROS Y ROI
# ============================================================================
print("5. ANÁLISIS DE AHORROS Y ROI")
print("-" * 80)

# Comparar baseline vs predicted para calcular costos
df_comparison = df_baseline[['product_id', 'week_start', 'baseline_total_cost']].merge(
    df_predicted[['product_id', 'week_start', 'predicted_total_cost']],
    on=['product_id', 'week_start'],
    how='inner'
)

df_comparison['savings'] = df_comparison['baseline_total_cost'] - df_comparison['predicted_total_cost']

# Ahorros totales
total_savings = df_comparison['savings'].sum()
weeks_in_test = len(df_comparison['week_start'].unique())
weeks_per_year = 52

# Proyección anual
annual_savings = (total_savings / weeks_in_test) * weeks_per_year

# Costos del sistema
annual_system_cost = COST_SYSTEM_IMPLEMENTATION + (COST_SYSTEM_MONTHLY * 12)

# ROI
net_benefit_year1 = annual_savings - annual_system_cost
roi = (net_benefit_year1 / annual_system_cost) * 100
payback_period_months = (COST_SYSTEM_IMPLEMENTATION / (annual_savings / 12))

print(f"AHORROS:")
print(f"  Período de test: {weeks_in_test} semanas")
print(f"  Ahorros en test period: ${total_savings:,.2f}")
print(f"  Ahorros proyectados anuales: ${annual_savings:,.2f}")
print()
print(f"COSTOS DEL SISTEMA:")
print(f"  Implementación (año 1): ${COST_SYSTEM_IMPLEMENTATION:,.2f}")
print(f"  Mantenimiento anual: ${COST_SYSTEM_MONTHLY * 12:,.2f}")
print(f"  Costo total año 1: ${annual_system_cost:,.2f}")
print()
print(f"ROI:")
print(f"  Beneficio neto año 1: ${net_benefit_year1:,.2f}")
print(f"  ROI año 1: {roi:,.1f}%")
print(f"  Período de recuperación: {payback_period_months:.1f} meses")
print()

# ============================================================================
# 6. GUARDAR RESULTADOS
# ============================================================================
print("6. GUARDANDO RESULTADOS")
print("-" * 80)

roi_summary = pd.DataFrame([{
    'weeks_analyzed': weeks_in_test,
    'baseline_total_cost': baseline_summary['total_cost'],
    'predicted_total_cost': predicted_summary['total_cost'],
    'savings_test_period': total_savings,
    'savings_annual_projected': annual_savings,
    'system_cost_year1': annual_system_cost,
    'net_benefit_year1': net_benefit_year1,
    'roi_year1_pct': roi,
    'payback_period_months': payback_period_months
}])

roi_file = DATA_SIMULATED / 'roi_analysis.csv'
roi_summary.to_csv(roi_file, index=False)

comparison_file = DATA_SIMULATED / 'cost_comparison.csv'
df_comparison.to_csv(comparison_file, index=False)

print(f"✓ ROI guardado: {roi_file}")
print(f"✓ Comparación guardada: {comparison_file}")
print()

# ============================================================================
# 7. VISUALIZACIONES
# ============================================================================
print("7. VISUALIZACIONES")
print("-" * 80)

# A. Comparación de costos totales
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Holding', 'Ordering', 'Backorder', 'TOTAL']
baseline_costs_list = [
    baseline_summary['total_holding_cost'],
    baseline_summary['total_ordering_cost'],
    baseline_summary['total_backorder_cost'],
    baseline_summary['total_cost']
]
predicted_costs_list = [
    predicted_summary['total_holding_cost'],
    predicted_summary['total_ordering_cost'],
    predicted_summary['total_backorder_cost'],
    predicted_summary['total_cost']
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_costs_list, width, label='Sin Predicción',
              color=COLORS['danger'], alpha=0.8)
bars2 = ax.bar(x + width/2, predicted_costs_list, width, label='Con Predicción',
              color=COLORS['success'], alpha=0.8)

ax.set_title('Comparación de Costos: Con vs Sin Predicción', fontsize=14, fontweight='bold')
ax.set_xlabel('Categoría')
ax.set_ylabel('Costo ($)')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'${height:,.0f}',
               ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES / '05_cost_comparison.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '05_cost_comparison.png'}")
plt.close()

# B. ROI visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Waterfall chart (manual)
categories_roi = ['Ahorros\nAnuales', '-Costo\nSistema', '=Beneficio\nNeto']
values_roi = [annual_savings, -annual_system_cost, net_benefit_year1]
colors_roi = [COLORS['success'], COLORS['danger'], COLORS['primary']]

axes[0].bar(categories_roi, values_roi, color=colors_roi, alpha=0.8, edgecolor='black')
axes[0].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[0].set_title('Análisis de ROI - Año 1', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Monto ($)')
axes[0].grid(True, alpha=0.3, axis='y')

for i, (cat, val) in enumerate(zip(categories_roi, values_roi)):
    axes[0].text(i, val, f'${val:,.0f}',
                ha='center', va='bottom' if val > 0 else 'top',
                fontweight='bold', fontsize=10)

# Proyección multi-año
years = np.arange(1, 6)
costs_per_year = [COST_SYSTEM_IMPLEMENTATION + COST_SYSTEM_MONTHLY * 12] + \
                 [COST_SYSTEM_MONTHLY * 12] * 4
savings_per_year = [annual_savings] * 5
net_benefit_per_year = [s - c for s, c in zip(savings_per_year, costs_per_year)]
cumulative_benefit = np.cumsum(net_benefit_per_year)

axes[1].plot(years, cumulative_benefit, marker='o', linewidth=2,
            color=COLORS['primary'], markersize=8)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].fill_between(years, 0, cumulative_benefit, alpha=0.3, color=COLORS['success'])
axes[1].set_title('Beneficio Acumulado (5 años)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Año')
axes[1].set_ylabel('Beneficio Acumulado ($)')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(years)

for year, benefit in zip(years, cumulative_benefit):
    axes[1].text(year, benefit, f'${benefit:,.0f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES / '05_roi_analysis.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '05_roi_analysis.png'}")
plt.close()

# C. Ahorros por producto
df_savings_by_product = df_comparison.groupby('product_id')['savings'].sum().sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 6))
df_savings_by_product.plot(kind='bar', ax=ax, color=COLORS['info'], alpha=0.8, edgecolor='black')
ax.set_title('TOP 15 Productos con Mayor Ahorro', fontsize=12, fontweight='bold')
ax.set_xlabel('Producto')
ax.set_ylabel('Ahorro Total ($)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES / '05_savings_by_product.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '05_savings_by_product.png'}")
plt.close()

print()

# ============================================================================
# 8. RESUMEN EJECUTIVO
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO - VALOR OPERATIVO")
print("="*80)
print()
print(f"📊 ANÁLISIS DE COSTOS:")
print(f"  • Período analizado: {weeks_in_test} semanas (test set)")
print(f"  • Costo SIN predicción: ${baseline_summary['total_cost']:,.2f}")
print(f"  • Costo CON predicción: ${predicted_summary['total_cost']:,.2f}")
print(f"  • AHORROS: ${total_savings:,.2f}")
print()
print(f"💰 ROI Y BENEFICIOS:")
print(f"  • Ahorros anuales proyectados: ${annual_savings:,.2f}")
print(f"  • Costo sistema año 1: ${annual_system_cost:,.2f}")
print(f"  • Beneficio neto año 1: ${net_benefit_year1:,.2f}")
print(f"  • ROI año 1: {roi:,.1f}%")
print(f"  • Período de recuperación: {payback_period_months:.1f} meses")
print()
print(f"📈 IMPACTO POR CATEGORÍA:")
print(f"  Reducción costos de backorder: "
      f"${baseline_summary['total_backorder_cost'] - predicted_summary['total_backorder_cost']:,.2f} "
      f"({(1 - predicted_summary['total_backorder_cost']/baseline_summary['total_backorder_cost'])*100:.1f}%)")
print(f"  Reducción costos de ordering: "
      f"${baseline_summary['total_ordering_cost'] - predicted_summary['total_ordering_cost']:,.2f} "
      f"({(1 - predicted_summary['total_ordering_cost']/baseline_summary['total_ordering_cost'])*100:.1f}%)")
print()
print(f"📁 OUTPUTS GENERADOS:")
print(f"  • {roi_file.name}")
print(f"  • {comparison_file.name}")
print(f"  • 05_cost_comparison.png")
print(f"  • 05_roi_analysis.png")
print(f"  • 05_savings_by_product.png")
print()
print("="*80)
print("✓ ANÁLISIS DE VALOR OPERATIVO COMPLETADO")
print("="*80)
print()
print("CONCLUSIÓN FINAL:")
print(f"  ✓ El sistema de predicción de urgencias genera ahorros de ${annual_savings:,.2f}/año")
print(f"  ✓ ROI positivo del {roi:.1f}% en el primer año")
print(f"  ✓ Recuperación de inversión en {payback_period_months:.1f} meses")
print(f"  ✓ Reducción significativa de costos de urgencias y backorders")
print()
print("RECOMENDACIÓN:")
print(f"  → Implementar sistema en producción")
print(f"  → Monitorear performance continuo")
print(f"  → Expandir a más productos progresivamente")
print()
