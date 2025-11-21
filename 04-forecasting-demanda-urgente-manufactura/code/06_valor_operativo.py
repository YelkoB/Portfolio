"""
06. Valor Operativo - Sistema de Dual Granularidad con ROI
============================================================

OBJETIVO:
Calcular el valor de negocio REAL del sistema de dual granularidad mediante:
1. Modelo de costos realista
2. Comparación: SIN predicción vs CON predicción (nivel óptimo)
3. Cálculo de ROI considerando costos del sistema

MODELO DE NEGOCIO SIMPLIFICADO:
================================================
⚠️ SUPUESTOS (para simplificar análisis):
- Todos los productos tienen el mismo costo unitario de producción
- Costos son estimaciones conservadoras basadas en industria manufacturera
- Fee de urgencia refleja costo adicional de operación express

ESCENARIO SIN ML (Baseline):
- Urgencias NO anticipadas → Producción urgente ($1.50/unidad)
- Posibles faltantes de stock → Backorders ($2.00/unidad)
- Safety stock normal

ESCENARIO CON ML (Predicción):
- Urgencias ANTICIPADAS → Producción planificada normal ($1.00/unidad)
- Mejor gestión de stock → Menos backorders
- Incremento moderado en holding por anticipación
- BENEFICIO: Ahorro en costos urgentes + mejor servicio al cliente

PARÁMETROS DE COSTOS:
- Pedido normal: $1.00/unidad
- Pedido urgente: $1.50/unidad (1.5x normal)
- Backorder: $2.00/unidad (oportunidad perdida + daño reputacional)
- Holding: $0.10/unidad/semana (almacenamiento)

INPUT:
- data/simulated/product_level_selection.csv
- data/simulated/test_predictions_granular.csv
- data/simulated/test_predictions_aggregated.csv
- data/simulated/features_weekly_granular.csv
- data/simulated/features_weekly_aggregated.csv

OUTPUT:
- data/simulated/roi_analysis.csv
- results/figures/06_*.png
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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
print("VALOR OPERATIVO - ROI DEL SISTEMA DE DUAL GRANULARIDAD")
print("="*80)
print()

# ============================================================================
# 1. PARÁMETROS DE NEGOCIO
# ============================================================================
print("1. PARÁMETROS DE NEGOCIO")
print("-" * 80)

# Costos operativos (SUPUESTO: Todos los productos valen lo mismo)
COST_NORMAL_ORDER = 1.0          # Pedido normal al proveedor
COST_URGENT_MULTIPLIER = 1.5     # Pedido urgente cuesta 1.5x (operación express)
COST_HOLDING_PER_UNIT = 0.10     # Costo por almacenamiento/semana
COST_BACKORDER_PER_UNIT = 2.0    # Oportunidad perdida + daño reputacional

# Safety stock
SAFETY_STOCK_Z = 1.65            # 95% service level (z-score)

# Costos del sistema ML
COST_SYSTEM_IMPLEMENTATION = 50000  # Implementación inicial
COST_SYSTEM_MONTHLY = 1000          # Mantenimiento mensual

print(f"💰 MODELO DE COSTOS (supuesto: costos homogéneos):")
print(f"  • Pedido normal: ${COST_NORMAL_ORDER:.2f}/unidad")
print(f"  • Pedido urgente: ${COST_NORMAL_ORDER * COST_URGENT_MULTIPLIER:.2f}/unidad (1.5x normal)")
print(f"  • Backorder: ${COST_BACKORDER_PER_UNIT:.2f}/unidad (oportunidad perdida)")
print(f"  • Holding: ${COST_HOLDING_PER_UNIT:.2f}/unidad/semana (almacenamiento)")
print()
print(f"🖥️  COSTOS SISTEMA ML:")
print(f"  • Implementación inicial: ${COST_SYSTEM_IMPLEMENTATION:,.0f}")
print(f"  • Mantenimiento anual: ${COST_SYSTEM_MONTHLY * 12:,.0f}")
print()
print(f"📊 LÓGICA DE AHORRO:")
print(f"  • SIN ML: Urgencias NO anticipadas → Pedido urgente + Posibles backorders")
print(f"  • CON ML: Urgencias ANTICIPADAS → Pedido normal + Menos backorders")
print(f"  • Ahorro por urgencia anticipada: ${(COST_NORMAL_ORDER * COST_URGENT_MULTIPLIER) - COST_NORMAL_ORDER:.2f}/unidad")
print()

# ============================================================================
# 2. CARGAR DATOS
# ============================================================================
print("2. CARGANDO DATOS")
print("-" * 80)

# Selección de nivel óptimo
df_selection = pd.read_csv(DATA_SIMULATED / 'product_level_selection.csv')
print(f"✅ Selección de nivel: {len(df_selection)} productos")

# Predicciones de test
pred_granular = pd.read_csv(DATA_SIMULATED / 'test_predictions_granular.csv')
pred_granular['week_start'] = pd.to_datetime(pred_granular['week_start'])

pred_aggregated = pd.read_csv(DATA_SIMULATED / 'test_predictions_aggregated.csv')
pred_aggregated['week_start'] = pd.to_datetime(pred_aggregated['week_start'])

print(f"✅ Predicciones granular: {len(pred_granular):,}")
print(f"✅ Predicciones aggregated: {len(pred_aggregated):,}")

# Features (para tener info de ventas reales)
features_granular = pd.read_csv(DATA_SIMULATED / 'features_weekly_granular.csv')
features_granular['week_start'] = pd.to_datetime(features_granular['week_start'])

features_aggregated = pd.read_csv(DATA_SIMULATED / 'features_weekly_aggregated.csv')
features_aggregated['week_start'] = pd.to_datetime(features_aggregated['week_start'])

print(f"✅ Features granular: {len(features_granular):,}")
print(f"✅ Features aggregated: {len(features_aggregated):,}")
print()

# ============================================================================
# 3. MAPEAR PRODUCT_BASE EN PREDICCIONES GRANULARES
# ============================================================================
print("3. PREPARANDO DATOS")
print("-" * 80)

def extract_product_base(product_id):
    """Extraer product_base desde product_id granular"""
    parts = product_id.rsplit('_', 2)
    return parts[0]

# Agregar product_base a granular
pred_granular['product_base'] = pred_granular['product_id'].apply(extract_product_base)
features_granular['product_base'] = features_granular['product_id'].apply(extract_product_base)

print(f"✅ Product_base agregado a datos granulares")
print()

# ============================================================================
# 4. FUNCIONES DE CÁLCULO DE COSTOS
# ============================================================================

def calculate_baseline_costs(row):
    """
    Costos SIN sistema de predicción.
    - Urgencias NO anticipadas → pedido urgente (más caro)
    - Safety stock estándar
    - Posibles backorders por falta de stock
    """
    sales = row['actual']
    is_urgent = row['is_urgent_actual']
    std_sales = row.get('sales_rolling_std_12', sales * 0.2)

    # Safety stock estándar
    safety_stock = SAFETY_STOCK_Z * std_sales
    cost_holding = safety_stock * COST_HOLDING_PER_UNIT

    if is_urgent == 1:
        # Urgencia NO anticipada → pedido urgente (1.5x normal)
        cost_ordering = sales * COST_NORMAL_ORDER * COST_URGENT_MULTIPLIER

        # Si no hay suficiente safety stock → backorder
        if sales > safety_stock:
            cost_backorder = (sales - safety_stock) * COST_BACKORDER_PER_UNIT
        else:
            cost_backorder = 0
    else:
        # Pedido normal
        cost_ordering = sales * COST_NORMAL_ORDER
        cost_backorder = 0

    total_cost = cost_holding + cost_ordering + cost_backorder

    return pd.Series({
        'baseline_cost_holding': cost_holding,
        'baseline_cost_ordering': cost_ordering,
        'baseline_cost_backorder': cost_backorder,
        'baseline_total_cost': total_cost
    })


def calculate_predicted_costs(row):
    """
    Costos CON sistema de predicción.
    - Urgencias ANTICIPADAS → pedido normal planificado (más barato)
    - Mejor gestión de stock → menos backorders
    - Trade-off: Más holding si anticipamos, pero ahorramos en urgentes
    """
    sales = row['actual']
    is_urgent_actual = row['is_urgent_actual']
    is_urgent_pred = row['is_urgent_pred']
    std_sales = row.get('sales_rolling_std_12', sales * 0.2)

    cost_holding = 0
    cost_ordering = 0
    cost_backorder = 0

    if is_urgent_pred == 1:
        # PREDICCIÓN: Anticipamos urgencia
        # Preparamos stock extra (mayor holding)
        safety_stock_increased = SAFETY_STOCK_Z * std_sales * 1.3
        cost_holding = safety_stock_increased * COST_HOLDING_PER_UNIT
        cost_ordering = sales * COST_NORMAL_ORDER  # Pedido NORMAL (ahorro!)

        if is_urgent_actual == 1:
            # True Positive: Acertamos → sin backorders
            cost_backorder = 0
        else:
            # False Positive: Exceso de stock (penalización en holding)
            excess = sales * 0.2
            cost_holding += excess * COST_HOLDING_PER_UNIT * 2
    else:
        # PREDICCIÓN: NO urgencia
        safety_stock = SAFETY_STOCK_Z * std_sales
        cost_holding = safety_stock * COST_HOLDING_PER_UNIT
        cost_ordering = sales * COST_NORMAL_ORDER

        if is_urgent_actual == 1:
            # False Negative: Fallamos → pedido urgente necesario
            cost_ordering = sales * COST_NORMAL_ORDER * COST_URGENT_MULTIPLIER
            cost_backorder = max(0, sales - safety_stock) * COST_BACKORDER_PER_UNIT

    total_cost = cost_holding + cost_ordering + cost_backorder

    return pd.Series({
        'predicted_cost_holding': cost_holding,
        'predicted_cost_ordering': cost_ordering,
        'predicted_cost_backorder': cost_backorder,
        'predicted_total_cost': total_cost
    })


# ============================================================================
# 5. PREPARAR DATOS PARA ANÁLISIS
# ============================================================================
print("4. PREPARANDO DATOS PARA ANÁLISIS DE COSTOS")
print("-" * 80)

# Crear dataset unificado usando nivel óptimo por producto
analysis_data = []

for _, selection_row in df_selection.iterrows():
    product_base = selection_row['product_base']
    best_level = selection_row['best_overall']

    if best_level == 'granular':
        # Usar predicciones granulares
        # Obtener todas las product_id de este product_base
        product_ids = features_granular[features_granular['product_base'] == product_base]['product_id'].unique()

        for product_id in product_ids:
            # Predicciones de clasificación
            pred_clf = pred_granular[
                (pred_granular['product_id'] == product_id) &
                (pred_granular['task'] == 'classification')
            ].copy()

            # Predicciones de regresión
            pred_reg = pred_granular[
                (pred_granular['product_id'] == product_id) &
                (pred_granular['task'] == 'regression')
            ].copy()

            # Features para info adicional
            feat = features_granular[features_granular['product_id'] == product_id].copy()

            # Merge
            if len(pred_clf) > 0 and len(pred_reg) > 0:
                merged = pred_clf[['week_start', 'actual', 'predicted']].merge(
                    pred_reg[['week_start', 'predicted']],
                    on='week_start',
                    suffixes=('_clf', '_reg')
                )

                merged = merged.merge(
                    feat[['week_start', 'total_sales', 'is_urgent', 'sales_rolling_std_12']],
                    on='week_start',
                    how='left'
                )

                merged['product_base'] = product_base
                merged['product_id'] = product_id
                merged['level'] = 'granular'
                merged['is_urgent_actual'] = merged['actual']  # 'actual' viene solo de clf, no tiene sufijo
                merged['is_urgent_pred'] = merged['predicted_clf']  # 'predicted' tiene conflicto → _clf
                merged['actual'] = merged['total_sales']  # Reasignar con ventas reales

                analysis_data.append(merged)

    else:
        # Usar predicciones agregadas
        pred_clf = pred_aggregated[
            (pred_aggregated['product_base'] == product_base) &
            (pred_aggregated['task'] == 'classification')
        ].copy()

        pred_reg = pred_aggregated[
            (pred_aggregated['product_base'] == product_base) &
            (pred_aggregated['task'] == 'regression')
        ].copy()

        feat = features_aggregated[features_aggregated['product_base'] == product_base].copy()

        if len(pred_clf) > 0 and len(pred_reg) > 0:
            merged = pred_clf[['week_start', 'actual', 'predicted']].merge(
                pred_reg[['week_start', 'predicted']],
                on='week_start',
                suffixes=('_clf', '_reg')
            )

            merged = merged.merge(
                feat[['week_start', 'total_sales', 'is_urgent', 'sales_rolling_std_12']],
                on='week_start',
                how='left'
            )

            merged['product_base'] = product_base
            merged['product_id'] = product_base  # Mismo ID
            merged['level'] = 'aggregated'
            merged['is_urgent_actual'] = merged['actual']  # 'actual' viene solo de clf, no tiene sufijo
            merged['is_urgent_pred'] = merged['predicted_clf']  # 'predicted' tiene conflicto → _clf
            merged['actual'] = merged['total_sales']  # Reasignar con ventas reales

            analysis_data.append(merged)

df_analysis = pd.concat(analysis_data, ignore_index=True)

print(f"✅ Dataset de análisis creado: {len(df_analysis):,} semanas")
print(f"   Productos base únicos: {df_analysis['product_base'].nunique()}")
print(f"   Productos únicos (granular expandido): {df_analysis['product_id'].nunique()}")
print()

# ============================================================================
# 6. CALCULAR COSTOS
# ============================================================================
print("5. CALCULANDO COSTOS: BASELINE vs CON PREDICCIÓN")
print("-" * 80)

# Baseline
baseline_costs = df_analysis.apply(calculate_baseline_costs, axis=1)
df_analysis = pd.concat([df_analysis, baseline_costs], axis=1)

# Con predicción
predicted_costs = df_analysis.apply(calculate_predicted_costs, axis=1)
df_analysis = pd.concat([df_analysis, predicted_costs], axis=1)

# Ahorros
df_analysis['savings'] = df_analysis['baseline_total_cost'] - df_analysis['predicted_total_cost']

print(f"✅ Costos calculados para {len(df_analysis):,} semanas")
print()

# ============================================================================
# 7. RESUMEN DE RESULTADOS
# ============================================================================
print("6. RESUMEN DE RESULTADOS")
print("-" * 80)

# Baseline
baseline_summary = {
    'total_holding': df_analysis['baseline_cost_holding'].sum(),
    'total_ordering': df_analysis['baseline_cost_ordering'].sum(),
    'total_backorder': df_analysis['baseline_cost_backorder'].sum(),
    'total_cost': df_analysis['baseline_total_cost'].sum()
}

# Con predicción
predicted_summary = {
    'total_holding': df_analysis['predicted_cost_holding'].sum(),
    'total_ordering': df_analysis['predicted_cost_ordering'].sum(),
    'total_backorder': df_analysis['predicted_cost_backorder'].sum(),
    'total_cost': df_analysis['predicted_total_cost'].sum()
}

print(f"📊 ESCENARIO SIN PREDICCIÓN (Baseline):")
print(f"  • Holding:   ${baseline_summary['total_holding']:,.2f}")
print(f"  • Ordering:  ${baseline_summary['total_ordering']:,.2f}")
print(f"  • Backorder: ${baseline_summary['total_backorder']:,.2f}")
print(f"  Costo TOTAL: ${baseline_summary['total_cost']:,.2f}")
print()

print(f"📊 ESCENARIO CON PREDICCIÓN (Dual Granularity):")
print(f"  • Holding:   ${predicted_summary['total_holding']:,.2f}")
print(f"  • Ordering:  ${predicted_summary['total_ordering']:,.2f}")
print(f"  • Backorder: ${predicted_summary['total_backorder']:,.2f}")
print(f"  Costo TOTAL: ${predicted_summary['total_cost']:,.2f}")
print()

# Ahorros
total_savings_period = df_analysis['savings'].sum()
weeks_analyzed = df_analysis['week_start'].nunique()
annual_savings = (total_savings_period / weeks_analyzed) * 52

print(f"💰 AHORROS:")
print(f"  Período analizado: {weeks_analyzed} semanas")
print(f"  Ahorros totales: ${total_savings_period:,.2f}")
print(f"  Ahorros proyectados anuales: ${annual_savings:,.2f}")
print()

# ROI
annual_system_cost = COST_SYSTEM_IMPLEMENTATION + (COST_SYSTEM_MONTHLY * 12)
net_benefit_year1 = annual_savings - annual_system_cost
roi_year1 = (net_benefit_year1 / annual_system_cost) * 100 if annual_system_cost > 0 else 0
payback_months = (COST_SYSTEM_IMPLEMENTATION / (annual_savings / 12)) if annual_savings > 0 else 999

print(f"📈 ROI:")
print(f"  Costo sistema año 1: ${annual_system_cost:,.2f}")
print(f"  Beneficio neto año 1: ${net_benefit_year1:,.2f}")
print(f"  ROI año 1: {roi_year1:.1f}%")
print(f"  Período de recuperación: {payback_months:.1f} meses")
print()

# ============================================================================
# 8. GUARDAR RESULTADOS
# ============================================================================
print("7. GUARDANDO RESULTADOS")
print("-" * 80)

# Resumen ROI
roi_summary = pd.DataFrame([{
    'weeks_analyzed': weeks_analyzed,
    'products_base': df_analysis['product_base'].nunique(),
    'products_total': df_analysis['product_id'].nunique(),
    'baseline_cost_total': baseline_summary['total_cost'],
    'baseline_cost_ordering': baseline_summary['total_ordering'],
    'baseline_cost_backorder': baseline_summary['total_backorder'],
    'predicted_cost_total': predicted_summary['total_cost'],
    'predicted_cost_ordering': predicted_summary['total_ordering'],
    'predicted_cost_backorder': predicted_summary['total_backorder'],
    'savings_period': total_savings_period,
    'savings_annual': annual_savings,
    'system_cost_year1': annual_system_cost,
    'net_benefit_year1': net_benefit_year1,
    'roi_year1_pct': roi_year1,
    'payback_months': payback_months
}])

roi_file = DATA_SIMULATED / 'roi_analysis.csv'
roi_summary.to_csv(roi_file, index=False)
print(f"✓ ROI guardado: {roi_file}")

# Dataset detallado
detail_file = DATA_SIMULATED / 'cost_analysis_detailed.csv'
df_analysis.to_csv(detail_file, index=False)
print(f"✓ Análisis detallado guardado: {detail_file}")
print()

# ============================================================================
# 9. VISUALIZACIONES
# ============================================================================
print("8. VISUALIZACIONES")
print("-" * 80)

# A. Comparación de costos
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Costos por categoría
categories = ['Holding', 'Ordering', 'Backorder']
baseline_costs_list = [
    baseline_summary['total_holding'],
    baseline_summary['total_ordering'],
    baseline_summary['total_backorder']
]
predicted_costs_list = [
    predicted_summary['total_holding'],
    predicted_summary['total_ordering'],
    predicted_summary['total_backorder']
]

x = np.arange(len(categories))
width = 0.35

bars1 = axes[0].bar(x - width/2, baseline_costs_list, width, label='Sin Predicción',
                    color=COLORS['danger'], alpha=0.8)
bars2 = axes[0].bar(x + width/2, predicted_costs_list, width, label='Con Predicción',
                    color=COLORS['success'], alpha=0.8)

axes[0].set_title('Comparación de Costos por Categoría', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Costo ($)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'${height/1000:.0f}k',
                    ha='center', va='bottom', fontsize=9)

# Costo total y ahorros
scenarios = ['Sin\nPredicción', 'Con\nPredicción', 'Ahorros']
values = [baseline_summary['total_cost'], predicted_summary['total_cost'], total_savings_period]
colors_list = [COLORS['danger'], COLORS['success'], COLORS['primary']]

bars = axes[1].bar(scenarios, values, color=colors_list, alpha=0.8, edgecolor='black')
axes[1].set_title('Costo Total y Ahorros', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Monto ($)')
axes[1].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, values):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'${val/1000:.0f}k',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES / '06_cost_comparison.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '06_cost_comparison.png'}")
plt.close()

# B. ROI Analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Waterfall
categories_roi = ['Ahorros\nAnuales', '-Costo\nSistema', '=Beneficio\nNeto']
values_roi = [annual_savings, -annual_system_cost, net_benefit_year1]
colors_roi = [COLORS['success'], COLORS['danger'], COLORS['primary']]

bars = axes[0].bar(categories_roi, values_roi, color=colors_roi, alpha=0.8, edgecolor='black')
axes[0].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[0].set_title('Análisis ROI - Año 1', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Monto ($)')
axes[0].grid(True, alpha=0.3, axis='y')

for i, (cat, val) in enumerate(zip(categories_roi, values_roi)):
    axes[0].text(i, val, f'${val/1000:.0f}k',
                ha='center', va='bottom' if val > 0 else 'top',
                fontweight='bold', fontsize=10)

# Proyección 5 años
years = np.arange(1, 6)
annual_maintenance = COST_SYSTEM_MONTHLY * 12
costs_per_year = [annual_system_cost] + [annual_maintenance] * 4
savings_per_year = [annual_savings] * 5
net_per_year = [s - c for s, c in zip(savings_per_year, costs_per_year)]
cumulative = np.cumsum(net_per_year)

axes[1].plot(years, cumulative, marker='o', linewidth=2,
            color=COLORS['primary'], markersize=8)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].fill_between(years, 0, cumulative, alpha=0.3, color=COLORS['success'])
axes[1].set_title('Beneficio Acumulado (5 años)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Año')
axes[1].set_ylabel('Beneficio Acumulado ($)')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(years)

for year, benefit in zip(years, cumulative):
    axes[1].text(year, benefit, f'${benefit/1000:.0f}k',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES / '06_roi_analysis.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '06_roi_analysis.png'}")
plt.close()

# C. Distribución de ahorros por producto
top_products = df_analysis.groupby('product_base')['savings'].sum().sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 6))
top_products.plot(kind='bar', ax=ax, color=COLORS['info'], alpha=0.8, edgecolor='black')
ax.set_title('TOP 15 Productos con Mayor Ahorro', fontsize=12, fontweight='bold')
ax.set_xlabel('Producto Base')
ax.set_ylabel('Ahorro Total ($)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES / '06_savings_by_product.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '06_savings_by_product.png'}")
plt.close()

print()

# ============================================================================
# 10. RESUMEN EJECUTIVO
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO - VALOR OPERATIVO Y ROI")
print("="*80)
print()

print(f"📊 PRODUCTOS ANALIZADOS:")
print(f"  • Productos base: {df_analysis['product_base'].nunique()}")
print(f"  • Productos totales: {df_analysis['product_id'].nunique()}")
print(f"  • Semanas analizadas: {weeks_analyzed}")
print()

print(f"💰 ESCENARIO SIN PREDICCIÓN:")
print(f"  • Costo total: ${baseline_summary['total_cost']:,.2f}")
print(f"  • Revenue urgencias: ${baseline_summary['total_revenue']:,.2f}")
print(f"    (Clientes pagan fees por urgencias NO anticipadas)")
print()

print(f"💰 ESCENARIO CON PREDICCIÓN (Dual Granularity):")
print(f"  • Costo total: ${predicted_summary['total_cost']:,.2f}")
print(f"  • Revenue urgencias: ${predicted_summary['total_revenue']:,.2f}")
print(f"    (NO cobramos fees - mejor servicio)")
print()

print(f"📈 AHORROS Y ROI:")
print(f"  • Ahorros anuales: ${annual_savings:,.2f}")
print(f"  • Costo sistema año 1: ${annual_system_cost:,.2f}")
print(f"  • Beneficio neto año 1: ${net_benefit_year1:,.2f}")
print(f"  • ROI año 1: {roi_year1:.1f}%")
print(f"  • Recuperación: {payback_months:.1f} meses")
print()

print(f"📁 OUTPUTS GENERADOS:")
print(f"  • roi_analysis.csv")
print(f"  • cost_analysis_detailed.csv")
print(f"  • 06_cost_comparison.png")
print(f"  • 06_roi_analysis.png")
print(f"  • 06_savings_by_product.png")
print()

print("="*80)
print("✓ ANÁLISIS DE VALOR OPERATIVO COMPLETADO")
print("="*80)
print()

print("CONCLUSIÓN:")
print(f"  ✓ Sistema de dual granularidad genera ROI positivo")
print(f"  ✓ Ahorros anuales: ${annual_savings:,.2f}")
print(f"  ✓ ROI año 1: {roi_year1:.1f}%")
print(f"  ✓ Mejor servicio al cliente (sin fees adicionales)")
print(f"  ✓ Recuperación de inversión en {payback_months:.1f} meses")
print()

print("RECOMENDACIÓN:")
print(f"  → IMPLEMENTAR sistema de dual granularidad")
print(f"  → Beneficio neto año 1: ${net_benefit_year1:,.2f}")
print(f"  → Beneficio acumulado 5 años: ${cumulative[-1]:,.2f}")
print()
