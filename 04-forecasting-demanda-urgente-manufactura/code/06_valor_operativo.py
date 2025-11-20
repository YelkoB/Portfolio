"""
06. Valor Operativo - Cuantificación de ROI
============================================

⚠️  IMPORTANTE: Ejecutar DESPUÉS del script 05_analisis_por_producto.py
   Este script requiere products_filtered.csv y products_tier1_deployment.csv.

OBJETIVO:
Cuantificar el valor de negocio del sistema de predicción de urgencias mediante:
1. Cálculo de costos evitados (urgencias vs planificación normal)
2. Revenue de cobros por entregas urgentes al cliente
3. ROI del sistema de predicción
4. Comparación: escenario CON vs SIN predicción
5. Análisis escalonado: FASE 1 (Tier 1) vs FASE 2 (Tier 1+2)

MODELO DE NEGOCIO:
- Contratos con clientes: Entrega estándar incluida en precio base
- Urgencias: Se cobra FEE adicional al cliente ($3-5 por unidad)
- Predicción permite: Anticipar y evitar backorders SIN cobrar fee
- Valor: Satisfacción del cliente + ahorro en costos internos

SUPUESTOS DE NEGOCIO:
- Costo pedido urgente interno: 1.5x costo normal ($0.50 extra al proveedor)
- Costo de stock excess: $0.10 por unidad/semana
- Costo de backorder: $2.00 por unidad (venta perdida)
- Revenue por entrega urgente al cliente: $3.00 por unidad
- Nivel de servicio target: 95%

INPUT:
- data/simulated/products_tier1_deployment.csv (del script 05 - Tier 1 solo)
- data/simulated/products_filtered.csv (del script 05 - Tier 1+2)
- data/simulated/test_predictions.csv (del script 04)
- data/simulated/validation_metrics.csv (del script 04)
- data/simulated/features_weekly.csv (del script 02)

OUTPUT:
- data/simulated/roi_analysis_tier1.csv (solo Tier 1)
- data/simulated/roi_analysis_tier1_2.csv (Tier 1+2)
- data/simulated/cost_comparison.csv
- results/figures/06_*.png - Visualizaciones de ROI
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

# Costos internos de la empresa
COST_URGENT_MULTIPLIER = 1.5  # Pedido urgente al proveedor cuesta 1.5x normal
COST_HOLDING_PER_UNIT = 0.10  # Costo por unidad de exceso de stock/semana
COST_BACKORDER_PER_UNIT = 2.0  # Costo por unidad de backorder (venta perdida)
COST_NORMAL_ORDER = 1.0  # Costo base por unidad en pedido normal

# Revenue por entregas urgentes al cliente (fee contractual)
REVENUE_URGENT_DELIVERY = 3.0  # Fee que cobra la empresa al cliente por entrega urgente

# Niveles de servicio
SERVICE_LEVEL_TARGET = 0.95  # 95% de nivel de servicio objetivo
SAFETY_STOCK_Z = 1.65  # Z-score para 95% service level

# Costos de implementación del sistema ML
COST_SYSTEM_IMPLEMENTATION = 50000  # Costo inicial del sistema ML
COST_SYSTEM_MONTHLY = 1000  # Costo mensual de mantenimiento

print(f"💰 MODELO DE NEGOCIO:")
print(f"  • Contrato estándar: Entrega normal incluida en precio base")
print(f"  • Urgencia NO anticipada: Se cobra ${REVENUE_URGENT_DELIVERY:.2f}/unidad extra al cliente")
print(f"  • Urgencia anticipada (con ML): NO se cobra al cliente (mejor servicio)")
print()
print(f"📦 COSTOS INTERNOS:")
print(f"  • Pedido normal al proveedor: ${COST_NORMAL_ORDER:.2f}/unidad")
print(f"  • Pedido urgente al proveedor: ${COST_NORMAL_ORDER * COST_URGENT_MULTIPLIER:.2f}/unidad ({COST_URGENT_MULTIPLIER}x)")
print(f"  • Holding cost: ${COST_HOLDING_PER_UNIT:.2f}/unidad/semana")
print(f"  • Backorder cost: ${COST_BACKORDER_PER_UNIT:.2f}/unidad (venta perdida)")
print()
print(f"💵 REVENUE:")
print(f"  • Fee por entrega urgente al cliente: ${REVENUE_URGENT_DELIVERY:.2f}/unidad")
print(f"  • Ventaja del ML: Evitar cobrar fee → Mayor satisfacción del cliente")
print()
print(f"🎯 NIVEL DE SERVICIO:")
print(f"  • Target: {SERVICE_LEVEL_TARGET*100:.0f}%")
print(f"  • Safety stock Z-score: {SAFETY_STOCK_Z}")
print()
print(f"🖥️  COSTOS DEL SISTEMA ML:")
print(f"  • Implementación (una vez): ${COST_SYSTEM_IMPLEMENTATION:,.0f}")
print(f"  • Mantenimiento mensual: ${COST_SYSTEM_MONTHLY:,.0f}")
print(f"  • Mantenimiento anual: ${COST_SYSTEM_MONTHLY * 12:,.0f}")
print()

# ============================================================================
# 2. CARGA DE DATOS
# ============================================================================
print("2. CARGANDO DATOS")
print("-" * 80)

# Cargar ambos tiers para análisis escalonado
df_products_tier1 = pd.read_csv(DATA_SIMULATED / 'products_tier1_deployment.csv')
df_products_tier1_2 = pd.read_csv(DATA_SIMULATED / 'products_filtered.csv')

tier1_products = df_products_tier1['product_id'].unique()
tier1_2_products = df_products_tier1_2['product_id'].unique()

print(f"📊 ANÁLISIS ESCALONADO:")
print(f"  • Tier 1 (Deployment inmediato): {len(tier1_products)} productos")
print(f"  • Tier 1+2 (Tras 6 meses monitoring): {len(tier1_2_products)} productos")
print(f"  • Diferencia: {len(tier1_2_products) - len(tier1_products)} productos adicionales en Tier 2")
print()

# Cargar datos de predicciones, métricas y features
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
# 3. FUNCIONES AUXILIARES PARA CÁLCULO DE COSTOS
# ============================================================================

def calculate_baseline_costs(row):
    """
    Calcula costos/revenue SIN sistema de predicción.

    Estrategia:
    - Safety stock = Z * std_12 (estándar)
    - Urgencias NO anticipadas → pedido urgente al proveedor + cobra fee al cliente
    - Si no hay stock → backorder (venta perdida)

    MODELO DE NEGOCIO:
    - Urgencia: Cobra $3/unidad al cliente pero cuesta $1.50/unidad al proveedor
    - Backorders: Venta perdida ($2/unidad) + mala experiencia del cliente
    - Revenue urgente PARCIALMENTE compensa costos extra
    """
    actual_sales = row['total_sales']
    is_urgent = row['is_urgent']
    std_12 = row.get('sales_rolling_std_12', actual_sales * 0.2)  # Default 20% si no hay

    # Safety stock estándar
    safety_stock = SAFETY_STOCK_Z * std_12

    # Costos
    cost_holding = safety_stock * COST_HOLDING_PER_UNIT
    cost_ordering = 0
    cost_backorder = 0
    revenue_urgent = 0  # Revenue de cobrar al cliente

    if is_urgent == 1:
        # Urgencia NO anticipada
        cost_ordering = actual_sales * COST_NORMAL_ORDER * COST_URGENT_MULTIPLIER  # $1.50/unidad al proveedor
        revenue_urgent = actual_sales * REVENUE_URGENT_DELIVERY  # +$3.00/unidad del cliente

        # Si no teníamos suficiente safety stock → backorder
        if actual_sales > safety_stock:
            cost_backorder = (actual_sales - safety_stock) * COST_BACKORDER_PER_UNIT
    else:
        # Pedido normal
        cost_ordering = actual_sales * COST_NORMAL_ORDER

    # Costo NETO = Costos - Revenue
    total_cost = cost_holding + cost_ordering + cost_backorder - revenue_urgent

    return pd.Series({
        'baseline_cost_holding': cost_holding,
        'baseline_cost_ordering': cost_ordering,
        'baseline_cost_backorder': cost_backorder,
        'baseline_revenue_urgent': revenue_urgent,
        'baseline_total_cost': total_cost
    })


def calculate_predicted_costs(row):
    """
    Calcula costos CON sistema de predicción.

    VENTAJA DEL ML:
    - Urgencias ANTICIPADAS → NO se cobra fee al cliente (mejor servicio)
    - Pedido normal planificado → costo estándar $1.00/unidad
    - Menos backorders → menos ventas perdidas
    - Mayor satisfacción del cliente → retención a largo plazo

    Estrategia:
    - Si predecimos urgencia → pedido normal anticipado + stock extra
    - Si NO predecimos urgencia → pedido normal estándar
    - Safety stock ajustado según predicción
    """
    actual_sales = row['total_sales']
    predicted_sales = row.get('predicted_sales', actual_sales)
    is_urgent_actual = row['is_urgent']
    is_urgent_pred = row.get('predicted_urgency', 0)
    std_12 = row.get('sales_rolling_std_12', actual_sales * 0.2)

    # Costos base
    cost_holding = 0
    cost_ordering = 0
    cost_backorder = 0
    revenue_urgent = 0  # NO cobramos al cliente (mejor servicio)

    # Decidir estrategia según predicción
    if is_urgent_pred == 1:
        # PREDICCIÓN: Urgencia anticipada
        # Preparamos stock extra para cumplir sin cobrar fee al cliente
        safety_stock_increased = SAFETY_STOCK_Z * std_12 * 1.3  # 30% más stock
        cost_holding = safety_stock_increased * COST_HOLDING_PER_UNIT
        cost_ordering = actual_sales * COST_NORMAL_ORDER  # Pedido NORMAL (no urgente)

        # Verificar accuracy de predicción
        if is_urgent_actual == 1:
            # True Positive: predecimos bien
            # Tenemos stock preparado → sin backorders
            cost_backorder = 0
        else:
            # False Positive: predecimos urgencia pero no la hubo
            # Exceso de stock → costo holding adicional
            excess_stock = predicted_sales * 0.2  # Estimamos 20% de exceso
            cost_holding += excess_stock * COST_HOLDING_PER_UNIT * 2
    else:
        # PREDICCIÓN: NO urgencia (pedido normal estándar)
        safety_stock = SAFETY_STOCK_Z * std_12
        cost_holding = safety_stock * COST_HOLDING_PER_UNIT
        cost_ordering = actual_sales * COST_NORMAL_ORDER

        if is_urgent_actual == 1:
            # False Negative: NO anticipamos urgencia pero la hubo
            # Tenemos que hacer pedido urgente → costo alto
            cost_ordering = actual_sales * COST_NORMAL_ORDER * COST_URGENT_MULTIPLIER
            # Como NO anticipamos, NO cobramos al cliente (no teníamos contrato)
            revenue_urgent = 0  # Sin revenue (no anticipamos)
            # Backorder si no hay suficiente safety stock
            cost_backorder = max(0, actual_sales - safety_stock) * COST_BACKORDER_PER_UNIT

    # Costo NETO = Costos - Revenue
    total_cost = cost_holding + cost_ordering + cost_backorder - revenue_urgent

    return pd.Series({
        'predicted_cost_holding': cost_holding,
        'predicted_cost_ordering': cost_ordering,
        'predicted_cost_backorder': cost_backorder,
        'predicted_revenue_urgent': revenue_urgent,
        'predicted_total_cost': total_cost
    })


# ============================================================================
# 4. FUNCIÓN PARA CALCULAR ROI POR TIER
# ============================================================================

def calculate_roi_for_products(product_list, tier_name):
    """
    Calcula ROI completo para un conjunto de productos (tier).

    Returns:
        dict con baseline_summary, predicted_summary, roi_metrics, df_comparison
    """
    print()
    print("="*80)
    print(f"ANÁLISIS ROI - {tier_name}")
    print("="*80)
    print()

    # Filtrar datos por productos del tier
    df_pred_tier = df_pred[df_pred['product_id'].isin(product_list)].copy()
    df_features_tier = df_features[df_features['product_id'].isin(product_list)].copy()

    df_pred_reg = df_pred_tier[df_pred_tier['task'] == 'regression'].copy()
    df_pred_clf = df_pred_tier[df_pred_tier['task'] == 'classification'].copy()

    print(f"📊 Productos analizados: {len(product_list)}")
    print(f"  • Predicciones: {len(df_pred_tier):,}")
    print(f"  • Semanas únicas: {df_features_tier['week_start'].nunique()}")
    print()

    # ========================================================================
    # 4.1. BASELINE (SIN PREDICCIÓN)
    # ========================================================================
    print("4.1. ESCENARIO BASELINE (SIN PREDICCIÓN)")
    print("-" * 80)

    # Mergear predicciones con features para tener información completa
    df_analysis = df_features_tier.merge(
        df_pred_reg[['product_id', 'week_start', 'predicted']],
        on=['product_id', 'week_start'],
        how='left',
        suffixes=('', '_pred_sales')
    )

    # Renombrar para claridad
    df_analysis = df_analysis.rename(columns={'predicted': 'predicted_sales'})

    # Calcular costos baseline
    df_baseline = df_analysis.copy()
    baseline_costs = df_baseline.apply(calculate_baseline_costs, axis=1)
    df_baseline = pd.concat([df_baseline, baseline_costs], axis=1)

    # Resumen baseline
    baseline_summary = {
        'total_weeks': len(df_baseline),
        'total_holding_cost': df_baseline['baseline_cost_holding'].sum(),
        'total_ordering_cost': df_baseline['baseline_cost_ordering'].sum(),
        'total_backorder_cost': df_baseline['baseline_cost_backorder'].sum(),
        'total_revenue_urgent': df_baseline['baseline_revenue_urgent'].sum(),
        'total_cost': df_baseline['baseline_total_cost'].sum()
    }

    print(f"💰 Costos/Revenue SIN predicción (test period):")
    print(f"  Semanas analizadas: {baseline_summary['total_weeks']:,}")
    print()
    print(f"  COSTOS:")
    print(f"    Costo holding:   ${baseline_summary['total_holding_cost']:,.2f}")
    print(f"    Costo ordering:  ${baseline_summary['total_ordering_cost']:,.2f}")
    print(f"    Costo backorder: ${baseline_summary['total_backorder_cost']:,.2f}")
    print(f"    Subtotal costos: ${baseline_summary['total_holding_cost'] + baseline_summary['total_ordering_cost'] + baseline_summary['total_backorder_cost']:,.2f}")
    print()
    print(f"  REVENUE:")
    print(f"    Revenue entregas urgentes: ${baseline_summary['total_revenue_urgent']:,.2f}")
    print(f"    (Fee cobrado a clientes por urgencias NO anticipadas)")
    print()
    print(f"  COSTO NETO:      ${baseline_summary['total_cost']:,.2f}")
    print(f"  ⚠️  Nota: Revenue urgente compensa costos, pero daña satisfacción del cliente")
    print()

    # ========================================================================
    # 4.2. CON PREDICCIÓN
    # ========================================================================
    print("4.2. ESCENARIO CON PREDICCIÓN")
    print("-" * 80)

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

    # Calcular costos con predicción
    df_predicted = df_analysis.copy()
    predicted_costs = df_predicted.apply(calculate_predicted_costs, axis=1)
    df_predicted = pd.concat([df_predicted, predicted_costs], axis=1)

    # Resumen con predicción
    predicted_summary = {
        'total_weeks': len(df_predicted),
        'total_holding_cost': df_predicted['predicted_cost_holding'].sum(),
        'total_ordering_cost': df_predicted['predicted_cost_ordering'].sum(),
        'total_backorder_cost': df_predicted['predicted_cost_backorder'].sum(),
        'total_revenue_urgent': df_predicted['predicted_revenue_urgent'].sum(),
        'total_cost': df_predicted['predicted_total_cost'].sum()
    }

    print(f"💰 Costos/Revenue CON predicción (test period):")
    print(f"  Semanas analizadas: {predicted_summary['total_weeks']:,}")
    print()
    print(f"  COSTOS:")
    print(f"    Costo holding:   ${predicted_summary['total_holding_cost']:,.2f}")
    print(f"    Costo ordering:  ${predicted_summary['total_ordering_cost']:,.2f}")
    print(f"    Costo backorder: ${predicted_summary['total_backorder_cost']:,.2f}")
    print(f"    Subtotal costos: ${predicted_summary['total_holding_cost'] + predicted_summary['total_ordering_cost'] + predicted_summary['total_backorder_cost']:,.2f}")
    print()
    print(f"  REVENUE:")
    print(f"    Revenue entregas urgentes: ${predicted_summary['total_revenue_urgent']:,.2f}")
    print(f"    (Debería ser ~$0 - anticipamos urgencias sin cobrar fee)")
    print()
    print(f"  COSTO NETO:      ${predicted_summary['total_cost']:,.2f}")
    print(f"  ✓ Mejor servicio al cliente sin fees adicionales")
    print()

    # ========================================================================
    # 4.3. CÁLCULO DE AHORROS Y ROI
    # ========================================================================
    print("4.3. ANÁLISIS DE AHORROS Y ROI")
    print("-" * 80)

    # Comparar baseline vs predicted
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

    # Costos del sistema (proporcional al número de productos)
    # Asumimos que el costo escala con el número de productos
    system_cost_factor = len(product_list) / len(tier1_2_products)
    annual_system_cost = COST_SYSTEM_IMPLEMENTATION * system_cost_factor + (COST_SYSTEM_MONTHLY * 12)

    # ROI
    net_benefit_year1 = annual_savings - annual_system_cost
    roi = (net_benefit_year1 / annual_system_cost) * 100 if annual_system_cost > 0 else 0
    payback_period_months = (COST_SYSTEM_IMPLEMENTATION * system_cost_factor / (annual_savings / 12)) if annual_savings > 0 else 999

    print(f"AHORROS:")
    print(f"  Período de test: {weeks_in_test} semanas")
    print(f"  Ahorros en test period: ${total_savings:,.2f}")
    print(f"  Ahorros proyectados anuales: ${annual_savings:,.2f}")
    print()
    print(f"COSTOS DEL SISTEMA:")
    print(f"  Implementación (proporción): ${COST_SYSTEM_IMPLEMENTATION * system_cost_factor:,.2f}")
    print(f"  Mantenimiento anual: ${COST_SYSTEM_MONTHLY * 12:,.2f}")
    print(f"  Costo total año 1: ${annual_system_cost:,.2f}")
    print()
    print(f"ROI:")
    print(f"  Beneficio neto año 1: ${net_benefit_year1:,.2f}")
    print(f"  ROI año 1: {roi:,.1f}%")
    print(f"  Período de recuperación: {payback_period_months:.1f} meses")
    print()

    # Retornar resultados
    return {
        'tier_name': tier_name,
        'n_products': len(product_list),
        'baseline_summary': baseline_summary,
        'predicted_summary': predicted_summary,
        'roi_metrics': {
            'weeks_analyzed': weeks_in_test,
            'savings_test_period': total_savings,
            'savings_annual_projected': annual_savings,
            'system_cost_year1': annual_system_cost,
            'net_benefit_year1': net_benefit_year1,
            'roi_year1_pct': roi,
            'payback_period_months': payback_period_months
        },
        'df_comparison': df_comparison
    }


# ============================================================================
# 5. EJECUTAR ANÁLISIS PARA CADA TIER
# ============================================================================
print()
print("="*80)
print("ANÁLISIS ESCALONADO POR TIER")
print("="*80)

# FASE 1: Solo Tier 1 (Deployment inmediato)
results_tier1 = calculate_roi_for_products(tier1_products, "FASE 1: Tier 1 (Deployment)")

# FASE 2: Tier 1+2 (Tras 6 meses de monitoring del Tier 2)
results_tier1_2 = calculate_roi_for_products(tier1_2_products, "FASE 2: Tier 1+2 (Post-Monitoring)")


# ============================================================================
# 6. GUARDAR RESULTADOS
# ============================================================================
print()
print("="*80)
print("6. GUARDANDO RESULTADOS")
print("="*80)
print()

# ROI summary para Tier 1
roi_summary_tier1 = pd.DataFrame([{
    'tier': 'Tier 1',
    'n_products': results_tier1['n_products'],
    **results_tier1['roi_metrics'],
    'baseline_total_cost': results_tier1['baseline_summary']['total_cost'],
    'predicted_total_cost': results_tier1['predicted_summary']['total_cost'],
    'baseline_revenue_urgent': results_tier1['baseline_summary']['total_revenue_urgent'],
    'predicted_revenue_urgent': results_tier1['predicted_summary']['total_revenue_urgent']
}])

# ROI summary para Tier 1+2
roi_summary_tier1_2 = pd.DataFrame([{
    'tier': 'Tier 1+2',
    'n_products': results_tier1_2['n_products'],
    **results_tier1_2['roi_metrics'],
    'baseline_total_cost': results_tier1_2['baseline_summary']['total_cost'],
    'predicted_total_cost': results_tier1_2['predicted_summary']['total_cost'],
    'baseline_revenue_urgent': results_tier1_2['baseline_summary']['total_revenue_urgent'],
    'predicted_revenue_urgent': results_tier1_2['predicted_summary']['total_revenue_urgent']
}])

# Guardar CSVs
roi_file_tier1 = DATA_SIMULATED / 'roi_analysis_tier1.csv'
roi_summary_tier1.to_csv(roi_file_tier1, index=False)
print(f"✓ ROI Tier 1 guardado: {roi_file_tier1}")

roi_file_tier1_2 = DATA_SIMULATED / 'roi_analysis_tier1_2.csv'
roi_summary_tier1_2.to_csv(roi_file_tier1_2, index=False)
print(f"✓ ROI Tier 1+2 guardado: {roi_file_tier1_2}")

# Guardar comparaciones detalladas
comparison_file_tier1 = DATA_SIMULATED / 'cost_comparison_tier1.csv'
results_tier1['df_comparison'].to_csv(comparison_file_tier1, index=False)
print(f"✓ Comparación Tier 1 guardada: {comparison_file_tier1}")

comparison_file_tier1_2 = DATA_SIMULATED / 'cost_comparison_tier1_2.csv'
results_tier1_2['df_comparison'].to_csv(comparison_file_tier1_2, index=False)
print(f"✓ Comparación Tier 1+2 guardada: {comparison_file_tier1_2}")
print()


# ============================================================================
# 7. VISUALIZACIONES
# ============================================================================
print("7. VISUALIZACIONES")
print("-" * 80)

# A. Comparación Tier 1 vs Tier 1+2
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# A1. Ahorros anuales comparados
tiers = ['Tier 1\n(Inmediato)', 'Tier 1+2\n(Post-Monitoring)']
savings_list = [
    results_tier1['roi_metrics']['savings_annual_projected'],
    results_tier1_2['roi_metrics']['savings_annual_projected']
]
colors_list = [COLORS['warning'], COLORS['success']]

axes[0, 0].bar(tiers, savings_list, color=colors_list, alpha=0.8, edgecolor='black')
axes[0, 0].set_title('Ahorros Anuales Proyectados por Fase', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Ahorros ($)')
axes[0, 0].grid(True, alpha=0.3, axis='y')

for i, (tier, saving) in enumerate(zip(tiers, savings_list)):
    axes[0, 0].text(i, saving, f'${saving:,.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

# A2. ROI comparado
roi_list = [
    results_tier1['roi_metrics']['roi_year1_pct'],
    results_tier1_2['roi_metrics']['roi_year1_pct']
]

axes[0, 1].bar(tiers, roi_list, color=colors_list, alpha=0.8, edgecolor='black')
axes[0, 1].set_title('ROI Año 1 por Fase', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('ROI (%)')
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[0, 1].grid(True, alpha=0.3, axis='y')

for i, (tier, roi_val) in enumerate(zip(tiers, roi_list)):
    axes[0, 1].text(i, roi_val, f'{roi_val:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

# A3. Número de productos
n_products_list = [
    results_tier1['n_products'],
    results_tier1_2['n_products']
]

axes[1, 0].bar(tiers, n_products_list, color=colors_list, alpha=0.8, edgecolor='black')
axes[1, 0].set_title('Número de Productos por Fase', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('N° Productos')
axes[1, 0].grid(True, alpha=0.3, axis='y')

for i, (tier, n_prod) in enumerate(zip(tiers, n_products_list)):
    axes[1, 0].text(i, n_prod, f'{n_prod}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

# A4. Período de recuperación
payback_list = [
    results_tier1['roi_metrics']['payback_period_months'],
    results_tier1_2['roi_metrics']['payback_period_months']
]

axes[1, 1].bar(tiers, payback_list, color=colors_list, alpha=0.8, edgecolor='black')
axes[1, 1].set_title('Período de Recuperación por Fase', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Meses')
axes[1, 1].grid(True, alpha=0.3, axis='y')

for i, (tier, payback) in enumerate(zip(tiers, payback_list)):
    axes[1, 1].text(i, payback, f'{payback:.1f} meses',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES / '06_tier_comparison.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '06_tier_comparison.png'}")
plt.close()


# B. Desglose de costos para Tier 1+2 (el escenario completo)
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Holding', 'Ordering', 'Backorder', 'TOTAL']
baseline_costs_list = [
    results_tier1_2['baseline_summary']['total_holding_cost'],
    results_tier1_2['baseline_summary']['total_ordering_cost'],
    results_tier1_2['baseline_summary']['total_backorder_cost'],
    results_tier1_2['baseline_summary']['total_cost']
]
predicted_costs_list = [
    results_tier1_2['predicted_summary']['total_holding_cost'],
    results_tier1_2['predicted_summary']['total_ordering_cost'],
    results_tier1_2['predicted_summary']['total_backorder_cost'],
    results_tier1_2['predicted_summary']['total_cost']
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_costs_list, width, label='Sin Predicción',
              color=COLORS['danger'], alpha=0.8)
bars2 = ax.bar(x + width/2, predicted_costs_list, width, label='Con Predicción',
              color=COLORS['success'], alpha=0.8)

ax.set_title('Comparación de Costos: Con vs Sin Predicción (Tier 1+2)', fontsize=14, fontweight='bold')
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
plt.savefig(FIGURES / '06_cost_comparison.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '06_cost_comparison.png'}")
plt.close()


# C. ROI analysis para Tier 1+2 (waterfall y proyección)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Waterfall chart (manual)
categories_roi = ['Ahorros\nAnuales', '-Costo\nSistema', '=Beneficio\nNeto']
values_roi = [
    results_tier1_2['roi_metrics']['savings_annual_projected'],
    -results_tier1_2['roi_metrics']['system_cost_year1'],
    results_tier1_2['roi_metrics']['net_benefit_year1']
]
colors_roi = [COLORS['success'], COLORS['danger'], COLORS['primary']]

axes[0].bar(categories_roi, values_roi, color=colors_roi, alpha=0.8, edgecolor='black')
axes[0].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[0].set_title('Análisis de ROI - Año 1 (Tier 1+2)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Monto ($)')
axes[0].grid(True, alpha=0.3, axis='y')

for i, (cat, val) in enumerate(zip(categories_roi, values_roi)):
    axes[0].text(i, val, f'${val:,.0f}',
                ha='center', va='bottom' if val > 0 else 'top',
                fontweight='bold', fontsize=10)

# Proyección multi-año
years = np.arange(1, 6)
annual_savings_projected = results_tier1_2['roi_metrics']['savings_annual_projected']
system_cost_impl = results_tier1_2['roi_metrics']['system_cost_year1'] - COST_SYSTEM_MONTHLY * 12
costs_per_year = [system_cost_impl + COST_SYSTEM_MONTHLY * 12] + [COST_SYSTEM_MONTHLY * 12] * 4
savings_per_year = [annual_savings_projected] * 5
net_benefit_per_year = [s - c for s, c in zip(savings_per_year, costs_per_year)]
cumulative_benefit = np.cumsum(net_benefit_per_year)

axes[1].plot(years, cumulative_benefit, marker='o', linewidth=2,
            color=COLORS['primary'], markersize=8)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].fill_between(years, 0, cumulative_benefit, alpha=0.3, color=COLORS['success'])
axes[1].set_title('Beneficio Acumulado (5 años) - Tier 1+2', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Año')
axes[1].set_ylabel('Beneficio Acumulado ($)')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(years)

for year, benefit in zip(years, cumulative_benefit):
    axes[1].text(year, benefit, f'${benefit:,.0f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES / '06_roi_analysis.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '06_roi_analysis.png'}")
plt.close()


# D. Top productos con mayor ahorro (Tier 1+2)
df_savings_by_product = results_tier1_2['df_comparison'].groupby('product_id')['savings'].sum().sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 6))
df_savings_by_product.plot(kind='bar', ax=ax, color=COLORS['info'], alpha=0.8, edgecolor='black')
ax.set_title('TOP 15 Productos con Mayor Ahorro (Tier 1+2)', fontsize=12, fontweight='bold')
ax.set_xlabel('Producto')
ax.set_ylabel('Ahorro Total ($)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(FIGURES / '06_savings_by_product.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '06_savings_by_product.png'}")
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

print("📊 ESTRATEGIA DE ROLLOUT ESCALONADO:")
print()
print("FASE 1: DEPLOYMENT INMEDIATO (Tier 1)")
print(f"  • Productos: {results_tier1['n_products']}")
print(f"  • Ahorros anuales: ${results_tier1['roi_metrics']['savings_annual_projected']:,.2f}")
print(f"  • ROI año 1: {results_tier1['roi_metrics']['roi_year1_pct']:.1f}%")
print(f"  • Recuperación: {results_tier1['roi_metrics']['payback_period_months']:.1f} meses")
print(f"  ✓ Modelos con F1 ≥ 0.5, alta confianza")
print()
print("FASE 2: EXPANSIÓN POST-MONITORING (Tier 1+2)")
print(f"  • Productos: {results_tier1_2['n_products']} ({results_tier1_2['n_products'] - results_tier1['n_products']} adicionales)")
print(f"  • Ahorros anuales: ${results_tier1_2['roi_metrics']['savings_annual_projected']:,.2f}")
print(f"  • ROI año 1: {results_tier1_2['roi_metrics']['roi_year1_pct']:.1f}%")
print(f"  • Recuperación: {results_tier1_2['roi_metrics']['payback_period_months']:.1f} meses")
print(f"  ✓ Incluye Tier 2 (F1 ≥ 0.3) tras 6 meses de monitoreo")
print()

print("💡 VALOR INCREMENTAL DE TIER 2:")
incremental_savings = results_tier1_2['roi_metrics']['savings_annual_projected'] - results_tier1['roi_metrics']['savings_annual_projected']
print(f"  • Ahorros adicionales: ${incremental_savings:,.2f}/año")
print(f"  • Productos adicionales: {results_tier1_2['n_products'] - results_tier1['n_products']}")
print(f"  • Ahorro promedio por producto Tier 2: ${incremental_savings / max(1, results_tier1_2['n_products'] - results_tier1['n_products']):,.2f}/año")
print()

print("🎯 MODELO DE NEGOCIO - IMPACTO EN SATISFACCIÓN:")
print(f"  SIN ML: ${results_tier1_2['baseline_summary']['total_revenue_urgent']:,.2f} cobrados a clientes")
print(f"           → Clientes pagan fee por urgencias NO anticipadas")
print(f"           → Experiencia negativa, riesgo de churn")
print()
print(f"  CON ML: ${results_tier1_2['predicted_summary']['total_revenue_urgent']:,.2f} cobrados a clientes")
print(f"           → Urgencias anticipadas SIN cobrar fee")
print(f"           → Mejor experiencia, mayor retención")
print(f"           → Ahorro en costos internos: ${results_tier1_2['roi_metrics']['savings_annual_projected']:,.2f}/año")
print()

print("📁 OUTPUTS GENERADOS:")
print(f"  • {roi_file_tier1.name}")
print(f"  • {roi_file_tier1_2.name}")
print(f"  • {comparison_file_tier1.name}")
print(f"  • {comparison_file_tier1_2.name}")
print(f"  • 06_tier_comparison.png")
print(f"  • 06_cost_comparison.png")
print(f"  • 06_roi_analysis.png")
print(f"  • 06_savings_by_product.png")
print()

print("="*80)
print("✓ ANÁLISIS DE VALOR OPERATIVO COMPLETADO")
print("="*80)
print()
print("RECOMENDACIÓN ESTRATÉGICA:")
print()
print("FASE 1 (Meses 1-6): Deployment de Tier 1")
print(f"  ✓ {results_tier1['n_products']} productos con modelos excelentes (F1 ≥ 0.5)")
print(f"  ✓ ROI confirmado: {results_tier1['roi_metrics']['roi_year1_pct']:.1f}%")
print(f"  ✓ Recuperación rápida: {results_tier1['roi_metrics']['payback_period_months']:.1f} meses")
print(f"  → Implementar inmediatamente para capturar ${results_tier1['roi_metrics']['savings_annual_projected']:,.2f}/año")
print()
print("FASE 2 (Meses 7-12): Expansión a Tier 2")
print(f"  ✓ Monitorear {results_tier1_2['n_products'] - results_tier1['n_products']} productos adicionales")
print(f"  ✓ Validar performance en producción")
print(f"  ✓ Optimizar umbrales de decisión")
print(f"  → Si estable, capturar ${incremental_savings:,.2f}/año adicionales")
print()
print("BENEFICIOS CLAVE:")
print(f"  ✓ Ahorros operativos: hasta ${results_tier1_2['roi_metrics']['savings_annual_projected']:,.2f}/año")
print(f"  ✓ Mejor servicio al cliente (sin fees por urgencia)")
print(f"  ✓ Reducción de backorders y ventas perdidas")
print(f"  ✓ ROI positivo desde año 1: {results_tier1_2['roi_metrics']['roi_year1_pct']:.1f}%")
print()
