"""
05. Análisis Detallado por Producto
====================================

OBJETIVO:
Analizar el rendimiento de cada producto individualmente, filtrar productos
con datos insuficientes, y generar métricas confiables para análisis de ROI.

ANÁLISIS:
1. Cargar métricas de validación y predicciones de test
2. Calcular estadísticas por producto (muestras, urgencias, distribución)
3. Filtrar productos con datos insuficientes
4. Identificar productos con mejor/peor rendimiento
5. Análisis de segmentos (categoría, tienda)
6. Generar dataset limpio para análisis de ROI

CRITERIOS DE FILTRADO:
- Mínimo 20 semanas en test set
- Mínimo 5 urgencias observadas en test (para clasificación)
- Al menos 1 modelo entrenado exitosamente

INPUT:
- data/simulated/validation_metrics.csv (del script 04)
- data/simulated/test_predictions.csv (del script 04)

OUTPUT:
- data/simulated/product_analysis.csv - Análisis completo por producto
- data/simulated/products_filtered.csv - Solo productos válidos (para script 06)
- results/figures/05_*.png - Visualizaciones de análisis
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
    DATA_SIMULATED, FIGURES, PROJECT_ROOT,
    FIGSIZE_STANDARD, FIGSIZE_WIDE,
    COLORS, RANDOM_SEED
)

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

print("="*80)
print("ANÁLISIS DETALLADO POR PRODUCTO")
print("="*80)
print()

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================
print("1. CARGANDO DATOS")
print("-" * 80)

# Cargar métricas de validación
df_metrics = pd.read_csv(DATA_SIMULATED / 'validation_metrics.csv')
print(f"✓ Métricas cargadas: {df_metrics.shape}")
print(f"  Productos evaluados: {df_metrics['product_id'].nunique()}")

# Cargar predicciones
df_preds = pd.read_csv(DATA_SIMULATED / 'test_predictions.csv')
print(f"✓ Predicciones cargadas: {len(df_preds):,}")

print()

# ============================================================================
# 2. CALCULAR ESTADÍSTICAS POR PRODUCTO
# ============================================================================
print("2. CALCULANDO ESTADÍSTICAS POR PRODUCTO")
print("-" * 80)

# Calcular estadísticas desde las predicciones
product_stats = df_preds.groupby(['product_id', 'task']).agg({
    'week_start': 'count',  # número de semanas
    'actual': ['sum', 'mean', 'std']
}).reset_index()

# Aplanar columnas multi-index
product_stats.columns = ['product_id', 'task', 'n_weeks', 'sum_actual', 'mean_actual', 'std_actual']

# Separar regresión y clasificación
regression_stats = product_stats[product_stats['task'] == 'regression'][['product_id', 'n_weeks']].copy()
regression_stats.columns = ['product_id', 'test_samples_reg']

classification_stats = product_stats[product_stats['task'] == 'classification'].copy()
classification_stats['test_urgencies'] = classification_stats['sum_actual'].astype(int)
classification_stats = classification_stats[['product_id', 'n_weeks', 'test_urgencies']]
classification_stats.columns = ['product_id', 'test_samples_clf', 'test_urgencies']

print(f"✓ Estadísticas de regresión calculadas: {len(regression_stats)} productos")
print(f"✓ Estadísticas de clasificación calculadas: {len(classification_stats)} productos")
print()

# ============================================================================
# 3. FUSIONAR DATOS
# ============================================================================
print("3. FUSIONANDO DATOS")
print("-" * 80)

# Merge con métricas
df_analysis = df_metrics.copy()

# Añadir estadísticas de regresión
df_analysis = df_analysis.merge(
    regression_stats,
    on='product_id',
    how='left'
)

# Añadir estadísticas de clasificación
df_analysis = df_analysis.merge(
    classification_stats,
    on='product_id',
    how='left'
)

# Rellenar NaN con 0
df_analysis['test_samples_reg'] = df_analysis['test_samples_reg'].fillna(0).astype(int)
df_analysis['test_samples_clf'] = df_analysis['test_samples_clf'].fillna(0).astype(int)
df_analysis['test_urgencies'] = df_analysis['test_urgencies'].fillna(0).astype(int)

# Crear columna unificada test_samples (usar el máximo)
df_analysis['test_samples'] = df_analysis[['test_samples_reg', 'test_samples_clf']].max(axis=1)

print(f"✓ Datos fusionados: {df_analysis.shape}")
print(f"  Productos totales: {df_analysis['product_id'].nunique()}")
print()

# ============================================================================
# 4. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================
print("4. ESTADÍSTICAS DESCRIPTIVAS")
print("-" * 80)

print("Distribución de muestras en test:")
print(df_analysis.groupby('task')['test_samples'].describe())
print()

print("Distribución de urgencias en test (clasificación):")
clf_products = df_analysis[df_analysis['task'] == 'classification']
print(clf_products['test_urgencies'].describe())
print()

# ============================================================================
# 5. FILTRADO BÁSICO (Datos suficientes)
# ============================================================================
print("5. FILTRADO BÁSICO - DATOS SUFICIENTES")
print("-" * 80)

MIN_TEST_SAMPLES = 20
MIN_TEST_URGENCIES = 5

print(f"⚠️  Criterios básicos:")
print(f"   • Mínimo de muestras en test: {MIN_TEST_SAMPLES}")
print(f"   • Mínimo de urgencias en test (clasificación): {MIN_TEST_URGENCIES}")
print()

# Aplicar filtros básicos
df_basic_filter = df_analysis.copy()

# Filtro 1: Mínimo de muestras
df_basic_filter = df_basic_filter[df_basic_filter['test_samples'] >= MIN_TEST_SAMPLES]

# Filtro 2: Para clasificación, mínimo de urgencias
mask_regression = df_basic_filter['task'] == 'regression'
mask_classification_valid = (
    (df_basic_filter['task'] == 'classification') &
    (df_basic_filter['test_urgencies'] >= MIN_TEST_URGENCIES)
)

df_basic_filter = df_basic_filter[mask_regression | mask_classification_valid]

print(f"✓ Productos antes del filtro básico: {len(df_analysis)}")
print(f"✓ Productos después del filtro básico: {len(df_basic_filter)}")
print(f"❌ Productos eliminados: {len(df_analysis) - len(df_basic_filter)} ({(len(df_analysis) - len(df_basic_filter))/len(df_analysis)*100:.1f}%)")
print()

# ============================================================================
# 6. FILTRADO POR CALIDAD DEL MODELO (Performance)
# ============================================================================
print("6. FILTRADO POR CALIDAD DEL MODELO")
print("-" * 80)

# Criterios de calidad
MIN_F1_TIER1 = 0.5  # Excelente
MIN_F1_TIER2 = 0.3  # Aceptable
MIN_RECALL_TIER1 = 0.4
MIN_RECALL_TIER2 = 0.2

MAX_RMSE_RATIO_TIER1 = 1.0  # RMSE/MAE < 1.0 (predicciones consistentes)
MAX_RMSE_RATIO_TIER2 = 1.5  # RMSE/MAE < 1.5 (aceptable)

print(f"📊 CRITERIOS DE CALIDAD:")
print(f"")
print(f"CLASIFICACIÓN (Predicción de urgencias):")
print(f"  Tier 1 (Deployment Inmediato):")
print(f"    • F1-Score ≥ {MIN_F1_TIER1}")
print(f"    • Recall ≥ {MIN_RECALL_TIER1}")
print(f"  Tier 2 (Monitoring):")
print(f"    • F1-Score ≥ {MIN_F1_TIER2}")
print(f"    • Recall ≥ {MIN_RECALL_TIER2}")
print(f"  Tier 3 (Rechazado): Resto")
print(f"")
print(f"REGRESIÓN (Predicción de ventas):")
print(f"  Tier 1: RMSE/MAE ≤ {MAX_RMSE_RATIO_TIER1} (predicciones consistentes)")
print(f"  Tier 2: RMSE/MAE ≤ {MAX_RMSE_RATIO_TIER2} (aceptable)")
print(f"  Tier 3 (Rechazado): Resto")
print()

# Separar por tarea
df_reg = df_basic_filter[df_basic_filter['task'] == 'regression'].copy()
df_clf = df_basic_filter[df_basic_filter['task'] == 'classification'].copy()

# Calcular RMSE/MAE ratio para regresión
if len(df_reg) > 0:
    df_reg['rmse_mae_ratio'] = df_reg['rmse'] / df_reg['mae']

    # Clasificar en tiers
    df_reg['tier'] = 'Tier 3 - Rechazado'
    df_reg.loc[df_reg['rmse_mae_ratio'] <= MAX_RMSE_RATIO_TIER2, 'tier'] = 'Tier 2 - Monitoring'
    df_reg.loc[df_reg['rmse_mae_ratio'] <= MAX_RMSE_RATIO_TIER1, 'tier'] = 'Tier 1 - Deployment'

# Clasificar en tiers para clasificación
if len(df_clf) > 0:
    df_clf['tier'] = 'Tier 3 - Rechazado'

    # Tier 2: F1 >= 0.3 Y Recall >= 0.2
    mask_tier2 = (df_clf['f1'] >= MIN_F1_TIER2) & (df_clf['recall'] >= MIN_RECALL_TIER2)
    df_clf.loc[mask_tier2, 'tier'] = 'Tier 2 - Monitoring'

    # Tier 1: F1 >= 0.5 Y Recall >= 0.4
    mask_tier1 = (df_clf['f1'] >= MIN_F1_TIER1) & (df_clf['recall'] >= MIN_RECALL_TIER1)
    df_clf.loc[mask_tier1, 'tier'] = 'Tier 1 - Deployment'

# Combinar
df_filtered = pd.concat([df_reg, df_clf], ignore_index=True)

# Estadísticas de filtrado
print("RESULTADOS DEL FILTRADO POR CALIDAD:")
print()

if len(df_reg) > 0:
    print("REGRESIÓN:")
    tier_counts_reg = df_reg['tier'].value_counts().sort_index()
    for tier, count in tier_counts_reg.items():
        pct = count / len(df_reg) * 100
        print(f"  {tier}: {count} productos ({pct:.1f}%)")
    print()

if len(df_clf) > 0:
    print("CLASIFICACIÓN:")
    tier_counts_clf = df_clf['tier'].value_counts().sort_index()
    for tier, count in tier_counts_clf.items():
        pct = count / len(df_clf) * 100
        print(f"  {tier}: {count} productos ({pct:.1f}%)")
    print()

# Productos aptos para deployment (Tier 1 + Tier 2)
df_deployment = df_filtered[df_filtered['tier'].isin(['Tier 1 - Deployment', 'Tier 2 - Monitoring'])].copy()
df_tier1 = df_filtered[df_filtered['tier'] == 'Tier 1 - Deployment'].copy()
df_rejected = df_filtered[df_filtered['tier'] == 'Tier 3 - Rechazado'].copy()

print(f"📊 RESUMEN GLOBAL:")
print(f"  Total productos analizados: {len(df_analysis)}")
print(f"  Tier 1 (Deployment Inmediato): {len(df_tier1)} ({len(df_tier1)/len(df_filtered)*100:.1f}%)")
print(f"  Tier 2 (Monitoring): {len(df_deployment) - len(df_tier1)} ({(len(df_deployment) - len(df_tier1))/len(df_filtered)*100:.1f}%)")
print(f"  Tier 3 (Rechazado): {len(df_rejected)} ({len(df_rejected)/len(df_filtered)*100:.1f}%)")
print(f"")
print(f"  ✅ APTOS PARA USAR: {len(df_deployment)} ({len(df_deployment)/len(df_filtered)*100:.1f}%)")
print(f"  ❌ NO USAR: {len(df_rejected)} ({len(df_rejected)/len(df_filtered)*100:.1f}%)")
print()

# ============================================================================
# 7. ANÁLISIS POR TIER
# ============================================================================
print("7. ANÁLISIS POR TIER")
print("-" * 80)

# Regresión
df_reg = df_filtered[df_filtered['task'] == 'regression'].copy()
if len(df_reg) > 0:
    print("REGRESIÓN:")
    print(f"  Productos válidos: {len(df_reg)}")
    print(f"  RMSE promedio: {df_reg['rmse'].mean():.2f}")
    print(f"  MAE promedio: {df_reg['mae'].mean():.2f}")
    print(f"  MAPE promedio: {df_reg['mape'].mean():.2f}%")
    print()

    # Top 10 mejores
    top_10_reg = df_reg.nsmallest(10, 'rmse')[['product_id', 'model', 'rmse', 'mae', 'test_samples']]
    print("  Top 10 productos (mejor RMSE):")
    print(top_10_reg.to_string(index=False))
    print()

# Clasificación
df_clf = df_filtered[df_filtered['task'] == 'classification'].copy()
if len(df_clf) > 0:
    print("CLASIFICACIÓN:")
    print(f"  Productos válidos: {len(df_clf)}")
    print(f"  F1-Score promedio: {df_clf['f1'].mean():.3f}")
    print(f"  Precision promedio: {df_clf['precision'].mean():.3f}")
    print(f"  Recall promedio: {df_clf['recall'].mean():.3f}")
    print(f"  ROC-AUC promedio: {df_clf['auc'].mean():.3f}")
    print()

    # Top 10 mejores
    top_10_clf = df_clf.nlargest(10, 'f1')[['product_id', 'model', 'f1', 'precision', 'recall', 'test_urgencies']]
    print("  Top 10 productos (mejor F1):")
    print(top_10_clf.to_string(index=False))
    print()

# ============================================================================
# 7. ANÁLISIS POR MODELO
# ============================================================================
print("7. COMPARACIÓN DE MODELOS")
print("-" * 80)

model_comparison = df_filtered.groupby(['task', 'model']).agg({
    'product_id': 'count',
    'rmse': 'mean',
    'mae': 'mean',
    'mape': 'mean',
    'f1': 'mean',
    'precision': 'mean',
    'recall': 'mean',
    'auc': 'mean'
}).round(3)

model_comparison.columns = ['n_products', 'rmse_avg', 'mae_avg', 'mape_avg', 'f1_avg', 'precision_avg', 'recall_avg', 'auc_avg']

print(model_comparison)
print()

# ============================================================================
# 8. GUARDAR RESULTADOS
# ============================================================================
print("8. GUARDANDO RESULTADOS")
print("-" * 80)

# Guardar análisis completo (con columna 'tier')
output_analysis = DATA_SIMULATED / 'product_analysis.csv'
df_filtered.to_csv(output_analysis, index=False)
print(f"✓ Análisis completo (con tiers) guardado: {output_analysis}")

# Guardar solo productos aptos (Tier 1 + Tier 2)
output_deployment = DATA_SIMULATED / 'products_filtered.csv'
df_deployment.to_csv(output_deployment, index=False)
print(f"✓ Productos APTOS (Tier 1+2) guardados: {output_deployment}")
print(f"  Total productos aptos: {len(df_deployment)}")

# Guardar Tier 1 (deployment inmediato)
output_tier1 = DATA_SIMULATED / 'products_tier1_deployment.csv'
df_tier1.to_csv(output_tier1, index=False)
print(f"✓ Productos Tier 1 (deployment) guardados: {output_tier1}")
print(f"  Total Tier 1: {len(df_tier1)}")

# Guardar rechazados para análisis
output_rejected = DATA_SIMULATED / 'products_tier3_rejected.csv'
df_rejected.to_csv(output_rejected, index=False)
print(f"✓ Productos Tier 3 (rechazados) guardados: {output_rejected}")
print(f"  Total rechazados: {len(df_rejected)}")
print()

# ============================================================================
# 9. VISUALIZACIONES
# ============================================================================
print("9. VISUALIZACIONES")
print("-" * 80)

# 9.1 Distribución de muestras en test
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma de test_samples
axes[0].hist(df_filtered['test_samples'], bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='black')
axes[0].axvline(MIN_TEST_SAMPLES, color='red', linestyle='--', linewidth=2, label=f'Mínimo: {MIN_TEST_SAMPLES}')
axes[0].set_xlabel('Número de semanas en test', fontsize=11)
axes[0].set_ylabel('Frecuencia', fontsize=11)
axes[0].set_title('Distribución de muestras en test set', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Histograma de urgencias (solo clasificación)
if len(df_clf) > 0:
    axes[1].hist(df_clf['test_urgencies'], bins=30, color=COLORS['secondary'], alpha=0.7, edgecolor='black')
    axes[1].axvline(MIN_TEST_URGENCIES, color='red', linestyle='--', linewidth=2, label=f'Mínimo: {MIN_TEST_URGENCIES}')
    axes[1].set_xlabel('Número de urgencias en test', fontsize=11)
    axes[1].set_ylabel('Frecuencia', fontsize=11)
    axes[1].set_title('Distribución de urgencias en test set', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / '05_data_distribution.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '05_data_distribution.png'}")
plt.close()

# 9.2 Rendimiento por modelo
if len(df_filtered) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regresión: RMSE por modelo
    if len(df_reg) > 0:
        df_reg.boxplot(column='rmse', by='model', ax=axes[0])
        axes[0].set_title('RMSE por Modelo (Regresión)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Modelo', fontsize=11)
        axes[0].set_ylabel('RMSE', fontsize=11)
        axes[0].get_figure().suptitle('')
        axes[0].grid(True, alpha=0.3, axis='y')

    # Clasificación: F1 por modelo
    if len(df_clf) > 0:
        df_clf.boxplot(column='f1', by='model', ax=axes[1])
        axes[1].set_title('F1-Score por Modelo (Clasificación)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Modelo', fontsize=11)
        axes[1].set_ylabel('F1-Score', fontsize=11)
        axes[1].get_figure().suptitle('')
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES / '05_model_performance.png', dpi=100, bbox_inches='tight')
    print(f"✓ Guardado: {FIGURES / '05_model_performance.png'}")
    plt.close()

# 9.3 Scatter: Muestras vs Performance
if len(df_reg) > 0 or len(df_clf) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Regresión: test_samples vs RMSE
    if len(df_reg) > 0:
        axes[0].scatter(df_reg['test_samples'], df_reg['rmse'], alpha=0.5, c=COLORS['primary'], s=50)
        axes[0].set_xlabel('Número de semanas en test', fontsize=11)
        axes[0].set_ylabel('RMSE', fontsize=11)
        axes[0].set_title('Muestras en Test vs RMSE', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

    # Clasificación: test_urgencies vs F1
    if len(df_clf) > 0:
        axes[1].scatter(df_clf['test_urgencies'], df_clf['f1'], alpha=0.5, c=COLORS['secondary'], s=50)
        axes[1].set_xlabel('Número de urgencias en test', fontsize=11)
        axes[1].set_ylabel('F1-Score', fontsize=11)
        axes[1].set_title('Urgencias en Test vs F1-Score', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / '05_samples_vs_performance.png', dpi=100, bbox_inches='tight')
    print(f"✓ Guardado: {FIGURES / '05_samples_vs_performance.png'}")
    plt.close()

print()

# ============================================================================
# 10. RESUMEN EJECUTIVO
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO - ANÁLISIS POR PRODUCTO")
print("="*80)
print()

print(f"📊 DATASET INICIAL:")
print(f"  • Total registros analizados: {len(df_analysis)}")
print(f"  • Productos únicos: {df_analysis['product_id'].nunique()}")
print()

# Stats por tier
df_reg_tier = df_reg if len(df_reg) > 0 else pd.DataFrame()
df_clf_tier = df_clf if len(df_clf) > 0 else pd.DataFrame()

tier1_reg = len(df_reg_tier[df_reg_tier['tier'] == 'Tier 1 - Deployment']) if len(df_reg_tier) > 0 else 0
tier2_reg = len(df_reg_tier[df_reg_tier['tier'] == 'Tier 2 - Monitoring']) if len(df_reg_tier) > 0 else 0
tier3_reg = len(df_reg_tier[df_reg_tier['tier'] == 'Tier 3 - Rechazado']) if len(df_reg_tier) > 0 else 0

tier1_clf = len(df_clf_tier[df_clf_tier['tier'] == 'Tier 1 - Deployment']) if len(df_clf_tier) > 0 else 0
tier2_clf = len(df_clf_tier[df_clf_tier['tier'] == 'Tier 2 - Monitoring']) if len(df_clf_tier) > 0 else 0
tier3_clf = len(df_clf_tier[df_clf_tier['tier'] == 'Tier 3 - Rechazado']) if len(df_clf_tier) > 0 else 0

print(f"🎯 CLASIFICACIÓN POR TIERS (según calidad del modelo):")
print(f"")
print(f"  REGRESIÓN:")
print(f"    ✅ Tier 1 (Deployment): {tier1_reg} productos ({tier1_reg/len(df_reg_tier)*100:.1f}% del total)" if len(df_reg_tier) > 0 else "    Sin datos")
print(f"    ⚠️  Tier 2 (Monitoring): {tier2_reg} productos ({tier2_reg/len(df_reg_tier)*100:.1f}%)" if len(df_reg_tier) > 0 else "")
print(f"    ❌ Tier 3 (Rechazado): {tier3_reg} productos ({tier3_reg/len(df_reg_tier)*100:.1f}%)" if len(df_reg_tier) > 0 else "")
print(f"")
print(f"  CLASIFICACIÓN:")
print(f"    ✅ Tier 1 (Deployment): {tier1_clf} productos ({tier1_clf/len(df_clf_tier)*100:.1f}% del total)" if len(df_clf_tier) > 0 else "    Sin datos")
print(f"    ⚠️  Tier 2 (Monitoring): {tier2_clf} productos ({tier2_clf/len(df_clf_tier)*100:.1f}%)" if len(df_clf_tier) > 0 else "")
print(f"    ❌ Tier 3 (Rechazado): {tier3_clf} productos ({tier3_clf/len(df_clf_tier)*100:.1f}%)" if len(df_clf_tier) > 0 else "")
print()

print(f"📈 PRODUCTOS APTOS PARA DEPLOYMENT:")
print(f"  • Total APTOS (Tier 1 + Tier 2): {len(df_deployment)} ({len(df_deployment)/len(df_filtered)*100:.1f}%)")
print(f"  • Tier 1 (usar ya): {len(df_tier1)} productos")
print(f"  • Tier 2 (monitorear): {len(df_deployment) - len(df_tier1)} productos")
print(f"  • Tier 3 (NO usar): {len(df_rejected)} productos ({len(df_rejected)/len(df_filtered)*100:.1f}%)")
print()

if len(df_clf_tier[df_clf_tier['tier'].isin(['Tier 1 - Deployment', 'Tier 2 - Monitoring'])]) > 0:
    df_clf_good = df_clf_tier[df_clf_tier['tier'].isin(['Tier 1 - Deployment', 'Tier 2 - Monitoring'])]
    print(f"🎯 MÉTRICAS - PRODUCTOS APTOS (Clasificación):")
    print(f"  • F1-Score promedio: {df_clf_good['f1'].mean():.3f}")
    print(f"  • Precision promedio: {df_clf_good['precision'].mean():.3f}")
    print(f"  • Recall promedio: {df_clf_good['recall'].mean():.3f}")
    print(f"  • ROC-AUC promedio: {df_clf_good['auc'].mean():.3f}")
    print()

print(f"📁 OUTPUTS GENERADOS:")
print(f"  • product_analysis.csv - Todos los productos con tier asignado")
print(f"  • products_filtered.csv - Solo Tier 1+2 (APTOS para usar): {len(df_deployment)} productos")
print(f"  • products_tier1_deployment.csv - Solo Tier 1: {len(df_tier1)} productos")
print(f"  • products_tier3_rejected.csv - Tier 3 (rechazados): {len(df_rejected)} productos")
print(f"  • 05_*.png (3 visualizaciones)")
print()

print("="*80)
print("✓ ANÁLISIS POR PRODUCTO COMPLETADO")
print("="*80)
print()

print("💡 RECOMENDACIÓN:")
print(f"  ✅ DEPLOYMENT INMEDIATO: {len(df_tier1)} productos Tier 1")
print(f"  ⚠️  MONITOREAR: {len(df_deployment) - len(df_tier1)} productos Tier 2")
print(f"  ❌ NO IMPLEMENTAR: {len(df_rejected)} productos Tier 3")
print()
print(f"  → Usar products_filtered.csv para análisis de ROI")
print(f"  → Usar products_tier1_deployment.csv para deployment inicial")
print()

print("PRÓXIMO PASO:")
print(f"  → Script 06: Valor Operativo (06_valor_operativo.py)")
print(f"  → Calcular ROI solo con productos APTOS (Tier 1+2)")
print()
