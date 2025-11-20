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
# 5. FILTRADO DE PRODUCTOS CON DATOS INSUFICIENTES
# ============================================================================
print("5. FILTRADO DE PRODUCTOS CON DATOS INSUFICIENTES")
print("-" * 80)

MIN_TEST_SAMPLES = 20
MIN_TEST_URGENCIES = 5

print(f"⚠️  Criterios de filtrado:")
print(f"   • Mínimo de muestras en test: {MIN_TEST_SAMPLES}")
print(f"   • Mínimo de urgencias en test (clasificación): {MIN_TEST_URGENCIES}")
print()

# Aplicar filtros
df_filtered = df_analysis.copy()

# Filtro 1: Mínimo de muestras
df_filtered = df_filtered[df_filtered['test_samples'] >= MIN_TEST_SAMPLES]

# Filtro 2: Para clasificación, mínimo de urgencias
mask_regression = df_filtered['task'] == 'regression'
mask_classification_valid = (
    (df_filtered['task'] == 'classification') &
    (df_filtered['test_urgencies'] >= MIN_TEST_URGENCIES)
)

df_filtered = df_filtered[mask_regression | mask_classification_valid]

print(f"✓ Productos antes del filtro: {len(df_analysis)}")
print(f"✓ Productos después del filtro: {len(df_filtered)}")
print(f"❌ Productos eliminados: {len(df_analysis) - len(df_filtered)} ({(len(df_analysis) - len(df_filtered))/len(df_analysis)*100:.1f}%)")
print()

# ============================================================================
# 6. ANÁLISIS POR TAREA
# ============================================================================
print("6. ANÁLISIS POR TAREA")
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

# Guardar análisis completo
output_analysis = DATA_SIMULATED / 'product_analysis.csv'
df_analysis.to_csv(output_analysis, index=False)
print(f"✓ Análisis completo guardado: {output_analysis}")

# Guardar solo productos filtrados (válidos)
output_filtered = DATA_SIMULATED / 'products_filtered.csv'
df_filtered.to_csv(output_filtered, index=False)
print(f"✓ Productos válidos guardados: {output_filtered}")
print(f"  Total productos válidos: {len(df_filtered)}")
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

print(f"📊 DATASET COMPLETO:")
print(f"  • Total registros (métricas): {len(df_analysis)}")
print(f"  • Productos únicos: {df_analysis['product_id'].nunique()}")
print()

print(f"✅ PRODUCTOS VÁLIDOS (después de filtrado):")
print(f"  • Total registros válidos: {len(df_filtered)}")
print(f"  • Productos únicos válidos: {df_filtered['product_id'].nunique()}")
print(f"  • Tasa de retención: {len(df_filtered)/len(df_analysis)*100:.1f}%")
print()

if len(df_reg) > 0:
    print(f"📈 REGRESIÓN (Predicción de ventas):")
    print(f"  • Productos evaluados: {len(df_reg)}")
    print(f"  • Mejor producto: {df_reg.nsmallest(1, 'rmse').iloc[0]['product_id']}")
    print(f"    RMSE: {df_reg['rmse'].min():.2f}")
    print(f"  • RMSE promedio: {df_reg['rmse'].mean():.2f} (std: {df_reg['rmse'].std():.2f})")
    print(f"  • MAE promedio: {df_reg['mae'].mean():.2f}")
    print()

if len(df_clf) > 0:
    print(f"🎯 CLASIFICACIÓN (Predicción de urgencias):")
    print(f"  • Productos evaluados: {len(df_clf)}")
    print(f"  • Mejor producto: {df_clf.nlargest(1, 'f1').iloc[0]['product_id']}")
    print(f"    F1-Score: {df_clf['f1'].max():.3f}")
    print(f"  • F1-Score promedio: {df_clf['f1'].mean():.3f} (std: {df_clf['f1'].std():.3f})")
    print(f"  • Precision promedio: {df_clf['precision'].mean():.3f}")
    print(f"  • Recall promedio: {df_clf['recall'].mean():.3f}")
    print(f"  • ROC-AUC promedio: {df_clf['auc'].mean():.3f}")
    print()

print(f"📁 OUTPUTS GENERADOS:")
print(f"  • product_analysis.csv ({len(df_analysis)} registros)")
print(f"  • products_filtered.csv ({len(df_filtered)} registros)")
print(f"  • 05_data_distribution.png")
print(f"  • 05_model_performance.png")
print(f"  • 05_samples_vs_performance.png")
print()

print("="*80)
print("✓ ANÁLISIS POR PRODUCTO COMPLETADO")
print("="*80)
print()

print("CONCLUSIÓN:")
print(f"  ✓ {len(df_filtered)} productos válidos identificados")
print(f"  ✓ {len(df_analysis) - len(df_filtered)} productos filtrados por datos insuficientes")
print(f"  ✓ Dataset limpio generado para análisis de ROI")
print()

print("PRÓXIMO PASO:")
print(f"  → Script 06: Valor Operativo (06_valor_operativo.py)")
print(f"  → Calcular ROI solo con productos confiables")
print()
