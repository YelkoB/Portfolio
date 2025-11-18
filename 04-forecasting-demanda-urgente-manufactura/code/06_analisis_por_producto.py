"""
06. Análisis Detallado por Producto
====================================

OBJETIVO:
Analizar rendimiento del modelo por producto individual para identificar:
- Productos con mejor predicción (AUC > 0.70)
- Productos donde NO funciona bien el modelo
- Patrones por categoría (FOODS, HOUSEHOLD, HOBBIES)
- Patrones por estado (CA, TX, WI)

OUTPUT:
- Ranking de productos por AUC
- Análisis por categoría y estado
- Recomendaciones de implementación selectiva
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
print("ANÁLISIS DETALLADO POR PRODUCTO")
print("="*80)
print()

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("1. CARGANDO DATOS")
print("-" * 80)

# Cargar métricas de validación
df_metrics = pd.read_csv(DATA_SIMULATED / 'validation_metrics.csv')
print(f"✓ Métricas cargadas: {df_metrics.shape}")
print(f"  Productos evaluados: {df_metrics['product_id'].nunique()}")

# Cargar predicciones para contar muestras por producto
df_predictions = pd.read_csv(DATA_SIMULATED / 'test_predictions.csv')
print(f"✓ Predicciones cargadas: {df_predictions.shape}")

# Contar muestras de test por producto
test_samples = df_predictions.groupby(['product_id', 'task']).size().reset_index(name='test_samples')
print(f"✓ Conteo de muestras por producto calculado")
print()

# ============================================================================
# 2. FUSIONAR CON CONTEO DE MUESTRAS Y FILTRAR
# ============================================================================
print("2. FUSIONANDO DATOS Y FILTRANDO PRODUCTOS")
print("-" * 80)

# Fusionar métricas con conteo de muestras
df_metrics = df_metrics.merge(test_samples, on=['product_id', 'task'], how='left')
df_metrics['test_samples'] = df_metrics['test_samples'].fillna(0).astype(int)

# Filtrar productos con pocas muestras
MIN_TEST_SAMPLES = 20
df_metrics_original = df_metrics.copy()
df_metrics = df_metrics[df_metrics['test_samples'] >= MIN_TEST_SAMPLES].copy()

productos_eliminados = len(df_metrics_original) - len(df_metrics)
print(f"⚠️  Mínimo de muestras en test: {MIN_TEST_SAMPLES}")
print(f"✓ Productos antes del filtro: {len(df_metrics_original)}")
print(f"✓ Productos después del filtro: {len(df_metrics)}")
print(f"❌ Productos eliminados (pocas muestras): {productos_eliminados} ({productos_eliminados/len(df_metrics_original)*100:.1f}%)")
print()

# ============================================================================
# 3. EXTRAER INFORMACIÓN DE PRODUCTO
# ============================================================================
print("3. PARSEANDO INFORMACIÓN DE PRODUCTOS")
print("-" * 80)

# Extraer categoría y estado del product_id
# Formato: CATEGORY_X_XXX_STATE_X
df_metrics['category'] = df_metrics['product_id'].str.split('_').str[0]
df_metrics['state'] = df_metrics['product_id'].str.split('_').str[-2]

print(f"✓ Categorías únicas: {df_metrics['category'].unique()}")
print(f"✓ Estados únicos: {df_metrics['state'].unique()}")
print()

# ============================================================================
# 4. RANKING DE PRODUCTOS POR AUC
# ============================================================================
print("4. RANKING DE PRODUCTOS (Solo productos con ≥{MIN_TEST_SAMPLES} muestras)")
print("-" * 80)

# Filtrar solo clasificación
df_clf = df_metrics[df_metrics['task'] == 'classification'].copy()

# Ordenar por AUC descendente
df_clf_sorted = df_clf.sort_values('auc', ascending=False).reset_index(drop=True)

# Estadísticas generales
print(f"AUC - Estadísticas:")
print(f"  Media:    {df_clf['auc'].mean():.3f}")
print(f"  Mediana:  {df_clf['auc'].median():.3f}")
print(f"  Std:      {df_clf['auc'].std():.3f}")
print(f"  Min:      {df_clf['auc'].min():.3f}")
print(f"  Max:      {df_clf['auc'].max():.3f}")
print()

# Categorías de rendimiento
auc_excellent = (df_clf['auc'] >= 0.80).sum()
auc_good = ((df_clf['auc'] >= 0.70) & (df_clf['auc'] < 0.80)).sum()
auc_acceptable = ((df_clf['auc'] >= 0.60) & (df_clf['auc'] < 0.70)).sum()
auc_poor = (df_clf['auc'] < 0.60).sum()

print("📊 CLASIFICACIÓN DE PRODUCTOS:")
print(f"  🌟 Excelente (AUC ≥ 0.80):   {auc_excellent:4d} ({auc_excellent/len(df_clf)*100:5.1f}%)")
print(f"  ✅ Bueno     (0.70 ≤ AUC < 0.80): {auc_good:4d} ({auc_good/len(df_clf)*100:5.1f}%)")
print(f"  ⚠️  Aceptable (0.60 ≤ AUC < 0.70): {auc_acceptable:4d} ({auc_acceptable/len(df_clf)*100:5.1f}%)")
print(f"  ❌ Pobre     (AUC < 0.60):   {auc_poor:4d} ({auc_poor/len(df_clf)*100:5.1f}%)")
print()

# Top 20 mejores productos
print("🏆 TOP 20 PRODUCTOS MÁS PREDECIBLES:")
print()
for idx, row in df_clf_sorted.head(20).iterrows():
    print(f"  {idx+1:2d}. {row['product_id']:30s} | "
          f"AUC: {row['auc']:.3f} | F1: {row['f1']:.3f} | "
          f"Precision: {row['precision']:.3f} | Recall: {row['recall']:.3f}")
print()

# Bottom 20 peores productos
print("⚠️  BOTTOM 20 PRODUCTOS MENOS PREDECIBLES:")
print()
for idx, row in df_clf_sorted.tail(20).iterrows():
    print(f"  {len(df_clf)-idx:2d}. {row['product_id']:30s} | "
          f"AUC: {row['auc']:.3f} | F1: {row['f1']:.3f} | "
          f"Precision: {row['precision']:.3f} | Recall: {row['recall']:.3f}")
print()

# ============================================================================
# 5. ANÁLISIS POR CATEGORÍA
# ============================================================================
print("5. ANÁLISIS POR CATEGORÍA")
print("-" * 80)

category_stats = df_clf.groupby('category')['auc'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(3)

category_stats['productos_buenos'] = df_clf[df_clf['auc'] >= 0.70].groupby('category').size()
category_stats['productos_buenos'] = category_stats['productos_buenos'].fillna(0).astype(int)
category_stats['pct_buenos'] = (category_stats['productos_buenos'] / category_stats['count'] * 100).round(1)

print(category_stats.to_string())
print()

# ============================================================================
# 6. ANÁLISIS POR ESTADO
# ============================================================================
print("6. ANÁLISIS POR ESTADO")
print("-" * 80)

state_stats = df_clf.groupby('state')['auc'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(3)

state_stats['productos_buenos'] = df_clf[df_clf['auc'] >= 0.70].groupby('state').size()
state_stats['productos_buenos'] = state_stats['productos_buenos'].fillna(0).astype(int)
state_stats['pct_buenos'] = (state_stats['productos_buenos'] / state_stats['count'] * 100).round(1)

print(state_stats.to_string())
print()

# ============================================================================
# 7. ANÁLISIS COMBINADO CATEGORÍA × ESTADO
# ============================================================================
print("7. ANÁLISIS COMBINADO CATEGORÍA × ESTADO")
print("-" * 80)

pivot_mean = df_clf.pivot_table(values='auc', index='category', columns='state', aggfunc='mean')
pivot_count = df_clf.pivot_table(values='auc', index='category', columns='state', aggfunc='count')

print("AUC Promedio por Categoría × Estado:")
print(pivot_mean.round(3).to_string())
print()
print("Número de Productos por Categoría × Estado:")
print(pivot_count.fillna(0).astype(int).to_string())
print()

# ============================================================================
# 8. GUARDAR RESULTADOS
# ============================================================================
print("8. GUARDANDO RESULTADOS")
print("-" * 80)

# Guardar ranking completo (con número de muestras)
df_clf_sorted.to_csv(DATA_SIMULATED / 'product_ranking_auc.csv', index=False)
print(f"✓ Ranking guardado: product_ranking_auc.csv")
print(f"  Incluye columna 'test_samples' para transparencia")

# Guardar productos recomendados (AUC >= 0.70)
df_recommended = df_clf_sorted[df_clf_sorted['auc'] >= 0.70].copy()
df_recommended.to_csv(DATA_SIMULATED / 'products_recommended.csv', index=False)
print(f"✓ Productos recomendados: {len(df_recommended)} productos con AUC ≥ 0.70")
print(f"  Rango de muestras: {df_recommended['test_samples'].min()}-{df_recommended['test_samples'].max()}")
print(f"  Media de muestras: {df_recommended['test_samples'].mean():.0f}")

# Guardar productos NO recomendados (AUC < 0.60)
df_not_recommended = df_clf_sorted[df_clf_sorted['auc'] < 0.60].copy()
df_not_recommended.to_csv(DATA_SIMULATED / 'products_not_recommended.csv', index=False)
print(f"✓ Productos NO recomendados: {len(df_not_recommended)} productos con AUC < 0.60")
print()

# ============================================================================
# 9. VISUALIZACIONES
# ============================================================================
print("9. GENERANDO VISUALIZACIONES")
print("-" * 80)

# 8.1 Distribución de AUC
fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_WIDE)

# Histograma
axes[0, 0].hist(df_clf['auc'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(0.70, color='red', linestyle='--', label='Umbral (0.70)')
axes[0, 0].axvline(df_clf['auc'].mean(), color='green', linestyle='--', label=f'Media ({df_clf["auc"].mean():.3f})')
axes[0, 0].set_xlabel('AUC')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución de AUC por Producto')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Boxplot por categoría
df_clf.boxplot(column='auc', by='category', ax=axes[0, 1])
axes[0, 1].set_xlabel('Categoría')
axes[0, 1].set_ylabel('AUC')
axes[0, 1].set_title('AUC por Categoría')
axes[0, 1].axhline(0.70, color='red', linestyle='--', alpha=0.5)
plt.sca(axes[0, 1])
plt.xticks(rotation=45)

# Boxplot por estado
df_clf.boxplot(column='auc', by='state', ax=axes[1, 0])
axes[1, 0].set_xlabel('Estado')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].set_title('AUC por Estado')
axes[1, 0].axhline(0.70, color='red', linestyle='--', alpha=0.5)

# Scatter F1 vs AUC
axes[1, 1].scatter(df_clf['auc'], df_clf['f1'], alpha=0.5)
axes[1, 1].axvline(0.70, color='red', linestyle='--', alpha=0.5, label='AUC=0.70')
axes[1, 1].axhline(0.60, color='orange', linestyle='--', alpha=0.5, label='F1=0.60')
axes[1, 1].set_xlabel('AUC')
axes[1, 1].set_ylabel('F1-Score')
axes[1, 1].set_title('Trade-off AUC vs F1')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / '06_product_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Guardado: 06_product_analysis.png")
plt.close()

# 8.2 Heatmap Categoría × Estado
fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap='RdYlGn', center=0.65,
            vmin=0.5, vmax=0.8, ax=ax)
ax.set_title('AUC Promedio por Categoría × Estado')
ax.set_xlabel('Estado')
ax.set_ylabel('Categoría')
plt.tight_layout()
plt.savefig(FIGURES / '06_heatmap_category_state.png', dpi=300, bbox_inches='tight')
print(f"✓ Guardado: 06_heatmap_category_state.png")
plt.close()

# 8.3 Top/Bottom productos
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

# Top 15
top15 = df_clf_sorted.head(15)
axes[0].barh(range(len(top15)), top15['auc'], color='green', alpha=0.7)
axes[0].set_yticks(range(len(top15)))
axes[0].set_yticklabels(top15['product_id'], fontsize=8)
axes[0].axvline(0.70, color='red', linestyle='--', label='Umbral (0.70)')
axes[0].set_xlabel('AUC')
axes[0].set_title('Top 15 Productos Más Predecibles')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='x')

# Bottom 15
bottom15 = df_clf_sorted.tail(15).iloc[::-1]  # Invertir para que el peor esté arriba
axes[1].barh(range(len(bottom15)), bottom15['auc'], color='red', alpha=0.7)
axes[1].set_yticks(range(len(bottom15)))
axes[1].set_yticklabels(bottom15['product_id'], fontsize=8)
axes[1].axvline(0.60, color='orange', linestyle='--', label='Mínimo aceptable')
axes[1].set_xlabel('AUC')
axes[1].set_title('Bottom 15 Productos Menos Predecibles')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(FIGURES / '06_top_bottom_products.png', dpi=300, bbox_inches='tight')
print(f"✓ Guardado: 06_top_bottom_products.png")
plt.close()

print()

# ============================================================================
# 9. RECOMENDACIONES
# ============================================================================
print("="*80)
print("RESUMEN EJECUTIVO Y RECOMENDACIONES")
print("="*80)
print()

print(f"📊 PRODUCTOS EVALUADOS: {len(df_clf)}")
print()

print("🎯 SEGMENTACIÓN POR RENDIMIENTO:")
print(f"  🌟 Implementar AHORA  (AUC ≥ 0.80): {auc_excellent:4d} productos ({auc_excellent/len(df_clf)*100:5.1f}%)")
print(f"  ✅ Implementar PRONTO (0.70-0.79):  {auc_good:4d} productos ({auc_good/len(df_clf)*100:5.1f}%)")
print(f"  ⚠️  Monitorear         (0.60-0.69):  {auc_acceptable:4d} productos ({auc_acceptable/len(df_clf)*100:5.1f}%)")
print(f"  ❌ NO implementar    (AUC < 0.60):  {auc_poor:4d} productos ({auc_poor/len(df_clf)*100:5.1f}%)")
print()

total_implementar = auc_excellent + auc_good
print(f"💡 RECOMENDACIÓN:")
print(f"  → Implementar modelo en {total_implementar} productos ({total_implementar/len(df_clf)*100:.1f}%)")
print(f"  → Usar método tradicional en {auc_poor} productos restantes")
print()

print("📈 MEJORES CATEGORÍAS:")
best_category = category_stats.sort_values('mean', ascending=False).head(3)
for idx, row in best_category.iterrows():
    print(f"  • {idx}: AUC medio {row['mean']:.3f} ({row['pct_buenos']:.1f}% buenos)")
print()

print("📁 OUTPUTS GENERADOS:")
print(f"  • product_ranking_auc.csv - Ranking completo")
print(f"  • products_recommended.csv - {len(df_recommended)} productos para implementar")
print(f"  • products_not_recommended.csv - {len(df_not_recommended)} productos a evitar")
print(f"  • 06_product_analysis.png")
print(f"  • 06_heatmap_category_state.png")
print(f"  • 06_top_bottom_products.png")
print()

print("="*80)
print("✓ ANÁLISIS POR PRODUCTO COMPLETADO")
print("="*80)
