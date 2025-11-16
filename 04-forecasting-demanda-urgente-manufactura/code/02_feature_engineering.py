"""
02. Feature Engineering - Variables Predictivas SIN Data Leakage
=================================================================

OBJETIVO:
Crear features temporales para predicción PROSPECTIVA de urgencias.

IMPORTANTE - EVITAR DATA LEAKAGE:
- Predecir is_urgent de semana N+1 usando SOLO datos hasta semana N
- NO usar ventas/stats de la semana actual
- Todos los rolling stats calculados HASTA t-1 (shift 1)
- Target shifted: predecir próxima semana

HORIZON:
- Horizonte de predicción: 1 semana adelante
- Con datos hasta HOY → predecir si PRÓXIMA semana es urgencia

FEATURES A CREAR (todos usando datos PASADOS):
1. Lags: ventas en t-1, t-2, t-4, t-52
2. Rolling stats SHIFTED: media, std, min, max hasta t-1
3. Features estacionales: mes, trimestre (de la semana TARGET)
4. Features de tendencia: índice temporal
5. Lags de urgencias: urgencias pasadas en t-1, t-2, t-4

INPUT:
- data/simulated/urgencias_weekly.csv
- data/simulated/products_predictability_ranking.csv

OUTPUT:
- data/simulated/features_weekly.csv
- data/simulated/feature_list.json
- results/figures/02_*.png
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from tqdm import tqdm

# Importar configuración
from config import (
    DATA_PROCESSED, DATA_SIMULATED, FIGURES,
    FIGSIZE_STANDARD, FIGSIZE_WIDE,
    COLORS, RANDOM_SEED
)

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

print("="*80)
print("FEATURE ENGINEERING - SIN DATA LEAKAGE")
print("="*80)
print()
print("⚠️  OBJETIVO: Predecir urgencia semana N+1 usando SOLO datos hasta N")
print()

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================
print("1. CARGANDO DATOS")
print("-" * 80)

# Cargar urgencias detectadas
df = pd.read_csv(DATA_SIMULATED / 'urgencias_weekly.csv')
df['week_start'] = pd.to_datetime(df['week_start'])

# Cargar ranking de predictibilidad
df_ranking = pd.read_csv(DATA_SIMULATED / 'products_predictability_ranking.csv')

print(f"✓ Urgencias cargadas: {df.shape}")
print(f"  Productos únicos: {df['product_id'].nunique()}")
print(f"  Período: {df['week_start'].min()} a {df['week_start'].max()}")
print()
print(f"✓ Ranking cargado: {df_ranking.shape}")
print()

# Seleccionar TOP 25 productos
TOP_N = 25
top_products = df_ranking.head(TOP_N)['product_id'].tolist()

print(f"TOP {TOP_N} productos seleccionados:")
for idx, row in df_ranking.head(TOP_N).iterrows():
    print(f"  {idx+1}. {row['product_id']:30s} | Score: {row['predictability_score']:5.0f} | "
          f"Urgencias: {row['n_urgencies']:3.0f} ({row['urgency_rate']*100:5.1f}%)")
print()

# Filtrar solo TOP productos
df_top = df[df['product_id'].isin(top_products)].copy()

print(f"✓ Datos filtrados a TOP {TOP_N}:")
print(f"  Registros: {len(df_top):,}")
print(f"  Productos: {df_top['product_id'].nunique()}")
print()

# ============================================================================
# 2. FUNCIÓN: CREAR FEATURES SIN DATA LEAKAGE
# ============================================================================

def create_features_for_product(product_df, product_id):
    """
    Crea features temporales SIN data leakage.

    ESTRATEGIA:
    - Todos los rolling stats se calculan y luego shift(1) → solo hasta t-1
    - Target = is_urgent shifted hacia ARRIBA (predecir próxima semana)
    - NO usamos total_sales de la semana actual en features

    Returns:
        DataFrame con features (solo datos pasados) y target (próxima semana)
    """
    df_prod = product_df.copy().sort_values('week_start').reset_index(drop=True)

    # =======================================================================
    # A. LAGS DE VENTAS (ya son pasado por definición)
    # =======================================================================
    df_prod['sales_lag_1'] = df_prod['total_sales'].shift(1)
    df_prod['sales_lag_2'] = df_prod['total_sales'].shift(2)
    df_prod['sales_lag_4'] = df_prod['total_sales'].shift(4)
    df_prod['sales_lag_52'] = df_prod['total_sales'].shift(52)

    # =======================================================================
    # B. ROLLING STATISTICS - SHIFTED (solo hasta t-1)
    # =======================================================================
    # Calcular rolling y luego shift(1) para que sean HASTA t-1

    # Ventana 4 semanas
    df_prod['sales_rolling_mean_4'] = df_prod['total_sales'].rolling(window=4, min_periods=1).mean().shift(1)
    df_prod['sales_rolling_std_4'] = df_prod['total_sales'].rolling(window=4, min_periods=1).std().shift(1)
    df_prod['sales_rolling_min_4'] = df_prod['total_sales'].rolling(window=4, min_periods=1).min().shift(1)
    df_prod['sales_rolling_max_4'] = df_prod['total_sales'].rolling(window=4, min_periods=1).max().shift(1)

    # Ventana 12 semanas
    df_prod['sales_rolling_mean_12'] = df_prod['total_sales'].rolling(window=12, min_periods=1).mean().shift(1)
    df_prod['sales_rolling_std_12'] = df_prod['total_sales'].rolling(window=12, min_periods=1).std().shift(1)
    df_prod['sales_rolling_min_12'] = df_prod['total_sales'].rolling(window=12, min_periods=1).min().shift(1)
    df_prod['sales_rolling_max_12'] = df_prod['total_sales'].rolling(window=12, min_periods=1).max().shift(1)

    # Ventana 52 semanas
    df_prod['sales_rolling_mean_52'] = df_prod['total_sales'].rolling(window=52, min_periods=1).mean().shift(1)
    df_prod['sales_rolling_std_52'] = df_prod['total_sales'].rolling(window=52, min_periods=1).std().shift(1)
    df_prod['sales_rolling_min_52'] = df_prod['total_sales'].rolling(window=52, min_periods=1).min().shift(1)
    df_prod['sales_rolling_max_52'] = df_prod['total_sales'].rolling(window=52, min_periods=1).max().shift(1)

    # =======================================================================
    # C. FEATURES DE TENDENCIA (solo pasado)
    # =======================================================================
    # Índice temporal (0, 1, 2, ..., n-1)
    df_prod['time_index'] = np.arange(len(df_prod))

    # Crecimiento semana t-1 vs t-2 (shift 1)
    df_prod['sales_diff_1'] = df_prod['total_sales'].diff(1).shift(1)

    # Aceleración (segunda derivada)
    df_prod['sales_diff_2'] = df_prod['sales_diff_1'].diff(1)

    # Tendencia normalizada
    df_prod['trend_normalized'] = df_prod['time_index'] / len(df_prod)

    # =======================================================================
    # D. LAGS DE URGENCIAS PASADAS
    # =======================================================================
    df_prod['urgent_lag_1'] = df_prod['is_urgent'].shift(1)
    df_prod['urgent_lag_2'] = df_prod['is_urgent'].shift(2)
    df_prod['urgent_lag_4'] = df_prod['is_urgent'].shift(4)

    # Conteo de urgencias en ventanas pasadas (shifted)
    df_prod['urgent_count_4'] = df_prod['is_urgent'].rolling(window=4, min_periods=1).sum().shift(1)
    df_prod['urgent_count_12'] = df_prod['is_urgent'].rolling(window=12, min_periods=1).sum().shift(1)

    # =======================================================================
    # E. RATIOS Y FEATURES DERIVADOS (usando solo datos pasados)
    # =======================================================================
    # Ratio de lag_1 respecto a media móvil
    df_prod['sales_ratio_mean_4'] = df_prod['sales_lag_1'] / df_prod['sales_rolling_mean_4']
    df_prod['sales_ratio_mean_12'] = df_prod['sales_lag_1'] / df_prod['sales_rolling_mean_12']

    # Distancias (usando lag_1)
    df_prod['sales_dist_max_4'] = df_prod['sales_rolling_max_4'] - df_prod['sales_lag_1']
    df_prod['sales_dist_min_4'] = df_prod['sales_lag_1'] - df_prod['sales_rolling_min_4']

    # Coeficiente de variación
    df_prod['cv_4'] = df_prod['sales_rolling_std_4'] / df_prod['sales_rolling_mean_4']
    df_prod['cv_12'] = df_prod['sales_rolling_std_12'] / df_prod['sales_rolling_mean_12']

    # =======================================================================
    # F. TARGET: SHIFT HACIA ARRIBA (predecir próxima semana)
    # =======================================================================
    # is_urgent_target = urgencia de la PRÓXIMA semana
    df_prod['is_urgent_target'] = df_prod['is_urgent'].shift(-1)

    # También guardamos sales target (para regresión prospectiva)
    df_prod['sales_target'] = df_prod['total_sales'].shift(-1)

    # Reemplazar infinitos y NaNs
    df_prod = df_prod.replace([np.inf, -np.inf], np.nan)

    return df_prod


# ============================================================================
# 3. PROCESAR TODOS LOS TOP PRODUCTOS
# ============================================================================
print("2. CREANDO FEATURES SIN DATA LEAKAGE")
print("-" * 80)
print("   ⚠️  Todos los rolling stats SHIFTED para usar solo datos hasta t-1")
print("   ⚠️  Target SHIFTED: predecir is_urgent de semana N+1")
print()

all_features = []

for product_id in tqdm(top_products, desc="Creando features"):
    product_df = df_top[df_top['product_id'] == product_id].copy()
    product_features = create_features_for_product(product_df, product_id)
    all_features.append(product_features)

# Concatenar todos
df_features = pd.concat(all_features, ignore_index=True)

# Eliminar última fila de cada producto (target es NaN)
df_features = df_features[df_features['is_urgent_target'].notna()].copy()

print()
print(f"✓ Features creadas SIN data leakage")
print(f"  Total registros: {len(df_features):,}")
print(f"  Total columnas: {len(df_features.columns)}")
print(f"  Productos: {df_features['product_id'].nunique()}")
print()

# Resumen de features
feature_cols = [col for col in df_features.columns if col not in [
    'product_id', 'item_id', 'store_id', 'week_id', 'week_start', 'week_num',
    'year', 'month', 'quarter', 'week_of_year', 'week_of_month',
    'total_sales', 'total_revenue', 'avg_price',
    'percentile_threshold', 'growth_rate',
    'urgent_criterio_a', 'urgent_criterio_b', 'is_urgent',
    'is_urgent_target', 'sales_target'
]]

print(f"Features creadas ({len(feature_cols)} total):")
print()

# Agrupar por tipo
lag_features = [col for col in feature_cols if 'lag' in col and 'urgent' not in col]
rolling_features = [col for col in feature_cols if 'rolling' in col]
trend_features = [col for col in feature_cols if 'trend' in col or 'diff' in col or 'time' in col]
urgency_features = [col for col in feature_cols if 'urgent' in col]
ratio_features = [col for col in feature_cols if 'ratio' in col or 'dist' in col or 'cv_' in col]
other_features = [col for col in feature_cols if col not in lag_features + rolling_features + trend_features + urgency_features + ratio_features]

print(f"  • Lags de ventas ({len(lag_features)}): {', '.join(lag_features)}")
print(f"  • Rolling stats ({len(rolling_features)}): {', '.join(rolling_features[:5])}...")
print(f"  • Tendencia ({len(trend_features)}): {', '.join(trend_features)}")
print(f"  • Urgencias pasadas ({len(urgency_features)}): {', '.join(urgency_features)}")
print(f"  • Ratios/Derivados ({len(ratio_features)}): {', '.join(ratio_features)}")
print()
print(f"⚠️  TARGET: is_urgent_target (urgencia de PRÓXIMA semana)")
print(f"⚠️  TARGET (regresión): sales_target (ventas de PRÓXIMA semana)")
print()

# ============================================================================
# 4. ANÁLISIS DE MISSING VALUES
# ============================================================================
print("3. ANÁLISIS DE MISSING VALUES")
print("-" * 80)

missing_summary = df_features[feature_cols].isnull().sum().sort_values(ascending=False)
missing_pct = (missing_summary / len(df_features) * 100).round(2)

print("Features con más missing values:")
top_missing = missing_pct[missing_pct > 0].head(10)
for feat, pct in top_missing.items():
    print(f"  {feat:40s}: {pct:6.2f}% ({int(missing_summary[feat]):,} registros)")

print()
print(f"Total features con NaNs: {(missing_pct > 0).sum()}")
print(f"Esto es ESPERADO debido a shifts y rolling windows en primeras semanas")
print()

# ============================================================================
# 5. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================
print("4. ESTADÍSTICAS DESCRIPTIVAS")
print("-" * 80)

# Seleccionar features numéricas importantes
key_features = [
    'sales_lag_1', 'sales_lag_4', 'sales_lag_52',
    'sales_rolling_mean_4', 'sales_rolling_mean_12', 'sales_rolling_std_12',
    'sales_diff_1', 'urgent_lag_1', 'urgent_count_12',
    'is_urgent_target', 'sales_target'
]

print("Estadísticas de features clave:")
print(df_features[key_features].describe())
print()

# ============================================================================
# 6. VISUALIZACIÓN: CORRELACIÓN DE FEATURES
# ============================================================================
print("5. VISUALIZACIONES")
print("-" * 80)

# Matriz de correlación (features clave vs target)
corr_features = [
    'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_52',
    'sales_rolling_mean_4', 'sales_rolling_mean_12', 'sales_rolling_std_12',
    'sales_rolling_max_4', 'sales_diff_1',
    'urgent_lag_1', 'urgent_lag_2', 'urgent_count_12',
    'sales_ratio_mean_12', 'cv_12', 'trend_normalized',
    'is_urgent_target'
]

# Filtrar solo features que existan
corr_features = [f for f in corr_features if f in df_features.columns]

# Calcular correlación
corr_matrix = df_features[corr_features].corr()

print("Graficando matriz de correlación...")

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlación - Features PASADOS vs Target FUTURO (sin leakage)',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(FIGURES / '02_correlation_matrix.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '02_correlation_matrix.png'}")
plt.close()

# Correlación con target
target_corr = corr_matrix['is_urgent_target'].drop('is_urgent_target').sort_values(ascending=False)
print()
print("Features más correlacionadas con is_urgent_target (próxima semana):")
print(target_corr.head(10))
print()
print("Features menos correlacionadas:")
print(target_corr.tail(5))
print()

# ============================================================================
# 7. VISUALIZACIÓN: DISTRIBUCIONES
# ============================================================================

print("Graficando distribuciones de features clave...")

hist_features = [
    'sales_lag_1', 'sales_rolling_mean_12',
    'sales_diff_1', 'sales_ratio_mean_12', 'cv_12', 'urgent_count_12'
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feat in enumerate(hist_features):
    if feat in df_features.columns:
        data = df_features[feat].dropna()

        axes[idx].hist(data, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='black')
        axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Media')
        axes[idx].axvline(data.median(), color='orange', linestyle='--', linewidth=2, label='Mediana')
        axes[idx].set_title(feat, fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Valor')
        axes[idx].set_ylabel('Frecuencia')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / '02_feature_distributions.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '02_feature_distributions.png'}")
plt.close()

# ============================================================================
# 8. VISUALIZACIÓN: EJEMPLO DE PREDICCIÓN (MEJOR PRODUCTO)
# ============================================================================

print("Graficando ejemplo de predicción para mejor producto...")

best_product_id = top_products[0]
df_best = df_features[df_features['product_id'] == best_product_id].copy()

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Ventas actuales y lag
axes[0].plot(df_best['week_start'], df_best['sales_lag_1'],
            linewidth=2, label='Ventas t-1 (feature)', color=COLORS['primary'])
axes[0].plot(df_best['week_start'], df_best['sales_target'],
            linewidth=2, label='Ventas t+1 (target)', alpha=0.7,
            linestyle='--', color=COLORS['danger'])
axes[0].set_title(f'Predicción Prospectiva - {best_product_id}', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Unidades')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Rolling means (pasado)
axes[1].plot(df_best['week_start'], df_best['sales_rolling_mean_4'],
            linewidth=2, label='Media móvil 4w (t-1)', color=COLORS['success'])
axes[1].plot(df_best['week_start'], df_best['sales_rolling_mean_12'],
            linewidth=2, label='Media móvil 12w (t-1)', color=COLORS['warning'])
axes[1].set_title('Features: Medias Móviles (solo datos pasados)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Unidades')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

# Target: urgencias FUTURAS
urgent_weeks = df_best[df_best['is_urgent_target'] == 1]
axes[2].plot(df_best['week_start'], df_best['sales_target'],
            linewidth=1, color='gray', alpha=0.5, label='Ventas target')
axes[2].scatter(urgent_weeks['week_start'], urgent_weeks['sales_target'],
               color=COLORS['danger'], s=100, marker='o',
               label=f'Urgencias FUTURAS (target, n={len(urgent_weeks)})',
               zorder=3, edgecolors='darkred', linewidths=2)
axes[2].set_title('Target: Urgencias de PRÓXIMA semana', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Fecha')
axes[2].set_ylabel('Unidades')
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES / '02_time_series_features.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '02_time_series_features.png'}")
plt.close()

print()

# ============================================================================
# 9. GUARDAR DATASET CON FEATURES
# ============================================================================
print("6. GUARDANDO DATASET CON FEATURES")
print("-" * 80)

output_file = DATA_SIMULATED / 'features_weekly.csv'
df_features.to_csv(output_file, index=False)

print(f"✓ Dataset guardado: {output_file}")
print(f"  Registros: {len(df_features):,}")
print(f"  Productos: {df_features['product_id'].nunique()}")
print(f"  Columnas totales: {len(df_features.columns)}")
print(f"  Features creados: {len(feature_cols)}")
print(f"  Tamaño: {output_file.stat().st_size / 1024:.2f} KB")
print()

# Guardar lista de features
feature_list = {
    'all_features': feature_cols,
    'lag_features': lag_features,
    'rolling_features': rolling_features,
    'trend_features': trend_features,
    'urgency_features': urgency_features,
    'ratio_features': ratio_features,
    'target_classification': 'is_urgent_target',
    'target_regression': 'sales_target'
}

import json
feature_list_file = DATA_SIMULATED / 'feature_list.json'
with open(feature_list_file, 'w') as f:
    json.dump(feature_list, f, indent=2)

print(f"✓ Lista de features guardada: {feature_list_file}")
print()

# ============================================================================
# 10. RESUMEN EJECUTIVO
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO")
print("="*80)
print()
print(f"📊 DATASET PROCESADO (SIN DATA LEAKAGE):")
print(f"  • TOP {TOP_N} productos seleccionados")
print(f"  • {len(df_features):,} registros totales")
print(f"  • {len(feature_cols)} features creados (SOLO datos pasados)")
print()
print(f"🔧 ESTRATEGIA ANTI-LEAKAGE:")
print(f"  ✅ Todos los rolling stats: .shift(1) → solo hasta t-1")
print(f"  ✅ Target shifted: is_urgent_target = is_urgent.shift(-1)")
print(f"  ✅ Horizonte: Predecir urgencia de semana N+1 con datos hasta N")
print(f"  ✅ NO se usa total_sales actual en features")
print()
print(f"🔧 FEATURES CREADOS:")
print(f"  • Lags: {len(lag_features)} (ventas en t-1, t-2, t-4, t-52)")
print(f"  • Rolling stats: {len(rolling_features)} (shifted, solo hasta t-1)")
print(f"  • Tendencia: {len(trend_features)} (índice, diferencias)")
print(f"  • Urgencias pasadas: {len(urgency_features)} (lags y conteos shifted)")
print(f"  • Ratios: {len(ratio_features)} (usando sales_lag_1)")
print()
print(f"📈 CORRELACIONES CON TARGET (is_urgent_target):")
if len(target_corr) > 0:
    top_positive = target_corr.head(3)
    for feat, corr in top_positive.items():
        print(f"  • {feat:40s}: {corr:+.3f}")
print()
print(f"📁 OUTPUTS GENERADOS:")
print(f"  • {output_file.name} - Dataset listo para modelización")
print(f"  • {feature_list_file.name} - Lista de features por tipo")
print(f"  • 02_correlation_matrix.png")
print(f"  • 02_feature_distributions.png")
print(f"  • 02_time_series_features.png")
print()
print("="*80)
print("✓ FEATURE ENGINEERING COMPLETADO (SIN DATA LEAKAGE)")
print("="*80)
print()
print("CONCLUSIÓN:")
print(f"  ✅ {len(feature_cols)} features creados SOLO con datos PASADOS")
print(f"  ✅ Target shifted: predecir urgencia de PRÓXIMA semana")
print(f"  ✅ CERO data leakage - listo para predicción real")
print()
print("PRÓXIMO PASO:")
print(f"  → Fase 3: Entrenar modelos con nuevos features (AUC esperado ~0.75-0.85)")
print(f"  → Métricas serán más realistas que antes")
print()
