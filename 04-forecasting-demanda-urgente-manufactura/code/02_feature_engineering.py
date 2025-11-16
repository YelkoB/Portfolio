"""
02. Feature Engineering - Variables Predictivas
================================================

OBJETIVO:
Crear features temporales para los TOP 25 productos más predecibles
que permitan modelizar la predicción de urgencias.

FEATURES A CREAR:
1. Lags: ventas en semanas anteriores (1, 2, 4, 52)
2. Rolling stats: media, std, min, max en ventanas móviles (4, 12, 52 semanas)
3. Features estacionales: mes, trimestre, semana del año, semana del mes
4. Features de tendencia: tendencia lineal, aceleración
5. Features de urgencias: lags de urgencias anteriores

INPUT:
- data/simulated/urgencias_weekly.csv (todos los productos con urgencias)
- data/simulated/products_predictability_ranking.csv (ranking)

OUTPUT:
- data/simulated/features_weekly.csv (dataset con features para TOP 25)
- results/figures/02_*.png (visualizaciones de features)
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

# tqdm may not be available in all environments; prefer tqdm.auto and fall back to a no-op wrapper
try:
    from tqdm.auto import tqdm
except Exception:
    try:
        from tqdm import tqdm
    except Exception:
        # fallback: simple passthrough iterable (no progress bar)
        def tqdm(iterable, **kwargs):
            return iterable

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
print("FEATURE ENGINEERING - VARIABLES PREDICTIVAS")
print("="*80)
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
# 2. FUNCIÓN: CREAR FEATURES POR PRODUCTO
# ============================================================================

def create_features_for_product(product_df, product_id):
    """
    Crea features temporales para un producto.

    Features:
    - Lags: 1, 2, 4, 52 semanas
    - Rolling stats: media, std, min, max (ventanas 4, 12, 52)
    - Estacionales: ya existen (month, quarter, week_of_year, week_of_month)
    - Tendencia: índice temporal, aceleración
    - Urgencias pasadas: lags de is_urgent

    Returns:
        DataFrame con features creadas
    """
    df_prod = product_df.copy().sort_values('week_start').reset_index(drop=True)

    # =======================================================================
    # A. LAGS DE VENTAS
    # =======================================================================
    df_prod['sales_lag_1'] = df_prod['total_sales'].shift(1)
    df_prod['sales_lag_2'] = df_prod['total_sales'].shift(2)
    df_prod['sales_lag_4'] = df_prod['total_sales'].shift(4)
    df_prod['sales_lag_52'] = df_prod['total_sales'].shift(52)  # Año anterior

    # =======================================================================
    # B. ROLLING STATISTICS
    # =======================================================================
    # Ventana 4 semanas (~1 mes)
    df_prod['sales_rolling_mean_4'] = df_prod['total_sales'].rolling(window=4, min_periods=1).mean()
    df_prod['sales_rolling_std_4'] = df_prod['total_sales'].rolling(window=4, min_periods=1).std()
    df_prod['sales_rolling_min_4'] = df_prod['total_sales'].rolling(window=4, min_periods=1).min()
    df_prod['sales_rolling_max_4'] = df_prod['total_sales'].rolling(window=4, min_periods=1).max()

    # Ventana 12 semanas (~3 meses)
    df_prod['sales_rolling_mean_12'] = df_prod['total_sales'].rolling(window=12, min_periods=1).mean()
    df_prod['sales_rolling_std_12'] = df_prod['total_sales'].rolling(window=12, min_periods=1).std()
    df_prod['sales_rolling_min_12'] = df_prod['total_sales'].rolling(window=12, min_periods=1).min()
    df_prod['sales_rolling_max_12'] = df_prod['total_sales'].rolling(window=12, min_periods=1).max()

    # Ventana 52 semanas (1 año)
    df_prod['sales_rolling_mean_52'] = df_prod['total_sales'].rolling(window=52, min_periods=1).mean()
    df_prod['sales_rolling_std_52'] = df_prod['total_sales'].rolling(window=52, min_periods=1).std()
    df_prod['sales_rolling_min_52'] = df_prod['total_sales'].rolling(window=52, min_periods=1).min()
    df_prod['sales_rolling_max_52'] = df_prod['total_sales'].rolling(window=52, min_periods=1).max()

    # =======================================================================
    # C. FEATURES DE TENDENCIA
    # =======================================================================
    # Índice temporal (0, 1, 2, ..., n-1)
    df_prod['time_index'] = np.arange(len(df_prod))

    # Crecimiento (diferencia vs semana anterior)
    df_prod['sales_diff_1'] = df_prod['total_sales'].diff(1)

    # Aceleración (segunda derivada)
    df_prod['sales_diff_2'] = df_prod['sales_diff_1'].diff(1)

    # Tendencia normalizada (posición en la serie)
    df_prod['trend_normalized'] = df_prod['time_index'] / len(df_prod)

    # =======================================================================
    # D. FEATURES ESTACIONALES (ya existen, pero añadimos algunas)
    # =======================================================================
    # One-hot encoding del mes (opcional, pero útil para algunos modelos)
    for m in range(1, 13):
        df_prod[f'month_{m}'] = (df_prod['month'] == m).astype(int)

    # Trimestre one-hot
    for q in range(1, 5):
        df_prod[f'quarter_{q}'] = (df_prod['quarter'] == q).astype(int)

    # Semana del mes one-hot
    for w in range(1, 6):
        df_prod[f'week_of_month_{w}'] = (df_prod['week_of_month'] == w).astype(int)

    # =======================================================================
    # E. LAGS DE URGENCIAS PASADAS
    # =======================================================================
    df_prod['urgent_lag_1'] = df_prod['is_urgent'].shift(1)
    df_prod['urgent_lag_2'] = df_prod['is_urgent'].shift(2)
    df_prod['urgent_lag_4'] = df_prod['is_urgent'].shift(4)

    # Urgencias en ventana móvil (¿cuántas urgencias en últimas N semanas?)
    df_prod['urgent_count_4'] = df_prod['is_urgent'].rolling(window=4, min_periods=1).sum()
    df_prod['urgent_count_12'] = df_prod['is_urgent'].rolling(window=12, min_periods=1).sum()

    # =======================================================================
    # F. RATIOS Y FEATURES DERIVADOS
    # =======================================================================
    # Ratio respecto a media móvil
    df_prod['sales_ratio_mean_4'] = df_prod['total_sales'] / df_prod['sales_rolling_mean_4']
    df_prod['sales_ratio_mean_12'] = df_prod['total_sales'] / df_prod['sales_rolling_mean_12']

    # Distancia a min/max de ventana
    df_prod['sales_dist_max_4'] = df_prod['sales_rolling_max_4'] - df_prod['total_sales']
    df_prod['sales_dist_min_4'] = df_prod['total_sales'] - df_prod['sales_rolling_min_4']

    # Coeficiente de variación rolling
    df_prod['cv_4'] = df_prod['sales_rolling_std_4'] / df_prod['sales_rolling_mean_4']
    df_prod['cv_12'] = df_prod['sales_rolling_std_12'] / df_prod['sales_rolling_mean_12']

    # Reemplazar infinitos y NaNs
    df_prod = df_prod.replace([np.inf, -np.inf], np.nan)

    return df_prod


# ============================================================================
# 3. PROCESAR TODOS LOS TOP PRODUCTOS
# ============================================================================
print("2. CREANDO FEATURES PARA TOP PRODUCTOS")
print("-" * 80)

all_features = []

for product_id in tqdm(top_products, desc="Creando features"):
    product_df = df_top[df_top['product_id'] == product_id].copy()
    product_features = create_features_for_product(product_df, product_id)
    all_features.append(product_features)

# Concatenar todos
df_features = pd.concat(all_features, ignore_index=True)

print()
print(f"✓ Features creadas")
print(f"  Total registros: {len(df_features):,}")
print(f"  Total columnas: {len(df_features.columns)}")
print(f"  Productos: {df_features['product_id'].nunique()}")
print()

# Resumen de features
feature_cols = [col for col in df_features.columns if col not in [
    'product_id', 'item_id', 'store_id', 'week_id', 'week_start', 'week_num',
    'year', 'month', 'quarter', 'week_of_year', 'week_of_month'
]]

print(f"Features creadas ({len(feature_cols)} total):")
print()

# Agrupar por tipo
lag_features = [col for col in feature_cols if 'lag' in col]
rolling_features = [col for col in feature_cols if 'rolling' in col]
trend_features = [col for col in feature_cols if 'trend' in col or 'diff' in col or 'time' in col]
seasonal_features = [col for col in feature_cols if 'month_' in col or 'quarter_' in col or 'week_of_month_' in col]
urgency_features = [col for col in feature_cols if 'urgent' in col]
ratio_features = [col for col in feature_cols if 'ratio' in col or 'dist' in col or 'cv_' in col]
base_features = [col for col in feature_cols if col not in lag_features + rolling_features + trend_features + seasonal_features + urgency_features + ratio_features]

print(f"  • Lags ({len(lag_features)}): {', '.join(lag_features[:5])}...")
print(f"  • Rolling stats ({len(rolling_features)}): {', '.join(rolling_features[:5])}...")
print(f"  • Tendencia ({len(trend_features)}): {', '.join(trend_features)}")
print(f"  • Estacionales ({len(seasonal_features)}): month_1..12, quarter_1..4, week_of_month_1..5")
print(f"  • Urgencias pasadas ({len(urgency_features)}): {', '.join(urgency_features)}")
print(f"  • Ratios/Derivados ({len(ratio_features)}): {', '.join(ratio_features[:5])}...")
print(f"  • Base ({len(base_features)}): {', '.join(base_features)}")
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
print(f"Esto es esperado debido a lags y rolling windows en las primeras semanas")
print()

# ============================================================================
# 5. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================
print("4. ESTADÍSTICAS DESCRIPTIVAS")
print("-" * 80)

# Seleccionar features numéricas importantes
key_features = [
    'total_sales', 'sales_lag_1', 'sales_lag_4', 'sales_lag_52',
    'sales_rolling_mean_4', 'sales_rolling_mean_12', 'sales_rolling_std_12',
    'growth_rate', 'sales_diff_1', 'percentile_threshold',
    'is_urgent'
]

print("Estadísticas de features clave:")
print(df_features[key_features].describe())
print()

# ============================================================================
# 6. VISUALIZACIÓN: CORRELACIÓN DE FEATURES
# ============================================================================
print("5. VISUALIZACIONES")
print("-" * 80)

# Matriz de correlación (features clave vs target is_urgent)
corr_features = [
    'total_sales', 'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_52',
    'sales_rolling_mean_4', 'sales_rolling_mean_12', 'sales_rolling_std_12',
    'sales_rolling_max_4', 'growth_rate', 'sales_diff_1',
    'urgent_lag_1', 'urgent_lag_2', 'urgent_count_12',
    'sales_ratio_mean_12', 'cv_12', 'trend_normalized',
    'is_urgent'
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
ax.set_title('Matriz de Correlación - Features Clave vs Urgencias',
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(FIGURES / '02_correlation_matrix.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '02_correlation_matrix.png'}")
plt.close()

# Correlación con target is_urgent
target_corr = corr_matrix['is_urgent'].drop('is_urgent').sort_values(ascending=False)
print()
print("Features más correlacionadas con is_urgent:")
print(target_corr.head(10))
print()
print("Features menos correlacionadas (negativas):")
print(target_corr.tail(5))
print()

# ============================================================================
# 7. VISUALIZACIÓN: DISTRIBUCIONES DE FEATURES
# ============================================================================

print("Graficando distribuciones de features clave...")

# Seleccionar features para histogramas
hist_features = [
    'total_sales', 'sales_lag_1', 'sales_rolling_mean_12',
    'growth_rate', 'sales_ratio_mean_12', 'cv_12'
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
# 8. VISUALIZACIÓN: SERIES TEMPORALES (MEJOR PRODUCTO)
# ============================================================================

print("Graficando series temporales para mejor producto...")

best_product_id = top_products[0]
df_best = df_features[df_features['product_id'] == best_product_id].copy()

fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Ventas y lags
axes[0].plot(df_best['week_start'], df_best['total_sales'],
            linewidth=2, label='Ventas actuales', color=COLORS['primary'])
axes[0].plot(df_best['week_start'], df_best['sales_lag_1'],
            linewidth=1, label='Lag 1', alpha=0.7, linestyle='--')
axes[0].plot(df_best['week_start'], df_best['sales_lag_4'],
            linewidth=1, label='Lag 4', alpha=0.7, linestyle='--')
axes[0].set_title(f'Ventas y Lags - {best_product_id}', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Unidades')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Rolling means
axes[1].plot(df_best['week_start'], df_best['total_sales'],
            linewidth=1, label='Ventas', alpha=0.5, color='gray')
axes[1].plot(df_best['week_start'], df_best['sales_rolling_mean_4'],
            linewidth=2, label='Media móvil 4w', color=COLORS['success'])
axes[1].plot(df_best['week_start'], df_best['sales_rolling_mean_12'],
            linewidth=2, label='Media móvil 12w', color=COLORS['warning'])
axes[1].set_title('Medias Móviles', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Unidades')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

# Volatilidad (rolling std)
axes[2].plot(df_best['week_start'], df_best['sales_rolling_std_4'],
            linewidth=1.5, label='Std 4w', color=COLORS['info'])
axes[2].plot(df_best['week_start'], df_best['sales_rolling_std_12'],
            linewidth=1.5, label='Std 12w', color=COLORS['danger'])
axes[2].set_title('Volatilidad (Desviación Estándar Móvil)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Desv. Est.')
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

# Urgencias
urgent_weeks = df_best[df_best['is_urgent'] == 1]
axes[3].plot(df_best['week_start'], df_best['total_sales'],
            linewidth=1, color='gray', alpha=0.5, label='Ventas')
axes[3].scatter(urgent_weeks['week_start'], urgent_weeks['total_sales'],
               color=COLORS['danger'], s=100, marker='o', label='Urgencias',
               zorder=3, edgecolors='darkred', linewidths=2)
axes[3].set_title('Urgencias Detectadas', fontsize=12, fontweight='bold')
axes[3].set_xlabel('Fecha')
axes[3].set_ylabel('Unidades')
axes[3].legend(loc='upper left')
axes[3].grid(True, alpha=0.3)

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

# Guardar también lista de features para uso posterior
feature_list = {
    'all_features': feature_cols,
    'lag_features': lag_features,
    'rolling_features': rolling_features,
    'trend_features': trend_features,
    'seasonal_features': seasonal_features,
    'urgency_features': urgency_features,
    'ratio_features': ratio_features,
    'base_features': base_features,
    'key_features': corr_features
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
print(f"📊 DATASET PROCESADO:")
print(f"  • TOP {TOP_N} productos seleccionados")
print(f"  • {len(df_features):,} registros totales")
print(f"  • {len(feature_cols)} features creados")
print()
print(f"🔧 FEATURES CREADOS:")
print(f"  • Lags: {len(lag_features)} (ventas y urgencias en t-1, t-2, t-4, t-52)")
print(f"  • Rolling stats: {len(rolling_features)} (media, std, min, max en ventanas 4, 12, 52)")
print(f"  • Tendencia: {len(trend_features)} (índice temporal, diferencias, aceleración)")
print(f"  • Estacionales: {len(seasonal_features)} (mes, trimestre, semana del mes)")
print(f"  • Urgencias pasadas: {len(urgency_features)} (lags y conteos)")
print(f"  • Ratios: {len(ratio_features)} (ratios respecto medias, distancias, CV)")
print()
print(f"📈 CORRELACIONES DESTACADAS:")
if len(target_corr) > 0:
    top_positive = target_corr.head(3)
    for feat, corr in top_positive.items():
        print(f"  • {feat:40s}: {corr:+.3f}")
print()
print(f"📁 OUTPUTS GENERADOS:")
print(f"  • {output_file.name} - Dataset con features")
print(f"  • {feature_list_file.name} - Lista de features por tipo")
print(f"  • 02_correlation_matrix.png")
print(f"  • 02_feature_distributions.png")
print(f"  • 02_time_series_features.png")
print()
print("="*80)
print("✓ FEATURE ENGINEERING COMPLETADO")
print("="*80)
print()
print("CONCLUSIÓN:")
print(f"  ✓ Creados {len(feature_cols)} features temporales para TOP {TOP_N} productos")
print(f"  ✓ Dataset listo para modelización")
print(f"  ✓ Features cubren: lags, rolling stats, tendencia, estacionalidad")
print()
print("PRÓXIMO PASO:")
print(f"  → Fase 3: Modelización (ARIMA, Prophet, ML)")
print(f"  → Train/Val/Test split temporal")
print(f"  → Comparación de modelos por métricas (RMSE, MAE, F1)")
print()
