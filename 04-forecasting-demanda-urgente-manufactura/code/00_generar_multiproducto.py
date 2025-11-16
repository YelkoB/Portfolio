"""
Generador de Datos Multi-Producto (Temporal)
=============================================

PROPÓSITO:
Convertir el archivo urgencias_weekly.csv existente (single-product)
en formato multi-product con 25 productos diferentes.

NOTA: Este es un workaround temporal para poder ejecutar los experimentos
cuando no se tiene acceso al dataset M5 original.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuración
DATA_SIMULATED = Path(__file__).parent.parent / 'data' / 'simulated'
RANDOM_SEED = 42
NUM_PRODUCTS = 25

np.random.seed(RANDOM_SEED)

print("="*80)
print("GENERADOR DE DATOS MULTI-PRODUCTO")
print("="*80)
print()

# 1. Cargar datos existentes (single-product)
print("1. Cargando datos existentes...")
df_base = pd.read_csv(DATA_SIMULATED / 'urgencias_weekly.csv')
df_base['week_start'] = pd.to_datetime(df_base['week_start'])
print(f"✓ Datos base cargados: {df_base.shape}")
print()

# 2. Generar 25 productos con variaciones
print(f"2. Generando {NUM_PRODUCTS} productos...")
all_products = []
predictability_scores = []

for product_id in range(NUM_PRODUCTS):
    # Copiar datos base
    df_prod = df_base.copy()
    df_prod['product_id'] = f'PRODUCT_{product_id:03d}'

    # Añadir variaciones aleatorias a las ventas (±20%)
    variation = np.random.uniform(0.8, 1.2, size=len(df_prod))
    df_prod['total_sales'] = (df_prod['total_sales'] * variation).astype(int)
    df_prod['total_revenue'] = df_prod['total_sales'] * df_prod['avg_price']

    # Recalcular urgencias con los nuevos valores
    # Criterio A: Top 15% en ventana de 12 semanas
    df_prod['percentile_threshold'] = df_prod['total_sales'].rolling(
        window=12, min_periods=1
    ).quantile(0.85)

    df_prod['urgent_criterio_a'] = (
        df_prod['total_sales'] > df_prod['percentile_threshold']
    ).astype(int)

    # Criterio B: Growth > 12%
    df_prod['growth_rate'] = df_prod['total_sales'].pct_change()
    df_prod['urgent_criterio_b'] = (df_prod['growth_rate'] > 0.12).astype(int)

    # Urgencia híbrida
    df_prod['is_urgent'] = (
        (df_prod['urgent_criterio_a'] == 1) |
        (df_prod['urgent_criterio_b'] == 1)
    ).astype(int)

    all_products.append(df_prod)

    # Calcular score de predictibilidad (simple)
    urgency_rate = df_prod['is_urgent'].mean()
    urgency_std = df_prod['is_urgent'].std()

    # Score: productos con urgencias moderadas (15-25%) y consistentes tienen mejor score
    target_rate = 0.20
    rate_score = 1 - abs(urgency_rate - target_rate) / target_rate
    consistency_score = 1 - urgency_std

    total_score = (rate_score * 0.6 + consistency_score * 0.4) * 100

    predictability_scores.append({
        'product_id': f'PRODUCT_{product_id:03d}',
        'predictability_score': total_score,
        'urgency_rate': urgency_rate,
        'urgency_std': urgency_std,
        'n_urgencies': df_prod['is_urgent'].sum()
    })

    print(f"  ✓ {df_prod['product_id'].iloc[0]}: {df_prod['is_urgent'].sum()} urgencias ({urgency_rate:.1%})")

# 3. Combinar todos los productos
print()
print("3. Combinando datos...")
df_multi = pd.concat(all_products, ignore_index=True)
print(f"✓ Dataset multi-producto: {df_multi.shape}")
print(f"  Productos: {df_multi['product_id'].nunique()}")
print(f"  Semanas por producto: {len(df_multi) // NUM_PRODUCTS}")
print()

# 4. Crear ranking
print("4. Creando ranking de predictibilidad...")
df_ranking = pd.DataFrame(predictability_scores)
df_ranking = df_ranking.sort_values('predictability_score', ascending=False)
df_ranking['rank'] = range(1, len(df_ranking) + 1)
print(f"✓ Ranking creado: {df_ranking.shape}")
print()
print("Top 5 productos más predecibles:")
print(df_ranking.head()[['rank', 'product_id', 'predictability_score', 'urgency_rate']].to_string(index=False))
print()

# 5. Guardar archivos
print("5. Guardando archivos...")
output_path_multi = DATA_SIMULATED / 'urgencias_weekly.csv'
output_path_ranking = DATA_SIMULATED / 'products_predictability_ranking.csv'

df_multi.to_csv(output_path_multi, index=False)
df_ranking.to_csv(output_path_ranking, index=False)

print(f"✓ {output_path_multi}")
print(f"✓ {output_path_ranking}")
print()
print("="*80)
print("✅ DATOS MULTI-PRODUCTO GENERADOS")
print("="*80)
