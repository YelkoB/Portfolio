"""
Generador de Datos Sintéticos Multi-Producto
=============================================

Genera múltiples productos con patrones predecibles para simular
urgencias que pueden ser forecasted.

Cada producto tiene:
- Tendencia única (crecimiento diferente)
- Estacionalidad anual y mensual
- Picos predecibles (urgencias)
- Ruido aleatorio

Output: sales_weekly.csv con formato (product_id × week)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Configuración
np.random.seed(42)

# Parámetros
START_DATE = '2011-01-01'
N_WEEKS = 278  # ~5.3 años
N_PRODUCTS = 30  # Productos sintéticos
BASE_SALES_RANGE = (5000, 50000)  # Rango ventas base

# Configuración de paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

print("="*80)
print("GENERANDO DATOS SINTÉTICOS MULTI-PRODUCTO")
print("="*80)
print(f"Productos: {N_PRODUCTS}")
print(f"Semanas: {N_WEEKS} (~{N_WEEKS/52:.1f} años)")
print(f"Fecha inicio: {START_DATE}")
print()

# Generar fechas base
start = pd.to_datetime(START_DATE)
dates = [start + timedelta(weeks=i) for i in range(N_WEEKS)]

# Generar múltiples productos
all_products = []

for prod_idx in range(N_PRODUCTS):
    product_id = f'PROD_{prod_idx+1:03d}'

    # Parámetros únicos por producto (variación controlada)
    base_sales = np.random.uniform(*BASE_SALES_RANGE)
    trend_growth = np.random.uniform(0.001, 0.005)  # 0.1-0.5% semanal
    seasonal_amp = base_sales * np.random.uniform(0.15, 0.30)  # 15-30% de la base
    monthly_amp = base_sales * np.random.uniform(0.05, 0.15)  # 5-15% de la base
    noise_std = base_sales * np.random.uniform(0.05, 0.15)  # 5-15% ruido
    peak_prob = np.random.uniform(0.10, 0.20)  # 10-20% semanas con pico

    # DataFrame para este producto
    df_prod = pd.DataFrame({'week_start': dates})
    df_prod['product_id'] = product_id
    df_prod['week_num'] = range(N_WEEKS)
    df_prod['week_id'] = df_prod['week_start'].dt.strftime('%y%W').astype(int)

    # Información temporal
    df_prod['year'] = df_prod['week_start'].dt.year
    df_prod['month'] = df_prod['week_start'].dt.month
    df_prod['quarter'] = df_prod['week_start'].dt.quarter
    df_prod['week_of_year'] = df_prod['week_start'].dt.isocalendar().week
    df_prod['week_of_month'] = (df_prod['week_start'].dt.day - 1) // 7 + 1

    # 1. Tendencia
    trend = base_sales * (1 + trend_growth) ** df_prod['week_num']

    # 2. Estacionalidad anual
    seasonal_annual = seasonal_amp * np.sin(2 * np.pi * df_prod['week_of_year'] / 52 - np.pi/2)

    # 3. Estacionalidad mensual (fin de mes)
    monthly_pattern = np.array([0.8, 0.9, 1.2, 1.1, 1.0])
    seasonal_monthly = df_prod['week_of_month'].apply(
        lambda x: monthly_amp * monthly_pattern[min(x, 4) - 1]
    )

    # 4. Picos predecibles (urgencias)
    # Meses con más urgencias (aleatorio por producto)
    peak_months = np.random.choice(range(1, 13), size=4, replace=False)
    df_prod['predictable_peak'] = 0
    for month in peak_months:
        mask = (df_prod['month'] == month) & (df_prod['week_of_month'].isin([3, 4]))
        df_prod.loc[mask, 'predictable_peak'] = base_sales * 0.3  # +30% en picos

    # 5. Ruido aleatorio
    noise = np.random.normal(0, noise_std, N_WEEKS)

    # Ventas totales
    df_prod['total_sales'] = (
        trend + seasonal_annual + seasonal_monthly + df_prod['predictable_peak'] + noise
    ).round().astype(int).clip(lower=int(base_sales * 0.3))  # Mínimo 30% de la base

    # Revenue y precio
    avg_price = np.random.uniform(2.0, 8.0)  # Precio base por producto
    price_variation = np.random.normal(avg_price, avg_price * 0.1, N_WEEKS)
    df_prod['avg_price'] = price_variation.clip(lower=avg_price * 0.5, upper=avg_price * 1.5)
    df_prod['total_revenue'] = (df_prod['total_sales'] * df_prod['avg_price']).round(2)

    # Agregar columnas adicionales para compatibilidad con M5
    df_prod['item_id'] = f'ITEM_{prod_idx+1:03d}'
    df_prod['store_id'] = 'STORE_SYNTH'

    all_products.append(df_prod)

# Combinar todos los productos
df = pd.concat(all_products, ignore_index=True)

print(f"✓ Datos generados:")
print(f"  Total registros: {len(df):,} (producto × semana)")
print(f"  Productos: {df['product_id'].nunique()}")
print(f"  Semanas por producto: {len(df) // df['product_id'].nunique()}")
print()

# Estadísticas
print("ESTADÍSTICAS POR PRODUCTO:")
print("="*80)
product_stats = df.groupby('product_id')['total_sales'].agg(['mean', 'std', 'min', 'max'])
print(product_stats.head(10))
print("...")
print(f"Promedio general: {df['total_sales'].mean():,.0f} unidades/semana")
print()

# Guardar
output_file = DATA_PROCESSED / 'sales_weekly.csv'
df_output = df[[
    'product_id', 'item_id', 'store_id', 'week_id', 'week_start',
    'total_sales', 'total_revenue', 'avg_price',
    'week_num', 'year', 'month', 'quarter', 'week_of_year', 'week_of_month'
]]
df_output.to_csv(output_file, index=False)

# Guardar lista de productos
products_file = DATA_PROCESSED / 'products_list.csv'
products_info = df.groupby('product_id').agg({
    'item_id': 'first',
    'store_id': 'first',
    'total_sales': 'sum',
    'week_start': 'count'
}).reset_index()
products_info.columns = ['product_id', 'item_id', 'store_id', 'total_sales', 'n_weeks']
products_info = products_info.sort_values('total_sales', ascending=False)
products_info.to_csv(products_file, index=False)

print(f"✓ Datos guardados: {output_file}")
print(f"  Tamaño: {output_file.stat().st_size / 1024:.2f} KB")
print()
print(f"✓ Lista productos: {products_file}")
print()

print("="*80)
print("✓ GENERACIÓN COMPLETADA")
print("="*80)
print()
print("PRÓXIMO PASO:")
print("  → Ejecutar: python code/02_deteccion_urgencias_predecibles.py")
print("  → Detectará urgencias POR producto y generará ranking")
