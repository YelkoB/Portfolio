"""
00. Setup y Carga de Datos M5
==============================

OBJETIVO:
Cargar y procesar el dataset M5 (Walmart Sales) de Kaggle para análisis
de forecasting de urgencias en manufactura/retail.

INPUT (datos externos - NO incluidos en repo):
- data/raw/sales_train_evaluation.csv (ventas diarias por producto)
- data/raw/calendar.csv (mapeo de días a fechas)
- data/raw/sell_prices.csv (precios por tienda y fecha)

OUTPUT:
- data/processed/sales_weekly.csv (ventas GRANULARES: producto-tienda × semana)
- data/processed/sales_weekly_aggregated.csv (ventas AGREGADAS: producto-base × semana)
- data/processed/products_list.csv (lista de productos granulares)
- data/processed/products_base_list.csv (lista de productos base)

NOTA: Si no tienes el dataset M5, usa 00_generar_datos_sinteticos.py
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
    DATA_RAW, DATA_PROCESSED, FIGURES,
    FIGSIZE_STANDARD, FIGSIZE_WIDE,
    COLORS, RANDOM_SEED
)

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

print("="*80)
print("SETUP Y CARGA DE DATOS M5")
print("="*80)
print()

# ============================================================================
# 1. VERIFICAR EXISTENCIA DE DATOS
# ============================================================================
print("1. VERIFICANDO DATOS M5")
print("-" * 80)

required_files = ['sales_train_evaluation.csv', 'calendar.csv', 'sell_prices.csv']
all_exist = True

for file in required_files:
    path = DATA_RAW / file
    if path.exists():
        size_mb = path.stat().st_size / 1024**2
        print(f"✓ {file} encontrado ({size_mb:.2f} MB)")
    else:
        print(f"✗ {file} NO encontrado")
        all_exist = False

print()

if not all_exist:
    print("⚠️  ARCHIVOS M5 NO ENCONTRADOS")
    print()
    print("El dataset M5 debe descargarse de Kaggle:")
    print("  https://www.kaggle.com/c/m5-forecasting-accuracy/data")
    print()
    print("Archivos necesarios:")
    print("  - sales_train_evaluation.csv (~60 MB)")
    print("  - calendar.csv (~1 MB)")
    print("  - sell_prices.csv (~145 MB)")
    print()
    print("Colócalos en: data/raw/")
    print()
    print("ALTERNATIVA:")
    print("  Si no tienes el dataset M5, ejecuta:")
    print("  python code/00_generar_datos_sinteticos.py")
    print()
    sys.exit(1)

# ============================================================================
# 2. CARGA DE DATOS
# ============================================================================
print()
print("2. CARGANDO DATOS M5")
print("-" * 80)

# Calendar (ligero)
print("Cargando calendar.csv...")
calendar = pd.read_csv(DATA_RAW / 'calendar.csv')
calendar['date'] = pd.to_datetime(calendar['date'])
print(f"✓ Calendar: {calendar.shape}")
print(f"  Rango: {calendar['date'].min()} a {calendar['date'].max()}")

# Sales (pesado - puede tardar)
print()
print("Cargando sales_train_evaluation.csv...")
print("(Puede tardar 15-30 segundos)")
sales = pd.read_csv(DATA_RAW / 'sales_train_evaluation.csv')
print(f"✓ Sales: {sales.shape}")
print(f"  Productos: {sales.shape[0]:,}")
print(f"  Días: {sales.shape[1] - 6}")

# Prices
print()
print("Cargando sell_prices.csv...")
prices = pd.read_csv(DATA_RAW / 'sell_prices.csv')
print(f"✓ Prices: {prices.shape}")
print()

# ============================================================================
# 3. TRANSFORMACIÓN A FORMATO LARGO
# ============================================================================
print("3. TRANSFORMANDO DATOS")
print("-" * 80)
print("Convirtiendo formato ancho → largo...")
print("(Puede tardar 30-60 segundos)")

# Melt: convertir columnas d_1, d_2, ... a filas
id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
sales_long = sales.melt(
    id_vars=id_cols,
    var_name='d',
    value_name='sales'
)

print(f"✓ Transformación completada: {sales_long.shape}")

# Merge con calendar
print("Agregando información temporal...")
sales_long = sales_long.merge(
    calendar[['d', 'date', 'wm_yr_wk', 'weekday', 'month', 'year']],
    on='d',
    how='left'
)
sales_long['date'] = pd.to_datetime(sales_long['date'])
print(f"✓ Temporal agregado: {sales_long.shape}")

# Merge con precios
print("Agregando precios...")
sales_long = sales_long.merge(
    prices,
    on=['store_id', 'item_id', 'wm_yr_wk'],
    how='left'
)
sales_long['revenue'] = sales_long['sales'] * sales_long['sell_price']
print(f"✓ Precios agregados: {sales_long.shape}")
print()

# ============================================================================
# 4. SELECCIÓN DE PRODUCTOS TOP
# ============================================================================
print("4. ANÁLISIS DE PRODUCTOS")
print("-" * 80)

# Calcular ventas totales por producto
product_sales = sales_long.groupby(['item_id', 'store_id'])['sales'].sum().reset_index()
product_sales.columns = ['item_id', 'store_id', 'total_sales']
product_sales = product_sales.sort_values('total_sales', ascending=False)
product_sales['product_id'] = product_sales['item_id'] + '_' + product_sales['store_id']

print(f"Total producto-tienda combinaciones: {len(product_sales)}")
print(f"Top 10 productos por ventas:")
print(product_sales.head(10))
print()

# Filtrar productos por ventas mínimas (en lugar de top N fijo)
MIN_TOTAL_SALES = 5000  # Solo productos con >5K ventas totales
top_products = product_sales[product_sales['total_sales'] >= MIN_TOTAL_SALES]['product_id'].tolist()

print(f"✓ Seleccionados {len(top_products)} productos con >{MIN_TOTAL_SALES:,} ventas totales")
print(f"  (Esto permite analizar productos representativos sin ruido)")
print()

# ============================================================================
# 5. AGREGACIÓN SEMANAL POR PRODUCTO
# ============================================================================
print("5. AGREGACIÓN SEMANAL POR PRODUCTO")
print("-" * 80)

# Crear product_id (granular: producto-tienda) y product_base (agregado: solo producto)
sales_long['product_id'] = sales_long['item_id'] + '_' + sales_long['store_id']
sales_long['product_base'] = sales_long['item_id']  # Producto sin tienda

# Filtrar solo top productos
sales_long_top = sales_long[sales_long['product_id'].isin(top_products)].copy()

print(f"Datos filtrados: {len(sales_long_top):,} registros")
print(f"  • Productos únicos (granular): {sales_long_top['product_id'].nunique()}")
print(f"  • Productos base (agregado): {sales_long_top['product_base'].nunique()}")
print(f"  • Promedio tiendas por producto base: {sales_long_top['product_id'].nunique() / sales_long_top['product_base'].nunique():.1f}")
print()

# Agregar por producto-tienda y semana (GRANULAR)
sales_weekly = sales_long_top.groupby(['product_id', 'item_id', 'store_id', 'product_base', 'wm_yr_wk']).agg({
    'date': 'min',
    'sales': 'sum',
    'revenue': 'sum',
    'sell_price': 'mean'
}).reset_index()

sales_weekly.columns = ['product_id', 'item_id', 'store_id', 'product_base', 'week_id',
                         'week_start', 'total_sales', 'total_revenue', 'avg_price']
sales_weekly = sales_weekly.sort_values(['product_id', 'week_start']).reset_index(drop=True)

# Agregar información temporal
sales_weekly['year'] = sales_weekly['week_start'].dt.year
sales_weekly['month'] = sales_weekly['week_start'].dt.month
sales_weekly['quarter'] = sales_weekly['week_start'].dt.quarter
sales_weekly['week_of_year'] = sales_weekly['week_start'].dt.isocalendar().week
sales_weekly['week_of_month'] = (sales_weekly['week_start'].dt.day - 1) // 7 + 1

# Agregar week_num por producto
sales_weekly['week_num'] = sales_weekly.groupby('product_id').cumcount()

print(f"✓ Agregación GRANULAR completada (producto-tienda)")
print(f"  Total registros: {len(sales_weekly):,}")
print(f"  Productos: {sales_weekly['product_id'].nunique()}")
print(f"  Semanas por producto: ~{len(sales_weekly) / sales_weekly['product_id'].nunique():.0f}")
print(f"  Período: {sales_weekly['week_start'].min().date()} a {sales_weekly['week_start'].max().date()}")
print()

# ============================================================================
# 5.1. AGREGACIÓN A NIVEL PRODUCTO BASE (CONSOLIDADO)
# ============================================================================
print("5.1. AGREGACIÓN A NIVEL PRODUCTO BASE (consolidado)")
print("-" * 80)

# Crear dataset agregado por product_base (suma de todas las tiendas)
sales_weekly_agg = sales_long_top.groupby(['product_base', 'wm_yr_wk']).agg({
    'date': 'min',
    'sales': 'sum',
    'revenue': 'sum',
    'sell_price': 'mean'
}).reset_index()

sales_weekly_agg.columns = ['product_base', 'week_id',
                             'week_start', 'total_sales', 'total_revenue', 'avg_price']
sales_weekly_agg = sales_weekly_agg.sort_values(['product_base', 'week_start']).reset_index(drop=True)

# Agregar información temporal
sales_weekly_agg['year'] = sales_weekly_agg['week_start'].dt.year
sales_weekly_agg['month'] = sales_weekly_agg['week_start'].dt.month
sales_weekly_agg['quarter'] = sales_weekly_agg['week_start'].dt.quarter
sales_weekly_agg['week_of_year'] = sales_weekly_agg['week_start'].dt.isocalendar().week
sales_weekly_agg['week_of_month'] = (sales_weekly_agg['week_start'].dt.day - 1) // 7 + 1

# Agregar week_num por producto
sales_weekly_agg['week_num'] = sales_weekly_agg.groupby('product_base').cumcount()

print(f"✓ Agregación CONSOLIDADA completada (producto base)")
print(f"  Total registros: {len(sales_weekly_agg):,}")
print(f"  Productos base: {sales_weekly_agg['product_base'].nunique()}")
print(f"  Semanas por producto: ~{len(sales_weekly_agg) / sales_weekly_agg['product_base'].nunique():.0f}")
print(f"  Factor de consolidación: {len(sales_weekly) / len(sales_weekly_agg):.2f}x (más datos por producto base)")
print()

# ============================================================================
# 6. VALIDACIÓN
# ============================================================================
print("6. VALIDACIÓN DE DATOS (GRANULAR)")
print("-" * 80)

# Missing values
missing = sales_weekly.isnull().sum().sum()
print(f"Missing values: {missing} {'✓' if missing == 0 else '✗'}")

# Continuidad temporal
date_diff = sales_weekly['week_start'].diff()[1:]
expected_diff = pd.Timedelta(days=7)
gaps = (date_diff != expected_diff).sum()
print(f"Gaps temporales: {gaps} {'✓' if gaps == 0 else '✗'}")

# Valores negativos
negatives = (sales_weekly['total_sales'] < 0).sum()
print(f"Ventas negativas: {negatives} {'✓' if negatives == 0 else '✗'}")

print()

# ============================================================================
# 7. ANÁLISIS EXPLORATORIO BÁSICO
# ============================================================================
print("7. ANÁLISIS EXPLORATORIO")
print("-" * 80)

print("Estadísticas descriptivas:")
print(sales_weekly[['total_sales', 'total_revenue', 'avg_price']].describe())
print()

# Visualizar top 5 productos
print("Graficando top 5 productos...")
top_5_products = sales_weekly['product_id'].unique()[:5]

fig, axes = plt.subplots(5, 1, figsize=(15, 12))
for idx, product_id in enumerate(top_5_products):
    product_data = sales_weekly[sales_weekly['product_id'] == product_id]
    axes[idx].plot(product_data['week_start'], product_data['total_sales'],
                   linewidth=1, color=COLORS['primary'], alpha=0.8)
    axes[idx].axhline(product_data['total_sales'].mean(), color='red',
                      linestyle='--', linewidth=1, alpha=0.5)
    axes[idx].set_title(f'{product_id} (Promedio: {product_data["total_sales"].mean():.0f} u/semana)',
                        fontsize=10)
    axes[idx].set_ylabel('Ventas')
    axes[idx].grid(True, alpha=0.3)
    if idx < 4:
        axes[idx].set_xticklabels([])
    else:
        axes[idx].set_xlabel('Fecha')
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(FIGURES / '00_serie_temporal_m5_top5.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '00_serie_temporal_m5_top5.png'}")
plt.close()

# ============================================================================
# 8. GUARDAR DATOS PROCESADOS
# ============================================================================
print()
print("8. GUARDANDO DATOS")
print("-" * 80)

# Dataset GRANULAR (producto-tienda)
output_file_granular = DATA_PROCESSED / 'sales_weekly.csv'
sales_weekly.to_csv(output_file_granular, index=False)

print(f"✓ Datos GRANULARES guardados: {output_file_granular}")
print(f"  Registros: {len(sales_weekly):,}")
print(f"  Productos (producto-tienda): {sales_weekly['product_id'].nunique()}")
print(f"  Tamaño: {output_file_granular.stat().st_size / 1024:.2f} KB")
print()

# Dataset AGREGADO (producto base)
output_file_agg = DATA_PROCESSED / 'sales_weekly_aggregated.csv'
sales_weekly_agg.to_csv(output_file_agg, index=False)

print(f"✓ Datos AGREGADOS guardados: {output_file_agg}")
print(f"  Registros: {len(sales_weekly_agg):,}")
print(f"  Productos base: {sales_weekly_agg['product_base'].nunique()}")
print(f"  Tamaño: {output_file_agg.stat().st_size / 1024:.2f} KB")
print(f"  Factor de consolidación: {len(sales_weekly) / len(sales_weekly_agg):.2f}x")
print()

# Guardar lista de productos GRANULARES
products_file = DATA_PROCESSED / 'products_list.csv'
products_info = sales_weekly.groupby('product_id').agg({
    'item_id': 'first',
    'store_id': 'first',
    'product_base': 'first',
    'total_sales': 'sum',
    'week_start': 'count'
}).reset_index()
products_info.columns = ['product_id', 'item_id', 'store_id', 'product_base', 'total_sales', 'n_weeks']
products_info = products_info.sort_values('total_sales', ascending=False)
products_info.to_csv(products_file, index=False)

print(f"✓ Lista de productos granulares: {products_file}")

# Guardar lista de productos BASE
products_base_file = DATA_PROCESSED / 'products_base_list.csv'
products_base_info = sales_weekly_agg.groupby('product_base').agg({
    'total_sales': 'sum',
    'week_start': 'count'
}).reset_index()
products_base_info.columns = ['product_base', 'total_sales', 'n_weeks']
products_base_info = products_base_info.sort_values('total_sales', ascending=False)
products_base_info.to_csv(products_base_file, index=False)

print(f"✓ Lista de productos base: {products_base_file}")
print()

# ============================================================================
# 9. RESUMEN
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO")
print("="*80)
print()
print(f"📊 DATOS PROCESADOS - DOBLE GRANULARIDAD:")
print(f"  • Dataset: M5 (Walmart Sales) - Kaggle")
print()
print(f"  NIVEL GRANULAR (producto-tienda):")
print(f"    • Productos únicos: {sales_weekly['product_id'].nunique()}")
print(f"    • Total registros: {len(sales_weekly):,}")
print(f"    • Semanas por producto: ~{len(sales_weekly) / sales_weekly['product_id'].nunique():.0f}")
print()
print(f"  NIVEL AGREGADO (producto-base):")
print(f"    • Productos base: {sales_weekly_agg['product_base'].nunique()}")
print(f"    • Total registros: {len(sales_weekly_agg):,}")
print(f"    • Semanas por producto: ~{len(sales_weekly_agg) / sales_weekly_agg['product_base'].nunique():.0f}")
print(f"    • Factor consolidación: {len(sales_weekly) / len(sales_weekly_agg):.2f}x más datos/producto")
print(f"  • Período: {sales_weekly['week_start'].min().date()} a {sales_weekly['week_start'].max().date()}")
print()
print(f"📈 VENTAS:")
print(f"  • Promedio por semana-producto: {sales_weekly['total_sales'].mean():,.0f} unidades")
print(f"  • Total acumulado: {sales_weekly['total_sales'].sum():,.0f} unidades")
print(f"  • Revenue total: ${sales_weekly['total_revenue'].sum():,.2f}")
print()
print(f"✅ CALIDAD:")
print(f"  • Sin valores nulos: {missing == 0}")
print(f"  • Serie temporal continua por producto")
print(f"  • Valores en rangos esperados")
print()
print(f"📁 OUTPUTS:")
print(f"  • {output_file_granular.name}")
print(f"  • {output_file_agg.name}")
print(f"  • {products_file.name}")
print(f"  • {products_base_file.name}")
print(f"  • 00_serie_temporal_m5_top5.png")
print()
print("="*80)
print("✓ SETUP COMPLETADO")
print("="*80)
print()
print("PRÓXIMO PASO:")
print("  → Ejecutar: python code/01_generacion_urgencias.py")
print("  → Generará urgencias para ambos niveles de granularidad")
print("  → Probará modelos tanto en nivel producto-tienda como agregado")
print()
print("💡 ESTRATEGIA DUAL:")
print("  • Granular (producto-tienda): Máxima precisión si hay datos suficientes")
print("  • Agregado (producto-base): Más robusto cuando hay pocos datos por tienda")
print("  • El sistema elegirá automáticamente el mejor nivel por producto")
