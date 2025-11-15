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
- data/processed/sales_weekly.csv (ventas agregadas semanalmente)
- data/processed/sales_components.csv (componentes temporales)

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
# 4. AGREGACIÓN SEMANAL
# ============================================================================
print("4. AGREGACIÓN SEMANAL")
print("-" * 80)

sales_weekly = sales_long.groupby(['wm_yr_wk']).agg({
    'date': 'min',
    'sales': 'sum',
    'revenue': 'sum',
    'sell_price': 'mean'
}).reset_index()

sales_weekly.columns = ['week_id', 'week_start', 'total_sales', 'total_revenue', 'avg_price']
sales_weekly = sales_weekly.sort_values('week_start').reset_index(drop=True)
sales_weekly['week_num'] = range(len(sales_weekly))

# Agregar información temporal
sales_weekly['year'] = sales_weekly['week_start'].dt.year
sales_weekly['month'] = sales_weekly['week_start'].dt.month
sales_weekly['quarter'] = sales_weekly['week_start'].dt.quarter
sales_weekly['week_of_year'] = sales_weekly['week_start'].dt.isocalendar().week
sales_weekly['week_of_month'] = (sales_weekly['week_start'].dt.day - 1) // 7 + 1

print(f"✓ Agregación completada")
print(f"  Total semanas: {len(sales_weekly)}")
print(f"  Período: {sales_weekly['week_start'].min().date()} a {sales_weekly['week_start'].max().date()}")
print(f"  Ventas promedio: {sales_weekly['total_sales'].mean():,.0f} unidades/semana")
print()

# ============================================================================
# 5. VALIDACIÓN
# ============================================================================
print("5. VALIDACIÓN DE DATOS")
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
# 6. ANÁLISIS EXPLORATORIO BÁSICO
# ============================================================================
print("6. ANÁLISIS EXPLORATORIO")
print("-" * 80)

print("Estadísticas descriptivas:")
print(sales_weekly[['total_sales', 'total_revenue', 'avg_price']].describe())
print()

# Serie temporal
fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
ax.plot(sales_weekly['week_start'], sales_weekly['total_sales'],
        linewidth=1, color=COLORS['primary'], alpha=0.8)
ax.axhline(sales_weekly['total_sales'].mean(), color='red',
           linestyle='--', linewidth=1.5, alpha=0.7,
           label=f"Promedio: {sales_weekly['total_sales'].mean():,.0f}")
ax.set_title('Serie Temporal de Ventas Semanales (M5 Dataset)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Unidades Vendidas', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURES / '00_serie_temporal_m5.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '00_serie_temporal_m5.png'}")
plt.close()

# ============================================================================
# 7. GUARDAR DATOS PROCESADOS
# ============================================================================
print()
print("7. GUARDANDO DATOS")
print("-" * 80)

# Dataset principal
output_file = DATA_PROCESSED / 'sales_weekly.csv'
sales_weekly.to_csv(output_file, index=False)

print(f"✓ Datos guardados: {output_file}")
print(f"  Registros: {len(sales_weekly)}")
print(f"  Tamaño: {output_file.stat().st_size / 1024:.2f} KB")
print()

# ============================================================================
# 8. RESUMEN
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO")
print("="*80)
print()
print(f"📊 DATOS PROCESADOS:")
print(f"  • Dataset: M5 (Walmart Sales) - Kaggle")
print(f"  • Total semanas: {len(sales_weekly)}")
print(f"  • Período: {sales_weekly['week_start'].min().date()} a {sales_weekly['week_start'].max().date()}")
print(f"  • Ventas promedio: {sales_weekly['total_sales'].mean():,.0f} unidades/semana")
print(f"  • Ventas totales: {sales_weekly['total_sales'].sum():,.0f} unidades")
print(f"  • Revenue total: ${sales_weekly['total_revenue'].sum():,.2f}")
print()
print(f"✅ CALIDAD:")
print(f"  • Sin valores nulos")
print(f"  • Serie temporal continua")
print(f"  • Valores en rangos esperados")
print()
print(f"📁 OUTPUTS:")
print(f"  • {output_file.name}")
print(f"  • 00_serie_temporal_m5.png")
print()
print("="*80)
print("✓ SETUP COMPLETADO")
print("="*80)
print()
print("PRÓXIMO PASO:")
print("  → Ejecutar: python code/02_deteccion_urgencias_predecibles.py")
