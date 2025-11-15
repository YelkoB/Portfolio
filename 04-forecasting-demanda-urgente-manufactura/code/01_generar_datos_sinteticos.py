"""
Generador de Datos Sintéticos de Ventas con Patrones Predecibles
==================================================================

Genera datos de ventas semanales con:
- Tendencia creciente
- Estacionalidad anual (picos en navidad, verano)
- Estacionalidad mensual (fin de mes)
- Ruido aleatorio
- Picos predecibles que simulan "urgencias"

Estos picos son predecibles pero pueden parecer urgencias impredecibles
para un comprador sin análisis temporal.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Configuración
np.random.seed(42)

# Parámetros de generación
START_DATE = '2011-01-01'
N_WEEKS = 278  # ~5.3 años de datos
BASE_SALES = 180000  # Ventas base semanales

# Componentes
TREND_GROWTH = 0.003  # 0.3% crecimiento semanal
SEASONAL_ANNUAL_AMP = 30000  # Amplitud estacional anual
SEASONAL_MONTHLY_AMP = 8000  # Amplitud estacional mensual
NOISE_STD = 8000  # Desviación estándar del ruido

# Picos predecibles (urgencias)
PEAK_MONTHS = [5, 6, 11, 12]  # Mayo, Junio, Noviembre, Diciembre
PEAK_WEEKS_OF_MONTH = [3, 4]  # Semanas 3 y 4 de cada mes
PEAK_AMPLITUDE = 25000  # Amplitud adicional en picos

print("="*70)
print("GENERANDO DATOS SINTÉTICOS DE VENTAS")
print("="*70)
print(f"Período: {N_WEEKS} semanas (~{N_WEEKS/52:.1f} años)")
print(f"Fecha inicio: {START_DATE}")
print(f"Ventas base: {BASE_SALES:,} unidades/semana")
print()

# Generar fechas
start = pd.to_datetime(START_DATE)
dates = [start + timedelta(weeks=i) for i in range(N_WEEKS)]
df = pd.DataFrame({'week_start': dates})
df['week_num'] = range(len(df))
df['week_id'] = df['week_start'].dt.strftime('%y%W').astype(int)

# Extraer información temporal
df['year'] = df['week_start'].dt.year
df['month'] = df['week_start'].dt.month
df['quarter'] = df['week_start'].dt.quarter
df['week_of_year'] = df['week_start'].dt.isocalendar().week
df['week_of_month'] = (df['week_start'].dt.day - 1) // 7 + 1

print("✓ Fechas generadas")

# 1. Componente de TENDENCIA (crecimiento lineal)
df['trend'] = BASE_SALES * (1 + TREND_GROWTH) ** df['week_num']

# 2. Componente ESTACIONAL ANUAL (patrón sinusoidal)
# Picos en verano (semana 26) y navidad (semana 52)
df['seasonal_annual'] = SEASONAL_ANNUAL_AMP * np.sin(2 * np.pi * df['week_of_year'] / 52 - np.pi/2)

# 3. Componente ESTACIONAL MENSUAL (fin de mes)
# Picos en semana 3-4 de cada mes
monthly_pattern = np.array([0.8, 0.9, 1.2, 1.1])  # Patrón relativo por semana del mes
df['seasonal_monthly'] = df['week_of_month'].apply(
    lambda x: SEASONAL_MONTHLY_AMP * monthly_pattern[min(x, 3) - 1] if x <= 4 else SEASONAL_MONTHLY_AMP * monthly_pattern[3]
)

# 4. PICOS PREDECIBLES (urgencias)
# Picos en meses específicos, semanas específicas
df['predictable_peak'] = 0
for month in PEAK_MONTHS:
    for week in PEAK_WEEKS_OF_MONTH:
        mask = (df['month'] == month) & (df['week_of_month'] == week)
        df.loc[mask, 'predictable_peak'] = PEAK_AMPLITUDE

# 5. RUIDO aleatorio
df['noise'] = np.random.normal(0, NOISE_STD, len(df))

# VENTAS TOTALES = suma de todos los componentes
df['total_sales'] = (
    df['trend'] +
    df['seasonal_annual'] +
    df['seasonal_monthly'] +
    df['predictable_peak'] +
    df['noise']
).round().astype(int)

# Asegurar que no hay ventas negativas
df['total_sales'] = df['total_sales'].clip(lower=50000)

print("✓ Componentes de ventas calculados:")
print(f"  - Tendencia: {df['trend'].min():,.0f} → {df['trend'].max():,.0f}")
print(f"  - Estacionalidad anual: ±{SEASONAL_ANNUAL_AMP:,}")
print(f"  - Estacionalidad mensual: ±{SEASONAL_MONTHLY_AMP:,}")
print(f"  - Picos predecibles: {df['predictable_peak'].sum() / PEAK_AMPLITUDE:.0f} semanas")
print(f"  - Ruido: ±{NOISE_STD:,}")
print()

# Generar revenue y precio promedio
BASE_PRICE = 4.0
PRICE_VARIATION = 0.3
df['avg_price'] = BASE_PRICE + np.random.normal(0, PRICE_VARIATION, len(df))
df['avg_price'] = df['avg_price'].clip(lower=2.5, upper=6.0)
df['total_revenue'] = (df['total_sales'] * df['avg_price']).round(2)

print("✓ Revenue y precios calculados")
print()

# Estadísticas finales
print("ESTADÍSTICAS DE VENTAS GENERADAS:")
print("="*70)
print(f"  Media: {df['total_sales'].mean():,.0f} unidades/semana")
print(f"  Mediana: {df['total_sales'].median():,.0f}")
print(f"  Desv. estándar: {df['total_sales'].std():,.0f}")
print(f"  Mínimo: {df['total_sales'].min():,.0f}")
print(f"  Máximo: {df['total_sales'].max():,.0f}")
print(f"  Coef. variación: {df['total_sales'].std() / df['total_sales'].mean() * 100:.1f}%")
print()

# Guardar datos
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# Dataset principal
output_file = DATA_PROCESSED / 'sales_weekly.csv'
df_output = df[[
    'week_id', 'week_start', 'total_sales', 'total_revenue',
    'avg_price', 'week_num', 'year', 'month', 'quarter',
    'week_of_year', 'week_of_month'
]]
df_output.to_csv(output_file, index=False)

print(f"✓ Datos guardados: {output_file}")
print(f"  - Tamaño: {output_file.stat().st_size / 1024:.2f} KB")
print(f"  - Registros: {len(df_output)}")
print()

# Guardar componentes (para análisis)
components_file = DATA_PROCESSED / 'sales_components.csv'
df_components = df[[
    'week_start', 'total_sales', 'trend', 'seasonal_annual',
    'seasonal_monthly', 'predictable_peak', 'noise'
]]
df_components.to_csv(components_file, index=False)

print(f"✓ Componentes guardados: {components_file}")
print()

# Análisis de picos predecibles
peak_weeks = df[df['predictable_peak'] > 0]
print("PICOS PREDECIBLES GENERADOS:")
print("="*70)
print(f"  Total semanas con picos: {len(peak_weeks)}")
print(f"  Proporción: {len(peak_weeks)/len(df)*100:.1f}%")
print(f"  Ventas promedio en picos: {peak_weeks['total_sales'].mean():,.0f}")
print(f"  Ventas promedio normal: {df[df['predictable_peak'] == 0]['total_sales'].mean():,.0f}")
print(f"  Ratio pico/normal: {peak_weeks['total_sales'].mean() / df[df['predictable_peak'] == 0]['total_sales'].mean():.2f}x")
print()

print("="*70)
print("✓ GENERACIÓN COMPLETADA")
print("="*70)
print()
print("PRÓXIMO PASO:")
print("  Ejecutar notebook: 02_simulacion_urgencias_eda.ipynb")
print("  para detectar urgencias predecibles en estos datos")
