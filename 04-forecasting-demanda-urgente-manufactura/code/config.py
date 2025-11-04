"""
Configuración global del proyecto de Forecasting de Demanda Urgente
Variables y constantes compartidas entre todos los notebooks
"""

from pathlib import Path

# ============================================
# PATHS DEL PROYECTO
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_SIMULATED = PROJECT_ROOT / 'data' / 'simulated'
RESULTS = PROJECT_ROOT / 'results'
FIGURES = RESULTS / 'figures'
OUTPUT = PROJECT_ROOT / 'output'

# Crear carpetas si no existen
for path in [DATA_RAW, DATA_PROCESSED, DATA_SIMULATED, RESULTS, FIGURES, OUTPUT]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================
# PARÁMETROS TEMPORALES
# ============================================
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
AGGREGATION = 'weekly'  # Agregación temporal
FORECAST_HORIZON = [1, 2, 4]  # Horizontes de predicción en semanas

# ============================================
# PARÁMETROS DE SIMULACIÓN DE URGENCIAS
# ============================================
URGENCY_THRESHOLD = 1.5  # Múltiplo de MA4 para considerar urgencia
SYNTHETIC_PROPORTION = 0.30  # Proporción de datos sintéticos
BASE_PROB_URGENT = 0.08  # Probabilidad base de urgencia
SEASONAL_AMPLITUDE = 0.04  # Amplitud de variación estacional

# ============================================
# PARÁMETROS DE VISUALIZACIÓN
# ============================================
FIGSIZE_STANDARD = (12, 6)
FIGSIZE_WIDE = (15, 5)
FIGSIZE_SQUARE = (10, 10)
COLOR_PALETTE = 'viridis'
DPI = 100

# Paleta de colores personalizada
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#6A4C93'
}

# ============================================
# PARÁMETROS DE MODELIZACIÓN
# ============================================
RANDOM_SEED = 42  # Semilla para reproducibilidad
CV_FOLDS = 5  # Folds para validación cruzada temporal
TEST_SIZE = 0.2  # Tamaño del conjunto de test

# Hiperparámetros base para modelos
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': RANDOM_SEED
}

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': RANDOM_SEED
}

# ============================================
# MÉTRICAS DE NEGOCIO
# ============================================
COST_FALSE_POSITIVE = 50  # Costo de falsa alarma (€)
COST_FALSE_NEGATIVE = 500  # Costo de urgencia no detectada (€)
COST_URGENT_PRODUCTION = 200  # Costo adicional producción urgente (€)

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def get_data_path(filename, data_type='processed'):
    """
    Obtener ruta completa a archivo de datos

    Args:
        filename (str): Nombre del archivo
        data_type (str): Tipo de datos ('raw', 'processed', 'simulated')

    Returns:
        Path: Ruta completa al archivo
    """
    data_dirs = {
        'raw': DATA_RAW,
        'processed': DATA_PROCESSED,
        'simulated': DATA_SIMULATED
    }
    return data_dirs[data_type] / filename


def get_figure_path(filename):
    """
    Obtener ruta para guardar figura

    Args:
        filename (str): Nombre del archivo de figura

    Returns:
        Path: Ruta completa para guardar figura
    """
    return FIGURES / filename


# ============================================
# VALIDACIÓN DE CONFIGURACIÓN
# ============================================

# Verificar que los ratios suman 1.0
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-10, \
    "Los ratios de train/val/test deben sumar 1.0"

# Verificar que los horizontes son positivos
assert all(h > 0 for h in FORECAST_HORIZON), \
    "Los horizontes de predicción deben ser positivos"

print(f"✓ Configuración cargada correctamente")
print(f"  - PROJECT_ROOT: {PROJECT_ROOT}")
print(f"  - RANDOM_SEED: {RANDOM_SEED}")
print(f"  - TRAIN/VAL/TEST: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
