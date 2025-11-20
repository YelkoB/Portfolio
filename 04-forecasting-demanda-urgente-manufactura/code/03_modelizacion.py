"""
03. Modelización - Comparación Multi-Modelo
============================================

OBJETIVO:
Entrenar y comparar múltiples modelos de forecasting para predecir:
1. Ventas futuras (regresión)
2. Urgencias futuras (clasificación binaria)

MODELOS EVALUADOS:
- ARIMA/SARIMA - Baseline estadístico
- Prophet - Forecasting con estacionalidad automática (Facebook)
- Random Forest - Ensemble de árboles de decisión
- XGBoost - Gradient boosting optimizado

ESTRATEGIA:
- Train/Val/Test split temporal (70/15/15)
- Entrenar cada modelo en TOP 25 productos
- Evaluar en validation set
- Comparar métricas y seleccionar mejor modelo por producto

MÉTRICAS:
Regresión (ventas):
- RMSE, MAE, MAPE

Clasificación (urgencias):
- Precision, Recall, F1-Score, ROC-AUC

INPUT:
- data/simulated/features_weekly_granular.csv (producto-tienda)
- data/simulated/features_weekly_aggregated.csv (producto-base) [PENDIENTE]
- data/simulated/feature_list.json

OUTPUT:
- models/granular/{product_id}_rf_reg.pkl - Modelos Random Forest (regresión)
- models/granular/{product_id}_xgb_reg.pkl - Modelos XGBoost (regresión)
- models/granular/{product_id}_rf_clf.pkl - Modelos Random Forest (clasificación)
- models/granular/{product_id}_xgb_clf.pkl - Modelos XGBoost (clasificación)
- models/aggregated/... [PENDIENTE]
- data/simulated/train_metrics_granular.csv - Métricas de entrenamiento
- results/figures/03_*.png - Visualizaciones de comparación
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
from tqdm import tqdm
import pickle
import json

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
print("MODELIZACIÓN - COMPARACIÓN MULTI-MODELO")
print("="*80)
print()

# Crear directorios para modelos (granular y agregado)
MODELS_DIR_GRANULAR = PROJECT_ROOT / 'models' / 'granular'
MODELS_DIR_AGGREGATED = PROJECT_ROOT / 'models' / 'aggregated'
MODELS_DIR_GRANULAR.mkdir(parents=True, exist_ok=True)
MODELS_DIR_AGGREGATED.mkdir(parents=True, exist_ok=True)

print(f"✓ Directorios de modelos creados:")
print(f"  • {models_dir}")
print(f"  • {MODELS_DIR_AGGREGATED}")
print()

# ============================================================================
# 1. CARGA DE DATOS (GRANULAR Y AGREGADO)
# ============================================================================
print("1. CARGANDO DATOS CON FEATURES")
print("-" * 80)

# Intentar cargar nivel GRANULAR
df_granular = None
granular_file = DATA_SIMULATED / 'features_weekly_granular.csv'
if granular_file.exists():
    df_granular = pd.read_csv(granular_file)
    df_granular['week_start'] = pd.to_datetime(df_granular['week_start'])
    print(f"✅ GRANULAR (producto-tienda): {df_granular.shape}")
    print(f"   Productos: {df_granular['product_id'].nunique()}")
    print(f"   Período: {df_granular['week_start'].min().date()} a {df_granular['week_start'].max().date()}")
else:
    print(f"❌ GRANULAR: Archivo no encontrado")

# Intentar cargar nivel AGREGADO
df_aggregated = None
aggregated_file = DATA_SIMULATED / 'features_weekly_aggregated.csv'
if aggregated_file.exists():
    df_aggregated = pd.read_csv(aggregated_file)
    df_aggregated['week_start'] = pd.to_datetime(df_aggregated['week_start'])
    print(f"✅ AGREGADO (producto-base): {df_aggregated.shape}")
    print(f"   Productos: {df_aggregated['product_base'].nunique()}")
    print(f"   Período: {df_aggregated['week_start'].min().date()} a {df_aggregated['week_start'].max().date()}")
else:
    print(f"⚠️  AGREGADO: Archivo no encontrado - ejecuta script 02 primero")

print()

# Verificar que al menos un dataset esté disponible
if df_granular is None and df_aggregated is None:
    raise FileNotFoundError("No hay datasets disponibles. Ejecuta script 02 primero.")

# Cargar lista de features
with open(DATA_SIMULATED / 'feature_list.json', 'r') as f:
    feature_list = json.load(f)

print(f"✓ Features cargadas: {len(feature_list['all_features'])} totales")
print()

# ============================================================================
# 2. VERIFICAR DATASETS DISPONIBLES
# ============================================================================
datasets_to_process = []

if df_granular is not None:
    datasets_to_process.append({
        'df': df_granular,
        'id_col': 'product_id',
        'level_name': 'GRANULAR (producto-tienda)',
        'models_dir': MODELS_DIR_GRANULAR,
        'suffix': 'granular'
    })

if df_aggregated is not None:
    datasets_to_process.append({
        'df': df_aggregated,
        'id_col': 'product_base',
        'level_name': 'AGREGADO (producto-base)',
        'models_dir': MODELS_DIR_AGGREGATED,
        'suffix': 'aggregated'
    })

print(f"📊 NIVELES A PROCESAR: {len(datasets_to_process)}")
for dataset in datasets_to_process:
    print(f"   • {dataset['level_name']}")
print()

if len(datasets_to_process) == 0:
    raise FileNotFoundError("No hay datasets disponibles para procesar.")

# ============================================================================
# 3. DEFINIR FEATURES Y TARGET
# ============================================================================
print("2. DEFINIENDO FEATURES Y TARGETS")
print("-" * 80)

# Features para modelos ML (excluir lags de target y features redundantes)
feature_cols = [
    # Lags de ventas
    'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_52',
    # Rolling stats
    'sales_rolling_mean_4', 'sales_rolling_std_4',
    'sales_rolling_mean_12', 'sales_rolling_std_12',
    'sales_rolling_mean_52', 'sales_rolling_std_52',
    'sales_rolling_min_4', 'sales_rolling_max_4',
    'sales_rolling_min_12', 'sales_rolling_max_12',
    # Tendencia
    'time_index', 'trend_normalized', 'sales_diff_1',
    # Ratios
    'sales_ratio_mean_4', 'sales_ratio_mean_12',
    'cv_4', 'cv_12',
    # Urgencias pasadas
    'urgent_lag_1', 'urgent_lag_2', 'urgent_lag_4',
    'urgent_count_4', 'urgent_count_12',
    # Estacionales (numeric)
    'month', 'quarter', 'week_of_year', 'week_of_month',
    # Threshold
    'percentile_threshold', 'growth_rate'
]

print(f"Features disponibles: {len(feature_cols)}")
print(f"  Ejemplos: {', '.join(feature_cols[:10])}...")
print()

# Targets (shifted - predecir próxima semana)
target_regression = 'sales_target'
target_classification = 'is_urgent_target'

print(f"Target regresión: {target_regression} (ventas próxima semana)")
print(f"Target clasificación: {target_classification} (urgencia próxima semana)")
print(f"⚠️  Horizonte de predicción: 1 semana adelante")
print()

# ============================================================================
# 4. TRAIN/TEST SPLIT TEMPORAL
# ============================================================================
print("3. TRAIN/TEST SPLIT TEMPORAL")
print("-" * 80)

def temporal_split(product_df, train_pct=0.80):
    """
    Split temporal: Train 80%, Test 20%
    Sin data leakage - respeta orden temporal
    """
    df_sorted = product_df.sort_values('week_start').reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_pct)

    train = df_sorted.iloc[:train_end]
    test = df_sorted.iloc[train_end:]

    return train, test

print("Split: Train 80% / Test 20%")
print()

# ============================================================================
# 4. FUNCIONES DE MODELIZACIÓN
# ============================================================================

def train_random_forest_regression(X_train, y_train, X_val, y_val):
    """Entrena Random Forest para regresión de ventas"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # Eliminar NaNs
    mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
    mask_val = ~(X_val.isna().any(axis=1) | y_val.isna())

    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_val_clean = X_val[mask_val]
    y_val_clean = y_val[mask_val]

    # Validar que hay suficientes datos en train Y validation
    if len(X_train_clean) < 10 or len(X_val_clean) < 5:
        return None, None, None

    model.fit(X_train_clean, y_train_clean)
    y_pred = model.predict(X_val_clean)

    # Métricas
    rmse = np.sqrt(mean_squared_error(y_val_clean, y_pred))
    mae = mean_absolute_error(y_val_clean, y_pred)

    # SMAPE (Symmetric Mean Absolute Percentage Error) - más robusto que MAPE
    # SMAPE = 100 * mean(2 * |actual - pred| / (|actual| + |pred|))
    # Evita división por cero y es simétrico
    denominator = np.abs(y_val_clean) + np.abs(y_pred)
    # Evitar división por 0: si ambos son 0, el error es 0
    smape = np.mean(np.where(denominator == 0, 0, 2 * np.abs(y_val_clean - y_pred) / denominator)) * 100

    metrics = {'rmse': rmse, 'mae': mae, 'mape': smape}  # Mantener nombre 'mape' para compatibilidad

    return model, metrics, y_pred


def train_xgboost_regression(X_train, y_train, X_val, y_val):
    """Entrena XGBoost para regresión de ventas"""
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0
    )

    # Eliminar NaNs
    mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
    mask_val = ~(X_val.isna().any(axis=1) | y_val.isna())

    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_val_clean = X_val[mask_val]
    y_val_clean = y_val[mask_val]

    # Validar que hay suficientes datos en train Y validation
    if len(X_train_clean) < 10 or len(X_val_clean) < 5:
        return None, None, None

    model.fit(X_train_clean, y_train_clean)
    y_pred = model.predict(X_val_clean)

    # Métricas
    rmse = np.sqrt(mean_squared_error(y_val_clean, y_pred))
    mae = mean_absolute_error(y_val_clean, y_pred)

    # SMAPE (Symmetric Mean Absolute Percentage Error) - más robusto que MAPE
    # SMAPE = 100 * mean(2 * |actual - pred| / (|actual| + |pred|))
    # Evita división por cero y es simétrico
    denominator = np.abs(y_val_clean) + np.abs(y_pred)
    # Evitar división por 0: si ambos son 0, el error es 0
    smape = np.mean(np.where(denominator == 0, 0, 2 * np.abs(y_val_clean - y_pred) / denominator)) * 100

    metrics = {'rmse': rmse, 'mae': mae, 'mape': smape}  # Mantener nombre 'mape' para compatibilidad

    return model, metrics, y_pred


def train_random_forest_classification(X_train, y_train, X_val, y_val):
    """Entrena Random Forest para clasificación de urgencias"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # Eliminar NaNs
    mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
    mask_val = ~(X_val.isna().any(axis=1) | y_val.isna())

    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_val_clean = X_val[mask_val]
    y_val_clean = y_val[mask_val]

    # Validar que hay suficientes datos y ambas clases en train y val
    n_positive_train = y_train_clean.sum()
    n_negative_train = len(y_train_clean) - n_positive_train
    n_positive_val = y_val_clean.sum()
    n_negative_val = len(y_val_clean) - n_positive_val

    # Necesitamos al menos 2 ejemplos de cada clase en train y al menos 1 de cada clase en val
    if (len(X_train_clean) < 10 or len(X_val_clean) < 5 or
        n_positive_train < 2 or n_negative_train < 2 or
        n_positive_val < 1 or n_negative_val < 1):
        return None, None, None

    model.fit(X_train_clean, y_train_clean)
    y_pred = model.predict(X_val_clean)

    # Manejar predict_proba cuando solo hay una clase (edge case)
    y_pred_proba_full = model.predict_proba(X_val_clean)
    if y_pred_proba_full.shape[1] == 2:
        y_pred_proba = y_pred_proba_full[:, 1]
    else:
        # Solo hay una clase predicha - usar esa probabilidad
        y_pred_proba = y_pred_proba_full[:, 0]

    # Métricas
    precision = precision_score(y_val_clean, y_pred, zero_division=0)
    recall = recall_score(y_val_clean, y_pred, zero_division=0)
    f1 = f1_score(y_val_clean, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_val_clean, y_pred_proba)
    except:
        auc = 0.5

    metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

    return model, metrics, y_pred


def train_xgboost_classification(X_train, y_train, X_val, y_val):
    """Entrena XGBoost para clasificación de urgencias"""
    # Calcular scale_pos_weight para balancear clases
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0
    )

    # Eliminar NaNs
    mask_train = ~(X_train.isna().any(axis=1) | y_train.isna())
    mask_val = ~(X_val.isna().any(axis=1) | y_val.isna())

    X_train_clean = X_train[mask_train]
    y_train_clean = y_train[mask_train]
    X_val_clean = X_val[mask_val]
    y_val_clean = y_val[mask_val]

    # Validar que hay suficientes datos y ambas clases en train y val
    n_positive_train = y_train_clean.sum()
    n_negative_train = len(y_train_clean) - n_positive_train
    n_positive_val = y_val_clean.sum()
    n_negative_val = len(y_val_clean) - n_positive_val

    # Necesitamos al menos 2 ejemplos de cada clase en train y al menos 1 de cada clase en val
    if (len(X_train_clean) < 10 or len(X_val_clean) < 5 or
        n_positive_train < 2 or n_negative_train < 2 or
        n_positive_val < 1 or n_negative_val < 1):
        return None, None, None

    model.fit(X_train_clean, y_train_clean)
    y_pred = model.predict(X_val_clean)

    # Manejar predict_proba cuando solo hay una clase (edge case)
    y_pred_proba_full = model.predict_proba(X_val_clean)
    if y_pred_proba_full.shape[1] == 2:
        y_pred_proba = y_pred_proba_full[:, 1]
    else:
        # Solo hay una clase predicha - usar esa probabilidad
        y_pred_proba = y_pred_proba_full[:, 0]

    # Métricas
    precision = precision_score(y_val_clean, y_pred, zero_division=0)
    recall = recall_score(y_val_clean, y_pred, zero_division=0)
    f1 = f1_score(y_val_clean, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_val_clean, y_pred_proba)
    except:
        auc = 0.5

    metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

    return model, metrics, y_pred


# ============================================================================
# 5. ENTRENAR MODELOS PARA TODOS LOS NIVELES
# ============================================================================

# Almacenar resultados de todos los niveles
all_results = {}

# ============================================================================
# PROCESAR CADA NIVEL (GRANULAR Y AGREGADO)
# ============================================================================
for dataset_info in datasets_to_process:
    df = dataset_info['df']
    id_col = dataset_info['id_col']
    level_name = dataset_info['level_name']
    models_dir = dataset_info['models_dir']
    suffix = dataset_info['suffix']

    print()
    print("="*80)
    print(f"PROCESANDO: {level_name}")
    print("="*80)
    print()

    # Verificar que features existan en este dataset
    feature_cols_clean = [f for f in feature_cols if f in df.columns]

    print(f"Features disponibles en {level_name}: {len(feature_cols_clean)}")
    print()

    results = []
    products = df[id_col].unique()

    print(f"4. ENTRENANDO MODELOS - {level_name}")
    print("-" * 80)
    print(f"Entrenando modelos para {len(products)} productos...")
    print()

    for product_id in tqdm(products, desc=f"Procesando {suffix}"):
        df_product = df[df[id_col] == product_id]

        # Split temporal (80/20)
        train, test = temporal_split(df_product)

        # Preparar datos
        X_train = train[feature_cols_clean]
        y_train_reg = train[target_regression]
        y_train_clf = train[target_classification]

        X_test = test[feature_cols_clean]
        y_test_reg = test[target_regression]
        y_test_clf = test[target_classification]

        # ========================================================================
        # A. RANDOM FOREST REGRESSION
        # ========================================================================
        model_rf_reg, metrics_rf_reg, _ = train_random_forest_regression(
            X_train, y_train_reg, X_test, y_test_reg
        )

        if model_rf_reg is not None:
            results.append({
                id_col: product_id,
                'model': 'RandomForest',
                'task': 'regression',
                **metrics_rf_reg
            })

            # Guardar modelo
            model_path = models_dir / f'{product_id}_rf_reg.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model_rf_reg, f)

        # ========================================================================
        # B. XGBOOST REGRESSION
        # ========================================================================
        model_xgb_reg, metrics_xgb_reg, _ = train_xgboost_regression(
            X_train, y_train_reg, X_test, y_test_reg
        )

        if model_xgb_reg is not None:
            results.append({
                id_col: product_id,
                'model': 'XGBoost',
                'task': 'regression',
                **metrics_xgb_reg
            })

            # Guardar modelo
            model_path = models_dir / f'{product_id}_xgb_reg.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model_xgb_reg, f)

        # ========================================================================
        # C. RANDOM FOREST CLASSIFICATION
        # ========================================================================
        model_rf_clf, metrics_rf_clf, _ = train_random_forest_classification(
            X_train, y_train_clf, X_test, y_test_clf
        )

        if model_rf_clf is not None:
            results.append({
                id_col: product_id,
                'model': 'RandomForest',
                'task': 'classification',
                **metrics_rf_clf
            })

            # Guardar modelo
            model_path = models_dir / f'{product_id}_rf_clf.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model_rf_clf, f)

        # ========================================================================
        # D. XGBOOST CLASSIFICATION
        # ========================================================================
        model_xgb_clf, metrics_xgb_clf, _ = train_xgboost_classification(
            X_train, y_train_clf, X_test, y_test_clf
        )

        if model_xgb_clf is not None:
            results.append({
                id_col: product_id,
                'model': 'XGBoost',
                'task': 'classification',
                **metrics_xgb_clf
            })

            # Guardar modelo
            model_path = models_dir / f'{product_id}_xgb_clf.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model_xgb_clf, f)

    # Crear DataFrame de resultados para este nivel
    df_results = pd.DataFrame(results)

    print()
    print(f"✓ Modelos entrenados: {len(results)}")
    print(f"  Productos procesados: {df_results[id_col].nunique()}")
    print(f"  Modelos guardados en: {models_dir}")
    print()

    # ============================================================================
    # 6. ANÁLISIS DE RESULTADOS
    # ============================================================================
    print("5. ANÁLISIS DE RESULTADOS")
    print("-" * 80)

    # Resumen por modelo y tarea
    summary = df_results.groupby(['model', 'task']).agg({
        'rmse': 'mean',
        'mae': 'mean',
        'mape': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'auc': 'mean'
    }).round(3)
    
    print("Rendimiento promedio por modelo:")
    print(summary)
    print()
    
    # Mejor modelo por producto (regresión)
    df_reg = df_results[df_results['task'] == 'regression'].copy()
    best_reg = pd.DataFrame()  # Inicializar como DataFrame vacío
    if len(df_reg) > 0:
        best_reg = df_reg.loc[df_reg.groupby(id_col)['rmse'].idxmin()]
        print(f"Mejor modelo de regresión por producto:")
        print(best_reg[[id_col, 'model', 'rmse', 'mae', 'mape']].head(10))
        print()

    # Mejor modelo por producto (clasificación)
    df_clf = df_results[df_results['task'] == 'classification'].copy()
    best_clf = pd.DataFrame()  # Inicializar como DataFrame vacío
    if len(df_clf) > 0:
        best_clf = df_clf.loc[df_clf.groupby(id_col)['f1'].idxmax()]
        print(f"Mejor modelo de clasificación por producto:")
        print(best_clf[[id_col, 'model', 'precision', 'recall', 'f1', 'auc']].head(10))
        print()
    
    # ============================================================================
    # 7. GUARDAR RESULTADOS
    # ============================================================================
    print(f"6. GUARDANDO RESULTADOS ({level_name})")
    print("-" * 80)
    
    # Guardar métricas de entrenamiento
    output_file = DATA_SIMULATED / f'train_metrics_{suffix}.csv'
    df_results.to_csv(output_file, index=False)
    print(f"✓ Métricas de entrenamiento guardadas: {output_file}")
    print(f"  Registros: {len(df_results)}")
    print(f"  Modelos entrenados en {models_dir}")
    
    # Guardar mejores modelos por producto
    best_models = []
    if len(best_reg) > 0:
        best_reg['metric_type'] = 'rmse'
        cols_reg = [id_col, 'model', 'task', 'metric_type', 'rmse', 'mae', 'mape']
        best_models.append(best_reg[cols_reg])
    if len(best_clf) > 0:
        best_clf['metric_type'] = 'f1'
        cols_clf = [id_col, 'model', 'task', 'metric_type', 'precision', 'recall', 'f1', 'auc']
        best_models.append(best_clf[cols_clf])
    
    if len(best_models) > 0:
        df_best = pd.concat(best_models, ignore_index=True)
        best_file = DATA_SIMULATED / f'best_models_{suffix}.csv'
        df_best.to_csv(best_file, index=False)
        print(f"✓ Mejores modelos guardados: {best_file}")

    # Almacenar resultados para comparación
    all_results[suffix] = {
        'df_results': df_results,
        'df_reg': df_reg,
        'df_clf': df_clf,
        'level_name': level_name,
        'n_products': len(products)
    }

    print()

# ============================================================================
# 6. COMPARACIÓN ENTRE NIVELES (GRANULAR VS AGREGADO)
# ============================================================================
if len(all_results) > 1:
    print()
    print("="*80)
    print("COMPARACIÓN: GRANULAR vs AGREGADO")
    print("="*80)
    print()

    for suffix, data in all_results.items():
        level_name = data['level_name']
        df_reg = data['df_reg']
        df_clf = data['df_clf']
        n_products = data['n_products']

        print(f"📊 {level_name}")
        print(f"   Productos: {n_products}")

        if len(df_reg) > 0:
            print(f"   REGRESIÓN:")
            print(f"      RMSE medio: {df_reg['rmse'].mean():.2f}")
            print(f"      MAE medio:  {df_reg['mae'].mean():.2f}")
            print(f"      MAPE medio: {df_reg['mape'].mean():.2f}%")

        if len(df_clf) > 0:
            print(f"   CLASIFICACIÓN:")
            print(f"      F1 medio:        {df_clf['f1'].mean():.3f}")
            print(f"      Precision media: {df_clf['precision'].mean():.3f}")
            print(f"      Recall medio:    {df_clf['recall'].mean():.3f}")
            print(f"      AUC medio:       {df_clf['auc'].mean():.3f}")

        print()

    print("💡 CONCLUSIÓN:")
    print("   • Ambos niveles procesados exitosamente")
    print("   • Script 05 decidirá qué modelo usar por producto")
    print("   • Criterio: Mejor métrica (F1 para clasificación, RMSE para regresión)")
    print()

# ============================================================================
# 7. VISUALIZACIONES (usar primer nivel procesado)
# ============================================================================
print("7. VISUALIZACIONES")
print("-" * 80)

# Usar el primer nivel para visualizaciones
first_result = list(all_results.values())[0]
df_reg_viz = first_result['df_reg']
df_clf_viz = first_result['df_clf']
level_viz = first_result['level_name']

print(f"Generando visualizaciones para: {level_viz}")
print()

# Comparación de modelos (regresión)
if len(df_reg_viz) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Detectar id_col dinámicamente
    id_col_viz = 'product_id' if 'product_id' in df_reg_viz.columns else 'product_base'

    df_reg_pivot = df_reg_viz.pivot_table(index=id_col_viz, columns='model', values='rmse')
    df_reg_pivot.plot(kind='bar', ax=axes[0], color=[COLORS['primary'], COLORS['secondary']])
    axes[0].set_title('RMSE por Producto (Regresión)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Producto')
    axes[0].set_ylabel('RMSE')
    axes[0].legend(title='Modelo')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].grid(True, alpha=0.3, axis='y')

    df_reg_viz.boxplot(column='mae', by='model', ax=axes[1])
    axes[1].set_title('MAE Distribución (Regresión)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Modelo')
    axes[1].set_ylabel('MAE')
    axes[1].get_figure().suptitle('')

    df_reg_viz.boxplot(column='mape', by='model', ax=axes[2])
    axes[2].set_title('MAPE Distribución (Regresión)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Modelo')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].get_figure().suptitle('')

    plt.tight_layout()
    plt.savefig(FIGURES / '03_regression_comparison.png', dpi=100, bbox_inches='tight')
    print(f"✓ Guardado: {FIGURES / '03_regression_comparison.png'}")
    plt.close()

# Comparación de modelos (clasificación)
if len(df_clf_viz) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    df_clf_viz.boxplot(column='precision', by='model', ax=axes[0, 0])
    axes[0, 0].set_title('Precision Distribución', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Modelo')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].get_figure().suptitle('')

    df_clf_viz.boxplot(column='recall', by='model', ax=axes[0, 1])
    axes[0, 1].set_title('Recall Distribución', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Modelo')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].get_figure().suptitle('')

    df_clf_viz.boxplot(column='f1', by='model', ax=axes[1, 0])
    axes[1, 0].set_title('F1-Score Distribución', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Modelo')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].get_figure().suptitle('')

    df_clf_viz.boxplot(column='auc', by='model', ax=axes[1, 1])
    axes[1, 1].set_title('ROC-AUC Distribución', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Modelo')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].get_figure().suptitle('')

    plt.tight_layout()
    plt.savefig(FIGURES / '03_classification_comparison.png', dpi=100, bbox_inches='tight')
    print(f"✓ Guardado: {FIGURES / '03_classification_comparison.png'}")
    plt.close()

print()

# ============================================================================
# 9. RESUMEN EJECUTIVO
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO")
print("="*80)
print()

print(f"📊 MODELIZACIÓN COMPLETADA:")
total_products = sum([data['n_products'] for data in all_results.values()])
total_models = sum([len(data['df_results']) for data in all_results.values()])
print(f"  • Niveles procesados: {len(all_results)}")
print(f"  • Productos procesados: {total_products}")
print(f"  • Total modelos entrenados: {total_models}")
print()

for suffix, data in all_results.items():
    level_name = data['level_name']
    n_products = data['n_products']
    n_models = len(data['df_results'])
    print(f"  {level_name}:")
    print(f"     • {n_products} productos")
    print(f"     • {n_models} modelos entrenados")
print()

print(f"🤖 MODELOS EVALUADOS:")
print(f"  • Random Forest (Regresión + Clasificación)")
print(f"  • XGBoost (Regresión + Clasificación)")
print()

print(f"📈 RENDIMIENTO PROMEDIO (primer nivel):")
if len(df_reg_viz) > 0:
    print(f"  Regresión (ventas):")
    for model in df_reg_viz['model'].unique():
        model_data = df_reg_viz[df_reg_viz['model'] == model]
        print(f"    {model:15s} - RMSE: {model_data['rmse'].mean():.2f}, "
              f"MAE: {model_data['mae'].mean():.2f}, "
              f"MAPE: {model_data['mape'].mean():.2f}%")
print()
if len(df_clf_viz) > 0:
    print(f"  Clasificación (urgencias):")
    for model in df_clf_viz['model'].unique():
        model_data = df_clf_viz[df_clf_viz['model'] == model]
        print(f"    {model:15s} - F1: {model_data['f1'].mean():.3f}, "
              f"Precision: {model_data['precision'].mean():.3f}, "
              f"Recall: {model_data['recall'].mean():.3f}, "
              f"AUC: {model_data['auc'].mean():.3f}")
print()

print(f"📁 OUTPUTS GENERADOS:")
for suffix in all_results.keys():
    print(f"  • train_metrics_{suffix}.csv")
    print(f"  • best_models_{suffix}.csv")
    print(f"  • models/{suffix}/*.pkl")
print(f"  • 03_regression_comparison.png")
print(f"  • 03_classification_comparison.png")
print()
print("="*80)
print("✓ MODELIZACIÓN COMPLETADA")
print("="*80)
print()
print("CONCLUSIÓN:")
print(f"  ✓ {len(results)} modelos entrenados exitosamente")
print(f"  ✓ Comparación de rendimiento completada")
print(f"  ✓ Modelos listos para predicción en producción")
print()
print("PRÓXIMO PASO:")
print(f"  → Fase 4: Validación en test set")
print(f"  → Análisis de errores y feature importance")
print(f"  → Predicciones en datos futuros")
print()
