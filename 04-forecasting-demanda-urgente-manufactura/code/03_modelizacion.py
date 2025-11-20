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
print(f"  • {MODELS_DIR_GRANULAR}")
print(f"  • {MODELS_DIR_AGGREGATED}")
print()

# ============================================================================
# 1. CARGA DE DATOS (GRANULAR)
# ============================================================================
print("1. CARGANDO DATOS CON FEATURES (GRANULAR)")
print("-" * 80)

df = pd.read_csv(DATA_SIMULATED / 'features_weekly_granular.csv')
df['week_start'] = pd.to_datetime(df['week_start'])

print(f"✓ Dataset GRANULAR cargado: {df.shape}")
print(f"  Productos: {df['product_id'].nunique()}")
print(f"  Período: {df['week_start'].min()} a {df['week_start'].max()}")
print()

# Cargar lista de features
with open(DATA_SIMULATED / 'feature_list.json', 'r') as f:
    feature_list = json.load(f)

print(f"✓ Features cargadas: {len(feature_list['all_features'])} totales")
print()

print("💡 NOTA: Este script procesa nivel GRANULAR (producto-tienda)")
print("   Nivel AGREGADO pendiente (requiere features_weekly_aggregated.csv)")
print()

# ============================================================================
# 2. DEFINIR FEATURES Y TARGET
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

# Verificar que features existan
feature_cols = [f for f in feature_cols if f in df.columns]

print(f"Features seleccionados para ML: {len(feature_cols)}")
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
# 3. TRAIN/VAL/TEST SPLIT TEMPORAL
# ============================================================================
print("3. TRAIN/TEST SPLIT TEMPORAL (80/20)")
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


# Ejemplo con primer producto
products = df['product_id'].unique()
example_product = products[0]
df_example = df[df['product_id'] == example_product]

train_ex, test_ex = temporal_split(df_example)

print(f"Split para producto ejemplo ({example_product}):")
print(f"  Train: {len(train_ex)} registros ({len(train_ex)/len(df_example)*100:.1f}%)")
print(f"    Período: {train_ex['week_start'].min().date()} a {train_ex['week_start'].max().date()}")
print(f"  Test:  {len(test_ex)} registros ({len(test_ex)/len(df_example)*100:.1f}%)")
print(f"    Período: {test_ex['week_start'].min().date()} a {test_ex['week_start'].max().date()}")
print()
print(f"📝 Selección de modelo: Time Series CV en train set")
print(f"   • 3 folds deslizantes dentro del 80% train")
print(f"   • Comparar RandomForest vs XGBoost")
print(f"   • Entrenar ganador en TODO el train set")
print()

# ============================================================================
# 4. FUNCIONES DE MODELIZACIÓN CON TIME SERIES CV
# ============================================================================

def time_series_cv_split(X, y, n_splits=3):
    """
    Time Series Cross-Validation manual
    Devuelve índices de train/val para cada fold
    """
    n = len(X)
    fold_size = n // (n_splits + 1)

    folds = []
    for i in range(n_splits):
        train_end = fold_size * (i + 2)  # Crece el train set
        val_start = train_end
        val_end = min(train_end + fold_size, n)

        if val_end > val_start:  # Verificar que hay datos en val
            train_idx = list(range(train_end))
            val_idx = list(range(val_start, val_end))
            folds.append((train_idx, val_idx))

    return folds


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


def select_best_model_regression(X_train, y_train, n_cv_splits=3):
    """
    Selecciona el mejor modelo de regresión usando Time Series CV
    Devuelve el modelo entrenado en TODO el train set
    """
    # Time Series CV
    folds = time_series_cv_split(X_train, y_train, n_splits=n_cv_splits)

    if len(folds) == 0:
        return None, None, 'RandomForest'  # Default

    # Evaluar RandomForest
    rf_rmses = []
    for train_idx, val_idx in folds:
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_train.iloc[val_idx]

        _, metrics, _ = train_random_forest_regression(X_tr, y_tr, X_va, y_va)
        if metrics is not None:
            rf_rmses.append(metrics['rmse'])

    # Evaluar XGBoost
    xgb_rmses = []
    for train_idx, val_idx in folds:
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_train.iloc[val_idx]

        _, metrics, _ = train_xgboost_regression(X_tr, y_tr, X_va, y_va)
        if metrics is not None:
            xgb_rmses.append(metrics['rmse'])

    # Comparar promedios
    rf_mean = np.mean(rf_rmses) if len(rf_rmses) > 0 else float('inf')
    xgb_mean = np.mean(xgb_rmses) if len(xgb_rmses) > 0 else float('inf')

    # Elegir el mejor
    if rf_mean <= xgb_mean:
        best_model_name = 'RandomForest'
        model, metrics, _ = train_random_forest_regression(X_train, y_train, X_train, y_train)
    else:
        best_model_name = 'XGBoost'
        model, metrics, _ = train_xgboost_regression(X_train, y_train, X_train, y_train)

    # Guardar métricas CV para referencia
    cv_metrics = {
        'rmse_cv': min(rf_mean, xgb_mean),
        'rf_rmse_cv': rf_mean,
        'xgb_rmse_cv': xgb_mean
    }

    return model, cv_metrics, best_model_name


def select_best_model_classification(X_train, y_train, n_cv_splits=3):
    """
    Selecciona el mejor modelo de clasificación usando Time Series CV
    Devuelve el modelo entrenado en TODO el train set
    """
    # Time Series CV
    folds = time_series_cv_split(X_train, y_train, n_splits=n_cv_splits)

    if len(folds) == 0:
        return None, None, 'RandomForest'  # Default

    # Evaluar RandomForest
    rf_aucs = []
    for train_idx, val_idx in folds:
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_train.iloc[val_idx]

        _, metrics, _ = train_random_forest_classification(X_tr, y_tr, X_va, y_va)
        if metrics is not None:
            rf_aucs.append(metrics['auc'])

    # Evaluar XGBoost
    xgb_aucs = []
    for train_idx, val_idx in folds:
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_train.iloc[val_idx]

        _, metrics, _ = train_xgboost_classification(X_tr, y_tr, X_va, y_va)
        if metrics is not None:
            xgb_aucs.append(metrics['auc'])

    # Comparar promedios
    rf_mean = np.mean(rf_aucs) if len(rf_aucs) > 0 else 0
    xgb_mean = np.mean(xgb_aucs) if len(xgb_aucs) > 0 else 0

    # Elegir el mejor
    if rf_mean >= xgb_mean:
        best_model_name = 'RandomForest'
        model, metrics, _ = train_random_forest_classification(X_train, y_train, X_train, y_train)
    else:
        best_model_name = 'XGBoost'
        model, metrics, _ = train_xgboost_classification(X_train, y_train, X_train, y_train)

    # Guardar métricas CV para referencia
    cv_metrics = {
        'auc_cv': max(rf_mean, xgb_mean),
        'rf_auc_cv': rf_mean,
        'xgb_auc_cv': xgb_mean
    }

    return model, cv_metrics, best_model_name


# ============================================================================
# 5. ENTRENAR MODELOS PARA TODOS LOS PRODUCTOS
# ============================================================================
print("4. ENTRENANDO MODELOS PARA TOP PRODUCTOS")
print("-" * 80)

results = []
products = df['product_id'].unique()

print(f"Entrenando modelos para {len(products)} productos...")
print()

for product_id in tqdm(products, desc="Procesando productos"):
    df_product = df[df['product_id'] == product_id]

    # Split temporal 80/20
    train, test = temporal_split(df_product)

    # Preparar datos de train
    X_train = train[feature_cols]
    y_train_reg = train[target_regression]
    y_train_clf = train[target_classification]

    # ========================================================================
    # A. REGRESSION: Seleccionar mejor modelo (RF vs XGB) con Time Series CV
    # ========================================================================
    model_reg, cv_metrics_reg, best_model_reg_name = select_best_model_regression(
        X_train, y_train_reg, n_cv_splits=3
    )

    if model_reg is not None:
        results.append({
            'product_id': product_id,
            'model': best_model_reg_name,
            'task': 'regression',
            **cv_metrics_reg
        })

        # Guardar modelo
        model_path = MODELS_DIR_GRANULAR / f'{product_id}_rf_reg.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_reg, f)

    # ========================================================================
    # B. CLASSIFICATION: Seleccionar mejor modelo (RF vs XGB) con Time Series CV
    # ========================================================================
    model_clf, cv_metrics_clf, best_model_clf_name = select_best_model_classification(
        X_train, y_train_clf, n_cv_splits=3
    )

    if model_clf is not None:
        results.append({
            'product_id': product_id,
            'model': 'XGBoost',
            'task': 'regression',
            **metrics_xgb_reg
        })

        # Guardar modelo
        model_path = MODELS_DIR_GRANULAR / f'{product_id}_xgb_reg.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_xgb_reg, f)

    # ========================================================================
    # C. RANDOM FOREST CLASSIFICATION
    # ========================================================================
    model_rf_clf, metrics_rf_clf, _ = train_random_forest_classification(
        X_train, y_train_clf, X_val, y_val_clf
    )

    if model_rf_clf is not None:
        results.append({
            'product_id': product_id,
            'model': 'RandomForest',
            'task': 'classification',
            **cv_metrics_clf
        })

        # Guardar modelo
        model_path = MODELS_DIR_GRANULAR / f'{product_id}_rf_clf.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_rf_clf, f)

    # ========================================================================
    # D. XGBOOST CLASSIFICATION
    # ========================================================================
    model_xgb_clf, metrics_xgb_clf, _ = train_xgboost_classification(
        X_train, y_train_clf, X_val, y_val_clf
    )

    if model_xgb_clf is not None:
        results.append({
            'product_id': product_id,
            'model': 'XGBoost',
            'task': 'classification',
            **metrics_xgb_clf
        })

        # Guardar modelo
        model_path = MODELS_DIR_GRANULAR / f'{product_id}_xgb_clf.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_xgb_clf, f)

# Crear DataFrame de resultados
df_results = pd.DataFrame(results)

print()
print(f"✓ Modelos entrenados: {len(results)}")
print(f"  Productos procesados: {df_results['product_id'].nunique()}")
print(f"  Modelos guardados en: {MODELS_DIR_GRANULAR}")
print()

# ============================================================================
# 6. ANÁLISIS DE RESULTADOS
# ============================================================================
print("5. ANÁLISIS DE RESULTADOS (MÉTRICAS DE TIME SERIES CV)")
print("-" * 80)

# Resumen por modelo seleccionado
model_counts = df_results.groupby(['task', 'model']).size().reset_index(name='count')
print("Modelos seleccionados (Time Series CV):")
print(model_counts)
print()

# Métricas promedio de CV
df_reg = df_results[df_results['task'] == 'regression'].copy()
best_reg = pd.DataFrame()  # Inicializar como DataFrame vacío
if len(df_reg) > 0:
    print("REGRESIÓN - Métricas CV promedio:")
    print(f"  RMSE CV promedio: {df_reg['rmse_cv'].mean():.2f}")
    print(f"  RandomForest vs XGBoost:")
    print(f"    RF elegido: {(df_reg['model'] == 'RandomForest').sum()} productos")
    print(f"    XGB elegido: {(df_reg['model'] == 'XGBoost').sum()} productos")
    print()

# Mejor modelo por producto (clasificación)
df_clf = df_results[df_results['task'] == 'classification'].copy()
best_clf = pd.DataFrame()  # Inicializar como DataFrame vacío
if len(df_clf) > 0:
    print("CLASIFICACIÓN - Métricas CV promedio:")
    print(f"  AUC CV promedio: {df_clf['auc_cv'].mean():.3f}")
    print(f"  RandomForest vs XGBoost:")
    print(f"    RF elegido: {(df_clf['model'] == 'RandomForest').sum()} productos")
    print(f"    XGB elegido: {(df_clf['model'] == 'XGBoost').sum()} productos")
    print()

# ============================================================================
# 7. GUARDAR RESULTADOS (GRANULAR)
# ============================================================================
print("6. GUARDANDO RESULTADOS (GRANULAR)")
print("-" * 80)

# Guardar métricas de entrenamiento (granular)
output_file = DATA_SIMULATED / 'train_metrics_granular.csv'
df_results.to_csv(output_file, index=False)
print(f"✓ Métricas de entrenamiento guardadas: {output_file}")
print(f"  Registros: {len(df_results)}")
print(f"  Modelos entrenados en {MODELS_DIR_GRANULAR}")

# Guardar mejores modelos por producto
best_models = []
if len(best_reg) > 0:
    best_reg['metric_type'] = 'rmse'
    best_models.append(best_reg[['product_id', 'model', 'task', 'metric_type', 'rmse', 'mae', 'mape']])
if len(best_clf) > 0:
    best_clf['metric_type'] = 'f1'
    best_models.append(best_clf[['product_id', 'model', 'task', 'metric_type', 'precision', 'recall', 'f1', 'auc']])

if len(best_models) > 0:
    df_best = pd.concat(best_models, ignore_index=True)
    best_file = DATA_SIMULATED / 'best_models_granular.csv'
    df_best.to_csv(best_file, index=False)
    print(f"✓ Mejores modelos guardados: {best_file}")

print()

# ============================================================================
# 8. VISUALIZACIONES
# ============================================================================
print("7. VISUALIZACIONES")
print("-" * 80)

# Visualización de selección de modelos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regresión: Conteo de modelos seleccionados
if len(df_reg) > 0:
    model_counts_reg = df_reg['model'].value_counts()
    model_counts_reg.plot(kind='bar', ax=axes[0], color=[COLORS['primary'], COLORS['secondary']])
    axes[0].set_title('Modelos Seleccionados - Regresión\n(Time Series CV)',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Modelo')
    axes[0].set_ylabel('Número de Productos')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Añadir valores encima de las barras
    for i, v in enumerate(model_counts_reg):
        axes[0].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

# Clasificación: Conteo de modelos seleccionados
if len(df_clf) > 0:
    model_counts_clf = df_clf['model'].value_counts()
    model_counts_clf.plot(kind='bar', ax=axes[1], color=[COLORS['primary'], COLORS['secondary']])
    axes[1].set_title('Modelos Seleccionados - Clasificación\n(Time Series CV)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Modelo')
    axes[1].set_ylabel('Número de Productos')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Añadir valores encima de las barras
    for i, v in enumerate(model_counts_clf):
        axes[1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES / '03_model_selection.png', dpi=100, bbox_inches='tight')
print(f"✓ Guardado: {FIGURES / '03_model_selection.png'}")
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
print(f"  • Productos procesados: {df_results['product_id'].nunique()}")
print(f"  • Total modelos entrenados: {len(results)}")
print(f"  • Modelos guardados en: {MODELS_DIR_GRANULAR}")
print()
print(f"🔬 METODOLOGÍA:")
print(f"  • Split: 80% Train / 20% Test")
print(f"  • Selección de modelo: Time Series CV (3 folds)")
print(f"  • Candidatos: RandomForest vs XGBoost")
print(f"  • Criterio: RMSE (regresión), AUC (clasificación)")
print()
print(f"🤖 MODELOS SELECCIONADOS:")
if len(df_reg) > 0:
    rf_count_reg = (df_reg['model'] == 'RandomForest').sum()
    xgb_count_reg = (df_reg['model'] == 'XGBoost').sum()
    print(f"  Regresión:")
    print(f"    • RandomForest: {rf_count_reg} productos")
    print(f"    • XGBoost: {xgb_count_reg} productos")
if len(df_clf) > 0:
    rf_count_clf = (df_clf['model'] == 'RandomForest').sum()
    xgb_count_clf = (df_clf['model'] == 'XGBoost').sum()
    print(f"  Clasificación:")
    print(f"    • RandomForest: {rf_count_clf} productos")
    print(f"    • XGBoost: {xgb_count_clf} productos")
print()
print(f"📈 RENDIMIENTO CV PROMEDIO:")
if len(df_reg) > 0:
    print(f"  Regresión:")
    print(f"    • RMSE CV: {df_reg['rmse_cv'].mean():.2f}")
if len(df_clf) > 0:
    print(f"  Clasificación:")
    print(f"    • AUC CV: {df_clf['auc_cv'].mean():.3f}")
print()
print(f"📁 OUTPUTS GENERADOS:")
print(f"  • {best_file.name}")
print(f"  • {len(list(MODELS_DIR_GRANULAR.glob('*.pkl')))} modelos guardados (.pkl)")
print(f"  • 03_regression_comparison.png")
print(f"  • 03_classification_comparison.png")
print()
print("="*80)
print("✓ MODELIZACIÓN COMPLETADA")
print("="*80)
print()
print("CONCLUSIÓN:")
print(f"  ✓ {len(results)} modelos entrenados exitosamente")
print(f"  ✓ Mejor modelo seleccionado por producto usando Time Series CV")
print(f"  ✓ Modelos entrenados en 80% de datos (máximo aprovechamiento)")
print()
print("PRÓXIMO PASO:")
print(f"  → Fase 4: Validación en test set (20%)")
print(f"  → Análisis de errores y feature importance")
print(f"  → Cuantificación de valor operativo")
print()
