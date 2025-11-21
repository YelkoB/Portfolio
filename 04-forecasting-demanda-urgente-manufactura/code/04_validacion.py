"""
04. Validación - Evaluación en Test Set
========================================

OBJETIVO:
Evaluar el rendimiento de los mejores modelos en el test set (datos no vistos)
y analizar errores, feature importance y predicciones futuras.

EVALUACIÓN:
1. Cargar mejores modelos seleccionados en Fase 3
2. Predecir en test set (15% final de datos)
3. Calcular métricas finales de rendimiento
4. Análisis de errores y residuos
5. Feature importance (modelos basados en árboles)
6. Predicciones para semanas futuras

INPUT:
- data/simulated/features_weekly_granular.csv (producto-tienda)
- data/simulated/features_weekly_aggregated.csv (producto-base)
- data/simulated/best_models_granular.csv
- data/simulated/best_models_aggregated.csv
- models/granular/*.pkl
- models/aggregated/*.pkl

OUTPUT:
- data/simulated/test_predictions_granular.csv
- data/simulated/test_predictions_aggregated.csv
- data/simulated/validation_metrics_granular.csv
- data/simulated/validation_metrics_aggregated.csv
- results/figures/04_*.png - Visualizaciones de validación
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
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
print("VALIDACIÓN - EVALUACIÓN EN TEST SET")
print("="*80)
print()

# Crear directorios para modelos (granular y agregado)
MODELS_DIR_GRANULAR = PROJECT_ROOT / 'models' / 'granular'
MODELS_DIR_AGGREGATED = PROJECT_ROOT / 'models' / 'aggregated'

# ============================================================================
# 1. CARGA DE DATOS (GRANULAR Y AGREGADO)
# ============================================================================
print("1. CARGANDO DATOS Y MODELOS")
print("-" * 80)

# Intentar cargar nivel GRANULAR
df_granular = None
best_models_granular = None
granular_file = DATA_SIMULATED / 'features_weekly_granular.csv'
best_granular_file = DATA_SIMULATED / 'best_models_granular.csv'

if granular_file.exists() and best_granular_file.exists():
    df_granular = pd.read_csv(granular_file)
    df_granular['week_start'] = pd.to_datetime(df_granular['week_start'])
    best_models_granular = pd.read_csv(best_granular_file)
    print(f"✅ GRANULAR (producto-tienda): {df_granular.shape}")
    print(f"   Productos: {df_granular['product_id'].nunique()}")
    print(f"   Mejores modelos: {len(best_models_granular)}")
else:
    print(f"❌ GRANULAR: Archivos no encontrados")

# Intentar cargar nivel AGREGADO
df_aggregated = None
best_models_aggregated = None
aggregated_file = DATA_SIMULATED / 'features_weekly_aggregated.csv'
best_aggregated_file = DATA_SIMULATED / 'best_models_aggregated.csv'

if aggregated_file.exists() and best_aggregated_file.exists():
    df_aggregated = pd.read_csv(aggregated_file)
    df_aggregated['week_start'] = pd.to_datetime(df_aggregated['week_start'])
    best_models_aggregated = pd.read_csv(best_aggregated_file)
    print(f"✅ AGREGADO (producto-base): {df_aggregated.shape}")
    print(f"   Productos: {df_aggregated['product_base'].nunique()}")
    print(f"   Mejores modelos: {len(best_models_aggregated)}")
else:
    print(f"⚠️  AGREGADO: Archivos no encontrados")

print()

# Verificar que al menos un dataset esté disponible
if df_granular is None and df_aggregated is None:
    raise FileNotFoundError("No hay datasets disponibles. Ejecuta script 03 primero.")

# Preparar datasets para procesar
datasets_to_process = []

if df_granular is not None and best_models_granular is not None:
    datasets_to_process.append({
        'df': df_granular,
        'df_best': best_models_granular,
        'id_col': 'product_id',
        'level_name': 'GRANULAR (producto-tienda)',
        'models_dir': MODELS_DIR_GRANULAR,
        'suffix': 'granular'
    })

if df_aggregated is not None and best_models_aggregated is not None:
    datasets_to_process.append({
        'df': df_aggregated,
        'df_best': best_models_aggregated,
        'id_col': 'product_base',
        'level_name': 'AGREGADO (producto-base)',
        'models_dir': MODELS_DIR_AGGREGATED,
        'suffix': 'aggregated'
    })

print(f"📊 NIVELES A VALIDAR: {len(datasets_to_process)}")
for dataset in datasets_to_process:
    print(f"   • {dataset['level_name']}")
print()

# Cargar feature list
with open(DATA_SIMULATED / 'feature_list.json', 'r') as f:
    feature_list = json.load(f)

feature_cols = [
    'sales_lag_1', 'sales_lag_2', 'sales_lag_4', 'sales_lag_52',
    'sales_rolling_mean_4', 'sales_rolling_std_4',
    'sales_rolling_mean_12', 'sales_rolling_std_12',
    'sales_rolling_mean_52', 'sales_rolling_std_52',
    'sales_rolling_min_4', 'sales_rolling_max_4',
    'sales_rolling_min_12', 'sales_rolling_max_12',
    'time_index', 'trend_normalized', 'sales_diff_1',
    'sales_ratio_mean_4', 'sales_ratio_mean_12',
    'cv_4', 'cv_12',
    'urgent_lag_1', 'urgent_lag_2', 'urgent_lag_4',
    'urgent_count_4', 'urgent_count_12',
    'month', 'quarter', 'week_of_year', 'week_of_month',
    'percentile_threshold', 'growth_rate'
]

# ============================================================================
# 2. TRAIN/TEST SPLIT
# ============================================================================

def temporal_split(product_df, train_pct=0.80):
    """Split temporal Train 80% / Test 20%"""
    df_sorted = product_df.sort_values('week_start').reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_pct)

    train = df_sorted.iloc[:train_end]
    test = df_sorted.iloc[train_end:]

    return train, test


# ============================================================================
# 3. EVALUACIÓN EN TEST SET (AMBOS NIVELES)
# ============================================================================
print("2. EVALUANDO MODELOS EN TEST SET")
print("-" * 80)

# Almacenar resultados de todos los niveles
all_results = {}

# ============================================================================
# LOOP SOBRE AMBOS NIVELES (GRANULAR Y AGREGADO)
# ============================================================================
for dataset_info in datasets_to_process:
    df = dataset_info['df']
    df_best = dataset_info['df_best']
    id_col = dataset_info['id_col']
    level_name = dataset_info['level_name']
    models_dir = dataset_info['models_dir']
    suffix = dataset_info['suffix']

    print(f"\n{'='*80}")
    print(f"PROCESANDO: {level_name}")
    print(f"{'='*80}")
    print(f"  • Productos: {df[id_col].nunique()}")
    print(f"  • Modelos dir: {models_dir}")
    print()

    # Filtrar feature_cols disponibles en este dataset
    feature_cols_filtered = [f for f in feature_cols if f in df.columns]

    test_results = []
    predictions = []

    products = df[id_col].unique()

    for product_id in tqdm(products, desc=f"Evaluando {suffix}"):
        df_product = df[df[id_col] == product_id]

        # Split (Train 80% / Test 20%)
        train, test = temporal_split(df_product)

        if len(test) < 5:
            continue

        # Preparar test set
        X_test = test[feature_cols_filtered]
        y_test_reg = test['sales_target']  # Predicción próxima semana
        y_test_clf = test['is_urgent_target']  # Urgencia próxima semana

        # Eliminar NaNs
        mask_test = ~(X_test.isna().any(axis=1))
        X_test_clean = X_test[mask_test]
        y_test_reg_clean = y_test_reg[mask_test]
        y_test_clf_clean = y_test_clf[mask_test]
        test_weeks = test.loc[mask_test, 'week_start']

        if len(X_test_clean) < 3:
            continue

        # ========================================================================
        # A. REGRESIÓN
        # ========================================================================
        best_reg_model = df_best[
            (df_best[id_col] == product_id) &
            (df_best['task'] == 'regression')
        ]

        if len(best_reg_model) > 0:
            model_name = best_reg_model.iloc[0]['model']
            model_type = 'rf' if model_name == 'RandomForest' else 'xgb'
            model_path = models_dir / f'{product_id}_{model_type}_reg.pkl'

        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Predecir
            y_pred_reg = model.predict(X_test_clean)

            # Métricas
            rmse = np.sqrt(mean_squared_error(y_test_reg_clean, y_pred_reg))
            mae = mean_absolute_error(y_test_reg_clean, y_pred_reg)
            mape = np.mean(np.abs((y_test_reg_clean - y_pred_reg) / (y_test_reg_clean + 1))) * 100

            test_results.append({
                id_col: product_id,
                'model': model_name,
                'task': 'regression',
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            })

            # Guardar predicciones
            for idx, (week, actual, pred) in enumerate(zip(test_weeks, y_test_reg_clean, y_pred_reg)):
                predictions.append({
                    id_col: product_id,
                    'week_start': week,
                    'task': 'regression',
                    'actual': actual,
                    'predicted': pred,
                    'error': actual - pred,
                    'abs_error': abs(actual - pred)
                })

        # ========================================================================
        # B. CLASIFICACIÓN
        # ========================================================================
        best_clf_model = df_best[
            (df_best[id_col] == product_id) &
            (df_best['task'] == 'classification')
        ]

        if len(best_clf_model) > 0:
            model_name = best_clf_model.iloc[0]['model']
            model_type = 'rf' if model_name == 'RandomForest' else 'xgb'
            model_path = models_dir / f'{product_id}_{model_type}_clf.pkl'

        if model_path.exists() and y_test_clf_clean.sum() > 0:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Predecir
            y_pred_clf = model.predict(X_test_clean)

            # Manejar predict_proba cuando solo hay una clase (edge case)
            y_pred_proba_full = model.predict_proba(X_test_clean)
            if y_pred_proba_full.shape[1] == 2:
                y_pred_proba = y_pred_proba_full[:, 1]
            else:
                # Solo hay una clase predicha - usar esa probabilidad
                y_pred_proba = y_pred_proba_full[:, 0]

            # Métricas
            precision = precision_score(y_test_clf_clean, y_pred_clf, zero_division=0)
            recall = recall_score(y_test_clf_clean, y_pred_clf, zero_division=0)
            f1 = f1_score(y_test_clf_clean, y_pred_clf, zero_division=0)
            try:
                auc = roc_auc_score(y_test_clf_clean, y_pred_proba)
            except:
                auc = 0.5

            test_results.append({
                id_col: product_id,
                'model': model_name,
                'task': 'classification',
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            })

            # Guardar predicciones
            for idx, (week, actual, pred, proba) in enumerate(zip(test_weeks, y_test_clf_clean, y_pred_clf, y_pred_proba)):
                predictions.append({
                    id_col: product_id,
                    'week_start': week,
                    'task': 'classification',
                    'actual': int(actual),
                    'predicted': int(pred),
                    'predicted_proba': proba,
                    'correct': int(actual == pred)
                })

    # Crear DataFrames para este nivel
    df_test_results = pd.DataFrame(test_results)
    df_predictions = pd.DataFrame(predictions)

    print()
    print(f"✓ Evaluación {suffix} completada")
    print(f"  Productos evaluados: {df_test_results[id_col].nunique()}")
    print(f"  Predicciones generadas: {len(df_predictions):,}")
    print()

    # ========================================================================
    # MÉTRICAS FINALES PARA ESTE NIVEL
    # ========================================================================
    print(f"{'='*80}")
    print(f"MÉTRICAS FINALES - {level_name}")
    print(f"{'='*80}")

    # Regresión
    df_reg_test = df_test_results[df_test_results['task'] == 'regression']
    if len(df_reg_test) > 0:
        print("REGRESIÓN (Predicción de ventas):")
        print(f"  Productos evaluados: {len(df_reg_test)}")
        print()
        summary_reg = df_reg_test.groupby('model')[['rmse', 'mae', 'mape']].mean()
        print(summary_reg)
        print()
        print(f"  Promedio general:")
        print(f"    RMSE: {df_reg_test['rmse'].mean():.2f}")
        print(f"    MAE:  {df_reg_test['mae'].mean():.2f}")
        print(f"    MAPE: {df_reg_test['mape'].mean():.2f}%")
        print()

    # Clasificación
    df_clf_test = df_test_results[df_test_results['task'] == 'classification']
    if len(df_clf_test) > 0:
        print("CLASIFICACIÓN (Predicción de urgencias):")
        print(f"  Productos evaluados: {len(df_clf_test)}")
        print()
        summary_clf = df_clf_test.groupby('model')[['precision', 'recall', 'f1', 'auc']].mean()
        print(summary_clf)
        print()
        print(f"  Promedio general:")
        print(f"    Precision: {df_clf_test['precision'].mean():.3f}")
        print(f"    Recall:    {df_clf_test['recall'].mean():.3f}")
        print(f"    F1-Score:  {df_clf_test['f1'].mean():.3f}")
        print(f"    ROC-AUC:   {df_clf_test['auc'].mean():.3f}")
        print()

    # ========================================================================
    # GUARDAR RESULTADOS PARA ESTE NIVEL
    # ========================================================================
    print(f"GUARDANDO RESULTADOS - {suffix}")
    print("-" * 80)

    # Métricas
    metrics_file = DATA_SIMULATED / f'validation_metrics_{suffix}.csv'
    df_test_results.to_csv(metrics_file, index=False)
    print(f"✓ Métricas guardadas: {metrics_file}")

    # Predicciones
    pred_file = DATA_SIMULATED / f'test_predictions_{suffix}.csv'
    df_predictions.to_csv(pred_file, index=False)
    print(f"✓ Predicciones guardadas: {pred_file}")
    print()

    # Guardar en diccionario para comparación posterior
    all_results[suffix] = {
        'level_name': level_name,
        'df_test_results': df_test_results,
        'df_predictions': df_predictions,
        'df_reg_test': df_reg_test,
        'df_clf_test': df_clf_test,
        'id_col': id_col,
        'models_dir': models_dir,
        'feature_cols': feature_cols_filtered
    }

print()
print("="*80)
print("EVALUACIÓN COMPLETADA PARA TODOS LOS NIVELES")
print("="*80)
print()

# ============================================================================
# 4. COMPARACIÓN ENTRE NIVELES (si hay más de uno)
# ============================================================================
if len(all_results) > 1:
    print("="*80)
    print("COMPARACIÓN: GRANULAR vs AGREGADO")
    print("="*80)
    print()

    for suffix, data in all_results.items():
        level_name = data['level_name']
        df_reg = data['df_reg_test']
        df_clf = data['df_clf_test']

        print(f"📊 {level_name}")

        if len(df_reg) > 0:
            print(f"   REGRESIÓN:")
            print(f"     • Productos evaluados: {len(df_reg)}")
            print(f"     • RMSE medio: {df_reg['rmse'].mean():.2f}")
            print(f"     • MAE medio:  {df_reg['mae'].mean():.2f}")
            print(f"     • MAPE medio: {df_reg['mape'].mean():.2f}%")

        if len(df_clf) > 0:
            print(f"   CLASIFICACIÓN:")
            print(f"     • Productos evaluados: {len(df_clf)}")
            print(f"     • Precision media: {df_clf['precision'].mean():.3f}")
            print(f"     • Recall medio:    {df_clf['recall'].mean():.3f}")
            print(f"     • F1 medio:        {df_clf['f1'].mean():.3f}")
            print(f"     • AUC medio:       {df_clf['auc'].mean():.3f}")
        print()

    print("-" * 80)
    print("INTERPRETACIÓN:")
    print("  • RMSE más alto en agregado es esperado (suma de 4-5 tiendas)")
    print("  • MAPE y F1 son comparables entre niveles")
    print("  • Script 05 decidirá qué nivel usar por producto basándose en métricas")
    print()

# ============================================================================
# 5. VISUALIZACIONES
# ============================================================================
print("5. VISUALIZACIONES")
print("-" * 80)

# Usar el primer nivel disponible para visualizaciones
first_suffix = list(all_results.keys())[0]
first_data = all_results[first_suffix]
df_reg_test = first_data['df_reg_test']
df_predictions = first_data['df_predictions']
id_col = first_data['id_col']
models_dir = first_data['models_dir']
feature_cols_filtered = first_data['feature_cols']

print(f"Generando visualizaciones para: {first_data['level_name']}")
print()

# A. Actual vs Predicted (Regresión) - Mejor producto
if len(df_reg_test) > 0:
    best_product_reg = df_reg_test.loc[df_reg_test['rmse'].idxmin(), id_col]
    df_pred_reg = df_predictions[
        (df_predictions[id_col] == best_product_reg) &
        (df_predictions['task'] == 'regression')
    ].copy()

    if len(df_pred_reg) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Actual vs Predicted
        axes[0].plot(df_pred_reg['week_start'], df_pred_reg['actual'],
                    linewidth=2, marker='o', label='Actual', color=COLORS['primary'])
        axes[0].plot(df_pred_reg['week_start'], df_pred_reg['predicted'],
                    linewidth=2, marker='s', label='Predicho', color=COLORS['danger'],
                    alpha=0.7, linestyle='--')
        axes[0].set_title(f'Predicción de Ventas - {best_product_reg} (Test Set)',
                         fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Unidades')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residuos
        axes[1].bar(df_pred_reg['week_start'], df_pred_reg['error'],
                   color=COLORS['secondary'], alpha=0.7)
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1].set_title('Errores de Predicción (Residuos)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Fecha')
        axes[1].set_ylabel('Error (Actual - Predicho)')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(FIGURES / '04_regression_predictions.png', dpi=100, bbox_inches='tight')
        print(f"✓ Guardado: {FIGURES / '04_regression_predictions.png'}")
        plt.close()

# B. Confusion Matrix (Clasificación)
if len(df_clf_test) > 0:
    # Agregado de todas las predicciones
    df_pred_clf = df_predictions[df_predictions['task'] == 'classification'].copy()

    if len(df_pred_clf) > 0:
        cm = confusion_matrix(df_pred_clf['actual'], df_pred_clf['predicted'])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Urgente'],
                   yticklabels=['Normal', 'Urgente'])
        ax.set_title('Matriz de Confusión - Clasificación de Urgencias (Test Set)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicho')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        plt.savefig(FIGURES / '04_confusion_matrix.png', dpi=100, bbox_inches='tight')
        print(f"✓ Guardado: {FIGURES / '04_confusion_matrix.png'}")
        plt.close()

# C. Distribución de errores (Regresión)
if len(df_reg_test) > 0:
    df_pred_reg_all = df_predictions[df_predictions['task'] == 'regression'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Histograma de errores
    axes[0].hist(df_pred_reg_all['error'], bins=50, color=COLORS['info'],
                alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Cero')
    axes[0].set_title('Distribución de Errores - Regresión', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Error (Actual - Predicho)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # QQ Plot (normalidad de errores)
    from scipy import stats as sp_stats
    sp_stats.probplot(df_pred_reg_all['error'].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot - Normalidad de Errores', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / '04_error_distribution.png', dpi=100, bbox_inches='tight')
    print(f"✓ Guardado: {FIGURES / '04_error_distribution.png'}")
    plt.close()

print()

# ============================================================================
# 6. FEATURE IMPORTANCE (Ejemplo con mejor producto)
# ============================================================================
print("6. FEATURE IMPORTANCE")
print("-" * 80)

if len(df_reg_test) > 0:
    best_product_reg = df_reg_test.loc[df_reg_test['rmse'].idxmin(), id_col]
    best_model_name = df_reg_test.loc[df_reg_test['rmse'].idxmin(), 'model']
    model_type = 'rf' if best_model_name == 'RandomForest' else 'xgb'
    model_path = models_dir / f'{best_product_reg}_{model_type}_reg.pkl'

    if model_path.exists():
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Obtener feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_cols_filtered,
                'importance': importances
            }).sort_values('importance', ascending=False).head(20)

            print(f"TOP 20 Features más importantes ({best_product_reg}):")
            print(feature_importance_df)
            print()

            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 8))
            feature_importance_df.plot(x='feature', y='importance', kind='barh',
                                      ax=ax, color=COLORS['success'], legend=False)
            ax.set_title(f'Feature Importance - {best_product_reg} ({best_model_name})',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Importancia')
            ax.set_ylabel('Feature')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(FIGURES / '04_feature_importance.png', dpi=100, bbox_inches='tight')
            print(f"✓ Guardado: {FIGURES / '04_feature_importance.png'}")
            plt.close()

# ============================================================================
# 7. RESUMEN EJECUTIVO
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO - VALIDACIÓN")
print("="*80)
print()

# Resumen por nivel
for suffix, data in all_results.items():
    level_name = data['level_name']
    df_test_results = data['df_test_results']
    df_predictions = data['df_predictions']
    df_reg_test = data['df_reg_test']
    df_clf_test = data['df_clf_test']
    id_col = data['id_col']

    print(f"📊 {level_name}:")
    print(f"  • Productos evaluados: {df_test_results[id_col].nunique()}")
    print(f"  • Total predicciones: {len(df_predictions):,}")
    print()

    if len(df_reg_test) > 0:
        print(f"  📈 REGRESIÓN:")
        print(f"    • RMSE promedio: {df_reg_test['rmse'].mean():.2f}")
        print(f"    • MAE promedio:  {df_reg_test['mae'].mean():.2f}")
        print(f"    • MAPE promedio: {df_reg_test['mape'].mean():.2f}%")
        best_rmse_idx = df_reg_test['rmse'].idxmin()
        print(f"    • Mejor producto: {df_reg_test.loc[best_rmse_idx, id_col]}")
        print(f"      RMSE: {df_reg_test.loc[best_rmse_idx, 'rmse']:.2f}")
        print()

    if len(df_clf_test) > 0:
        print(f"  🎯 CLASIFICACIÓN:")
        print(f"    • Precision promedio: {df_clf_test['precision'].mean():.3f}")
        print(f"    • Recall promedio:    {df_clf_test['recall'].mean():.3f}")
        print(f"    • F1-Score promedio:  {df_clf_test['f1'].mean():.3f}")
        print(f"    • ROC-AUC promedio:   {df_clf_test['auc'].mean():.3f}")
        best_f1_idx = df_clf_test['f1'].idxmax()
        print(f"    • Mejor producto: {df_clf_test.loc[best_f1_idx, id_col]}")
        print(f"      F1-Score: {df_clf_test.loc[best_f1_idx, 'f1']:.3f}")
        print()

print(f"📁 OUTPUTS GENERADOS:")
for suffix in all_results.keys():
    print(f"  • validation_metrics_{suffix}.csv")
    print(f"  • test_predictions_{suffix}.csv")
print(f"  • 04_regression_predictions.png")
print(f"  • 04_confusion_matrix.png")
print(f"  • 04_error_distribution.png")
print(f"  • 04_feature_importance.png")
print()
print("="*80)
print("✓ VALIDACIÓN COMPLETADA")
print("="*80)
print()
print("CONCLUSIÓN:")
print(f"  ✓ Modelos validados en datos no vistos (test set)")
print(f"  ✓ Métricas finales calculadas")
print(f"  ✓ Errores y residuos analizados")
print(f"  ✓ Feature importance identificado")
print()
print("PRÓXIMO PASO:")
print(f"  → Script 05: Análisis por producto (05_analisis_por_producto.py)")
print(f"  → Script 06: Valor Operativo (06_valor_operativo.py)")
print()
