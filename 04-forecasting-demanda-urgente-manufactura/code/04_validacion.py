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
- data/simulated/features_weekly.csv
- data/simulated/best_models.csv
- models/*.pkl

OUTPUT:
- data/simulated/test_predictions.csv - Predicciones en test set
- data/simulated/validation_metrics.csv - Métricas finales
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

MODELS_DIR = PROJECT_ROOT / 'models'

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================
print("1. CARGANDO DATOS Y MODELOS")
print("-" * 80)

df = pd.read_csv(DATA_SIMULATED / 'features_weekly.csv')
df['week_start'] = pd.to_datetime(df['week_start'])

df_best = pd.read_csv(DATA_SIMULATED / 'best_models.csv')

print(f"✓ Dataset cargado: {df.shape}")
print(f"  Productos: {df['product_id'].nunique()}")
print()
print(f"✓ Mejores modelos: {len(df_best)}")
print(f"  Regresión: {(df_best['task'] == 'regression').sum()}")
print(f"  Clasificación: {(df_best['task'] == 'classification').sum()}")
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

feature_cols = [f for f in feature_cols if f in df.columns]

# ============================================================================
# 2. TRAIN/VAL/TEST SPLIT
# ============================================================================

def temporal_split(product_df, train_pct=0.70, val_pct=0.15):
    """Split temporal sin data leakage"""
    df_sorted = product_df.sort_values('week_start').reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]

    return train, val, test


# ============================================================================
# 3. EVALUACIÓN EN TEST SET
# ============================================================================
print("2. EVALUANDO MODELOS EN TEST SET")
print("-" * 80)

test_results = []
predictions = []

products = df['product_id'].unique()

for product_id in tqdm(products, desc="Evaluando productos"):
    df_product = df[df['product_id'] == product_id]

    # Split
    train, val, test = temporal_split(df_product)

    if len(test) < 5:
        continue

    # Preparar test set
    X_test = test[feature_cols]
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
        (df_best['product_id'] == product_id) &
        (df_best['task'] == 'regression')
    ]

    if len(best_reg_model) > 0:
        model_name = best_reg_model.iloc[0]['model']
        model_type = 'rf' if model_name == 'RandomForest' else 'xgb'
        model_path = MODELS_DIR / f'{product_id}_{model_type}_reg.pkl'

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
                'product_id': product_id,
                'model': model_name,
                'task': 'regression',
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            })

            # Guardar predicciones
            for idx, (week, actual, pred) in enumerate(zip(test_weeks, y_test_reg_clean, y_pred_reg)):
                predictions.append({
                    'product_id': product_id,
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
        (df_best['product_id'] == product_id) &
        (df_best['task'] == 'classification')
    ]

    if len(best_clf_model) > 0:
        model_name = best_clf_model.iloc[0]['model']
        model_type = 'rf' if model_name == 'RandomForest' else 'xgb'
        model_path = MODELS_DIR / f'{product_id}_{model_type}_clf.pkl'

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
                'product_id': product_id,
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
                    'product_id': product_id,
                    'week_start': week,
                    'task': 'classification',
                    'actual': int(actual),
                    'predicted': int(pred),
                    'predicted_proba': proba,
                    'correct': int(actual == pred)
                })

# Crear DataFrames
df_test_results = pd.DataFrame(test_results)
df_predictions = pd.DataFrame(predictions)

print()
print(f"✓ Evaluación completada")
print(f"  Productos evaluados: {df_test_results['product_id'].nunique()}")
print(f"  Predicciones generadas: {len(df_predictions):,}")
print()

# ============================================================================
# 4. MÉTRICAS FINALES
# ============================================================================
print("3. MÉTRICAS FINALES EN TEST SET")
print("-" * 80)

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

# ============================================================================
# 5. GUARDAR RESULTADOS
# ============================================================================
print("4. GUARDANDO RESULTADOS")
print("-" * 80)

# Métricas
metrics_file = DATA_SIMULATED / 'validation_metrics.csv'
df_test_results.to_csv(metrics_file, index=False)
print(f"✓ Métricas guardadas: {metrics_file}")

# Predicciones
pred_file = DATA_SIMULATED / 'test_predictions.csv'
df_predictions.to_csv(pred_file, index=False)
print(f"✓ Predicciones guardadas: {pred_file}")
print()

# ============================================================================
# 6. VISUALIZACIONES
# ============================================================================
print("5. VISUALIZACIONES")
print("-" * 80)

# A. Actual vs Predicted (Regresión) - Mejor producto
if len(df_reg_test) > 0:
    best_product_reg = df_reg_test.loc[df_reg_test['rmse'].idxmin(), 'product_id']
    df_pred_reg = df_predictions[
        (df_predictions['product_id'] == best_product_reg) &
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
# 7. FEATURE IMPORTANCE (Ejemplo con mejor producto)
# ============================================================================
print("6. FEATURE IMPORTANCE")
print("-" * 80)

if len(df_reg_test) > 0:
    best_product_reg = df_reg_test.loc[df_reg_test['rmse'].idxmin(), 'product_id']
    best_model_name = df_reg_test.loc[df_reg_test['rmse'].idxmin(), 'model']
    model_type = 'rf' if best_model_name == 'RandomForest' else 'xgb'
    model_path = MODELS_DIR / f'{best_product_reg}_{model_type}_reg.pkl'

    if model_path.exists():
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Obtener feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_cols,
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
# 8. RESUMEN EJECUTIVO
# ============================================================================
print()
print("="*80)
print("RESUMEN EJECUTIVO - VALIDACIÓN")
print("="*80)
print()
print(f"📊 EVALUACIÓN EN TEST SET:")
print(f"  • Productos evaluados: {df_test_results['product_id'].nunique()}")
print(f"  • Total predicciones: {len(df_predictions):,}")
print()

if len(df_reg_test) > 0:
    print(f"📈 REGRESIÓN (Predicción de ventas):")
    print(f"  • RMSE promedio: {df_reg_test['rmse'].mean():.2f}")
    print(f"  • MAE promedio:  {df_reg_test['mae'].mean():.2f}")
    print(f"  • MAPE promedio: {df_reg_test['mape'].mean():.2f}%")
    best_rmse_idx = df_reg_test['rmse'].idxmin()
    print(f"  • Mejor producto: {df_reg_test.loc[best_rmse_idx, 'product_id']}")
    print(f"    RMSE: {df_reg_test.loc[best_rmse_idx, 'rmse']:.2f}")
    print()

if len(df_clf_test) > 0:
    print(f"🎯 CLASIFICACIÓN (Predicción de urgencias):")
    print(f"  • Precision promedio: {df_clf_test['precision'].mean():.3f}")
    print(f"  • Recall promedio:    {df_clf_test['recall'].mean():.3f}")
    print(f"  • F1-Score promedio:  {df_clf_test['f1'].mean():.3f}")
    print(f"  • ROC-AUC promedio:   {df_clf_test['auc'].mean():.3f}")
    best_f1_idx = df_clf_test['f1'].idxmax()
    print(f"  • Mejor producto: {df_clf_test.loc[best_f1_idx, 'product_id']}")
    print(f"    F1-Score: {df_clf_test.loc[best_f1_idx, 'f1']:.3f}")
    print()

print(f"📁 OUTPUTS GENERADOS:")
print(f"  • {metrics_file.name}")
print(f"  • {pred_file.name}")
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
print(f"  → Script 06: Análisis por producto (filtrado de productos confiables)")
print(f"  → Script 05: Valor Operativo (ROI, costos evitados)")
print()
