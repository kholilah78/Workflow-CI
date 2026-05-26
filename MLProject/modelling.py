"""
modelling.py (MLProject version)
==================================
Script training yang digunakan dalam MLflow Project (Kriteria 3).

Author  : Kholilah Nurafifah
Dataset : Heart Disease UCI
"""

import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
TRAIN_PATH      = os.path.join('heart_preprocessing', 'heart_train.csv')
TEST_PATH       = os.path.join('heart_preprocessing', 'heart_test.csv')
TARGET_COL      = 'target'
EXPERIMENT_NAME = 'Heart_Disease_CI_Pipeline'
ARTIFACTS_DIR   = 'artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# ARGPARSE — positional args (tanpa --)
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Heart Disease Model Training')
    parser.add_argument('n_estimators',       type=int,   nargs='?', default=100)
    parser.add_argument('max_depth',          type=float, nargs='?', default=0,
                        help='0 berarti None')
    parser.add_argument('min_samples_split',  type=int,   nargs='?', default=2)
    parser.add_argument('min_samples_leaf',   type=int,   nargs='?', default=1)
    parser.add_argument('random_state',       type=int,   nargs='?', default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def load_data(train_path, test_path, target_col):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    X_train  = train_df.drop(target_col, axis=1)
    y_train  = train_df[target_col]
    X_test   = test_df.drop(target_col, axis=1)
    y_test   = test_df[target_col]
    print(f"Data: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def plot_roc_curve(y_true, y_proba, roc_auc, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def plot_feature_importance(model, feature_names, path):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices],
            color='#2196F3', edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    max_depth = None if args.max_depth == 0 else int(args.max_depth)

    print("=" * 55)
    print("  CI PIPELINE - HEART DISEASE CLASSIFICATION")
    print("  Author: Kholilah Nurafifah")
    print("=" * 55)

    X_train, X_test, y_train, y_test = load_data(
        TRAIN_PATH, TEST_PATH, TARGET_COL
    )

    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'mlruns'))
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Pastikan tidak ada run yang masih aktif
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name='RandomForest_CI'):

        # Log params
        mlflow.log_param('n_estimators',      args.n_estimators)
        mlflow.log_param('max_depth',         str(max_depth))
        mlflow.log_param('min_samples_split', args.min_samples_split)
        mlflow.log_param('min_samples_leaf',  args.min_samples_leaf)
        mlflow.log_param('random_state',      args.random_state)

        # Training
        model = RandomForestClassifier(
            n_estimators      = args.n_estimators,
            max_depth         = max_depth,
            min_samples_split = args.min_samples_split,
            min_samples_leaf  = args.min_samples_leaf,
            random_state      = args.random_state,
            n_jobs            = -1
        )
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

        # Log metrics
        mlflow.log_metric('accuracy',   acc)
        mlflow.log_metric('precision',  precision)
        mlflow.log_metric('recall',     recall)
        mlflow.log_metric('f1_score',   f1)
        mlflow.log_metric('roc_auc',    roc_auc)
        mlflow.log_metric('cv_f1_mean', cv_scores.mean())
        mlflow.log_metric('cv_f1_std',  cv_scores.std())

        print(f"\nAccuracy : {acc:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")

        # Log model
        mlflow.sklearn.log_model(
            model, 'model',
            registered_model_name='HeartDiseaseClassifier'
        )

        # Artefak tambahan
        cm_path     = os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png')
        roc_path    = os.path.join(ARTIFACTS_DIR, 'roc_curve.png')
        fi_path     = os.path.join(ARTIFACTS_DIR, 'feature_importance.png')
        report_path = os.path.join(ARTIFACTS_DIR, 'classification_report.txt')

        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_roc_curve(y_test, y_proba, roc_auc, roc_path)
        plot_feature_importance(model, list(X_train.columns), fi_path)

        report = classification_report(
            y_test, y_pred, target_names=['No Disease', 'Disease']
        )
        with open(report_path, 'w') as f:
            f.write(report)

        mlflow.log_artifact(cm_path,     'plots')
        mlflow.log_artifact(roc_path,    'plots')
        mlflow.log_artifact(fi_path,     'plots')
        mlflow.log_artifact(report_path, 'reports')

        mlflow.set_tag('author',  'Kholilah Nurafifah')
        mlflow.set_tag('dataset', 'Heart Disease UCI')

        run_id = mlflow.active_run().info.run_id
        print(f"\n[MLflow] Run ID: {run_id}")
        print("[CI] Training selesai!")


if __name__ == '__main__':
    main()