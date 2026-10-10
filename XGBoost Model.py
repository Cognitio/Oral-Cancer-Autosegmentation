#!/usr/bin/env python3
"""
radiomics_excel_xgboost_pipeline.py

Pipeline features:
  1. Direct ingestion of clinical/radiomics Excel spreadsheets (.xlsx / .xls)
  2. Configurable hyperparameter tuning: optimize for either ROC-AUC or PR-AUC (Average Precision)
  3. Formatted out-of-fold Confusion Matrix output with full diagnostic metrics
  4. Leak-Free Cross-Validation (nested scaling and mRMR / LASSO feature selection)
  5. Empirical 95% Confidence Intervals via non-parametric bootstrapping
  6. SHAP Interpretability plots
"""

import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
)
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# =========================================================================
# CONFIGURATION
# =========================================================================
EXCEL_PATH = Path(r"C:\Code\Radiomics\interface_radiomics_features.xlsx")

# Neck Dissection Path
#EXCEL_PATH = Path(r"C:\Code\Radiomics\Radiomics and Clinical Data.xlsx")

SHEET_NAME: Optional[str] = 0           # Name or index of Excel sheet
ID_COL_INDEX: int = 0                   # First column (Col 0) = Case identifiers
LABEL_COL: str = "Mandible Involvement Ground Truth"  # Name of your target binary column (1/0)
#LABEL_COL: str = "Neck Dissection"  # Name of your target binary column (1/0)


# Optimization Target: "pr_auc" (Average Precision) or "roc_auc"
OPTIMIZE_METRIC: str = "roc_auc"

N_SPLITS = 5
N_TOP_FEATURES = 10
SELECTION_METHOD = "lasso"               # Options: "mrmr" or "lasso"
N_BOOTSTRAPS = 2000                     # Iterations for 95% Confidence Intervals
DECISION_THRESHOLD = 0.5                # Probability threshold for binary classification
RANDOM_STATE = 42

# =========================================================================
# 1. EXCEL DATA INGESTION
# =========================================================================

def load_dataset_from_excel(
    file_path: Path,
    label_col: str,
    id_col_idx: int = 0,
    sheet_name: Any = 0
) -> Tuple[pd.DataFrame, str, str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Spreadsheet '{file_path}' not found.")

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    id_col_name = df.columns[id_col_idx]

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in Excel columns: {df.columns.tolist()}")

    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in [id_col_name, label_col]]

    print(f"Loaded dataset from '{file_path}':")
    print(f"  • Total Cases: {len(df)}")
    print(f"  • Candidate Features: {len(feature_cols)}")
    print(f"  • Class Distribution (1 vs 0): {np.sum(df[label_col] == 1)} vs {np.sum(df[label_col] == 0)}\n")

    return df[[id_col_name, label_col] + feature_cols], id_col_name, label_col


# =========================================================================
# 2. LEAK-FREE FEATURE SELECTION MODULES
# =========================================================================

def select_features_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    n_features: int = N_TOP_FEATURES
) -> List[str]:
    for c_val in np.logspace(-2, 1, 30):
        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=c_val,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
        clf.fit(X_train, y_train)
        non_zero_idx = np.where(clf.coef_[0] != 0)[0]
        if len(non_zero_idx) >= n_features:
            sorted_idx = non_zero_idx[np.argsort(np.abs(clf.coef_[0][non_zero_idx]))[::-1]]
            return [feature_names[i] for i in sorted_idx[:n_features]]

    clf = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    sorted_idx = np.argsort(np.abs(clf.coef_[0]))[::-1]
    return [feature_names[i] for i in sorted_idx[:n_features]]


def select_features_mrmr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    n_features: int = N_TOP_FEATURES
) -> List[str]:
    n_samples, n_feats = X_train.shape
    mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    corr_matrix = np.abs(np.corrcoef(X_train, rowvar=False))
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    selected_indices: List[int] = []
    candidates = list(range(n_feats))

    first_feat = int(np.argmax(mi_scores))
    selected_indices.append(first_feat)
    candidates.remove(first_feat)

    while len(selected_indices) < min(n_features, n_feats):
        mrmr_scores = []
        for c in candidates:
            rel = mi_scores[c]
            red = np.mean([corr_matrix[c, s] for s in selected_indices])
            mrmr_scores.append(rel / (red + 1e-5))

        best_candidate = candidates[int(np.argmax(mrmr_scores))]
        selected_indices.append(best_candidate)
        candidates.remove(best_candidate)

    return [feature_names[i] for i in selected_indices]


# =========================================================================
# 3. STATISTICAL BOOTSTRAPPING & CONFUSION MATRIX REPORTING
# =========================================================================

def print_full_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = DECISION_THRESHOLD):
    """Prints a detailed 2x2 confusion matrix with derived clinical metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0          # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0
    total = len(y_true)

    print("\n" + "="*70)
    print(f"OUT-OF-FOLD CONFUSION MATRIX (Decision Threshold = {threshold:.2f})")
    print("="*70)
    print(f"{'':22} | {'Predicted: Negative (0)':<22} | {'Predicted: Positive (1)':<22}")
    print("-" * 70)
    print(f"{'Actual: Negative (0)':<22} | {f'TN = {tn:<4} ({tn/total*100:4.1f}%)':<22} | {f'FP = {fp:<4} ({fp/total*100:4.1f}%)':<22}")
    print(f"{'Actual: Positive (1)':<22} | {f'FN = {fn:<4} ({fn/total*100:4.1f}%)':<22} | {f'TP = {tp:<4} ({tp/total*100:4.1f}%)':<22}")
    print("-" * 70)
    print(f"Operational Metrics:")
    print(f"  • Sensitivity / Recall:    {sensitivity:.4f}  [{tp}/{tp+fn}]")
    print(f"  • Specificity (TNR):       {specificity:.4f}  [{tn}/{tn+fp}]")
    print(f"  • Positive Pred Val (PPV): {ppv:.4f}  [{tp}/{tp+fp}]")
    print(f"  • Negative Pred Val (NPV): {npv:.4f}  [{tn}/{tn+fn}]")
    print(f"  • F1-Score:                {f1:.4f}")
    print("="*70 + "\n")


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = RANDOM_STATE
) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    metrics_records = {
        "ROC-AUC": [],
        "PR-AUC": [],
        "Accuracy": [],
        "Sensitivity": [],
        "Specificity": [],
        "PPV (Precision)": [],
        "NPV": [],
        "Brier Score": []
    }

    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue

        y_t = y_true[idx]
        y_p = y_probs[idx]
        y_pred = (y_p >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_t, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        metrics_records["ROC-AUC"].append(roc_auc_score(y_t, y_p))
        metrics_records["PR-AUC"].append(average_precision_score(y_t, y_p))
        metrics_records["Accuracy"].append(accuracy_score(y_t, y_pred))
        metrics_records["Sensitivity"].append(sens)
        metrics_records["Specificity"].append(spec)
        metrics_records["PPV (Precision)"].append(ppv)
        metrics_records["NPV"].append(npv)
        metrics_records["Brier Score"].append(brier_score_loss(y_t, y_p))

    ci_summary = {}
    for metric_name, values in metrics_records.items():
        arr = np.array(values)
        ci_summary[metric_name] = {
            "mean": float(np.mean(arr)),
            "lower_95": float(np.percentile(arr, 2.5)),
            "upper_95": float(np.percentile(arr, 97.5))
        }

    return ci_summary


# =========================================================================
# 4. NESTED CV PIPELINE (ROC-AUC / PR-AUC OPTIMIZED)
# =========================================================================

def run_nested_cv_pipeline(df: pd.DataFrame, id_col: str, label_col: str, optimize_metric: str = OPTIMIZE_METRIC):
    X_raw = df.drop(columns=[id_col, label_col])
    y = df[label_col].values
    feature_names_all = X_raw.columns.tolist()

    # Map optimization metric string to scikit-learn scoring string
    scoring_name = "average_precision" if optimize_metric.lower() == "pr_auc" else "roc_auc"
    display_name = "PR-AUC (Average Precision)" if scoring_name == "average_precision" else "ROC-AUC"

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(df))
    selected_features_across_folds = []

    print(f"Starting {N_SPLITS}-Fold Stratified Nested CV | Optimizing Hyperparameters for: [{display_name}]...")

    param_distributions = {
        "n_estimators": [75, 100, 150, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma": [0.0, 0.1, 0.2],
        "min_child_weight": [1, 2, 3]
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y), 1):
        X_train_df, X_val_df = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 1. Variance Filter (Fit on train only)
        var_thresh = VarianceThreshold(threshold=0.0)
        X_train_var = var_thresh.fit_transform(X_train_df)
        X_val_var = var_thresh.transform(X_val_df)
        retained_cols = [feature_names_all[i] for i in var_thresh.get_support(indices=True)]

        # 2. Standardize Features (Fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_var)
        X_val_scaled = scaler.transform(X_val_var)

        # 3. Feature Selection (Fit on train only)
        if SELECTION_METHOD == "lasso":
            chosen_feats = select_features_lasso(X_train_scaled, y_train, retained_cols, N_TOP_FEATURES)
        else:
            chosen_feats = select_features_mrmr(X_train_scaled, y_train, retained_cols, N_TOP_FEATURES)

        selected_features_across_folds.append(chosen_feats)
        feat_indices = [retained_cols.index(f) for f in chosen_feats]

        X_train_sel = X_train_scaled[:, feat_indices]
        X_val_sel = X_val_scaled[:, feat_indices]

        # 4. Class Imbalance Weight Calculation
        scale_pos_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))

        # 5. Inner Hyperparameter Tuning targeting specified metric
        base_xgb = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            use_label_encoder=False
        )

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + fold)
        tuner = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_distributions,
            n_iter=20,
            scoring=scoring_name,
            cv=inner_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        tuner.fit(X_train_sel, y_train)
        best_model = tuner.best_estimator_

        # 6. Out-of-fold Prediction
        val_probs = best_model.predict_proba(X_val_sel)[:, 1]
        oof_probs[val_idx] = val_probs

        fold_roc_auc = roc_auc_score(y_val, val_probs)
        fold_pr_auc = average_precision_score(y_val, val_probs)
        print(f"Fold {fold} — Val ROC-AUC: {fold_roc_auc:.3f} | Val PR-AUC: {fold_pr_auc:.3f} (Max Depth: {best_model.max_depth}, LR: {best_model.learning_rate})")

# Calculate Feature Stability
    all_selected = [f for sublist in selected_features_across_folds for f in sublist]
    stability_counts = pd.Series(all_selected).value_counts()
    top_stable_features = stability_counts.index.tolist()[:N_TOP_FEATURES]

    return oof_probs, y, top_stable_features, stability_counts


# =========================================================================
# 5. CONSOLE SHAP & FEATURE IMPORTANCE REPORTING
# =========================================================================

def explain_and_print_shap_to_console(
    df: pd.DataFrame,
    id_col: str,
    label_col: str,
    top_features: List[str],
    stability_counts: pd.Series
):
    """
    Fits final interpretable model, computes TreeSHAP values, and outputs
    both global importance and patient-level decision tables directly to the console.
    """
    X = df[top_features]
    y = df[label_col].values

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=top_features)
    scale_pos_weight = float(np.sum(y == 0) / max(1, np.sum(y == 1)))

    final_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE
    )
    final_model.fit(X_scaled, y)

    # Compute SHAP Values
    explainer = shap.TreeExplainer(final_model)
    shap_explanation = explainer(X_scaled)
    shap_matrix = shap_explanation.values  # Shape: (N, num_features)

    # 1. Global Mean |SHAP| Impact & Directionality Calculation
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    total_shap_impact = np.sum(mean_abs_shap)
    relative_contributions = (mean_abs_shap / (total_shap_impact + 1e-8)) * 100

    # Determine correlation between raw feature value and SHAP value (Direction of Effect)
    directions = []
    for i, col in enumerate(top_features):
        corr, _ = spearmanr(X[col], shap_matrix[:, i])
        if np.isnan(corr) or abs(corr) < 0.05:
            directions.append("Non-linear / Complex")
        elif corr > 0:
            directions.append("Higher Value -> Increases Risk (+)")
        else:
            directions.append("Higher Value -> Decreases Risk (-)")

    # Build Global Summary Table
    shap_df = pd.DataFrame({
        "Feature": top_features,
        "CV Stability": [f"{stability_counts.get(f, 0)}/{N_SPLITS} folds" for f in top_features],
        "Mean |SHAP|": mean_abs_shap,
        "Relative Impact (%)": relative_contributions,
        "Clinical Direction": directions
    }).sort_values(by="Mean |SHAP|", ascending=False).reset_index(drop=True)

    print("\n" + "="*95)
    print("GLOBAL MODEL INTERPRETABILITY (SHAP FEATURE ATTRIBUTION TABLE)")
    print("="*95)
    print(f"{'Rank':<5} | {'Feature Name':<35} | {'CV Stability':<13} | {'Mean |SHAP|':<11} | {'Impact %':<9} | {'Directionality'}")
    print("-" * 95)
    for idx, row in shap_df.iterrows():
        print(
            f"{idx+1:<5} | "
            f"{row['Feature'][:35]:<35} | "
            f"{row['CV Stability']:<13} | "
            f"{row['Mean |SHAP|']:<11.4f} | "
            f"{row['Relative Impact (%)']:>6.2f}%   | "
            f"{row['Clinical Direction']}"
        )
    print("="*95)

    # 2. Local Patient-Level Case Breakdown (Console Demonstration)
    print("\n" + "="*95)
    print("PATIENT-LEVEL CASE DEMONSTRATION (LOCAL SHAP CONTRIBUTIONS)")
    print("="*95)
    
    # Pick first positive invasion case and first negative case
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    sample_indices = [pos_indices[0], neg_indices[0]] if len(pos_indices) and len(neg_indices) else [0]

    for s_idx in sample_indices:
        case_name = str(df[id_col].iloc[s_idx])
        true_status = "Invasion (1)" if y[s_idx] == 1 else "No Invasion (0)"
        base_val = explainer.expected_value
        pred_prob = final_model.predict_proba(X_scaled.iloc[[s_idx]])[0, 1]

        print(f"\nCase ID: {case_name} | Ground Truth: {true_status} | Model Predicted Probability: {pred_prob:.4f}")
        print(f"Base Value (log-odds): {base_val:.4f}")
        print(f"{'Feature Name':<35} | {'Patient Raw Value':<18} | {'SHAP Contribution (log-odds)':<25}")
        print("-" * 85)

        # Sort features by absolute local impact for this patient
        patient_shaps = shap_matrix[s_idx]
        sorted_local_idx = np.argsort(np.abs(patient_shaps))[::-1]

        for rank_i in sorted_local_idx[:5]:  # Top 5 most influential for this case
            feat_name = top_features[rank_i]
            raw_val = X.iloc[s_idx, rank_i]
            shap_val = patient_shaps[rank_i]
            sign_str = f"+{shap_val:.4f} (Raises Risk)" if shap_val > 0 else f"{shap_val:.4f} (Lowers Risk)"
            print(f"{feat_name[:35]:<35} | {raw_val:<18.3f} | {sign_str}")
    print("="*95 + "\n")

    # 3. Save Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_explanation, X_scaled, show=False)
    plt.title("SHAP Summary: Neck Dissection", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(r"C:\Code\Radiomics\shap_summary_plot_neck.png", dpi=300)
    plt.close()
    print("Saved global SHAP visualization to 'shap_summary_plot.png'.")


# =========================================================================
# MAIN EXECUTION
# =========================================================================

if __name__ == "__main__":
    if not EXCEL_PATH.exists():
        print(f"'{EXCEL_PATH}' not found. Generating sample Excel spreadsheet...")
        np.random.seed(42)
        mock_excel = {
            "case_id": [f"case_{i:03d}" for i in range(1, 132)],
            "contact_surface_area_mm2": np.random.exponential(110, 131),
            "overlap_volume_mm3": np.random.exponential(45, 131),
            "min_distance_to_bone_mm": np.random.exponential(1.2, 131),
            "tumor_volume_cm3": np.random.normal(16, 4, 131),
            "interface_firstorder_Mean": np.random.normal(120, 30, 131),
            "interface_glcm_Contrast": np.random.normal(35, 10, 131),
            "interface_glszm_ZoneEntropy": np.random.normal(6.2, 0.7, 131),
            "interface_glrlm_RunLengthNonUniformity": np.random.normal(250, 40, 131),
            "interface_ngtdm_Coarseness": np.random.exponential(0.05, 131),
            LABEL_COL: np.random.choice([0, 1], size=131, p=[0.70, 0.30])
        }
        pd.DataFrame(mock_excel).to_excel(EXCEL_PATH, index=False)

    df_loaded, id_col, label_col = load_dataset_from_excel(
        file_path=EXCEL_PATH,
        label_col=LABEL_COL,
        id_col_idx=ID_COL_INDEX,
        sheet_name=SHEET_NAME
    )

    oof_probs, y_true, top_features, stability_counts = run_nested_cv_pipeline(
        df_loaded, id_col, label_col, optimize_metric=OPTIMIZE_METRIC
    )

    oof_predictions = (oof_probs >= DECISION_THRESHOLD).astype(int)
    print_full_confusion_matrix(y_true, oof_predictions, threshold=DECISION_THRESHOLD)

    ci_results = compute_bootstrap_confidence_intervals(y_true, oof_probs, threshold=DECISION_THRESHOLD)
    print("="*65)
    print("OUT-OF-FOLD PERFORMANCE METRICS (95% EMPIRICAL BOOTSTRAP CI)")
    print("="*65)
    print(f"{'Metric':<18} | {'Mean Value':<12} | {'95% Confidence Interval':<25}")
    print("-" * 65)
    for metric, vals in ci_results.items():
        print(f"{metric:<18} | {vals['mean']:<12.4f} | [{vals['lower_95']:.4f} – {vals['upper_95']:.4f}]")
    print("="*65)

    explain_and_print_shap_to_console(
        df=df_loaded,
        id_col=id_col,
        label_col=label_col,
        top_features=top_features,
        stability_counts=stability_counts
    )
