"""
MAST IQ — Model Training Pipeline v3
=====================================
Per-target XGBoost regression with Optuna hyperparameter optimization,
stratified k-fold cross-validation, comprehensive evaluation, and SHAP analysis.

Features:
  - 10 elemental composition features (C, Si, Mn, P, S, Ni, Cr, Cu, Mo, Fe)
  - 4 process parameters (HT_Temp, Soak_Time, Tempering_Temp, Tempering_Time)
  - 7 engineered features (Carbon_Equiv, A3_Temp, Delta_HT_A3, Hollomon_Jaffe,
    C_x_Cr, Cooling_Rate_Est, Total_Alloy)
  - One-hot encoded Process (4) and Cooling_Medium (6)

Targets: Tensile_MPa, Yield_MPa, Hardness_HB, Elongation_pct, Fatigue_MPa

Usage:
    python train_model.py
"""

import os
import json
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    learning_curve,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MAST-IQ-Train")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
DATA_FILE = "steel_heat_treatment.csv"
OUTPUT_DIR = "models"
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_CV_FOLDS = 5
OPTUNA_TRIALS = 80        # per target
EARLY_STOPPING = 50       # XGBoost early stopping rounds

TARGET_COLS = [
    "Tensile_MPa",
    "Yield_MPa",
    "Hardness_HB",
    "Elongation_pct",
    "Fatigue_MPa",
]

# Estimated cooling rates (°C/s) for each medium — used as engineered feature
COOLING_RATES = {
    "Water": 80.0,
    "Oil": 25.0,
    "Polymer": 35.0,
    "Salt Bath": 15.0,
    "Air": 2.0,
    "Furnace": 0.1,
}


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
def carbon_equiv(row):
    """IIW carbon equivalent — hardenability proxy."""
    return (row.C + row.Mn / 6 + row.Si / 24
            + row.Ni / 40 + row.Cr / 5 + row.Mo / 4 + row.Cu / 15)


def a3_temp(row):
    """Andrews (1965) empirical A3 temperature."""
    return (912.0 - 203.0 * np.sqrt(max(row.C, 1e-6))
            - 30.0 * row.Mn + 44.7 * row.Si
            - 15.2 * row.Ni + 31.5 * row.Mo)


def hollomon_jaffe(row):
    """Tempering parameter H = T(K) * (18 + log10(t_hours))."""
    if row.Tempering_Temp_C <= 0 or row.Tempering_Time_min <= 0:
        return 0.0
    t_h = max(row.Tempering_Time_min / 60.0, 0.001)
    return (row.Tempering_Temp_C + 273.15) * (18.0 + np.log10(t_h))


def engineer_features(df):
    """Add all derived features to the dataframe."""
    # Fe content (balance element) — add if missing
    alloy_cols = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo"]
    if "Fe" not in df.columns:
        df["Fe"] = 100.0 - df[alloy_cols].sum(axis=1)

    # IIW Carbon Equivalent
    df["Carbon_Equiv"] = df.apply(carbon_equiv, axis=1)

    # A3 transformation temperature
    df["A3_Temp_C"] = df.apply(a3_temp, axis=1)

    # Delta between HT temp and A3 (positive = fully austenitized)
    df["Delta_HT_A3"] = df["HT_Temp_C"] - df["A3_Temp_C"]

    # Hollomon-Jaffe tempering parameter
    df["Hollomon_Jaffe"] = df.apply(hollomon_jaffe, axis=1)

    # C × Cr interaction — carbide-forming tendency
    df["C_x_Cr"] = df["C"] * df["Cr"]

    # Estimated cooling rate based on medium
    df["Cooling_Rate_Est"] = df["Cooling_Medium"].map(COOLING_RATES).fillna(2.0)

    # Total alloy content (non-Fe, non-trace)
    df["Total_Alloy"] = df[["C", "Si", "Mn", "Ni", "Cr", "Mo", "Cu"]].sum(axis=1)

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD & PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_data():
    log.info(f"Loading {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    log.info(f"  Raw shape: {df.shape}")
    log.info(f"  Process distribution:\n{df['Process'].value_counts().to_string()}")
    log.info(f"  Cooling medium distribution:\n{df['Cooling_Medium'].value_counts().to_string()}")

    # Engineer features
    df = engineer_features(df)

    # One-hot encode categorical columns
    df_enc = pd.get_dummies(df, columns=["Process", "Cooling_Medium"], drop_first=False)

    # Define feature columns in canonical order
    feature_cols = [
        # Chemistry (10 elements including Fe)
        "C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo", "Fe",
        # Engineered (7)
        "Carbon_Equiv", "A3_Temp_C", "Delta_HT_A3", "Hollomon_Jaffe",
        "C_x_Cr", "Cooling_Rate_Est", "Total_Alloy",
        # Process parameters (4)
        "HT_Temp_C", "Soaking_Time_min", "Tempering_Temp_C", "Tempering_Time_min",
        # Process dummies (4)
        "Process_Quench_Temper", "Process_Normalizing",
        "Process_Annealing", "Process_Stress_Relief",
        # Cooling medium dummies (6)
        "Cooling_Medium_Water", "Cooling_Medium_Oil", "Cooling_Medium_Polymer",
        "Cooling_Medium_Air", "Cooling_Medium_Furnace", "Cooling_Medium_Salt Bath",
    ]
    # Keep only columns present in data
    feature_cols = [c for c in feature_cols if c in df_enc.columns]
    log.info(f"  Total features: {len(feature_cols)}")

    X = df_enc[feature_cols].astype(float)
    y = df_enc[TARGET_COLS].astype(float)

    return X, y, feature_cols, df


# ═══════════════════════════════════════════════════════════════════════════
#  OPTUNA OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════
def make_objective(X_train, y_train_col, feature_cols):
    """Create an Optuna objective for a single target."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample": trial.suggest_float("subsample", 0.50, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.40, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            model, X_train, y_train_col,
            cv=kf, scoring="r2", n_jobs=-1,
        )
        return scores.mean()
    return objective


# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN + EVALUATE
# ═══════════════════════════════════════════════════════════════════════════
def train_and_evaluate(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    log.info(f"Train: {X_train.shape}  Test: {X_test.shape}")

    models = {}
    best_params = {}
    eval_results = []
    cv_results = []

    for tgt in TARGET_COLS:
        log.info(f"\n{'='*60}")
        log.info(f"Training model for: {tgt}")
        log.info(f"{'='*60}")

        y_tr = y_train[tgt]
        y_te = y_test[tgt]

        # ── Optuna hyperparameter search ──
        log.info(f"  Running Optuna ({OPTUNA_TRIALS} trials, {N_CV_FOLDS}-fold CV)...")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(
            make_objective(X_train, y_tr, feature_cols),
            n_trials=OPTUNA_TRIALS,
            show_progress_bar=True,
        )
        bp = study.best_params
        best_params[tgt] = bp
        log.info(f"  Best CV R²: {study.best_value:.4f}")
        log.info(f"  Best params: {json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in bp.items()}, indent=4)}")

        # ── Train final model with early stopping ──
        final_model = XGBRegressor(
            **bp,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=EARLY_STOPPING,
        )
        final_model.fit(
            X_train, y_tr,
            eval_set=[(X_test, y_te)],
            verbose=False,
        )
        models[tgt] = final_model

        # ── Test set evaluation ──
        y_pred = final_model.predict(X_test)
        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mape = mean_absolute_percentage_error(y_te, y_pred) * 100
        n = len(y_te)
        p = X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        eval_results.append({
            "Target": tgt,
            "R²": round(r2, 4),
            "Adj_R²": round(adj_r2, 4),
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "Best_Iter": final_model.best_iteration if hasattr(final_model, 'best_iteration') else bp.get("n_estimators"),
        })
        log.info(f"  Test R² = {r2:.4f}  Adj_R² = {adj_r2:.4f}  MAE = {mae:.2f}  RMSE = {rmse:.2f}  MAPE = {mape:.2f}%")

        # ── Cross-validation on full training set ──
        cv_model = XGBRegressor(
            **bp,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(cv_model, X_train, y_tr, cv=kf, scoring="r2", n_jobs=-1)
        cv_results.append({
            "Target": tgt,
            "CV_Mean_R²": round(cv_scores.mean(), 4),
            "CV_Std_R²": round(cv_scores.std(), 4),
            "CV_Min_R²": round(cv_scores.min(), 4),
            "CV_Max_R²": round(cv_scores.max(), 4),
        })
        log.info(f"  {N_CV_FOLDS}-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return models, best_params, eval_results, cv_results, X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════
#  SAVE MODELS & METADATA
# ═══════════════════════════════════════════════════════════════════════════
def save_outputs(models, best_params, eval_results, cv_results,
                 feature_cols, X_train, X_test):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save per-target XGBoost models
    for tgt, model in models.items():
        fname = os.path.join(OUTPUT_DIR, f"xgb_{tgt.lower()}.json")
        model.save_model(fname)
        log.info(f"  Saved {fname}")

    # Also save to root directory for legacy compatibility
    for tgt, model in models.items():
        fname = f"xgb_{tgt.lower()}.json"
        model.save_model(fname)

    # Compute training ranges for all features
    training_ranges = {}
    for col in feature_cols:
        training_ranges[col] = {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
        }

    # Metadata JSON
    metadata = {
        "model_version": "v3",
        "trained_at": datetime.now().isoformat(),
        "features": feature_cols,
        "targets": TARGET_COLS,
        "training_ranges": training_ranges,
        "model_metrics": eval_results,
        "cv_metrics": cv_results,
        "best_params": best_params,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(feature_cols),
        "optuna_trials": OPTUNA_TRIALS,
        "cv_folds": N_CV_FOLDS,
        "early_stopping_rounds": EARLY_STOPPING,
    }

    meta_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  Saved {meta_path}")

    # Also save to root for legacy compatibility
    with open("model_metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Feature order file
    feat_path = os.path.join(OUTPUT_DIR, "feature_order.json")
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)
    log.info(f"  Saved {feat_path}")

    return metadata


# ═══════════════════════════════════════════════════════════════════════════
#  SHAP ANALYSIS (optional, runs if shap is installed)
# ═══════════════════════════════════════════════════════════════════════════
def run_shap_analysis(models, X_test, feature_cols):
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping SHAP analysis")
        return None

    log.info("\nRunning SHAP analysis...")
    shap_importance = {}
    for tgt, model in models.items():
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_test)
        mean_abs = np.abs(sv).mean(axis=0)
        shap_importance[tgt] = {
            feat: round(float(val), 4)
            for feat, val in sorted(
                zip(feature_cols, mean_abs),
                key=lambda x: x[1], reverse=True,
            )[:15]
        }
        log.info(f"  {tgt} top-5 SHAP: {list(shap_importance[tgt].items())[:5]}")

    # Save SHAP results
    shap_path = os.path.join(OUTPUT_DIR, "shap_importance.json")
    with open(shap_path, "w") as f:
        json.dump(shap_importance, f, indent=2)
    log.info(f"  Saved {shap_path}")
    return shap_importance


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("MAST IQ — Model Training Pipeline v3")
    log.info("=" * 70)

    # Load and prepare data
    X, y, feature_cols, df_raw = load_data()

    # Print dataset summary
    log.info(f"\nDataset summary:")
    log.info(f"  Samples: {len(X)}")
    log.info(f"  Features: {len(feature_cols)}")
    log.info(f"  Targets: {len(TARGET_COLS)}")
    log.info(f"  Fe range: {X['Fe'].min():.2f} — {X['Fe'].max():.2f} wt%")

    # Target statistics
    log.info(f"\nTarget statistics:")
    for tgt in TARGET_COLS:
        log.info(f"  {tgt:20s}  mean={y[tgt].mean():8.1f}  std={y[tgt].std():8.1f}  "
                 f"range=[{y[tgt].min():.1f}, {y[tgt].max():.1f}]")

    # Train and evaluate
    models, best_params, eval_results, cv_results, X_train, X_test, y_train, y_test = \
        train_and_evaluate(X, y, feature_cols)

    # Print evaluation summary
    log.info(f"\n{'='*70}")
    log.info("FINAL EVALUATION SUMMARY")
    log.info(f"{'='*70}")
    eval_df = pd.DataFrame(eval_results)
    log.info(f"\nTest Set Metrics:\n{eval_df.to_string(index=False)}")
    cv_df = pd.DataFrame(cv_results)
    log.info(f"\nCross-Validation Metrics:\n{cv_df.to_string(index=False)}")

    # Save outputs
    log.info(f"\nSaving models and metadata...")
    metadata = save_outputs(models, best_params, eval_results, cv_results,
                            feature_cols, X_train, X_test)

    # SHAP analysis
    run_shap_analysis(models, X_test, feature_cols)

    # Final summary
    mean_r2 = np.mean([r["R²"] for r in eval_results])
    mean_adj_r2 = np.mean([r["Adj_R²"] for r in eval_results])
    mean_mape = np.mean([r["MAPE (%)"] for r in eval_results])
    log.info(f"\n{'='*70}")
    log.info(f"TRAINING COMPLETE")
    log.info(f"  Mean R²     : {mean_r2:.4f}")
    log.info(f"  Mean Adj R² : {mean_adj_r2:.4f}")
    log.info(f"  Mean MAPE   : {mean_mape:.2f}%")
    log.info(f"  Models saved to: {OUTPUT_DIR}/")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
