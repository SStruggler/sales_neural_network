"""
train_pipeline.py
=================
Retail Sales Prediction - Neural Network Training Pipeline

Builds, trains, evaluates, and saves an ANN for retail sales regression.
Includes early stopping, loss curve visualization, and evaluation metrics.

Author: Sales Neural Network Project
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "sales_model.keras"
PROCESSED_DIR = Path("data/processed")

EPOCHS        = 200
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
PATIENCE      = 10
VALIDATION_SPLIT = 0.15
RANDOM_STATE  = 42

# ──────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────

def build_model(input_dim: int) -> keras.Model:
    """
    Construct the ANN architecture for regression:

        Input → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
              → Dense(64,  ReLU) → Dropout(0.2)
              → Output(1, Linear)

    Args:
        input_dim: Number of input features after scaling.

    Returns:
        Compiled Keras Model ready for training.
    """
    tf.random.set_seed(RANDOM_STATE)

    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,), name="input_layer"),

            # Block 1
            layers.Dense(128, activation="relu", name="dense_1"),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.Dropout(0.3, name="dropout_1"),

            # Block 2
            layers.Dense(64, activation="relu", name="dense_2"),
            layers.Dropout(0.2, name="dropout_2"),

            # Output — linear for regression (unbounded prediction)
            layers.Dense(1, activation="linear", name="output"),
        ],
        name="SalesANN",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    return model


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def get_callbacks() -> list:
    """
    Return training callbacks:
    - EarlyStopping: stops training if val_loss doesn't improve for `patience` epochs.
    - ModelCheckpoint: saves the single best model weights during training.

    Returns:
        List of Keras callback instances.
    """
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=0,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    return [early_stop, checkpoint, reduce_lr]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: Optional[int] = None,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Build and train the ANN model.

    Args:
        X_train: Scaled training feature matrix.
        y_train: Training target values.
        input_dim: Number of input features (auto-inferred if None).

    Returns:
        (trained_model, training_history)
    """
    if input_dim is None:
        input_dim = X_train.shape[1]

    print("=" * 60)
    print("  RETAIL SALES — NEURAL NETWORK TRAINING")
    print("=" * 60)

    model = build_model(input_dim)
    model.summary()

    print(f"\n  Training on {X_train.shape[0]:,} samples | {input_dim} features")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Val split: {VALIDATION_SPLIT}")
    print("-" * 60)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=get_callbacks(),
        verbose=1,
    )

    print(f"\n  ✓  Training complete. Model saved → {MODEL_PATH}")
    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate the trained model on the held-out test set.

    Metrics returned:
    - MAE  : Mean Absolute Error (in $ units)
    - RMSE : Root Mean Squared Error (in $ units)
    - R²   : Coefficient of Determination (1.0 = perfect)

    Args:
        model: Trained Keras model.
        X_test: Scaled test feature matrix.
        y_test: True test target values.

    Returns:
        Dictionary of metric names → values.
    """
    y_pred = model.predict(X_test, verbose=0).flatten()

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION — TEST SET")
    print("=" * 60)
    print(f"  MAE  : ${mae:>10,.2f}   (avg prediction error in dollars)")
    print(f"  RMSE : ${rmse:>10,.2f}   (penalizes large errors more)")
    print(f"  R²   : {r2:>11.4f}   (1.0 = perfect fit)")
    print("=" * 60)

    return metrics, y_pred


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: keras.callbacks.History, save_path: Optional[str] = None) -> None:
    """
    Plot training & validation loss (MSE) and MAE curves side by side.

    Args:
        history: Keras History object from model.fit().
        save_path: If provided, saves the figure to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Neural Network Training History", fontsize=15, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("#0f1117")

    style = dict(linewidth=2)
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Loss curve
    axes[0].plot(history.history["loss"],     label="Train Loss", color="#00d4ff", **style)
    axes[0].plot(history.history["val_loss"], label="Val Loss",   color="#ff6b6b", **style)
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Squared Error")
    axes[0].legend(facecolor="#1a1d27", labelcolor="white")

    # MAE curve
    axes[1].plot(history.history["mae"],     label="Train MAE", color="#00d4ff", **style)
    axes[1].plot(history.history["val_mae"], label="Val MAE",   color="#ff6b6b", **style)
    axes[1].set_title("MAE Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Absolute Error ($)")
    axes[1].legend(facecolor="#1a1d27", labelcolor="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Training curves saved → {save_path}")
    plt.close()


def plot_residuals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Generate a 4-panel diagnostic residual plot:
    1. Actual vs Predicted scatter
    2. Residual distribution histogram
    3. Residuals vs Predicted (heteroskedasticity check)
    4. Cumulative residual error

    Args:
        y_test: Ground-truth sales values.
        y_pred: Model predictions.
        save_path: If provided, saves to disk.
    """
    residuals = y_test - y_pred

    fig = plt.figure(figsize=(16, 10), facecolor="#0f1117")
    fig.suptitle("Model Residual Diagnostics", fontsize=16, fontweight="bold", color="white", y=1.01)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaa")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.4, color="#00d4ff", s=15, edgecolors="none")
    lim = max(y_test.max(), y_pred.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect Fit")
    axes[0].set_title("Actual vs Predicted")
    axes[0].set_xlabel("Actual Sales ($)")
    axes[0].set_ylabel("Predicted Sales ($)")
    axes[0].legend(facecolor="#1a1d27", labelcolor="white")

    # 2. Residual Distribution
    axes[1].hist(residuals, bins=50, color="#7c5cbf", edgecolor="#0f1117", linewidth=0.3)
    axes[1].axvline(0, color="#ff6b6b", linewidth=1.5, linestyle="--")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Count")

    # 3. Residuals vs Predicted
    axes[2].scatter(y_pred, residuals, alpha=0.4, color="#ffa500", s=15, edgecolors="none")
    axes[2].axhline(0, color="#ff6b6b", linewidth=1.5, linestyle="--")
    axes[2].set_title("Residuals vs Predicted (Bias Check)")
    axes[2].set_xlabel("Predicted Sales ($)")
    axes[2].set_ylabel("Residual ($)")

    # 4. Cumulative absolute error
    sorted_preds = np.sort(np.abs(residuals))
    cum_pct = np.arange(1, len(sorted_preds) + 1) / len(sorted_preds)
    axes[3].plot(sorted_preds, cum_pct * 100, color="#00ff88", linewidth=2)
    axes[3].set_title("Cumulative Absolute Error")
    axes[3].set_xlabel("Absolute Error ($)")
    axes[3].set_ylabel("% of Predictions")
    axes[3].grid(True, alpha=0.2, color="#444")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Residual plot saved → {save_path}")
    plt.close()


def export_predictions_for_powerbi(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "data/processed/predictions.csv",
) -> None:
    """
    Export actual vs predicted values to CSV for Power BI Page 5 visualization.

    Args:
        y_test: Ground-truth values from test set.
        y_pred: Model predictions on test set.
        save_path: Output CSV path.
    """
    out = pd.DataFrame({
        "Actual_Sales":    y_test,
        "Predicted_Sales": y_pred,
        "Residual":        y_test - y_pred,
        "Abs_Error":       np.abs(y_test - y_pred),
        "Index":           np.arange(len(y_test)),
    })
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    print(f"  Predictions exported → {save_path}  ({len(out):,} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# FULL TRAINING RUN
# ──────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    save_plots: bool = True,
) -> Tuple[keras.Model, Dict[str, float], np.ndarray]:
    """
    Execute the complete training, evaluation, and export pipeline.

    Args:
        X_train, X_test: Scaled feature arrays.
        y_train, y_test: Target arrays.
        save_plots: Whether to save diagnostic plots to disk.

    Returns:
        (model, metrics_dict, y_pred)
    """
    # Train
    model, history = train_model(X_train, y_train)

    # Evaluate
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # Visualize
    plots_dir = Path("data/processed")
    plots_dir.mkdir(parents=True, exist_ok=True)

    if save_plots:
        plot_training_curves(history, save_path=str(plots_dir / "training_curves.png"))
        plot_residuals(y_test, y_pred, save_path=str(plots_dir / "residual_diagnostics.png"))

    # Export predictions for Power BI
    export_predictions_for_powerbi(y_test, y_pred)

    return model, metrics, y_pred


if __name__ == "__main__":
    # Quick smoke-test using the preprocessing pipeline
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.preprocessing import run_preprocessing_pipeline

    X_train, X_test, y_train, y_test, feature_names, scaler = run_preprocessing_pipeline()
    model, metrics, y_pred = run_training_pipeline(X_train, X_test, y_train, y_test)
    print("\nDone! Model and artifacts saved to models/ and data/processed/")
