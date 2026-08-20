"""Shared helpers for training and validation workflows."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from config import CONSCIOUS_CONDITIONS, RANDOM_STATE, RESULTS_DIR, SUBJECTS
from data_loader import load_subject_all_conditions
from features import extract_all_features


def _json_ready(value: Any) -> Any:
    """Convert NumPy-heavy result payloads into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def select_subjects(subjects: Optional[Iterable[str]] = None, max_subjects: Optional[int] = None) -> list[str]:
    """Return the ordered subject list, optionally truncated for smoke runs."""
    selected_subjects = list(SUBJECTS if subjects is None else subjects)
    if max_subjects is not None:
        selected_subjects = selected_subjects[:max_subjects]
    return selected_subjects


def build_feature_dataset(
    subjects: Optional[Iterable[str]] = None,
    max_subjects: Optional[int] = None,
    on_subject_loaded: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Load matrices and extract raw, un-fitted features for the design matrix.

    This returns engineered and connectivity features before any imputation,
    scaling, or PCA. Those steps must be fit per cross-validation fold (see
    ``fit_fold_features``) so that no held-out subject's data leaks into the
    statistics used to build features for that same subject.
    """
    selected_subjects = select_subjects(subjects=subjects, max_subjects=max_subjects)
    if not selected_subjects:
        raise ValueError("At least one subject is required to build the feature dataset")

    all_features = []
    all_connectivity_matrices = []

    for idx, subject in enumerate(selected_subjects):
        if on_subject_loaded is not None:
            on_subject_loaded(idx, len(selected_subjects), subject)

        subject_connectivity_matrices = load_subject_all_conditions(subject)
        for condition_idx in range(7):
            connectivity_matrix = subject_connectivity_matrices[condition_idx]
            if np.isnan(connectivity_matrix).all():
                # Scan missing or fully motion-censored: exclude the sample
                # rather than fabricating a zero-connectivity observation.
                continue

            features = extract_all_features(connectivity_matrix)

            all_connectivity_matrices.append(features["connectivity"])
            features["subject"] = subject
            features["condition"] = condition_idx
            features["label"] = 1 if condition_idx in CONSCIOUS_CONDITIONS else 0
            all_features.append(features)

    if not all_features:
        raise ValueError("No valid samples found: every connectivity matrix was missing or fully censored")

    feature_names_basic = [
        key for key in all_features[0].keys()
        if key not in ["subject", "condition", "label", "connectivity"]
    ]
    x_basic = np.array([[feature[name] for name in feature_names_basic] for feature in all_features])
    subject_ids = np.array([feature["subject"] for feature in all_features])
    labels = np.array([feature["label"] for feature in all_features])

    x_deviations = np.zeros_like(x_basic)
    for subject in np.unique(subject_ids):
        subject_mask = subject_ids == subject
        subject_data = x_basic[subject_mask]
        conscious_mask = labels[subject_mask] == 1
        baseline = subject_data[conscious_mask].mean(axis=0) if conscious_mask.sum() > 0 else subject_data.mean(axis=0)
        x_deviations[subject_mask] = subject_data - baseline

    x_engineered = np.hstack([x_basic, x_deviations])
    x_connectivity = np.array(all_connectivity_matrices)

    return {
        "subjects": selected_subjects,
        "subject_ids": subject_ids,
        "labels": labels,
        "x_engineered": x_engineered,
        "x_connectivity": x_connectivity,
        "engineered_feature_count": int(x_engineered.shape[1]),
    }


def fit_fold_features(
    x_engineered_train: np.ndarray,
    x_connectivity_train: np.ndarray,
    x_engineered_test: np.ndarray,
    x_connectivity_test: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit imputation and PCA on the training fold only, then transform both folds.

    Fitting these on the full dataset before cross-validation would let each
    held-out subject's own connectivity data shape the PCA components and
    median-imputation values used to build that same subject's test features.
    Restricting the fit to the training fold keeps the held-out subject's data
    genuinely unseen.
    """
    engineered_imputer = SimpleImputer(strategy="median")
    x_engineered_train_clean = engineered_imputer.fit_transform(x_engineered_train)
    x_engineered_test_clean = engineered_imputer.transform(x_engineered_test)

    connectivity_imputer = SimpleImputer(strategy="median")
    x_connectivity_train_clean = connectivity_imputer.fit_transform(x_connectivity_train)
    x_connectivity_test_clean = connectivity_imputer.transform(x_connectivity_test)

    n_components = max(1, min(50, x_connectivity_train_clean.shape[0] - 1, x_connectivity_train_clean.shape[1]))
    pca = PCA(n_components=n_components, random_state=random_state)
    x_connectivity_train_pca = pca.fit_transform(x_connectivity_train_clean)
    x_connectivity_test_pca = pca.transform(x_connectivity_test_clean)

    x_train_combined = np.hstack([x_engineered_train_clean, x_connectivity_train_pca])
    x_test_combined = np.hstack([x_engineered_test_clean, x_connectivity_test_pca])

    metadata = {
        "connectivity_components": int(n_components),
        "connectivity_variance_explained": float(pca.explained_variance_ratio_.sum()),
    }
    return x_train_combined, x_test_combined, metadata


def train_xgboost_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: Optional[np.ndarray] = None,
) -> tuple[xgb.XGBClassifier, Optional[np.ndarray], StandardScaler]:
    """Train the default XGBoost classifier with SMOTE and scaling."""
    minority_count = int(np.sum(y_train == 0))
    if minority_count >= 2:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, minority_count - 1))
        try:
            x_train, y_train = smote.fit_resample(x_train, y_train)
        except ValueError as error:
            warnings.warn(
                f"SMOTE resampling failed; continuing without oversampling: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    positive_count = max(int(np.sum(y_train == 1)), 1)
    negative_count = int(np.sum(y_train == 0))
    scale_pos_weight = negative_count / positive_count

    classifier = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    classifier.fit(x_train_scaled, y_train, verbose=0)

    if x_test is None:
        return classifier, None, scaler

    x_test_scaled = scaler.transform(x_test)
    probabilities = classifier.predict_proba(x_test_scaled)[:, 1]
    return classifier, probabilities, scaler


def optimize_threshold(y_true: np.ndarray, y_probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Select the threshold that maximizes balanced accuracy."""
    best_threshold = 0.5
    best_balanced_accuracy = -1.0
    best_metrics = None

    for threshold in np.arange(0.1, 0.95, 0.05):
        predictions = (y_probabilities >= threshold).astype(int)
        balanced_accuracy = balanced_accuracy_score(y_true, predictions)
        if balanced_accuracy <= best_balanced_accuracy:
            continue

        matrix = confusion_matrix(y_true, predictions)
        best_balanced_accuracy = balanced_accuracy
        best_threshold = float(threshold)
        best_metrics = {
            "balanced_acc": float(balanced_accuracy),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "recall_unconscious": float(recall_score(y_true, predictions, pos_label=0, zero_division=0)),
            "recall_conscious": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_probabilities)),
            "confusion_matrix": matrix,
        }

    return best_threshold, best_metrics


def save_results(payload: dict, prefix: str = "results", output_dir: Path = RESULTS_DIR) -> Path:
    """Persist a JSON result payload under results/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{prefix}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)
        handle.write("\n")
    return output_path