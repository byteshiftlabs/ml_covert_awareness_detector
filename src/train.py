#!/usr/bin/env python3
"""Train the default XGBoost pipeline for covert-awareness detection."""

import argparse
import shutil
import sys
import time
import warnings

import numpy as np

from pipeline import (
    build_feature_dataset,
    fit_fold_features,
    optimize_threshold,
    save_results,
    train_xgboost_classifier,
)

warnings.filterwarnings('ignore')


def progress_bar(current, total, start_time, prefix='', bar_length=None):
    """Display a progress bar with ETA (only on terminal, written to stderr)."""
    # Write to stderr so it appears on terminal but not in log files
    if not sys.stderr.isatty():
        return
    
    if bar_length is None:
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        bar_length = max(20, terminal_width - 55)

    elapsed = time.time() - start_time
    if current > 0:
        eta = elapsed / current * (total - current)
        eta_str = time.strftime('%M:%S', time.gmtime(eta))
    else:
        eta_str = '--:--'

    elapsed_str = time.strftime('%M:%S', time.gmtime(elapsed))
    fraction = current / total
    filled = int(bar_length * fraction)
    bar = '█' * filled + '░' * (bar_length - filled)
    pct = fraction * 100

    sys.stderr.write(
        f'\r  {prefix} [{bar}] {pct:5.1f}% '
        f'({current}/{total}) '
        f'elapsed {elapsed_str} · ETA {eta_str}'
    )
    sys.stderr.flush()
    if current == total:
        sys.stderr.write('\n')


def parse_args(argv=None):
    """Parse CLI flags for full or quick smoke training."""
    parser = argparse.ArgumentParser(description="Train the XGBoost covert-awareness classifier")
    parser.add_argument(
        "--max-subjects",
        type=int,
        help="Limit training to the first N ordered subjects for a quick smoke run",
    )
    args = parser.parse_args(argv)
    if args.max_subjects is not None and args.max_subjects <= 0:
        parser.error("--max-subjects must be a positive integer")
    return args


def main(argv=None):
    args = parse_args(argv)

    print("="*70)
    print("ADVANCED CONSCIOUSNESS DETECTION")
    print("="*70)
    print("XGBoost + PCA + SMOTE + Threshold Tuning")
    if args.max_subjects is not None:
        print(f"Quick mode: first {args.max_subjects} subjects")
    print()

    # ========================================================================
    # STEP 1: Load data with FULL connectivity
    # ========================================================================
    print("Loading data...")
    load_start = time.time()
    dataset = build_feature_dataset(
        max_subjects=args.max_subjects,
        on_subject_loaded=lambda idx, total, _subject: progress_bar(idx, total, load_start, prefix='Loading'),
    )
    progress_bar(len(dataset['subjects']), len(dataset['subjects']), load_start, prefix='Loading')
    x_engineered = dataset['x_engineered']
    x_connectivity = dataset['x_connectivity']
    subject_ids = dataset['subject_ids']
    labels = dataset['labels']
    print(f"✓ Loaded {x_engineered.shape[0]} samples from {len(dataset['subjects'])} subjects\n")

    # ========================================================================
    # STEP 2: Feature engineering
    # ========================================================================
    print("Feature engineering...")
    print(f"✓ Engineered {dataset['engineered_feature_count']} features\n")
    print(f"Conscious: {np.sum(labels == 1)}, Unconscious: {np.sum(labels == 0)}\n")

    # ========================================================================
    # STEP 3: Train with LOSO-CV (imputation + PCA refit per fold)
    # ========================================================================
    print("Training with LOSO-CV...")
    print("Imputation and PCA are refit inside each fold so the held-out")
    print("subject's data never shapes its own features.\n")

    unique_subjects = np.unique(subject_ids)
    all_labels = []
    all_probas = []
    fold_components = []
    fold_variance_explained = []

    cv_start = time.time()
    for i, test_subject in enumerate(unique_subjects):
        progress_bar(i, len(unique_subjects), cv_start, prefix='Training')
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask

        x_train, x_test, fold_meta = fit_fold_features(
            x_engineered[train_mask],
            x_connectivity[train_mask],
            x_engineered[test_mask],
            x_connectivity[test_mask],
        )
        y_train = labels[train_mask]
        y_test = labels[test_mask]

        fold_components.append(fold_meta['connectivity_components'])
        fold_variance_explained.append(fold_meta['connectivity_variance_explained'])

        _, y_proba, _ = train_xgboost_classifier(x_train, y_train, x_test)
        all_labels.extend(y_test)
        all_probas.extend(y_proba)

    progress_bar(len(unique_subjects), len(unique_subjects), cv_start, prefix='Training')
    cv_elapsed = time.time() - cv_start
    print(f"✓ Completed {len(unique_subjects)} LOSO-CV folds in {time.strftime('%M:%S', time.gmtime(cv_elapsed))}")
    print(
        f"  PCA per fold: {np.mean(fold_components):.0f} components "
        f"({np.mean(fold_variance_explained):.1%} mean variance explained)\n"
    )

    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)

    # ========================================================================
    # STEP 6: Optimize threshold
    # ========================================================================
    print("Optimizing threshold...")

    best_threshold, best_metrics = optimize_threshold(all_labels, all_probas)
    print(f"✓ Optimal threshold: {best_threshold:.2f} (balanced acc: {best_metrics['balanced_acc']:.3f})\n")

    # ========================================================================
    # STEP 7: Results
    # ========================================================================
    print(f"{'='*70}")
    print("FINAL RESULTS")
    print('='*70)

    print(f"\nOptimized XGBoost (threshold {best_threshold:.2f})")
    print("-" * 70)
    print(f"Accuracy:             {best_metrics['accuracy']:.3f}")
    print(f"Balanced Accuracy:    {best_metrics['balanced_acc']:.3f}")
    print(f"Recall (Unconscious): {best_metrics['recall_unconscious']:.3f}")
    print(f"Recall (Conscious):   {best_metrics['recall_conscious']:.3f}")
    print(f"F1 Score:             {best_metrics['f1']:.3f}")
    print(f"ROC-AUC:              {best_metrics['roc_auc']:.3f}")

    cm = np.asarray(best_metrics['confusion_matrix'], dtype=int)
    print("\nConfusion Matrix:")
    print("                    Predicted")
    print("              Unconscious  Conscious")
    print(
        f"Unconscious      {cm[0, 0]:5d}       "
        f"{cm[0, 1]:5d}"
    )
    print(
        f"Conscious        {cm[1, 0]:5d}       "
        f"{cm[1, 1]:5d}"
    )

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"• Detection: {cm[0, 0]}/{cm[0, 0] + cm[0, 1]} unconscious states correctly identified")
    print(f"• Balanced accuracy: {best_metrics['balanced_acc']*100:.1f}%")
    print(f"• Optimal decision threshold: {best_threshold:.2f}")
    print()
    print("Key techniques:")
    print(f"  - Full connectivity (99K features) → PCA (~{np.mean(fold_components):.0f} components, refit per LOSO fold)")
    print("  - XGBoost classifier with SMOTE oversampling")
    print("  - Per-subject deviation features")
    print("  - Threshold tuning for class balance")

    results_path = save_results(
        {
            'model': 'xgboost',
            'subject_count': len(dataset['subjects']),
            'subjects': dataset['subjects'],
            'sample_count': int(x_engineered.shape[0]),
            'engineered_feature_count': dataset['engineered_feature_count'],
            'pca_components_mean': float(np.mean(fold_components)),
            'connectivity_variance_explained_mean': float(np.mean(fold_variance_explained)),
            'decision_threshold': best_threshold,
            'metrics': best_metrics,
        }
    )
    print(f"\n✓ Saved metrics to {results_path}")


if __name__ == '__main__':
    main()
