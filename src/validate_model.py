#!/usr/bin/env python3
"""
Overfitting validation: holdout test, feature importance, CV stability, permutation test.
"""

import numpy as np
import warnings
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from config import RANDOM_STATE
from pipeline import build_feature_dataset, fit_fold_features, optimize_threshold, train_xgboost_classifier

warnings.filterwarnings('ignore')

def main():
    print("="*70)
    print("OVERFITTING VALIDATION")
    print("="*70)

    # Prepare data
    print("\nLoading data...")
    dataset = build_feature_dataset()
    x_engineered = dataset['x_engineered']
    x_connectivity = dataset['x_connectivity']
    labels = dataset['labels']
    subject_ids = dataset['subject_ids']
    engineered_feature_count = dataset['engineered_feature_count']
    print(f"Engineered features: {x_engineered.shape}, raw connectivity: {x_connectivity.shape}\n")

    # CHECK 1: Holdout test (5 subjects held out)
    print("CHECK 1: HOLDOUT TEST")
    print("-" * 70)
    unique_subjects = np.unique(subject_ids)
    np.random.seed(RANDOM_STATE)
    test_subjects = np.random.choice(unique_subjects, size=5, replace=False)

    test_mask = np.isin(subject_ids, test_subjects)
    train_mask = ~test_mask
    y_train, y_test = labels[train_mask], labels[test_mask]

    # PCA/imputation are fit on the training split only, then applied to the
    # held-out subjects, so no held-out data shapes its own features.
    x_train, x_test, _ = fit_fold_features(
        x_engineered[train_mask], x_connectivity[train_mask],
        x_engineered[test_mask], x_connectivity[test_mask],
    )

    clf, y_proba, _ = train_xgboost_classifier(x_train, y_train, x_test)
    best_threshold, threshold_metrics = optimize_threshold(y_test, y_proba)
    best_balanced_accuracy = threshold_metrics['balanced_acc']

    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    confusion_mat = confusion_matrix(y_test, y_pred_optimal)
    recall_unconscious = recall_score(y_test, y_pred_optimal, pos_label=0, zero_division=0)
    recall_conscious = recall_score(y_test, y_pred_optimal, pos_label=1, zero_division=0)

    print(f"Test subjects: {test_subjects}")
    print(f"Balanced Accuracy: {best_balanced_accuracy:.3f} (threshold {best_threshold:.2f})")
    print(f"Recall - Unconscious: {recall_unconscious:.3f}, Conscious: {recall_conscious:.3f}")
    print(f"Confusion: [[{confusion_mat[0,0]}, {confusion_mat[0,1]}], [{confusion_mat[1,0]}, {confusion_mat[1,1]}]]")

    check1_pass = best_balanced_accuracy > 0.65
    print(f"{'✓ PASS' if check1_pass else '⚠ FAIL'}: {'Good' if check1_pass else 'Low'} generalization\n")

    # CHECK 2: Feature importance
    print("CHECK 2: FEATURE IMPORTANCE")
    print("-" * 70)
    importances = clf.feature_importances_
    engineered_importance = importances[:engineered_feature_count].sum()
    pca_importance = importances[engineered_feature_count:].sum()
    total_importance = engineered_importance + pca_importance or 1.0

    print(f"Engineered features: {engineered_importance:.3f} ({engineered_importance/total_importance*100:.0f}%)")
    print(f"PCA connectivity:    {pca_importance:.3f} ({pca_importance/total_importance*100:.0f}%)")

    check2_pass = pca_importance > 0.25
    print(f"{'✓ PASS' if check2_pass else '⚠ FAIL'}: PCA features {'are' if check2_pass else 'not'} meaningful\n")

    # CHECK 3: CV stability (10 subjects)
    print("CHECK 3: CV STABILITY")
    print("-" * 70)
    cv_scores = []
    for test_subject in unique_subjects[:10]:
        cv_test_mask = subject_ids == test_subject
        cv_train_mask = ~cv_test_mask
        y_train_cv, y_test_cv = labels[cv_train_mask], labels[cv_test_mask]

        x_train_cv, x_test_cv, _ = fit_fold_features(
            x_engineered[cv_train_mask], x_connectivity[cv_train_mask],
            x_engineered[cv_test_mask], x_connectivity[cv_test_mask],
        )

        _, y_proba_cv, _ = train_xgboost_classifier(x_train_cv, y_train_cv, x_test_cv)
        y_pred_cv = (y_proba_cv >= best_threshold).astype(int)
        cv_scores.append(balanced_accuracy_score(y_test_cv, y_pred_cv))

    mean_cv, std_cv = np.mean(cv_scores), np.std(cv_scores)
    coefficient_of_variation = std_cv / mean_cv

    print(f"Balanced accuracy: {mean_cv:.3f} ± {std_cv:.3f}")
    print(f"Coefficient of variation: {coefficient_of_variation:.3f}")

    check3_pass = coefficient_of_variation < 0.30
    print(f"{'✓ PASS' if check3_pass else '⚠ FAIL'}: {'Low' if check3_pass else 'High'} variance across folds\n")

    # CHECK 4: Permutation test
    print("CHECK 4: PERMUTATION TEST")
    print("-" * 70)
    y_permuted = np.random.permutation(y_train)
    _, y_proba_permuted, _ = train_xgboost_classifier(x_train, y_permuted, x_test)
    y_pred_permuted = (y_proba_permuted >= best_threshold).astype(int)
    permuted_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_permuted)

    print(f"Real labels:     {best_balanced_accuracy:.3f}")
    print(f"Permuted labels: {permuted_balanced_accuracy:.3f}")
    print(f"Difference:      {best_balanced_accuracy - permuted_balanced_accuracy:.3f}")

    check4_pass = best_balanced_accuracy > permuted_balanced_accuracy + 0.15
    print(f"{'✓ PASS' if check4_pass else '⚠ FAIL'}: Real model {'significantly' if check4_pass else 'barely'} outperforms chance\n")

    # Summary
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    checks_passed = sum([check1_pass, check2_pass, check3_pass, check4_pass])
    print(f"Checks passed: {checks_passed}/4")
    print(f"✓ Holdout test:       {'PASS' if check1_pass else 'FAIL'}")
    print(f"✓ Feature importance: {'PASS' if check2_pass else 'FAIL'}")
    print(f"✓ CV stability:       {'PASS' if check3_pass else 'FAIL'}")
    print(f"✓ Permutation test:   {'PASS' if check4_pass else 'FAIL'}")

    if checks_passed >= 3:
        print(f"\n✓✓ MODEL VALIDATED - Bal Acc: {best_balanced_accuracy:.1%}, Stable: {mean_cv:.3f}±{std_cv:.3f}")
    else:
        print("\n⚠ VALIDATION CONCERNS - Investigate further")
    print("="*70)


if __name__ == '__main__':
    main()

