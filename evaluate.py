"""Metric calculations shared by training and the Streamlit app.

Precision, recall, and F1 use a macro average so each bean variety counts
equally. On a single-label problem, weighted recall is identical to accuracy,
which would make two columns of the required comparison table duplicates.
AUC is macro one-versus-rest. The SVM was saved without probability
estimates, so its AUC uses decision-function scores. The other four models
use predict_proba. Rank-based AUC is valid for both.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

USER_ID = "2026ak05304"
TARGET_COLUMN = "Class"
AVERAGE = "macro"
METRIC_KEYS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def final_estimator(model):
    if hasattr(model, "named_steps"):
        return model.named_steps["clf"]
    return model


def has_probabilities(model):
    """True when the fitted estimator can return class probabilities.

    SVC keeps a probability flag that is False, or the string "deprecated"
    when probabilities were not requested. Calling predict_proba in that
    case raises, so AUC falls back to the decision function.
    """
    classifier = final_estimator(model)
    probability = getattr(classifier, "probability", None)
    if probability in (False, "deprecated"):
        return False
    return True


def class_scores(model, features):
    """Return an (n_samples, n_classes) score matrix aligned with classes_."""
    if has_probabilities(model):
        scores = model.predict_proba(features)
    else:
        scores = model.decision_function(features)
    if np.ndim(scores) == 1:
        scores = np.column_stack([-scores, scores])
    return scores


def macro_ovr_auc(target, scores, labels):
    """Unweighted mean of one-versus-rest ROC AUCs.

    Each column of ``scores`` is ranked on its own, so the column may contain
    probabilities or raw decision scores. The multiclass helper in scikit-learn
    rejects decision scores because they do not sum to one.
    """
    binary_target = label_binarize(target, classes=labels)
    if binary_target.shape[1] != len(labels):
        raise ValueError("AUC could not be aligned with the trained class labels.")
    class_aucs = []
    for index, label in enumerate(labels):
        observed = binary_target[:, index]
        if observed.min() == observed.max():
            raise ValueError(
                f"AUC is undefined for {label} because this file does not contain "
                "both examples of that variety and examples of other varieties."
            )
        class_aucs.append(roc_auc_score(observed, scores[:, index]))
    return float(np.mean(class_aucs))


def classification_metrics(model, features, target, raise_on_auc_error=True):
    labels = list(model.classes_)
    unknown = sorted(set(map(str, target)) - set(map(str, labels)))
    if unknown:
        raise ValueError(
            "The file contains class labels the model did not see in training: "
            + ", ".join(unknown)
        )

    predicted = model.predict(features)
    scores = {
        "Accuracy": float(accuracy_score(target, predicted)),
        "Precision": float(
            precision_score(
                target,
                predicted,
                average=AVERAGE,
                labels=labels,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                target,
                predicted,
                average=AVERAGE,
                labels=labels,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                target,
                predicted,
                average=AVERAGE,
                labels=labels,
                zero_division=0,
            )
        ),
        "MCC": float(matthews_corrcoef(target, predicted)),
    }
    try:
        ranking_scores = class_scores(model, features)
        scores["AUC"] = macro_ovr_auc(target, ranking_scores, labels)
        if has_probabilities(model):
            library_auc = roc_auc_score(
                target,
                ranking_scores,
                multi_class="ovr",
                average=AVERAGE,
                labels=labels,
            )
            if abs(library_auc - scores["AUC"]) > 1e-8:
                raise ValueError("The one-versus-rest AUC helper drifted from scikit-learn.")
    except ValueError:
        if raise_on_auc_error:
            raise
        scores["AUC"] = None
    return scores


def prediction_details(model, features, target):
    labels = list(model.classes_)
    predicted = model.predict(features)
    report = classification_report(
        target,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(target, predicted, labels=labels)
    return predicted, report, matrix, labels
