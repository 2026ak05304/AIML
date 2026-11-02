"""Train five dry-bean classifiers and save the holdout test file.

Run this from the project folder on the BITS Virtual Lab:

    python train_models.py

The script fits each model on the training split only, writes joblib
pipelines under model/, and writes test_data.csv from the untouched holdout.
Nothing in the Streamlit app refits these pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from evaluate import (
    METRIC_KEYS,
    MODEL_FILES,
    TARGET_COLUMN,
    USER_ID,
    classification_metrics,
    prediction_details,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Dry_Bean_Dataset.csv"
MODEL_DIR = ROOT / "model"
TEST_PATH = ROOT / "test_data.csv"
REPORT_PATH = MODEL_DIR / "training_report.json"
CHART_PATH = MODEL_DIR / "metric_comparison.png"
RANDOM_STATE = 42


def load_dataset():
    frame = pd.read_csv(DATA_PATH)
    frame.columns = [str(column).strip() for column in frame.columns]
    feature_columns = [column for column in frame.columns if column != TARGET_COLUMN]
    if len(feature_columns) < 12:
        raise SystemExit(f"Expected at least 12 features, found {len(feature_columns)}.")
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).str.strip()
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise SystemExit("Measurement columns contain non-numeric values.")
    cleaned = features.copy()
    cleaned[TARGET_COLUMN] = frame[TARGET_COLUMN]
    return cleaned, feature_columns


def make_pipeline(classifier):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def logistic_model(inverse_regularization):
    return make_pipeline(
        LogisticRegression(
            C=inverse_regularization,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    )


def tree_model(max_depth):
    return make_pipeline(
        DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    )


def knn_model(neighbors):
    return make_pipeline(
        KNeighborsClassifier(
            n_neighbors=neighbors,
            weights="distance",
            metric="euclidean",
        )
    )


def naive_bayes_model():
    return make_pipeline(GaussianNB())


def svm_model():
    return make_pipeline(
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            cache_size=1000,
        )
    )


def macro_f1(pipeline, features, target, folds):
    scores = cross_val_score(
        pipeline,
        features,
        target,
        cv=folds,
        scoring="f1_macro",
        n_jobs=-1,
    )
    return float(scores.mean()), float(scores.std())


def select_setting(name, candidates, builder, features, target, folds):
    """Pick the training setting with the highest mean macro F1.

    Candidates are tried from simpler to more flexible. A tie keeps the
    simpler setting because replacement requires a strictly higher score.
    The holdout test set is not used here.
    """
    print(f"\nSelecting {name} on the training split")
    best_value = None
    best_score = -1.0
    history = []
    for value in candidates:
        mean_score, std_score = macro_f1(builder(value), features, target, folds)
        history.append({"value": value, "mean_macro_f1": mean_score, "std": std_score})
        label = "unlimited" if value is None else value
        print(f"  {label}: macro F1 {mean_score:.4f} ± {std_score:.4f}")
        if mean_score > best_score:
            best_value = value
            best_score = mean_score
    print(f"  chosen: {'unlimited' if best_value is None else best_value}")
    return best_value, history


def strongest_correlations(frame, feature_columns, limit=6):
    correlation = frame[feature_columns].corr().abs()
    pairs = []
    for index, left in enumerate(feature_columns):
        for right in feature_columns[index + 1 :]:
            pairs.append((float(correlation.loc[left, right]), left, right))
    pairs.sort(reverse=True)
    return [
        {"feature_a": left, "feature_b": right, "abs_correlation": value}
        for value, left, right in pairs[:limit]
    ]


def class_profiles(frame):
    columns = ["Area", "AspectRation", "Compactness", "roundness"]
    profiles = {}
    grouped = frame.groupby(TARGET_COLUMN)[columns].mean()
    for variety, row in grouped.iterrows():
        profiles[str(variety)] = {column: float(row[column]) for column in columns}
    return profiles


def top_confusions(matrix, labels, limit=4):
    pairs = []
    for actual_index, actual in enumerate(labels):
        for predicted_index, predicted in enumerate(labels):
            count = int(matrix[actual_index, predicted_index])
            if actual_index != predicted_index and count > 0:
                pairs.append((count, str(actual), str(predicted)))
    pairs.sort(reverse=True)
    return [
        {"count": count, "actual": actual, "predicted": predicted}
        for count, actual, predicted in pairs[:limit]
    ]


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def print_metrics(metrics):
    header = f"{'ML Model':<28}" + "".join(f"{name:>12}" for name in METRIC_KEYS)
    print("\n" + header)
    print("-" * len(header))
    for name, scores in metrics.items():
        cells = "".join(f"{scores[key]:12.4f}" for key in METRIC_KEYS)
        print(f"{name:<28}{cells}")


def save_chart(metrics):
    names = list(metrics)
    fig, axis = plt.subplots(figsize=(9.2, 4.6))
    positions = range(len(names))
    width = 0.25
    colors = {"F1": "#8C4A2F", "MCC": "#C4A35A", "Accuracy": "#3E6B56"}
    for offset, key in enumerate(("F1", "MCC", "Accuracy")):
        values = [metrics[name][key] for name in names]
        axis.bar(
            [pos + (offset - 1) * width for pos in positions],
            values,
            width=width,
            label=key,
            color=colors[key],
        )
    axis.set_xticks(list(positions))
    axis.set_xticklabels(names, rotation=12, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Holdout score")
    axis.set_title(f"Dry bean classifiers on the holdout split ({USER_ID})")
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=140)
    plt.close(fig)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"User ID: {USER_ID}")
    print("Machine Learning Assignment 2 — dry bean variety classification")
    frame, feature_columns = load_dataset()
    features = frame[feature_columns]
    target = frame[TARGET_COLUMN]
    print(f"Rows: {len(frame):,}    Features: {len(feature_columns)}")
    print(target.value_counts().to_string())

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    logistic_c, logistic_history = select_setting(
        "logistic regression C",
        (0.1, 1.0, 10.0),
        logistic_model,
        train_features,
        train_target,
        folds,
    )
    tree_depth, tree_history = select_setting(
        "decision tree max_depth",
        (6, 10, 14, 20, None),
        tree_model,
        train_features,
        train_target,
        folds,
    )
    knn_k, knn_history = select_setting(
        "k in KNN",
        (3, 5, 7, 11, 15),
        knn_model,
        train_features,
        train_target,
        folds,
    )

    fitted = {
        "Logistic Regression": logistic_model(logistic_c),
        "Decision Tree": tree_model(tree_depth),
        "KNN": knn_model(knn_k),
        "Naive Bayes": naive_bayes_model(),
        "Support Vector Machine": svm_model(),
    }

    metrics = {}
    reports = {}
    confusions = {}
    for name, pipeline in fitted.items():
        print(f"\nFitting {name} on {len(train_features):,} training rows...")
        pipeline.fit(train_features, train_target)
        metrics[name] = classification_metrics(pipeline, test_features, test_target)
        _, report, matrix, labels = prediction_details(
            pipeline, test_features, test_target
        )
        reports[name] = report
        confusions[name] = top_confusions(matrix, labels)
        path = MODEL_DIR / MODEL_FILES[name]
        joblib.dump(pipeline, path, compress=3)
        reloaded = joblib.load(path)
        reloaded_scores = classification_metrics(reloaded, test_features, test_target)
        for key in METRIC_KEYS:
            if abs(reloaded_scores[key] - metrics[name][key]) > 1e-9:
                raise SystemExit(f"Reloaded {name} did not match in-memory {key}.")
        print(f"  saved {path.name}")

    test_frame = test_features.copy()
    test_frame[TARGET_COLUMN] = test_target.to_numpy()
    test_frame.to_csv(TEST_PATH, index=False)
    roundtrip = pd.read_csv(TEST_PATH)
    roundtrip_scores = classification_metrics(
        joblib.load(MODEL_DIR / MODEL_FILES["Logistic Regression"]),
        roundtrip[feature_columns],
        roundtrip[TARGET_COLUMN].astype(str),
    )
    for key in METRIC_KEYS:
        if abs(roundtrip_scores[key] - metrics["Logistic Regression"][key]) > 1e-9:
            raise SystemExit("test_data.csv did not reproduce logistic regression scores.")

    importances = fitted["Decision Tree"].named_steps["clf"].feature_importances_
    importance_rows = [
        {"feature": feature, "importance": float(score)}
        for feature, score in sorted(
            zip(feature_columns, importances),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    winner = max(metrics, key=lambda name: (metrics[name]["MCC"], metrics[name]["F1"]))
    report = {
        "user_id": USER_ID,
        "dataset": {
            "name": "Dry Bean Dataset",
            "source": "UCI Machine Learning Repository, id 602",
            "citation": (
                "Koklu, M. and Ozkan, I. A. (2020). Multiclass classification of "
                "dry beans using computer vision and machine learning techniques. "
                "Computers and Electronics in Agriculture, 174, 105507."
            ),
            "rows": int(len(frame)),
            "feature_count": int(len(feature_columns)),
            "class_counts": {
                str(variety): int(count)
                for variety, count in target.value_counts().items()
            },
            "class_profiles": class_profiles(frame),
            "strongest_correlations": strongest_correlations(frame, feature_columns),
        },
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_rows": int(len(train_features)),
            "test_rows": int(len(test_features)),
        },
        "metric_definition": {
            "precision_recall_f1": "macro",
            "auc": "macro one-vs-rest",
            "svm_auc_scores": "decision_function",
            "other_auc_scores": "predict_proba",
            "winner_rule": "highest holdout MCC, then higher macro F1",
        },
        "hyperparameters": {
            "logistic_regression": {
                "C": logistic_c,
                "max_iter": 2000,
                "class_weight": "balanced",
                "cv_macro_f1": logistic_history,
            },
            "decision_tree": {
                "max_depth": tree_depth,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
                "cv_macro_f1": tree_history,
            },
            "knn": {
                "n_neighbors": knn_k,
                "weights": "distance",
                "metric": "euclidean",
                "cv_macro_f1": knn_history,
            },
            "naive_bayes": {"model": "GaussianNB", "defaults": True},
            "support_vector_machine": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "class_weight": "balanced",
                "probability": False,
                "selection": "fixed before looking at the test set",
            },
            "preprocessing": "StandardScaler fit on the training fold only",
        },
        "metrics": metrics,
        "classification_reports": reports,
        "top_confusions": confusions,
        "tree_feature_importance": importance_rows,
        "winner": winner,
        "feature_columns": feature_columns,
    }
    REPORT_PATH.write_text(json.dumps(to_builtin(report), indent=2))
    save_chart(metrics)
    print(f"\nUser ID: {USER_ID}")
    print_metrics(metrics)
    print(f"\nWinner by holdout MCC: {winner}")
    print(f"Test file: {TEST_PATH}")
    print(f"Report: {REPORT_PATH}")
    for name, pairs in confusions.items():
        print(f"\n{name} most common confusions:")
        for pair in pairs:
            print(f"  {pair['actual']} -> {pair['predicted']}: {pair['count']}")
    print("\nMean measurements by variety:")
    print(pd.DataFrame(report["dataset"]["class_profiles"]).T.round(3).to_string())
    print("\nStrongest absolute correlations:")
    for pair in report["dataset"]["strongest_correlations"]:
        print(
            f"  {pair['feature_a']} vs {pair['feature_b']}: "
            f"{pair['abs_correlation']:.3f}"
        )
    print("\nDecision tree importances:")
    for row in importance_rows[:6]:
        print(f"  {row['feature']}: {row['importance']:.3f}")


if __name__ == "__main__":
    main()
