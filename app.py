"""Interactive comparison of five saved dry-bean classifiers.

Upload a test CSV, choose a model, and read accuracy, AUC, precision,
recall, F1, MCC, a confusion matrix, and a per-variety report. The files
in model/ are loaded as already trained pipelines. This page never calls fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from evaluate import (
    METRIC_KEYS,
    MODEL_FILES,
    TARGET_COLUMN,
    USER_ID,
    classification_metrics,
    prediction_details,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_CSV = ROOT / "test_data.csv"
REPORT_PATH = MODEL_DIR / "training_report.json"

VARIETY_COLORS = {
    "SEKER": "#C4A35A",
    "BARBUNYA": "#8C3A4B",
    "BOMBAY": "#E0C068",
    "CALI": "#A0673B",
    "HOROZ": "#D4A017",
    "SIRA": "#E7D7A1",
    "DERMASON": "#6B4F3A",
}

FEATURE_HELP = {
    "Area": "Pixels inside the bean outline.",
    "Perimeter": "Length of the outline.",
    "MajorAxisLength": "Longest distance across the bean.",
    "MinorAxisLength": "Longest distance perpendicular to the major axis.",
    "AspectRation": "Major axis divided by minor axis. The UCI file uses this spelling.",
    "Eccentricity": "How stretched the ellipse with the same moments is.",
    "ConvexArea": "Pixels in the smallest convex polygon around the bean.",
    "EquivDiameter": "Diameter of a circle with the same area.",
    "Extent": "Bean area divided by the area of its bounding box.",
    "Solidity": "Bean area divided by convex area.",
    "roundness": "4π × area / perimeter².",
    "Compactness": "Equivalent diameter divided by the major axis.",
    "ShapeFactor1": "Shape descriptor from the Koklu and Ozkan measurements.",
    "ShapeFactor2": "Shape descriptor from the Koklu and Ozkan measurements.",
    "ShapeFactor3": "Shape descriptor from the Koklu and Ozkan measurements.",
    "ShapeFactor4": "Shape descriptor from the Koklu and Ozkan measurements.",
}


def depth_label(value):
    if value is None:
        return "unlimited"
    return str(value)


def model_caption(name, report):
    settings = report["hyperparameters"]
    if name == "Logistic Regression":
        chosen = settings["logistic_regression"]["C"]
        return (
            f"Standardized measurements, balanced class weights, and C = {chosen}. "
            "C was chosen by 5-fold macro F1 on the training split."
        )
    if name == "Decision Tree":
        depth = depth_label(settings["decision_tree"]["max_depth"])
        return (
            f"Balanced class weights, at least 5 grains per leaf, and max depth {depth}. "
            "Depth was chosen on the training split. Scaling does not change the splits, "
            "but the saved pipeline still scales so every model accepts raw measurements."
        )
    if name == "KNN":
        neighbors = settings["knn"]["n_neighbors"]
        return (
            f"{neighbors} neighbours with distance weights in standardized shape space. "
            "The neighbourhood size was chosen on the training split."
        )
    if name == "Naive Bayes":
        return (
            "Gaussian Naive Bayes with the estimated class prior. "
            "It treats the 16 measurements as independent, even though area, "
            "perimeter, and diameter describe the same grain."
        )
    return (
        "RBF support vector machine with C = 1 and balanced class weights. "
        "Those settings were fixed before the test split was scored. "
        "AUC uses decision scores because the saved model does not store probabilities."
    )


@st.cache_resource
def load_saved_models():
    models = {
        name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()
    }
    report = json.loads(REPORT_PATH.read_text())
    return models, report


def prepare_frame(frame, feature_columns):
    renamed = frame.copy()
    renamed.columns = [str(column).strip() for column in renamed.columns]
    required = feature_columns + [TARGET_COLUMN]
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise ValueError(
            "This CSV is missing required columns: " + ", ".join(missing)
        )
    features = renamed[feature_columns].apply(pd.to_numeric, errors="coerce")
    bad_columns = features.columns[features.isna().any()].tolist()
    if bad_columns:
        raise ValueError(
            "These measurement columns contain non-numeric or empty values: "
            + ", ".join(bad_columns)
        )
    target = renamed[TARGET_COLUMN].astype(str).str.strip()
    if target.eq("").any() or target.str.lower().isin(["nan", "none"]).any():
        raise ValueError("The Class column has empty values.")
    return features, target


def score_every_model(models, features, target):
    rows = []
    for name, model in models.items():
        scores = classification_metrics(
            model, features, target, raise_on_auc_error=False
        )
        rows.append({"Model": name, **scores})
    return pd.DataFrame(rows)


def draw_class_balance(target):
    counts = target.value_counts()
    fig, axis = plt.subplots(figsize=(7.4, 3.6))
    colors = [VARIETY_COLORS.get(variety, "#8C4A2F") for variety in counts.index]
    axis.bar(counts.index, counts.values, color=colors)
    axis.set_ylabel("Grains in this file")
    axis.set_xlabel("")
    axis.set_title("Varieties in the file being scored")
    fig.tight_layout()
    return fig


def draw_comparison(table):
    fig, axis = plt.subplots(figsize=(8.6, 4.2))
    plot_frame = table.set_index("Model")[["Accuracy", "F1", "MCC"]]
    plot_frame.plot(
        kind="bar",
        ax=axis,
        color=["#3E6B56", "#8C4A2F", "#C4A35A"],
        rot=12,
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score on the current file")
    axis.set_xlabel("")
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    return fig


def draw_confusion(matrix, labels):
    fig, axis = plt.subplots(figsize=(8.2, 6.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlOrBr",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
        cbar=False,
    )
    axis.set_xlabel("Predicted variety")
    axis.set_ylabel("Actual variety")
    axis.set_title("Confusion matrix")
    fig.tight_layout()
    return fig


def report_frame(report, labels):
    rows = []
    for label in list(labels) + ["macro avg", "weighted avg"]:
        stats = report[label]
        rows.append(
            {
                "Variety": label,
                "Precision": stats["precision"],
                "Recall": stats["recall"],
                "F1": stats["f1-score"],
                "Support": int(stats["support"]),
            }
        )
    return pd.DataFrame(rows)


def show_figure(fig):
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(
        page_title=f"Dry bean classifier — {USER_ID}",
        page_icon="🫘",
        layout="wide",
    )
    missing = [
        filename
        for filename in list(MODEL_FILES.values()) + ["training_report.json"]
        if not (MODEL_DIR / filename).exists()
    ]
    if missing or not DEFAULT_CSV.exists():
        st.error(
            "Saved models or test_data.csv are missing. From this folder run "
            "`python train_models.py`, then open the app again."
        )
        st.stop()

    models, report = load_saved_models()
    feature_columns = report["feature_columns"]

    st.title("Dry bean variety classifier")
    st.caption(f"User ID: {USER_ID}")
    st.write(
        "Seven registered dry-bean varieties are separated from 16 size and "
        "shape measurements taken from a grain image. Pick one of the five "
        "saved classifiers, or upload another test CSV. Scoring uses the "
        "saved pipelines only. The page does not retrain them."
    )

    with st.sidebar:
        st.markdown(f"**User ID:** {USER_ID}")
        st.header("Score a test file")
        selected = st.selectbox("Classification model", list(MODEL_FILES))
        upload = st.file_uploader("Upload test data (CSV)", type=["csv"])
        st.caption(
            "The file needs the 16 measurement columns and a Class column. "
            "A full comparison is meaningful when all seven varieties are present."
        )
        st.download_button(
            "Download the holdout test CSV",
            data=DEFAULT_CSV.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
        st.divider()
        st.markdown(model_caption(selected, report))
        counts = report["dataset"]["class_counts"]
        st.caption(
            f"Training used {report['split']['train_rows']:,} grains and the "
            f"bundled file holds the other {report['split']['test_rows']:,}. "
            f"Bombay is the smallest variety ({counts['BOMBAY']:,} grains) and "
            f"Dermason is the largest ({counts['DERMASON']:,})."
        )

    if upload is None:
        frame = pd.read_csv(DEFAULT_CSV)
        source_name = "test_data.csv, the bundled holdout split"
        st.info(
            "Scoring the bundled holdout file. Upload a CSV if you want to "
            "try a different test set with the same saved models."
        )
    else:
        frame = pd.read_csv(upload)
        source_name = upload.name
        st.success(
            f"Loaded {source_name}. The metrics below come from the saved models. "
            "Uploading a file does not retrain them."
        )

    try:
        features, target = prepare_frame(frame, feature_columns)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    missing_varieties = sorted(set(map(str, models[selected].classes_)) - set(map(str, target)))
    if missing_varieties:
        st.warning(
            "This file does not contain every variety the models were trained on ("
            + ", ".join(missing_varieties)
            + "). Macro averages will count a missing variety as a zero, so "
            "these scores are not directly comparable to the README table."
        )

    st.subheader(f"Current file: {source_name}")
    count_column, chart_column = st.columns([1, 2])
    count_column.metric("Grains", f"{len(features):,}")
    count_column.metric("Varieties present", target.nunique())
    count_column.metric("Measurements", len(feature_columns))
    with chart_column:
        show_figure(draw_class_balance(target))

    try:
        comparison = score_every_model(models, features, target)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.subheader("All five models on this file")
    st.caption(
        f"On the original {report['split']['test_rows']:,}-row holdout, "
        f"{report['winner']} had the highest Matthews correlation. "
        "The numbers in this section are computed again from the file loaded now."
    )
    st.caption(
        "Precision, recall, and F1 are macro averages. AUC is macro one-versus-rest. "
        "The SVM AUC is based on decision scores; the other models use predicted probabilities."
    )
    display = comparison.copy()
    for key in METRIC_KEYS:
        display[key] = display[key].map(lambda value: None if value is None else round(float(value), 4))
    st.dataframe(display, hide_index=True, use_container_width=True)
    winner = comparison.sort_values(["MCC", "F1"], ascending=False).iloc[0]
    st.write(
        f"On this file, **{winner['Model']}** has the highest Matthews correlation "
        f"({winner['MCC']:.4f})."
    )
    show_figure(draw_comparison(comparison))

    st.subheader(selected)
    st.write(model_caption(selected, report))
    selected_scores = comparison.loc[comparison["Model"] == selected].iloc[0]
    metric_columns = st.columns(6)
    for column, key in zip(metric_columns, METRIC_KEYS):
        value = selected_scores[key]
        column.metric(key, "n/a" if pd.isna(value) else f"{float(value):.4f}")

    _, class_report, matrix, labels = prediction_details(
        models[selected], features, target
    )
    matrix_column, report_column = st.columns(2)
    with matrix_column:
        show_figure(draw_confusion(matrix, labels))
    with report_column:
        st.markdown("**Classification report**")
        per_variety = report_frame(class_report, labels)
        st.dataframe(
            per_variety.style.format(
                {"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("A few scored grains")
    predicted = models[selected].predict(features)
    preview = features[["Area", "AspectRation", "Compactness", "roundness"]].head(12).copy()
    preview.insert(0, "Actual", target.head(12).to_numpy())
    preview.insert(1, "Predicted", predicted[:12])
    preview.insert(2, "Match", preview["Actual"].to_numpy() == preview["Predicted"])
    st.dataframe(preview, hide_index=True, use_container_width=True)

    with st.expander("What the 16 measurements mean"):
        glossary = pd.DataFrame(
            [{"Measurement": name, "Meaning": FEATURE_HELP[name]} for name in feature_columns]
        )
        st.dataframe(glossary, hide_index=True, use_container_width=True)
        st.caption(
            "Source: UCI Dry Bean Dataset, Koklu and Ozkan, 2020. "
            "The original column AspectRation is spelled that way in the data file."
        )
    st.caption(f"Machine Learning Assignment 2 · User ID {USER_ID}")


main()
