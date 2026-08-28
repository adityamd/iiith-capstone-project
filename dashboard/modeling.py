from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


KEY = ["code_module", "code_presentation", "id_student"]
COURSE_KEY = ["code_module", "code_presentation"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "histogram_boosting": "Histogram Boosting",
    "neural_network": "Neural Network",
}
MODEL_COLORS = {
    "logistic_regression": "#4F7CAC",
    "histogram_boosting": "#E07A5F",
    "neural_network": "#6A994E",
}
PRIMARY_FAIRNESS_ATTRIBUTES = [
    "gender",
    "disability",
    "imd_band",
    "gender_x_disability",
    "disability_x_deprivation",
]
THRESHOLD = 0.5
FEATURE_GROUPS = {
    "Early VLE engagement": [
        "total_clicks_25pct", "vle_events_25pct", "active_days_25pct",
        "distinct_sites_25pct", "last_activity_day_25pct",
        "days_since_last_activity_at_checkpoint", "no_early_vle_activity_flag",
    ],
    "Early assessment behavior": [
        "early_assessments_due_count", "early_assessments_submitted_count",
        "early_assessments_missing_count", "mean_early_score", "min_early_score",
        "max_early_score", "submitted_early_weight_total", "weighted_early_score",
        "average_submission_delay", "late_submission_count", "banked_assessment_count",
    ],
    "Student profile": [
        "gender", "region", "highest_education", "imd_band", "age_band",
        "num_of_prev_attempts", "studied_credits", "disability",
    ],
    "Course context": [
        "code_module", "code_presentation", "module_presentation_length", "checkpoint_day",
    ],
    "Registration timing": [
        "date_registration", "registration_missing_flag",
        "registered_before_start_days", "registered_after_start_flag",
    ],
}
WHAT_IF_FEATURES = [
    "active_days_25pct", "total_clicks_25pct",
    "days_since_last_activity_at_checkpoint", "early_assessments_missing_count",
    "mean_early_score", "studied_credits", "num_of_prev_attempts",
    "average_submission_delay",
]


def find_data_dir(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    candidates = []
    for parent in [start, *start.parents]:
        candidates.extend([
            parent / "open+university+learning+analytics+dataset",
            parent / "data" / "open+university+learning+analytics+dataset",
        ])
    for candidate in candidates:
        if (candidate / "studentVle.csv").exists() and (candidate / "studentInfo.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the OULAD dataset folder. Expected "
        "'open+university+learning+analytics+dataset' in this directory or a parent."
    )


def _read_csv(data_dir: Path, name: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(data_dir / name, na_values=["?"], **kwargs)


def build_modeling_dataset(data_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Reproduce the leakage-safe 25%-checkpoint table from checkpoint1_EDA_v3."""
    assessments = _read_csv(data_dir, "assessments.csv")
    courses = _read_csv(data_dir, "courses.csv")
    student_assessment = _read_csv(data_dir, "studentAssessment.csv")
    student_info = _read_csv(data_dir, "studentInfo.csv")
    student_registration = _read_csv(data_dir, "studentRegistration.csv")
    vle = _read_csv(data_dir, "vle.csv")

    base = student_info.copy()
    base["target_dropout"] = (base["final_result"] == "Withdrawn").astype(int)
    base = base.merge(courses, on=COURSE_KEY, how="left")
    base["checkpoint_day"] = np.floor(base["module_presentation_length"] * 0.25).astype(int)

    registration = student_registration[KEY + ["date_registration"]].copy()
    registration["registration_missing_flag"] = registration["date_registration"].isna().astype(int)
    registration["registered_before_start_days"] = (-registration["date_registration"]).clip(lower=0)
    registration["registered_after_start_flag"] = np.where(
        registration["date_registration"].isna(),
        0,
        (registration["date_registration"] > 0).astype(int),
    )

    checkpoint_lookup = base[KEY + ["checkpoint_day"]].copy()
    vle_meta = vle[["id_site", "code_module", "code_presentation", "activity_type"]].copy()
    top_activity_types = vle["activity_type"].value_counts(dropna=False).head(12).index.tolist()
    event_parts: list[pd.DataFrame] = []
    last_parts: list[pd.DataFrame] = []
    day_parts: list[pd.DataFrame] = []
    site_parts: list[pd.DataFrame] = []
    activity_parts: list[pd.DataFrame] = []
    usecols = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    for chunk in _read_csv(data_dir, "studentVle.csv", usecols=usecols, chunksize=1_000_000):
        chunk = chunk.merge(checkpoint_lookup, on=KEY, how="inner")
        chunk = chunk[chunk["date"] <= chunk["checkpoint_day"]]
        if chunk.empty:
            continue
        event_parts.append(
            chunk.groupby(KEY).agg(
                total_clicks_25pct=("sum_click", "sum"),
                vle_events_25pct=("sum_click", "size"),
            ).reset_index()
        )
        last_parts.append(
            chunk.groupby(KEY, as_index=False)["date"].max().rename(
                columns={"date": "last_activity_day_25pct"}
            )
        )
        day_parts.append(chunk[KEY + ["date"]].drop_duplicates())
        site_parts.append(chunk[KEY + ["id_site"]].drop_duplicates())
        typed = chunk.merge(vle_meta, on=["id_site", *COURSE_KEY], how="left")
        typed["activity_type"] = typed["activity_type"].where(
            typed["activity_type"].isin(top_activity_types), "other_activity"
        )
        activity_parts.append(
            typed.groupby(KEY + ["activity_type"])["sum_click"].sum().reset_index()
            .pivot_table(index=KEY, columns="activity_type", values="sum_click", fill_value=0)
            .reset_index()
        )

    vle_events = pd.concat(event_parts, ignore_index=True).groupby(KEY, as_index=False).sum()
    vle_last = pd.concat(last_parts, ignore_index=True).groupby(KEY, as_index=False).max()
    vle_days = (
        pd.concat(day_parts, ignore_index=True).drop_duplicates().groupby(KEY).size()
        .reset_index(name="active_days_25pct")
    )
    vle_sites = (
        pd.concat(site_parts, ignore_index=True).drop_duplicates().groupby(KEY).size()
        .reset_index(name="distinct_sites_25pct")
    )
    vle_activity = pd.concat(activity_parts, ignore_index=True).groupby(KEY, as_index=False).sum()
    vle_features = (
        vle_events.merge(vle_days, on=KEY, how="left")
        .merge(vle_sites, on=KEY, how="left")
        .merge(vle_last, on=KEY, how="left")
        .merge(vle_activity, on=KEY, how="left")
        .merge(checkpoint_lookup, on=KEY, how="left")
    )
    vle_features["days_since_last_activity_at_checkpoint"] = (
        vle_features["checkpoint_day"] - vle_features["last_activity_day_25pct"]
    )
    vle_features = vle_features.drop(columns=["checkpoint_day"])
    vle_features = vle_features.rename(columns={
        col: f"clicks_{str(col).replace('-', '_').replace(' ', '_')}_25pct"
        for col in vle_activity.columns if col not in KEY
    })

    assessment_defs = assessments.merge(
        courses.assign(
            checkpoint_day=np.floor(courses["module_presentation_length"] * 0.25).astype(int)
        ),
        on=COURSE_KEY,
        how="left",
    )
    early_defs = assessment_defs[
        assessment_defs["date"].notna()
        & (assessment_defs["date"] <= assessment_defs["checkpoint_day"])
    ].copy()
    early_due = early_defs.groupby(COURSE_KEY).agg(
        early_assessments_due_count=("id_assessment", "nunique"),
        early_assessment_weight_total=("weight", "sum"),
    ).reset_index()
    early_submissions = student_assessment.merge(
        early_defs[[
            "id_assessment", "code_module", "code_presentation", "assessment_type",
            "date", "weight", "checkpoint_day",
        ]],
        on="id_assessment",
        how="inner",
    )
    early_submissions = early_submissions[
        early_submissions["date_submitted"] <= early_submissions["checkpoint_day"]
    ].copy()
    early_submissions["submission_delay"] = (
        early_submissions["date_submitted"] - early_submissions["date"]
    )
    early_submissions["late_submission_flag"] = (
        early_submissions["submission_delay"] > 0
    ).astype(int)
    early_submissions["weighted_score_component"] = (
        early_submissions["score"] * early_submissions["weight"]
    )
    assessment_features = early_submissions.groupby(KEY).agg(
        early_assessments_submitted_count=("id_assessment", "nunique"),
        mean_early_score=("score", "mean"),
        min_early_score=("score", "min"),
        max_early_score=("score", "max"),
        early_score_missing_count=("score", lambda values: values.isna().sum()),
        weighted_early_score_num=("weighted_score_component", "sum"),
        submitted_early_weight_total=("weight", "sum"),
        average_submission_delay=("submission_delay", "mean"),
        late_submission_count=("late_submission_flag", "sum"),
        banked_assessment_count=("is_banked", "sum"),
    ).reset_index()
    assessment_features["weighted_early_score"] = np.where(
        assessment_features["submitted_early_weight_total"] > 0,
        assessment_features["weighted_early_score_num"]
        / assessment_features["submitted_early_weight_total"],
        np.nan,
    )
    assessment_features = assessment_features.drop(columns=["weighted_early_score_num"])
    assessment_features = assessment_features.merge(
        base[KEY].drop_duplicates(), on=KEY, how="right"
    ).merge(early_due, on=COURSE_KEY, how="left")
    for column in [
        "early_assessments_due_count", "early_assessment_weight_total",
        "early_assessments_submitted_count",
    ]:
        assessment_features[column] = assessment_features[column].fillna(0)
    assessment_features["early_assessments_missing_count"] = (
        assessment_features["early_assessments_due_count"]
        - assessment_features["early_assessments_submitted_count"]
    ).clip(lower=0)

    profile_cols = [
        "code_module", "code_presentation", "id_student", "gender", "region",
        "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
        "studied_credits", "disability", "module_presentation_length",
        "checkpoint_day", "target_dropout",
    ]
    frame = base[profile_cols].copy()
    frame["imd_band"] = frame["imd_band"].fillna("Unknown")
    frame = frame.merge(registration, on=KEY, how="left")
    frame = frame.merge(vle_features, on=KEY, how="left")
    frame = frame.merge(assessment_features, on=KEY, how="left")
    frame["no_early_vle_activity_flag"] = frame["last_activity_day_25pct"].isna().astype(int)
    zero_prefixes = [
        "total_clicks", "vle_events", "active_days", "distinct_sites", "clicks_",
        "early_assessments", "late_submission", "banked_assessment",
        "early_score_missing", "early_assessment_weight", "submitted_early_weight",
    ]
    for column in frame.columns:
        if any(column.startswith(prefix) for prefix in zero_prefixes):
            frame[column] = frame[column].fillna(0)
    for column in [
        "mean_early_score", "min_early_score", "max_early_score",
        "weighted_early_score", "average_submission_delay",
    ]:
        frame[f"{column}_missing_flag"] = frame[column].isna().astype(int)

    numeric = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in ["id_student", "target_dropout"]
    ]
    categorical = list(frame.select_dtypes(include=["object"]).columns)
    assert len(frame) == 32_593
    assert not frame.duplicated(KEY).any()
    assert len(numeric) == 46 and len(categorical) == 8
    return frame, numeric, categorical


def grouped_split(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    x = frame[features].copy()
    y = frame["target_dropout"].copy()
    groups = frame["id_student"].copy()
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)
    assignment = np.full(len(frame), -1, dtype=int)
    for fold, (_, indices) in enumerate(splitter.split(x, y, groups=groups)):
        assignment[indices] = fold
    test_mask = np.isin(assignment, [0, 1, 2])
    eval_mask = np.isin(assignment, [3, 4, 5])
    train_mask = ~(test_mask | eval_mask)
    result = {
        "train": frame.loc[train_mask].copy(),
        "evaluation": frame.loc[eval_mask].copy(),
        "test": frame.loc[test_mask].copy(),
    }
    student_sets = {name: set(part["id_student"]) for name, part in result.items()}
    assert student_sets["train"].isdisjoint(student_sets["evaluation"])
    assert student_sets["train"].isdisjoint(student_sets["test"])
    assert student_sets["evaluation"].isdisjoint(student_sets["test"])
    return result


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]), categorical),
    ])


def _linear(values: np.ndarray, state: dict[str, np.ndarray], index: int) -> np.ndarray:
    return values @ state[f"network.{index}.weight"].T + state[f"network.{index}.bias"]


def _batch_norm(values: np.ndarray, state: dict[str, np.ndarray], index: int) -> np.ndarray:
    mean = state[f"network.{index}.running_mean"]
    variance = state[f"network.{index}.running_var"]
    weight = state[f"network.{index}.weight"]
    bias = state[f"network.{index}.bias"]
    return ((values - mean) / np.sqrt(variance + 1e-5)) * weight + bias


def predict_mlp_numpy(states: list[dict[str, np.ndarray]], matrix: np.ndarray) -> np.ndarray:
    """Run the exported PyTorch network exactly in evaluation mode using NumPy."""
    inputs = np.asarray(matrix, dtype=np.float32)
    probabilities = []
    for state in states:
        hidden = np.maximum(_batch_norm(_linear(inputs, state, 0), state, 1), 0)
        hidden = np.maximum(_batch_norm(_linear(hidden, state, 4), state, 5), 0)
        hidden = np.maximum(_linear(hidden, state, 8), 0)
        logits = _linear(hidden, state, 11).reshape(-1)
        probabilities.append(1.0 / (1.0 + np.exp(-logits)))
    return np.mean(np.vstack(probabilities), axis=0)


def metric_record(model_id: str, actual: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, prediction, labels=[0, 1]).ravel()
    return {
        "model_id": model_id,
        "model": MODEL_LABELS[model_id],
        "accuracy": accuracy_score(actual, prediction),
        "balanced_accuracy": balanced_accuracy_score(actual, prediction),
        "precision": precision_score(actual, prediction, zero_division=0),
        "recall": recall_score(actual, prediction, zero_division=0),
        "f1": f1_score(actual, prediction, zero_division=0),
        "roc_auc": roc_auc_score(actual, probability),
        "pr_auc": average_precision_score(actual, probability),
        "brier_score": brier_score_loss(actual, probability),
        "log_loss": log_loss(actual, probability),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def curve_record(actual: np.ndarray, probability: np.ndarray) -> dict[str, list[float]]:
    fpr, tpr, _ = roc_curve(actual, probability)
    precision, recall, _ = precision_recall_curve(actual, probability)
    fraction_positive, mean_predicted = calibration_curve(actual, probability, n_bins=10)
    return {
        "roc_fpr": fpr.tolist(), "roc_tpr": tpr.tolist(),
        "pr_recall": recall.tolist(), "pr_precision": precision.tolist(),
        "calibration_predicted": mean_predicted.tolist(),
        "calibration_observed": fraction_positive.tolist(),
    }


def _case_id(row: pd.Series) -> str:
    raw = f"{row['code_module']}|{row['code_presentation']}|{int(row['id_student'])}"
    return "CASE-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:6].upper()


def select_curated_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    probability_columns = [f"prob_{model_id}" for model_id in MODEL_LABELS]
    for model_id in MODEL_LABELS:
        frame[f"pred_{model_id}"] = (frame[f"prob_{model_id}"] >= THRESHOLD).astype(int)
    frame["mean_probability"] = frame[probability_columns].mean(axis=1)
    frame["spread"] = frame[probability_columns].max(axis=1) - frame[probability_columns].min(axis=1)
    selected: list[tuple[str, str, pd.Series]] = []
    used: set[str] = set()

    def add(kind: str, title: str, candidates: pd.DataFrame, sort_column: str, ascending: bool) -> None:
        available = candidates[~candidates["case_id"].isin(used)].sort_values(sort_column, ascending=ascending)
        if available.empty:
            return
        row = available.iloc[0]
        used.add(row["case_id"])
        selected.append((kind, title, row))

    unanimous_high = frame[
        (frame["actual"] == 1)
        & frame[[f"pred_{m}" for m in MODEL_LABELS]].eq(1).all(axis=1)
    ]
    add("unanimous_high", "Clear high-risk withdrawal", unanimous_high, "mean_probability", False)
    unanimous_low = frame[
        (frame["actual"] == 0)
        & frame[[f"pred_{m}" for m in MODEL_LABELS]].eq(0).all(axis=1)
    ]
    add("unanimous_low", "Clear low-risk continuation", unanimous_low, "mean_probability", True)
    hgb_catch = frame[
        (frame["actual"] == 1)
        & (frame["pred_histogram_boosting"] == 1)
        & ((frame["pred_logistic_regression"] == 0) | (frame["pred_neural_network"] == 0))
    ].copy()
    hgb_catch["selection_score"] = (
        hgb_catch["prob_histogram_boosting"]
        - hgb_catch[["prob_logistic_regression", "prob_neural_network"]].min(axis=1)
    )
    add("hgb_catch", "HGB catches a withdrawal others miss", hgb_catch, "selection_score", False)
    other_catch = frame[
        (frame["actual"] == 1)
        & (frame["pred_histogram_boosting"] == 0)
        & ((frame["pred_logistic_regression"] == 1) | (frame["pred_neural_network"] == 1))
    ]
    add("other_catch", "Another model catches an HGB miss", other_catch, "spread", False)
    frame["boundary_distance"] = (frame["mean_probability"] - THRESHOLD).abs()
    add("borderline", "Borderline threshold-sensitive case", frame, "boundary_distance", True)
    if len(selected) < 5:
        add("largest_disagreement", "Largest remaining model disagreement", frame, "spread", False)

    rows = []
    for order, (kind, title, row) in enumerate(selected, start=1):
        item = row.to_dict()
        item.update({"case_type": kind, "case_title": title, "display_order": order})
        rows.append(item)
    return pd.DataFrame(rows)


def feature_references(development: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    for column in numeric:
        reference[column] = float(development[column].median())
    for column in categorical:
        modes = development[column].mode(dropna=True)
        reference[column] = str(modes.iloc[0]) if len(modes) else "Unknown"
    return reference


def input_ranges(development: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    ranges = {}
    for column in features:
        values = pd.to_numeric(development[column], errors="coerce").dropna()
        ranges[column] = {
            "min": float(values.min()), "max": float(values.max()),
            "step": 1.0 if np.allclose(values, values.round()) else 0.1,
        }
    return ranges


def fairness_audit_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the primary audit groups used by the dedicated DPD/EOD notebook."""
    audit = pd.DataFrame(index=frame.index)
    audit["gender"] = frame["gender"].fillna("Unknown").astype(str)
    audit["disability"] = frame["disability"].fillna("Unknown").astype(str)
    audit["imd_band"] = frame["imd_band"].fillna("Unknown").astype(str)
    deprived_bands = {"0-10%", "10-20", "10-20%", "20-30%"}
    audit["deprivation_group"] = np.where(
        audit["imd_band"].eq("Unknown"),
        "Unknown IMD",
        np.where(audit["imd_band"].isin(deprived_bands), "Lowest 30% IMD", "Other known IMD"),
    )
    audit["gender_x_disability"] = (
        audit["gender"] + " | disability=" + audit["disability"]
    )
    audit["disability_x_deprivation"] = (
        audit["disability"] + " | " + audit["deprivation_group"]
    )
    return audit.reset_index(drop=True)


def subgroup_records(test: pd.DataFrame, probabilities: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    actual = test["target_dropout"].to_numpy()
    audit = fairness_audit_groups(test)
    for model_id, probability in probabilities.items():
        prediction = (probability >= THRESHOLD).astype(int)
        for attribute in PRIMARY_FAIRNESS_ATTRIBUTES:
            attribute_values = audit[attribute].to_numpy()
            for group in sorted(audit[attribute].unique()):
                mask = attribute_values == group
                y_group, pred_group = actual[mask], prediction[mask]
                tn, fp, fn, tp = confusion_matrix(y_group, pred_group, labels=[0, 1]).ravel()
                positives, negatives, flagged = tp + fn, tn + fp, tp + fp
                eligible = positives >= 30 and negatives >= 30
                if attribute == "imd_band" and group == "Unknown":
                    eligible = False
                if attribute == "disability_x_deprivation" and "Unknown IMD" in group:
                    eligible = False
                rows.append({
                    "model_id": model_id, "model": MODEL_LABELS[model_id],
                    "attribute": attribute, "group": group,
                    "records": int(mask.sum()),
                    "actual_withdrawn": int(positives),
                    "actual_not_withdrawn": int(negatives),
                    "withdrawal_rate": positives / mask.sum(),
                    "selection_rate": flagged / mask.sum(),
                    "tpr_recall": tp / positives if positives else np.nan,
                    "fnr": fn / positives if positives else np.nan,
                    "fpr": fp / negatives if negatives else np.nan,
                    "precision": tp / flagged if flagged else np.nan,
                    "accuracy": (tp + tn) / mask.sum(),
                    "eligible": bool(eligible),
                })
    return pd.DataFrame(rows)


def fairness_summary(subgroup_metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate notebook-equivalent DPD and equal-opportunity differences."""
    rows = []
    for (model_id, model, attribute), groups in subgroup_metrics.groupby(
        ["model_id", "model", "attribute"], sort=False,
    ):
        eligible = groups[groups["eligible"].astype(bool)]
        selection = eligible["selection_rate"].dropna()
        recall = eligible["tpr_recall"].dropna()
        rows.append({
            "model_id": model_id,
            "model": model,
            "attribute": attribute,
            "eligible_groups": int(len(eligible)),
            "dpd": float(selection.max() - selection.min()) if len(selection) >= 2 else np.nan,
            "eod": float(recall.max() - recall.min()) if len(recall) >= 2 else np.nan,
            "lowest_selection_group": (
                eligible.loc[selection.idxmin(), "group"] if len(selection) else None
            ),
            "highest_selection_group": (
                eligible.loc[selection.idxmax(), "group"] if len(selection) else None
            ),
            "lowest_recall_group": (
                eligible.loc[recall.idxmin(), "group"] if len(recall) else None
            ),
            "highest_recall_group": (
                eligible.loc[recall.idxmax(), "group"] if len(recall) else None
            ),
        })
    return pd.DataFrame(rows)


def dataset_fingerprint(data_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in [
        "assessments.csv", "courses.csv", "studentAssessment.csv", "studentInfo.csv",
        "studentRegistration.csv", "studentVle.csv", "vle.csv",
    ]:
        path = data_dir / name
        stat = path.stat()
        digest.update(f"{name}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()[:16]


def benchmark_and_export(data_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    from dashboard.torch_backend import export_numpy_states, predict_mlp, train_mlp
    frame, numeric, categorical = build_modeling_dataset(data_dir)
    features = numeric + categorical
    partitions = grouped_split(frame, features)
    development = pd.concat([partitions["train"], partitions["evaluation"]], axis=0)
    test = partitions["test"].copy()
    x_dev, y_dev = development[features], development["target_dropout"].to_numpy()
    x_test, y_test = test[features], test["target_dropout"].to_numpy()

    preprocessor = make_preprocessor(numeric, categorical)
    dev_matrix = preprocessor.fit_transform(x_dev)
    test_matrix = preprocessor.transform(x_test)
    assert dev_matrix.shape[1] == 93

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    lr.fit(dev_matrix, y_dev)
    hgb = HistGradientBoostingClassifier(
        loss="log_loss", random_state=42, learning_rate=0.08, max_iter=200
    )
    hgb_weights = np.where(y_dev == 1, 3.0, 1.0).astype(float)
    hgb_weights /= hgb_weights.mean()
    hgb.fit(dev_matrix, y_dev, sample_weight=hgb_weights)
    mlp_models = [train_mlp(dev_matrix, y_dev, seed) for seed in [42, 43, 44]]
    mlp_states = export_numpy_states(mlp_models)

    probabilities = {
        "logistic_regression": lr.predict_proba(test_matrix)[:, 1],
        "histogram_boosting": hgb.predict_proba(test_matrix)[:, 1],
        "neural_network": predict_mlp(mlp_models, test_matrix),
    }
    metrics = pd.DataFrame([
        metric_record(model_id, y_test, probability)
        for model_id, probability in probabilities.items()
    ])
    curves = {
        model_id: curve_record(y_test, probability)
        for model_id, probability in probabilities.items()
    }
    predictions = pd.DataFrame({
        "case_id": test.apply(_case_id, axis=1).to_numpy(),
        "actual": y_test,
        **{f"prob_{model_id}": probability for model_id, probability in probabilities.items()},
    })
    cases = select_curated_cases(predictions)
    case_positions = {case_id: position for position, case_id in enumerate(predictions["case_id"])}
    case_records = []
    for _, case in cases.iterrows():
        position = case_positions[case["case_id"]]
        raw = test.iloc[position][features].to_dict()
        raw.update({
            "case_id": case["case_id"], "case_type": case["case_type"],
            "case_title": case["case_title"], "display_order": int(case["display_order"]),
            "actual": int(case["actual"]),
            **{f"prob_{model_id}": float(case[f"prob_{model_id}"]) for model_id in MODEL_LABELS},
        })
        case_records.append(raw)

    partition_audit = []
    for name, part in partitions.items():
        partition_audit.append({
            "partition": name.title(), "rows": len(part),
            "unique_students": int(part["id_student"].nunique()),
            "withdrawal_prevalence": float(part["target_dropout"].mean()),
        })
    bundle = {
        "artifact_version": "1.1.0",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "dataset_fingerprint": dataset_fingerprint(data_dir),
        "feature_columns": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "encoded_feature_count": int(dev_matrix.shape[1]),
        "threshold": THRESHOLD,
        "preprocessor": preprocessor,
        "sklearn_models": {"logistic_regression": lr, "histogram_boosting": hgb},
        "metrics": metrics,
        "curves": curves,
        "predictions": predictions,
        "case_records": pd.DataFrame(case_records).sort_values("display_order"),
        "subgroup_metrics": subgroup_records(test, probabilities),
        "feature_reference": feature_references(development, numeric, categorical),
        "feature_groups": {
            group: [feature for feature in group_features if feature in features]
            for group, group_features in FEATURE_GROUPS.items()
        },
        "what_if_features": WHAT_IF_FEATURES,
        "input_ranges": input_ranges(development, WHAT_IF_FEATURES),
        "partition_audit": pd.DataFrame(partition_audit),
        "model_config": {
            "logistic_regression": "Balanced class weights; linear log-odds; max_iter=1000",
            "histogram_boosting": "Withdrawal weight=3.0; learning_rate=0.08; max_iter=200",
            "neural_network": "93→128→64→32→1; dropout=0.20; 3-seed mean; 9 epochs",
        },
    }
    joblib.dump(bundle, artifact_dir / "dashboard_bundle.joblib", compress=3)
    joblib.dump(mlp_states, artifact_dir / "mlp_states.joblib", compress=3)
    metrics.to_csv(artifact_dir / "benchmark_metrics.csv", index=False)
    summary = {
        "artifact_version": bundle["artifact_version"],
        "dataset_fingerprint": bundle["dataset_fingerprint"],
        "rows": len(frame), "features": len(features), "encoded_features": dev_matrix.shape[1],
        "metrics": metrics.set_index("model_id").to_dict(orient="index"),
    }
    (artifact_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    return summary


@dataclass
class RuntimeModels:
    bundle: dict[str, Any]
    mlp_states: list[dict[str, np.ndarray]]

    @classmethod
    def load(cls, artifact_dir: Path) -> "RuntimeModels":
        bundle = joblib.load(artifact_dir / "dashboard_bundle.joblib")
        states = joblib.load(artifact_dir / "mlp_states.joblib")
        return cls(bundle=bundle, mlp_states=states)

    def predict(self, model_id: str, raw_frame: pd.DataFrame) -> tuple[np.ndarray, float]:
        ordered = raw_frame[self.bundle["feature_columns"]]
        matrix = self.bundle["preprocessor"].transform(ordered)
        started = time.perf_counter()
        if model_id == "neural_network":
            probability = predict_mlp_numpy(self.mlp_states, matrix)
        else:
            probability = self.bundle["sklearn_models"][model_id].predict_proba(matrix)[:, 1]
        latency_ms = (time.perf_counter() - started) * 1000
        return probability, latency_ms

    def sensitivities(self, model_id: str, raw_frame: pd.DataFrame) -> pd.DataFrame:
        original, _ = self.predict(model_id, raw_frame)
        rows = []
        for group, features in self.bundle["feature_groups"].items():
            comparison = raw_frame.copy()
            for feature in features:
                comparison.loc[:, feature] = self.bundle["feature_reference"][feature]
            reference_probability, _ = self.predict(model_id, comparison)
            rows.append({
                "feature_group": group,
                "probability_delta": float(original[0] - reference_probability[0]),
            })
        return pd.DataFrame(rows).sort_values("probability_delta", key=lambda s: s.abs(), ascending=False)
