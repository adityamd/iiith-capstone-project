from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.modeling import (
    MODEL_COLORS,
    MODEL_LABELS,
    PRIMARY_FAIRNESS_ATTRIBUTES,
    RuntimeModels,
    fairness_summary,
)


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
LABELS = {
    "active_days_25pct": "Active learning days",
    "total_clicks_25pct": "Total VLE clicks",
    "days_since_last_activity_at_checkpoint": "Days since last activity",
    "early_assessments_missing_count": "Missing early assessments",
    "mean_early_score": "Mean early assessment score",
    "studied_credits": "Studied credits",
    "num_of_prev_attempts": "Previous attempts",
    "average_submission_delay": "Average submission delay",
}

st.set_page_config(
    page_title="OULAD Model Inference Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root { --ink:#152238; --muted:#65758B; --paper:#F5F7FA; --line:#DDE4ED; --navy:#162B4D; --blue:#4F7CAC; --coral:#E07A5F; --green:#6A994E; }
.stApp { background: linear-gradient(145deg, #F8FAFD 0%, #F3F6FA 48%, #EEF3F7 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-0.025em; color:var(--ink) !important; }
[data-testid="stSidebar"] { background:#12233F; }
[data-testid="stSidebar"] * { color:#EEF4FF !important; }
[data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] * { color:#17243A !important; }
[data-testid="stMetric"] { background:#FFFFFF; border:1px solid var(--line); border-radius:16px; padding:18px; box-shadow:0 6px 20px rgba(31,48,73,.05); }
[data-testid="stMetricLabel"] { color:var(--muted); }
.hero { background:linear-gradient(115deg,#142846 0%,#1D3D67 62%,#315A7D 100%); border-radius:24px; padding:30px 34px; color:#F7FAFF; margin-bottom:20px; box-shadow:0 16px 42px rgba(18,43,77,.18); }
.hero h1 { color:#FFFFFF !important; margin:0 0 8px 0; font-size:2.15rem; }
.hero p { color:#C9D8EB; margin:0; font-size:1.02rem; max-width:900px; }
.eyebrow { color:#8FD4C1; text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:700; margin-bottom:8px; }
.model-card { background:#FFFFFF; border:1px solid var(--line); border-top:5px solid var(--model-color); border-radius:18px; padding:20px; min-height:190px; box-shadow:0 8px 22px rgba(33,48,72,.06); }
.model-name { color:var(--muted); font-size:.78rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.probability { font-family:'Manrope',sans-serif; font-size:2.25rem; font-weight:800; color:var(--ink); line-height:1.15; margin:10px 0 5px; }
.decision { display:inline-block; border-radius:999px; padding:5px 10px; font-weight:700; font-size:.76rem; }
.decision-high { background:#FBE8E2; color:#A9412B; }
.decision-low { background:#E8F2E4; color:#3F6D2F; }
.card-detail { color:var(--muted); font-size:.8rem; margin-top:12px; line-height:1.55; }
.pending { color:#9AA6B5; font-size:1.8rem; font-family:'Manrope',sans-serif; margin:22px 0 7px; }
.callout { background:#FFF8EF; border:1px solid #F3D7AE; border-left:5px solid #D99A45; padding:14px 16px; border-radius:12px; color:#66491F; margin:10px 0 18px; }
.case-strip { background:#FFFFFF; border:1px solid var(--line); border-radius:16px; padding:17px 20px; margin-bottom:14px; }
.tiny { color:var(--muted); font-size:.78rem; }
.stButton > button { border-radius:11px; font-weight:700; min-height:44px; border:1px solid #CDD7E3; }
.stButton > button[kind="primary"] { background:#183A61; border-color:#183A61; }
div[data-testid="stTabs"] button { font-weight:700; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained models…")
def load_runtime() -> RuntimeModels:
    return RuntimeModels.load(ARTIFACT_DIR)


def record_signature(frame: pd.DataFrame) -> str:
    return hashlib.sha1(frame.to_json().encode("utf-8")).hexdigest()


def model_card(model_id: str, result: dict | None, actual: int) -> None:
    color = MODEL_COLORS[model_id]
    label = MODEL_LABELS[model_id]
    if result is None:
        body = "<div class='pending'>Not run yet</div><div class='card-detail'>Use the model button to reveal this prediction.</div>"
    else:
        probability = result["probability"]
        predicted = result["prediction"]
        risk_class = "decision-high" if predicted else "decision-low"
        decision = "Withdrawal risk" if predicted else "Likely to continue"
        correct = "Matches outcome" if predicted == actual else "Differs from outcome"
        body = (
            f"<div class='probability'>{probability:.1%}</div>"
            f"<span class='decision {risk_class}'>{decision}</span>"
            f"<div class='card-detail'>{correct}<br>"
            f"Distance from threshold: {abs(probability - result['threshold']):.1%}<br>"
            f"Inference: {result['latency_ms']:.2f} ms</div>"
        )
    st.markdown(
        f"<div class='model-card' style='--model-color:{color}'>"
        f"<div class='model-name'>{label}</div>{body}</div>",
        unsafe_allow_html=True,
    )


def probability_chart(results: dict[str, dict | None], threshold: float) -> go.Figure:
    rows = [
        {"model": MODEL_LABELS[model_id], "probability": result["probability"], "model_id": model_id}
        for model_id, result in results.items() if result is not None
    ]
    figure = go.Figure()
    for row in rows:
        figure.add_bar(
            x=[row["probability"]], y=[row["model"]], orientation="h",
            marker_color=MODEL_COLORS[row["model_id"]], name=row["model"],
            text=[f"{row['probability']:.1%}"], textposition="outside", showlegend=False,
        )
    figure.add_vline(x=threshold, line_dash="dash", line_color="#A9412B", annotation_text="Decision threshold")
    figure.update_layout(
        height=270, margin=dict(l=10, r=30, t=25, b=15), xaxis_tickformat=".0%",
        xaxis_range=[0, 1.02], xaxis_title="Predicted withdrawal probability",
        yaxis_title=None, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def curve_figure(curves: dict, kind: str) -> go.Figure:
    figure = go.Figure()
    for model_id, curve in curves.items():
        if kind == "roc":
            x, y = curve["roc_fpr"], curve["roc_tpr"]
            x_title, y_title = "False-positive rate", "True-positive rate"
        elif kind == "pr":
            x, y = curve["pr_recall"], curve["pr_precision"]
            x_title, y_title = "Recall", "Precision"
        else:
            x, y = curve["calibration_predicted"], curve["calibration_observed"]
            x_title, y_title = "Mean predicted probability", "Observed withdrawal rate"
        figure.add_scatter(
            x=x, y=y, mode="lines+markers" if kind == "calibration" else "lines",
            name=MODEL_LABELS[model_id], line=dict(color=MODEL_COLORS[model_id], width=3),
        )
    if kind in {"roc", "calibration"}:
        figure.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#AAB4C2", dash="dot"), name="Reference")
    figure.update_layout(
        height=390, margin=dict(l=25, r=15, t=30, b=25), xaxis_title=x_title,
        yaxis_title=y_title, xaxis_range=[0, 1], yaxis_range=[0, 1],
        legend=dict(orientation="h", y=1.12), plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def comparison_story(case_row: pd.Series) -> str:
    probabilities = {model_id: float(case_row[f"prob_{model_id}"]) for model_id in MODEL_LABELS}
    highest = max(probabilities, key=probabilities.get)
    lowest = min(probabilities, key=probabilities.get)
    spread = probabilities[highest] - probabilities[lowest]
    return (
        f"{MODEL_LABELS[highest]} assigns the highest risk and {MODEL_LABELS[lowest]} the lowest, "
        f"a {spread:.1%} probability spread. This is a model-behavior comparison, not a causal claim."
    )


if not (ARTIFACT_DIR / "dashboard_bundle.joblib").exists():
    st.error("Dashboard artifacts are missing. Run `dashboard/train_dashboard_models.py` in the MLDL environment first.")
    st.stop()

runtime = load_runtime()
bundle = runtime.bundle
cases = bundle["case_records"].copy()
metrics = bundle["metrics"].copy()
threshold = float(bundle["threshold"])

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">OULAD · 25% course checkpoint</div>
  <h1>Student withdrawal inference lab</h1>
  <p>Three models see the same early-learning evidence. Compare where their probabilities agree, where their decisions split, and what that means for intervention.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Demo controls")
    case_options = dict(zip(cases["case_title"], cases.index))
    chosen_title = st.selectbox("Curated student case", list(case_options), index=0)
    selected = cases.loc[case_options[chosen_title]].copy()
    st.caption(f"{selected['case_id']} · anonymized held-out record")
    actual_label = "Withdrawn" if int(selected["actual"]) else "Not withdrawn"
    st.markdown(f"**Known outcome:** {actual_label}")
    what_if = st.toggle("Enable what-if editing", value=False)
    st.markdown("### Decision threshold")
    decision_threshold = st.slider(
        "Shared what-if threshold", 0.20, 0.80, threshold, 0.01,
        help="Published benchmark metrics remain fixed at 0.50.",
    )
    if decision_threshold != threshold:
        st.caption("What-if threshold active. Benchmark charts still use 0.50.")
    st.divider()
    st.caption(
        f"Artifact {bundle['artifact_version']} · {len(bundle['feature_columns'])} raw features · "
        f"{bundle['encoded_feature_count']} encoded inputs"
    )

raw_record = pd.DataFrame([{column: selected[column] for column in bundle["feature_columns"]}])
if what_if:
    with st.expander("Adjust interpretable features", expanded=True):
        st.caption("Changes create a synthetic what-if profile. All other fields remain anchored to the selected record.")
        columns = st.columns(2)
        for position, feature in enumerate(bundle["what_if_features"]):
            range_info = bundle["input_ranges"][feature]
            value = raw_record.iloc[0][feature]
            if pd.isna(value):
                value = bundle["feature_reference"][feature]
            kwargs = {
                "label": LABELS.get(feature, feature),
                "min_value": float(range_info["min"]),
                "max_value": float(range_info["max"]),
                "value": float(value),
                "step": float(range_info["step"]),
                "key": f"what_if_{selected['case_id']}_{feature}",
            }
            changed = columns[position % 2].number_input(**kwargs)
            raw_record.loc[0, feature] = changed

signature = record_signature(raw_record) + f"|threshold={decision_threshold:.2f}"
if st.session_state.get("record_signature") != signature:
    st.session_state["record_signature"] = signature
    st.session_state["model_results"] = {model_id: None for model_id in MODEL_LABELS}
results = st.session_state.setdefault("model_results", {model_id: None for model_id in MODEL_LABELS})

tab_inference, tab_performance, tab_differences, tab_gallery = st.tabs([
    "Inference lab", "Performance", "Prediction differences", "Case gallery",
])

with tab_inference:
    st.markdown(
        f"<div class='case-strip'><strong>{selected['case_title']}</strong><br>"
        f"<span class='tiny'>{comparison_story(selected)}</span></div>",
        unsafe_allow_html=True,
    )
    button_columns = st.columns([1, 1, 1, .8, .65])
    clicked = {}
    for position, model_id in enumerate(MODEL_LABELS):
        clicked[model_id] = button_columns[position].button(
            f"Run {MODEL_LABELS[model_id]}", width="stretch",
            type="primary" if model_id == "histogram_boosting" else "secondary",
        )
    run_all = button_columns[3].button("Run all", width="stretch")
    reset = button_columns[4].button("Reset", width="stretch")
    if reset:
        st.session_state["model_results"] = {model_id: None for model_id in MODEL_LABELS}
        results = st.session_state["model_results"]
    for model_id in MODEL_LABELS:
        if clicked[model_id] or run_all:
            with st.spinner(f"Running {MODEL_LABELS[model_id]}…"):
                probability, latency_ms = runtime.predict(model_id, raw_record)
                results[model_id] = {
                    "probability": float(probability[0]),
                    "prediction": int(probability[0] >= decision_threshold),
                    "threshold": decision_threshold,
                    "latency_ms": latency_ms,
                }
    card_columns = st.columns(3)
    for position, model_id in enumerate(MODEL_LABELS):
        with card_columns[position]:
            model_card(model_id, results[model_id], int(selected["actual"]))
    if any(result is not None for result in results.values()):
        st.plotly_chart(probability_chart(results, decision_threshold), width="stretch")
        completed = [model_id for model_id, result in results.items() if result is not None]
        explanation_model = st.selectbox(
            "Explain model sensitivity", completed,
            format_func=lambda model_id: MODEL_LABELS[model_id],
        )
        sensitivity = runtime.sensitivities(explanation_model, raw_record)
        sensitivity["direction"] = np.where(
            sensitivity["probability_delta"] >= 0, "Raises risk", "Lowers risk"
        )
        figure = px.bar(
            sensitivity, x="probability_delta", y="feature_group", orientation="h",
            color="direction", color_discrete_map={"Raises risk": "#E07A5F", "Lowers risk": "#6A994E"},
            labels={"probability_delta": "Probability change vs training reference", "feature_group": ""},
        )
        figure.update_layout(
            height=330, margin=dict(l=10, r=20, t=20, b=20),
            xaxis_tickformat="+.1%", legend_title=None, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "Sensitivity replaces one feature group with its training reference and re-scores the case. "
            "It describes model behavior, not why a student withdrew."
        )
    else:
        st.info("Run one model at a time to build the comparison, or use Run all.")

with tab_performance:
    best_recall = metrics.sort_values("recall", ascending=False).iloc[0]
    best_pr = metrics.sort_values("pr_auc", ascending=False).iloc[0]
    fewest_fn = metrics.sort_values("fn").iloc[0]
    top_metrics = st.columns(4)
    top_metrics[0].metric("Test enrollments", f"{int(metrics.iloc[0].tn + metrics.iloc[0].fp + metrics.iloc[0].fn + metrics.iloc[0].tp):,}")
    top_metrics[1].metric("Best PR-AUC", f"{best_pr.pr_auc:.3f}", best_pr.model)
    top_metrics[2].metric("Best withdrawal recall", f"{best_recall.recall:.3f}", best_recall.model)
    top_metrics[3].metric("Fewest false negatives", f"{int(fewest_fn.fn):,}", fewest_fn.model)
    st.markdown(
        "<div class='callout'><strong>Read the trade-off:</strong> Similar ranking performance does not imply identical intervention lists. "
        "For this project, false negatives are students who withdrew but would not have been flagged at the 0.50 threshold.</div>",
        unsafe_allow_html=True,
    )
    display_metrics = metrics[[
        "model", "pr_auc", "roc_auc", "balanced_accuracy", "precision", "recall",
        "f1", "fp", "fn",
    ]].rename(columns={
        "model": "Model", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
        "balanced_accuracy": "Balanced accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1",
        "fp": "False positives", "fn": "False negatives",
    })
    st.dataframe(
        display_metrics.style.format({
            "PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}", "Balanced accuracy": "{:.3f}",
            "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}",
        }).highlight_max(subset=["PR-AUC", "ROC-AUC", "Balanced accuracy", "Precision", "Recall", "F1"], color="#DDEED6")
        .highlight_min(subset=["False positives", "False negatives"], color="#DDEED6"),
        width="stretch", hide_index=True,
    )
    st.markdown("### Fairness summary")
    st.markdown(
        "<div class='callout'><strong>How to read these fairness gaps:</strong> "
        "DPD compares how often eligible groups are flagged. EOD compares how often actual "
        "withdrawals are detected. A value of 0 means no measured gap; larger values mean "
        "larger differences. Groups need at least 30 withdrawals and 30 non-withdrawals to "
        "define either gap.</div>",
        unsafe_allow_html=True,
    )
    fairness = fairness_summary(bundle["subgroup_metrics"]).rename(columns={
        "model": "Model",
        "attribute": "Audit attribute",
        "eligible_groups": "Eligible groups",
        "dpd": "DPD",
        "eod": "EOD",
        "lowest_selection_group": "Lowest alert group",
        "highest_selection_group": "Highest alert group",
        "lowest_recall_group": "Lowest recall group",
        "highest_recall_group": "Highest recall group",
    })
    attribute_labels = {
        "gender": "Gender",
        "disability": "Disability",
        "imd_band": "IMD band",
        "gender_x_disability": "Gender × disability",
        "disability_x_deprivation": "Disability × deprivation",
    }
    fairness["Audit attribute"] = fairness["Audit attribute"].map(attribute_labels)
    fairness = fairness[[
        "Model", "Audit attribute", "Eligible groups", "DPD", "Lowest alert group",
        "Highest alert group", "EOD", "Lowest recall group", "Highest recall group",
    ]]
    st.dataframe(
        fairness.style.format({"DPD": "{:.3f}", "EOD": "{:.3f}"}),
        width="stretch", hide_index=True,
    )
    st.caption(
        "DPD is Demographic Parity Difference: highest minus lowest eligible-group selection rate. "
        "EOD is Equal Opportunity Difference: highest minus lowest eligible-group withdrawal recall. "
        "These descriptive test-set checks do not establish fairness or discrimination."
    )
    with st.expander("Group-level fairness audit"):
        detail = bundle["subgroup_metrics"].copy()
        selector_columns = st.columns(2)
        selected_fairness_model = selector_columns[0].selectbox(
            "Model",
            options=list(MODEL_LABELS),
            format_func=lambda model_id: MODEL_LABELS[model_id],
            key="fairness_detail_model",
        )
        selected_fairness_attribute = selector_columns[1].selectbox(
            "Audit attribute",
            options=PRIMARY_FAIRNESS_ATTRIBUTES,
            format_func=lambda attribute: attribute_labels[attribute],
            key="fairness_detail_attribute",
        )
        detail = detail[
            (detail["model_id"] == selected_fairness_model)
            & (detail["attribute"] == selected_fairness_attribute)
        ][[
            "group", "records", "actual_withdrawn", "actual_not_withdrawn",
            "withdrawal_rate", "selection_rate", "tpr_recall", "fnr", "fpr",
            "precision", "accuracy", "eligible",
        ]].rename(columns={
            "group": "Group", "records": "Records",
            "actual_withdrawn": "Withdrawals",
            "actual_not_withdrawn": "Non-withdrawals",
            "withdrawal_rate": "Withdrawal rate", "selection_rate": "Alert rate",
            "tpr_recall": "Recall / TPR", "fnr": "FNR", "fpr": "FPR",
            "precision": "Precision", "accuracy": "Accuracy", "eligible": "Eligible",
        })
        st.dataframe(
            detail.style.format({
                "Withdrawal rate": "{:.1%}", "Alert rate": "{:.1%}",
                "Recall / TPR": "{:.1%}", "FNR": "{:.1%}", "FPR": "{:.1%}",
                "Precision": "{:.1%}", "Accuracy": "{:.1%}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Ineligible groups remain visible for context but do not contribute to DPD or EOD. "
            "Unknown IMD categories are always excluded from those gap calculations."
        )
    curve_columns = st.columns(2)
    curve_columns[0].plotly_chart(curve_figure(bundle["curves"], "pr"), width="stretch")
    curve_columns[1].plotly_chart(curve_figure(bundle["curves"], "roc"), width="stretch")
    st.markdown("### Calibration")
    calibration_columns = st.columns([1.4, 1])
    calibration_columns[0].plotly_chart(curve_figure(bundle["curves"], "calibration"), width="stretch")
    calibration_columns[1].markdown(
        "**Why calibration matters**\n\nPR-AUC and ROC-AUC measure ranking. The calibration curve asks whether a displayed 70% risk "
        "corresponds to withdrawal about 70% of the time. Lower Brier score is better."
    )
    with st.expander("Data partition audit"):
        st.dataframe(bundle["partition_audit"], hide_index=True, width="stretch")
        st.caption("Student overlap across training, evaluation, and test partitions is zero.")

with tab_differences:
    predictions = bundle["predictions"].copy()
    probability_columns = {f"prob_{model_id}": MODEL_LABELS[model_id] for model_id in MODEL_LABELS}
    melted = predictions.melt(
        id_vars=["case_id", "actual"], value_vars=list(probability_columns),
        var_name="model_key", value_name="probability",
    )
    melted["model"] = melted["model_key"].map(probability_columns)
    histogram = px.histogram(
        melted, x="probability", color="model", barmode="overlay", nbins=35,
        color_discrete_map={MODEL_LABELS[key]: value for key, value in MODEL_COLORS.items()},
        labels={"probability": "Predicted withdrawal probability", "count": "Test enrollments"},
    )
    histogram.add_vline(x=threshold, line_dash="dash", line_color="#A9412B")
    histogram.update_layout(height=390, legend_title=None, xaxis_tickformat=".0%")
    st.plotly_chart(histogram, width="stretch")

    pair_columns = st.columns(3)
    pairs = [
        ("logistic_regression", "histogram_boosting"),
        ("logistic_regression", "neural_network"),
        ("histogram_boosting", "neural_network"),
    ]
    for column, (left, right) in zip(pair_columns, pairs):
        left_probability = predictions[f"prob_{left}"]
        right_probability = predictions[f"prob_{right}"]
        agreement = ((left_probability >= threshold) == (right_probability >= threshold)).mean()
        mean_delta = (left_probability - right_probability).abs().mean()
        column.metric(f"{MODEL_LABELS[left]} ↔ {MODEL_LABELS[right]}", f"{agreement:.1%} agreement", f"Mean |Δ| {mean_delta:.1%}")

    scatter_columns = st.columns(2)
    x_id, y_id = "histogram_boosting", "neural_network"
    scatter = px.scatter(
        predictions, x=f"prob_{x_id}", y=f"prob_{y_id}", color=predictions["actual"].map({0: "Not withdrawn", 1: "Withdrawn"}),
        opacity=.45, color_discrete_map={"Not withdrawn": "#4F7CAC", "Withdrawn": "#E07A5F"},
        labels={f"prob_{x_id}": MODEL_LABELS[x_id], f"prob_{y_id}": MODEL_LABELS[y_id], "color": "Outcome"},
    )
    scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="#AAB4C2", dash="dot"))
    scatter.add_vline(x=threshold, line_dash="dash", line_color="#D6A8A0")
    scatter.add_hline(y=threshold, line_dash="dash", line_color="#D6A8A0")
    scatter.update_layout(height=440, xaxis_tickformat=".0%", yaxis_tickformat=".0%", legend_title=None)
    scatter_columns[0].plotly_chart(scatter, width="stretch")
    scatter_columns[1].markdown(
        "### Why probabilities separate\n\n"
        "- **Logistic Regression** adds linear feature effects. It is transparent, but sharp thresholds and interactions must be specified explicitly.\n\n"
        "- **Histogram Boosting** learns threshold-like rules and interactions, such as disengagement combined with missed assessments.\n\n"
        "- **Neural Network** learns nonlinear combinations, but the same 54 aggregate predictors limit the extra signal it can recover.\n\n"
        "The models can therefore have similar PR-AUC while flagging different individual students."
    )

with tab_gallery:
    st.markdown("### Evidence-selected cases")
    st.caption("Cases are selected deterministically from held-out predictions by correctness, decision disagreement, and probability spread.")
    for _, case in cases.sort_values("display_order").iterrows():
        probabilities = pd.DataFrame({
            "Model": list(MODEL_LABELS.values()),
            "Withdrawal probability": [float(case[f"prob_{model_id}"]) for model_id in MODEL_LABELS],
            "Color": list(MODEL_COLORS.values()),
        })
        with st.expander(
            f"{int(case['display_order'])}. {case['case_title']} · {case['case_id']}",
            expanded=int(case["display_order"]) == 1,
        ):
            left, right = st.columns([1.1, 1.7])
            left.markdown(f"**Known outcome:** {'Withdrawn' if int(case['actual']) else 'Not withdrawn'}")
            left.markdown(comparison_story(case))
            left.caption("Predictions use the common 0.50 benchmark threshold.")
            gallery_figure = go.Figure(go.Bar(
                x=probabilities["Withdrawal probability"], y=probabilities["Model"],
                orientation="h", marker_color=probabilities["Color"],
                text=[f"{value:.1%}" for value in probabilities["Withdrawal probability"]], textposition="outside",
            ))
            gallery_figure.add_vline(x=threshold, line_dash="dash", line_color="#A9412B")
            gallery_figure.update_layout(
                height=250, margin=dict(l=10, r=30, t=10, b=10), xaxis_range=[0, 1.02],
                xaxis_tickformat=".0%", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            right.plotly_chart(gallery_figure, width="stretch")

st.caption(
    "Decision-support demonstration only · Predictions indicate model-estimated association, not student intent or causation · "
    f"Dataset fingerprint {bundle['dataset_fingerprint']}"
)
