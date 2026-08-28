from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.modeling import (
    MODEL_LABELS,
    PRIMARY_FAIRNESS_ATTRIBUTES,
    RuntimeModels,
    fairness_summary,
    select_curated_cases,
    subgroup_records,
)


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


class CuratedCaseTests(unittest.TestCase):
    def test_case_selection_is_unique_and_includes_boundary(self) -> None:
        rows = []
        for index in range(20):
            actual = index % 2
            rows.append({
                "case_id": f"CASE-{index:02d}", "actual": actual,
                "prob_logistic_regression": min(.95, .08 + index * .045),
                "prob_histogram_boosting": min(.98, .05 + index * .052),
                "prob_neural_network": min(.96, .10 + index * .043),
            })
        cases = select_curated_cases(pd.DataFrame(rows))
        self.assertTrue(cases["case_id"].is_unique)
        self.assertIn("borderline", set(cases["case_type"]))


class FairnessSummaryTests(unittest.TestCase):
    def test_calculates_notebook_dpd_and_equal_opportunity_difference(self) -> None:
        subgroup = pd.DataFrame([
            {
                "model_id": "model_a", "model": "Model A", "attribute": "gender", "group": "F",
                "selection_rate": 0.25, "tpr_recall": 0.80, "fpr": 0.10, "eligible": True,
            },
            {
                "model_id": "model_a", "model": "Model A", "attribute": "gender", "group": "M",
                "selection_rate": 0.40, "tpr_recall": 0.65, "fpr": 0.90, "eligible": True,
            },
            {
                "model_id": "model_a", "model": "Model A", "attribute": "gender", "group": "Unknown",
                "selection_rate": 0.95, "tpr_recall": 0.10, "fpr": 0.95, "eligible": False,
            },
        ])

        result = fairness_summary(subgroup).iloc[0]

        self.assertAlmostEqual(result["dpd"], 0.15)
        self.assertAlmostEqual(result["eod"], 0.15)
        self.assertEqual(result["eligible_groups"], 2)
        self.assertEqual(result["lowest_selection_group"], "F")
        self.assertEqual(result["highest_selection_group"], "M")
        self.assertEqual(result["lowest_recall_group"], "M")
        self.assertEqual(result["highest_recall_group"], "F")

    def test_returns_nan_when_fewer_than_two_groups_are_available(self) -> None:
        subgroup = pd.DataFrame([{
            "model_id": "model_a", "model": "Model A", "attribute": "gender", "group": "F",
            "selection_rate": 0.25, "tpr_recall": 0.80, "eligible": True,
        }])

        result = fairness_summary(subgroup).iloc[0]

        self.assertTrue(np.isnan(result["dpd"]))
        self.assertTrue(np.isnan(result["eod"]))

    def test_group_eligibility_and_unknown_imd_exclusion(self) -> None:
        rows = []
        for group, positives, negatives, imd_band in [
            ("F", 30, 30, "0-10%"),
            ("M", 29, 30, np.nan),
        ]:
            for actual in ([1] * positives + [0] * negatives):
                rows.append({
                    "target_dropout": actual, "gender": group, "disability": "N",
                    "imd_band": imd_band,
                })
        frame = pd.DataFrame(rows)
        probabilities = {model_id: np.full(len(frame), 0.6) for model_id in MODEL_LABELS}

        groups = subgroup_records(frame, probabilities)
        logistic = groups[groups["model_id"] == "logistic_regression"]

        gender = logistic[logistic["attribute"] == "gender"].set_index("group")
        self.assertTrue(gender.loc["F", "eligible"])
        self.assertFalse(gender.loc["M", "eligible"])
        imd = logistic[logistic["attribute"] == "imd_band"].set_index("group")
        self.assertFalse(imd.loc["Unknown", "eligible"])


@unittest.skipUnless((ARTIFACT_DIR / "dashboard_bundle.joblib").exists(), "Artifacts not built")
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = RuntimeModels.load(ARTIFACT_DIR)

    def test_artifact_contract(self) -> None:
        bundle = self.runtime.bundle
        self.assertEqual(len(bundle["feature_columns"]), 54)
        self.assertEqual(bundle["encoded_feature_count"], 93)
        self.assertEqual(set(bundle["metrics"]["model_id"]), set(MODEL_LABELS))
        self.assertEqual(set(bundle["sklearn_models"]), {"logistic_regression", "histogram_boosting"})
        self.assertEqual(bundle["artifact_version"], "1.1.0")
        subgroup = bundle["subgroup_metrics"]
        self.assertEqual(set(subgroup["attribute"]), set(PRIMARY_FAIRNESS_ATTRIBUTES))
        self.assertTrue({
            "actual_withdrawn", "actual_not_withdrawn", "withdrawal_rate",
            "selection_rate", "tpr_recall", "fnr", "fpr", "precision",
            "accuracy", "eligible",
        }.issubset(subgroup.columns))

    def test_curated_probabilities_match_loaded_models(self) -> None:
        bundle = self.runtime.bundle
        case = bundle["case_records"].iloc[[0]]
        raw = case[bundle["feature_columns"]]
        for model_id in MODEL_LABELS:
            probability, _ = self.runtime.predict(model_id, raw)
            expected = float(case.iloc[0][f"prob_{model_id}"])
            self.assertTrue(np.isclose(probability[0], expected, atol=1e-6))

    def test_sensitivity_contract(self) -> None:
        bundle = self.runtime.bundle
        case = bundle["case_records"].iloc[[0]][bundle["feature_columns"]]
        result = self.runtime.sensitivities("histogram_boosting", case)
        self.assertEqual(set(result["feature_group"]), set(bundle["feature_groups"]))
        self.assertTrue(np.isfinite(result["probability_delta"]).all())


if __name__ == "__main__":
    unittest.main()
