from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.modeling import MODEL_LABELS, RuntimeModels, select_curated_cases


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
