"""Automated tests for V1 contract, notebook parity, input validation, and HTTP endpoints."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import ChurnAppState, create_server
from export_model_from_notebook import EXPECTED_TEST_CM, EXPECTED_TEST_METRICS, METRIC_ATOL, run_notebook_dependency_chain
from validator import InputValidationError, validate_and_build_dataframe


class ChurnFrontendBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract_path = PROJECT_ROOT / "artifacts" / "model_contract.json"
        cls.model_path = PROJECT_ROOT / "artifacts" / "netflix_churn_rf_pipeline.joblib"
        with open(cls.contract_path, "r", encoding="utf-8") as f:
            cls.contract = json.load(f)
        cls.model = joblib.load(cls.model_path)

        cls.server, cls.state = create_server(host="127.0.0.1", port=0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def _request(self, method: str, path: str, body: dict | str | None = None) -> tuple[int, dict | str]:
        url = f"http://127.0.0.1:{self.port}{path}"
        data = None
        headers = {}
        if body is not None:
            raw = body if isinstance(body, str) else json.dumps(body)
            data = raw.encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=5) as resp:
                raw_resp = resp.read().decode("utf-8")
                ctype = resp.headers.get("Content-Type", "")
                parsed = json.loads(raw_resp) if "application/json" in ctype else raw_resp
                return resp.status, parsed
        except HTTPError as err:
            raw_err = err.read().decode("utf-8")
            return err.code, json.loads(raw_err)

    def _valid_payload(self) -> dict:
        # Synthetic test payload from contract presets
        return dict(self.contract["synthetic_presets"]["moderate_risk_demo"]["inputs"])

    def test_01_exported_model_matches_notebook_x_test(self) -> None:
        ns = run_notebook_dependency_chain()
        X_test = ns["X_test"]
        y_test = ns["y_test"]
        assert list(X_test.columns) == self.contract["feature_order"]

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        cm = ns["confusion_matrix"] if "confusion_matrix" in ns else __import__("sklearn.metrics").metrics.confusion_matrix
        actual_cm = cm(y_test, y_pred)
        self.assertTrue(np.array_equal(actual_cm, EXPECTED_TEST_CM))

        acc = ns["accuracy_score"](y_test, y_pred)
        prec = ns["precision_score"](y_test, y_pred, zero_division=0)
        rec = ns["recall_score"](y_test, y_pred, zero_division=0)
        f1 = ns["f1_score"](y_test, y_pred, zero_division=0)
        roc = ns["roc_auc_score"](y_test, y_prob)
        ap = ns["average_precision_score"](y_test, y_prob)
        trap = ns["pr_auc_trap_func"](y_test, y_prob)

        for name, val in [
            ("accuracy", acc),
            ("precision", prec),
            ("recall", rec),
            ("f1", f1),
            ("roc_auc", roc),
            ("average_precision", ap),
            ("pr_auc_trapezoidal", trap),
        ]:
            self.assertAlmostEqual(val, EXPECTED_TEST_METRICS[name], delta=METRIC_ATOL)

    def test_02_contract_24_features_and_independent_rating_liked(self) -> None:
        payload = self._valid_payload()
        # Verify rating=1 and liked="Yes" work independently without forced mapping
        payload["rating"] = 1
        payload["liked"] = "Yes"
        df_row, derived = validate_and_build_dataframe(payload, self.contract)
        self.assertEqual(list(df_row.columns), self.contract["feature_order"])
        self.assertEqual(df_row.shape, (1, 24))
        self.assertEqual(derived["monthly_fee"], 13.99)
        self.assertEqual(derived["region"], "North America")
        self.assertEqual(derived["watch_time_per_session"], 42.5)
        self.assertEqual(int(df_row.iloc[0]["rating"]), 1)
        self.assertEqual(df_row.iloc[0]["liked"], "Yes")

    def test_03_frontend_static_assets_and_health_contract(self) -> None:
        for path in ["/", "/styles.css", "/app.js", "/health", "/contract"]:
            status, body = self._request("GET", path)
            self.assertEqual(status, 200, f"Failed GET {path}")
            if path == "/":
                self.assertIn("Estimated Churn Probability", body)
                self.assertNotIn("Retention Confidence", body)

    def test_04_valid_predict_http_200(self) -> None:
        for preset_key in ("high_risk_demo", "moderate_risk_demo", "low_risk_demo"):
            payload = dict(self.contract["synthetic_presets"][preset_key]["inputs"])
            status, data = self._request("POST", "/predict", payload)
            self.assertEqual(status, 200)
            self.assertGreaterEqual(data["estimated_churn_probability"], 0.0)
            self.assertLessEqual(data["estimated_churn_probability"], 1.0)
            self.assertIn(data["risk_category"], ("Low", "Moderate", "High"))
            self.assertIn("not a calibrated", data["probability_disclaimer"])

    def test_05_subscription_start_date_tenure_derivation(self) -> None:
        payload = self._valid_payload()
        payload.pop("tenure_days")
        payload["subscription_start_date"] = "2024-04-14"  # 365 days before 2025-04-14
        status, data = self._request("POST", "/predict", payload)
        self.assertEqual(status, 200)
        self.assertEqual(data["derived_features"]["tenure_days"], 365.0)

        # Future date after reference_date must fail with 400
        payload["subscription_start_date"] = "2026-01-01"
        status, err = self._request("POST", "/predict", payload)
        self.assertEqual(status, 400)
        self.assertIn("subscription_start_date", err["field_errors"])

    def test_06_validation_rejects_missing_invalid_zero_and_bad_types(self) -> None:
        # Missing field
        p_missing = self._valid_payload()
        del p_missing["genre"]
        status, err = self._request("POST", "/predict", p_missing)
        self.assertEqual(status, 400)
        self.assertIn("genre", err["field_errors"])

        # Invalid category
        p_cat = self._valid_payload()
        p_cat["subscription_plan"] = "UltraVIP"
        status, err = self._request("POST", "/predict", p_cat)
        self.assertEqual(status, 400)
        self.assertIn("subscription_plan", err["field_errors"])

        # Zero sessions (zero-division protection)
        p_zero = self._valid_payload()
        p_zero["session_count"] = 0
        status, err = self._request("POST", "/predict", p_zero)
        self.assertEqual(status, 400)
        self.assertIn("session_count", err["field_errors"])

        # Incorrect data type & boolean masquerading as number
        p_type = self._valid_payload()
        p_type["watch_time_minutes"] = "not-a-number"
        p_type["rating"] = True
        status, err = self._request("POST", "/predict", p_type)
        self.assertEqual(status, 400)
        self.assertIn("watch_time_minutes", err["field_errors"])
        self.assertIn("rating", err["field_errors"])

        # Out-of-range numeric values
        p_range = self._valid_payload()
        p_range["completion_percentage"] = 140
        p_range["content_diversity_score"] = -0.1
        status, err = self._request("POST", "/predict", p_range)
        self.assertEqual(status, 400)
        self.assertIn("completion_percentage", err["field_errors"])
        self.assertIn("content_diversity_score", err["field_errors"])

        # Mismatched derived fee/region
        p_mismatch = self._valid_payload()
        p_mismatch["monthly_fee"] = 8.99  # Standard must be 13.99
        p_mismatch["region"] = "Asia"     # USA must be North America
        status, err = self._request("POST", "/predict", p_mismatch)
        self.assertEqual(status, 400)
        self.assertIn("monthly_fee", err["field_errors"])
        self.assertIn("region", err["field_errors"])

    def test_07_model_loading_error_returns_503(self) -> None:
        bad_server, bad_state = create_server(
            host="127.0.0.1",
            port=0,
            model_path=PROJECT_ROOT / "artifacts" / "does_not_exist.joblib",
            contract_path=self.contract_path,
        )
        self.assertIsNotNone(bad_state.load_error)
        t = threading.Thread(target=bad_server.serve_forever, daemon=True)
        t.start()
        try:
            b_port = bad_server.server_address[1]
            req = Request(
                f"http://127.0.0.1:{b_port}/predict",
                data=json.dumps(self._valid_payload()).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as ctx:
                urlopen(req, timeout=5)
            self.assertEqual(ctx.exception.code, 503)
        finally:
            bad_server.shutdown()
            bad_server.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

