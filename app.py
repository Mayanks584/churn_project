"""Localhost-only HTTP server for Netflix Customer Churn Prediction (Version 1)."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib

from validator import InputValidationError, validate_and_build_dataframe

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = (PROJECT_ROOT / "frontend").resolve()
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "netflix_churn_rf_pipeline.joblib"
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "artifacts" / "model_contract.json"


class ChurnAppState:
    """Holds the loaded ML Pipeline and contract, or records load error cleanly."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, contract_path: Path = DEFAULT_CONTRACT_PATH):
        self.model_path = Path(model_path)
        self.contract_path = Path(contract_path)
        self.model = None
        self.contract: dict | None = None
        self.load_error: str | None = None
        self.reload()

    def reload(self) -> None:
        self.model = None
        self.contract = None
        self.load_error = None
        try:
            if not self.contract_path.is_file():
                raise FileNotFoundError(f"Missing contract file: {self.contract_path}")
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Missing model artifact: {self.model_path}")
            with open(self.contract_path, "r", encoding="utf-8") as f:
                self.contract = json.load(f)
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"


def classify_risk_band(prob: float, risk_bands: list[dict]) -> dict:
    for band in risk_bands:
        if band["min_prob_inclusive"] <= prob < band["max_prob_exclusive"]:
            return band
    return risk_bands[-1]


def build_descriptive_context(df_row, contract: dict) -> list[dict]:
    """Compare submitted numeric inputs against X_train Active/Cancelled descriptive means (non-causal)."""
    stats = contract["train_numeric_reference_stats"]
    focus_cols = [
        ("monthly_fee", "Monthly Fee ($)"),
        ("avg_weekly_watch_time", "Weekly Watch Time (min)"),
        ("watch_time_minutes", "Session Watch Time (min)"),
        ("session_count", "Session Count"),
        ("watch_time_per_session", "Watch Time / Session (min)"),
        ("completion_percentage", "Completion (%)"),
        ("days_since_last_watch", "Days Since Last Watch"),
        ("tenure_days", "Tenure (days)"),
    ]
    rows = []
    for col, label in focus_cols:
        val = round(float(df_row.iloc[0][col]), 2)
        st = stats[col]
        rows.append({
            "feature": col,
            "label": label,
            "submitted_value": val,
            "train_active_mean": st["active_mean"],
            "train_cancelled_mean": st["cancelled_mean"],
        })
    return rows


def make_handler(state: ChurnAppState, frontend_dir: Path = FRONTEND_DIR):
    class ChurnRequestHandler(BaseHTTPRequestHandler):
        server_version = "NetflixChurnV1/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Keep server output concise during tests & local runs
            pass

        def _send_json(self, status: int, body: dict) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _serve_frontend_file(self, rel_path: str) -> None:
            clean_rel = "index.html" if rel_path in ("", "/") else rel_path.lstrip("/")
            candidate = (frontend_dir / clean_rel).resolve()
            if frontend_dir not in candidate.parents and candidate != frontend_dir:
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "Forbidden path."})
                return
            if not candidate.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Asset not found: {clean_rel}"})
                return
            ctype, _ = mimetypes.guess_type(str(candidate))
            data = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{ctype or 'application/octet-stream'}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/health":
                if state.load_error:
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"status": "error", "error": state.load_error},
                    )
                else:
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "model_name": state.contract["model_name"],
                            "model_version": state.contract["model_version"],
                        },
                    )
                return

            if path == "/contract":
                if state.load_error or state.contract is None:
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": state.load_error or "Model contract unavailable."},
                    )
                    return
                self._send_json(HTTPStatus.OK, state.contract)
                return

            if path in ("/", "/index.html", "/styles.css", "/app.js"):
                self._serve_frontend_file(path)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path != "/predict":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
                return

            if state.load_error or state.model is None or state.contract is None:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": "Model artifact is unavailable or failed to load.",
                        "details": state.load_error,
                    },
                )
                return

            try:
                content_len = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length header."})
                return

            if content_len <= 0 or content_len > 1_000_000:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Request body must be non-empty JSON under 1 MB."},
                )
                return

            raw_bytes = self.rfile.read(content_len)
            try:
                payload = json.loads(raw_bytes.decode("utf-8"))
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Malformed JSON payload: {exc}"})
                return

            try:
                df_row, derived = validate_and_build_dataframe(payload, state.contract)
            except InputValidationError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Input validation failed.", "field_errors": exc.errors},
                )
                return

            try:
                prob = float(state.model.predict_proba(df_row)[0, 1])
                pred_cls = int(state.model.predict(df_row)[0])
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Inference failure: {type(exc).__name__}: {exc}"},
                )
                return

            band = classify_risk_band(prob, state.contract["risk_bands"])
            response = {
                "model_name": state.contract["model_name"],
                "model_version": state.contract["model_version"],
                "estimated_churn_probability": round(prob, 4),
                "estimated_churn_probability_pct": round(prob * 100, 2),
                "predicted_class": pred_cls,
                "predicted_label": "Cancelled" if pred_cls == 1 else "Active",
                "risk_category": band["name"],
                "risk_band_details": band,
                "probability_disclaimer": (
                    "This is an Estimated Churn Probability from a class-balanced Random Forest "
                    "(class_weight='balanced'). It is not a calibrated survival probability or guarantee."
                ),
                "derived_features": derived,
                "descriptive_context": build_descriptive_context(df_row, state.contract),
            }
            self._send_json(HTTPStatus.OK, response)

    return ChurnRequestHandler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_path: Path = DEFAULT_MODEL_PATH,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> tuple[ThreadingHTTPServer, ChurnAppState]:
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Security policy restricts server binding to localhost (127.0.0.1) only.")
    state = ChurnAppState(model_path=model_path, contract_path=contract_path)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    return server, state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Netflix Churn Prediction V1 frontend server.")
    parser.add_argument("--port", type=int, default=8000, help="Localhost port (default: 8000)")
    args = parser.parse_args()

    server, state = create_server(host="127.0.0.1", port=args.port)
    if state.load_error:
        print(f"WARNING: Started in degraded mode ({state.load_error})")
    else:
        print(f"Loaded {state.contract['model_name']} ({state.contract['model_version']})")
    print(f"Serving Version 1 Frontend at http://127.0.0.1:{args.port}/ (localhost only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down local server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

