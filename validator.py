"""Strict input contract validation and 24-feature DataFrame builder."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd


class InputValidationError(ValueError):
    """Raised when a prediction request violates the model input contract."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


def _parse_number(val: Any, field: str, integer_only: bool = False) -> tuple[float | int | None, str | None]:
    if isinstance(val, bool) or val is None:
        return None, f"{field} must be a valid {'integer' if integer_only else 'number'} (not boolean or null)."
    if isinstance(val, str) and not val.strip():
        return None, f"{field} cannot be empty."
    try:
        num_val = float(val)
    except (TypeError, ValueError):
        return None, f"{field} must be numeric (got {val!r})."
    if not math.isfinite(num_val):
        return None, f"{field} must be finite (NaN/Infinity are not allowed)."
    if integer_only:
        if not num_val.is_integer():
            return None, f"{field} must be a whole integer (got {val!r})."
        return int(num_val), None
    return float(num_val), None


def validate_and_build_dataframe(payload: Any, contract: dict) -> tuple[pd.DataFrame, dict]:
    """Validate raw request JSON and build a 1-row DataFrame matching the exact 24-feature contract."""
    if not isinstance(payload, dict):
        raise InputValidationError({"payload": "Request body must be a JSON object."})

    errors: dict[str, str] = {}
    bounds = contract["numeric_bounds"]
    allowed_cats = contract["allowed_categories"]
    plan_to_fee = contract["plan_to_fee"]
    country_to_region = contract["country_to_region"]

    # Allowed input keys (20 direct user inputs + optional derived/date overrides)
    user_numeric_keys = [
        "watch_time_minutes",
        "session_count",
        "completion_percentage",
        "rating",
        "days_since_last_watch",
        "avg_weekly_watch_time",
        "content_diversity_score",
    ]
    user_cat_keys = [c for c in contract["categorical_features"] if c != "region"]
    optional_keys = {"monthly_fee", "region", "tenure_days", "subscription_start_date", "watch_time_per_session"}
    known_keys = set(user_numeric_keys) | set(user_cat_keys) | optional_keys

    unknown = sorted(set(payload.keys()) - known_keys)
    if unknown:
        errors["unknown_fields"] = f"Unexpected input field(s): {', '.join(unknown)}"

    cleaned: dict[str, Any] = {}

    # 1. Validate categorical fields (including integer-encoded age_group)
    for col in user_cat_keys:
        if col not in payload or payload[col] is None or (isinstance(payload[col], str) and not payload[col].strip()):
            errors[col] = f"{col} is required."
            continue
        raw_v = payload[col]
        if col == "age_group":
            parsed_age, err = _parse_number(raw_v, col, integer_only=True)
            if err:
                errors[col] = err
            elif parsed_age not in allowed_cats["age_group"]:
                errors[col] = f"age_group must be one of {allowed_cats['age_group']} (got {parsed_age})."
            else:
                cleaned[col] = int(parsed_age)
        else:
            if not isinstance(raw_v, str):
                errors[col] = f"{col} must be a string."
            else:
                s_val = raw_v.strip()
                if s_val not in allowed_cats[col]:
                    errors[col] = f"{col} must be one of {allowed_cats[col]} (got {s_val!r})."
                else:
                    cleaned[col] = s_val

    # 2. Validate & verify derived fields: monthly_fee (from subscription_plan) & region (from country)
    if "subscription_plan" in cleaned:
        expected_fee = float(plan_to_fee[cleaned["subscription_plan"]])
        if "monthly_fee" in payload and payload["monthly_fee"] is not None:
            supplied_fee, fee_err = _parse_number(payload["monthly_fee"], "monthly_fee")
            if fee_err:
                errors["monthly_fee"] = fee_err
            elif abs(float(supplied_fee) - expected_fee) > 1e-6:
                errors["monthly_fee"] = (
                    f"monthly_fee ({supplied_fee}) does not match {cleaned['subscription_plan']} plan (${expected_fee})."
                )
        cleaned["monthly_fee"] = expected_fee

    if "country" in cleaned:
        expected_region = country_to_region[cleaned["country"]]
        if "region" in payload and payload["region"] is not None and str(payload["region"]).strip():
            supplied_region = str(payload["region"]).strip()
            if supplied_region != expected_region:
                errors["region"] = (
                    f"region ({supplied_region!r}) does not match country {cleaned['country']} ({expected_region!r})."
                )
        cleaned["region"] = expected_region

    # 3. Validate direct numerical fields
    for col in user_numeric_keys:
        if col not in payload:
            errors[col] = f"{col} is required."
            continue
        is_int = bounds[col]["type"] == "int"
        num_v, err = _parse_number(payload[col], col, integer_only=is_int)
        if err:
            errors[col] = err
            continue
        b_min, b_max = bounds[col]["min"], bounds[col]["max"]
        if num_v < b_min or num_v > b_max:
            errors[col] = f"{col} must be between {b_min} and {b_max} (got {num_v})."
        else:
            cleaned[col] = num_v

    # 4. Validate tenure_days OR subscription_start_date
    has_tenure = "tenure_days" in payload and payload["tenure_days"] is not None and str(payload["tenure_days"]).strip() != ""
    has_start_date = "subscription_start_date" in payload and payload["subscription_start_date"] is not None and str(payload["subscription_start_date"]).strip() != ""
    if not has_tenure and not has_start_date:
        errors["tenure_days"] = "Either tenure_days or subscription_start_date is required."
    elif has_tenure:
        t_val, err = _parse_number(payload["tenure_days"], "tenure_days")
        if err:
            errors["tenure_days"] = err
        elif t_val < bounds["tenure_days"]["min"] or t_val > bounds["tenure_days"]["max"]:
            errors["tenure_days"] = f"tenure_days must be between {bounds['tenure_days']['min']} and {bounds['tenure_days']['max']}."
        else:
            cleaned["tenure_days"] = float(t_val)
    else:
        raw_dt = str(payload["subscription_start_date"]).strip()
        try:
            start_dt = datetime.strptime(raw_dt, "%Y-%m-%d")
            ref_dt = datetime.strptime(contract["reference_date"], "%Y-%m-%d")
            diff_days = (ref_dt - start_dt).days
            if diff_days < 0:
                errors["subscription_start_date"] = f"subscription_start_date cannot be after reference_date ({contract['reference_date']})."
            elif diff_days > bounds["tenure_days"]["max"]:
                errors["subscription_start_date"] = f"Computed tenure_days ({diff_days}) exceeds max bound ({bounds['tenure_days']['max']})."
            else:
                cleaned["tenure_days"] = float(diff_days)
        except ValueError:
            errors["subscription_start_date"] = "subscription_start_date must be YYYY-MM-DD."

    if errors:
        raise InputValidationError(errors)

    # 5. Compute watch_time_per_session (session_count >= 1 already enforced)
    wt_per_session = float(cleaned["watch_time_minutes"]) / float(cleaned["session_count"])
    if "watch_time_per_session" in payload and payload["watch_time_per_session"] is not None:
        sup_wtps, err = _parse_number(payload["watch_time_per_session"], "watch_time_per_session")
        if err:
            raise InputValidationError({"watch_time_per_session": err})
        if abs(float(sup_wtps) - wt_per_session) > 1e-4:
            raise InputValidationError({
                "watch_time_per_session": f"watch_time_per_session ({sup_wtps}) must equal watch_time_minutes / session_count ({wt_per_session:.4f})."
            })
    cleaned["watch_time_per_session"] = wt_per_session

    # 6. Build 1-row DataFrame in EXACT 24-column notebook order and verify contract
    feature_order = contract["feature_order"]
    ordered_row = {col: cleaned[col] for col in feature_order}
    df_row = pd.DataFrame([ordered_row], columns=feature_order)

    for col in contract["numeric_features"]:
        df_row[col] = df_row[col].astype("float64")
    df_row["age_group"] = df_row["age_group"].astype("int64")

    assert list(df_row.columns) == feature_order, "Constructed DataFrame column order mismatch!"
    assert df_row.shape == (1, 24), f"Expected (1, 24) DataFrame, got {df_row.shape}"

    derived = {
        "monthly_fee": cleaned["monthly_fee"],
        "region": cleaned["region"],
        "tenure_days": round(float(cleaned["tenure_days"]), 2),
        "watch_time_per_session": round(wt_per_session, 2),
    }
    return df_row, derived

