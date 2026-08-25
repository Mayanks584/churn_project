
data prepos :- steps 

netflix_large_dataset_cleaned.csv
        ↓
Data quality checks
        ↓
Convert dates
        ↓
Clean strings
        ↓
Check repeated users
        ↓
Check churn conflicts
        ↓
Aggregate event data → 1 row per customer
        ↓
Create useful customer-level features
        ↓
Remove IDs / leakage-prone raw fields
        ↓
Save:
netflix_customer_features.csv


code i did not run 

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("netflix_large_dataset_cleaned.csv")
OUTPUT_FILE = Path("netflix_customer_features.csv")

TARGET_COLUMN = "churn_status"
ID_COLUMN = "user_id"

# Columns that should not become model features directly.
DROP_RAW_COLUMNS = [
    ID_COLUMN,
    "subscription_end_date",   # Potential leakage / post-churn information
    "title",                   # Event/content-level; aggregated instead
    "date_watched",            # Raw event date; aggregated instead
]

# Binary-like categorical columns that we will aggregate into rates.
BINARY_COLUMNS = [
    "liked",
    "discount_applied",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from string/object columns.

    We do not force categories into lowercase because that could
    accidentally change legitimate category names.
    """
    df = df.copy()

    string_columns = df.select_dtypes(include=["object", "string"]).columns

    for col in string_columns:
        df[col] = df[col].astype("string").str.strip()

    return df


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date columns to pandas datetime.
    Invalid values become NaT so they can be inspected.
    """
    df = df.copy()

    date_columns = [
        "subscription_start_date",
        "subscription_end_date",
        "date_watched",
    ]

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )

    return df


def report_date_conversion(df: pd.DataFrame) -> None:
    """Print invalid date counts after conversion."""

    print("\n========== DATE VALIDATION ==========")

    date_columns = [
        "subscription_start_date",
        "subscription_end_date",
        "date_watched",
    ]

    for col in date_columns:
        if col in df.columns:
            invalid = df[col].isna().sum()
            print(f"{col}: invalid/missing after conversion = {invalid}")


def validate_numeric_ranges(df: pd.DataFrame) -> None:
    """
    Print suspicious values.
    We do not automatically delete outliers.
    """

    print("\n========== NUMERIC VALIDATION ==========")

    checks = {
        "age_group": (None, None),
        "monthly_fee": (0, None),
        "release_year": (1900, None),
        "watch_time_minutes": (0, None),
        "session_count": (0, None),
        "completion_percentage": (0, 100),
        "rating": (0, 5),
        "days_since_last_watch": (0, None),
        "avg_weekly_watch_time": (0, None),
        "content_diversity_score": (0, None),
    }

    for column, (minimum, maximum) in checks.items():

        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce")

        invalid = pd.Series(False, index=df.index)

        if minimum is not None:
            invalid |= series < minimum

        if maximum is not None:
            invalid |= series > maximum

        print(f"{column}: suspicious rows = {invalid.sum()}")


def report_dataset_quality(df: pd.DataFrame) -> None:
    """Print basic dataset-quality information."""

    print("\n========== DATASET QUALITY ==========")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print(f"Duplicate full rows: {df.duplicated().sum():,}")

    if ID_COLUMN in df.columns:

        unique_users = df[ID_COLUMN].nunique()

        duplicated_users = df[ID_COLUMN].duplicated().sum()

        print(f"Unique users: {unique_users:,}")
        print(f"Repeated user rows: {duplicated_users:,}")

    print("\nMissing values:")

    missing = df.isna().sum()

    print(missing[missing > 0].sort_values(ascending=False))


def validate_target(df: pd.DataFrame) -> None:
    """
    Verify target values and check whether a user has conflicting
    churn labels.
    """

    print("\n========== TARGET VALIDATION ==========")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    print("Target values:")
    print(df[TARGET_COLUMN].value_counts(dropna=False))

    print("\nTarget distribution:")
    print(
        df[TARGET_COLUMN]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    # Check target conflicts at user level.
    user_target_counts = (
        df.groupby(ID_COLUMN)[TARGET_COLUMN]
        .nunique(dropna=False)
    )

    conflicting_users = user_target_counts[
        user_target_counts > 1
    ]

    print(
        f"\nUsers with multiple different churn labels: "
        f"{len(conflicting_users):,}"
    )

    if len(conflicting_users) > 0:

        print(
            "\nWARNING: Some users have conflicting churn_status "
            "values."
        )

        print(
            conflicting_users.head(10)
        )

        raise ValueError(
            "Conflicting churn labels detected for some users. "
            "Resolve this before creating the customer-level dataset."
        )


def mode_or_nan(series: pd.Series):
    """
    Return the most frequent non-null value.
    """
    series = series.dropna()

    if series.empty:
        return np.nan

    mode = series.mode()

    if mode.empty:
        return series.iloc[0]

    return mode.iloc[0]


def binary_rate(series: pd.Series) -> float:
    """
    Convert common binary values into a numeric rate.

    Handles values such as:
    Yes/No
    True/False
    Y/N
    1/0
    """
    if series.empty:
        return np.nan

    values = series.astype("string").str.strip().str.lower()

    positive_values = {
        "yes",
        "y",
        "true",
        "1",
        "liked",
        "active",
        "cancelled"   # Not normally used here, but harmless.
    }

    negative_values = {
        "no",
        "n",
        "false",
        "0",
        "disliked",
        "inactive",
    }

    positive = values.isin(positive_values)
    negative = values.isin(negative_values)

    recognized = positive | negative

    if recognized.sum() == 0:
        return np.nan

    return positive[recognized].mean()


def safe_nunique(series: pd.Series) -> int:
    """Number of unique non-null values."""
    return int(series.dropna().nunique())


# ============================================================
# CUSTOMER-LEVEL FEATURE ENGINEERING
# ============================================================

def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert event-level Netflix records into one row per user.
    """

    print("\n========== CUSTOMER FEATURE ENGINEERING ==========")

    # Ensure dates are sorted.
    if "date_watched" in df.columns:
        df = df.sort_values(
            by=[ID_COLUMN, "date_watched"]
        )

    grouped = df.groupby(
        ID_COLUMN,
        sort=False,
        dropna=False
    )

    # --------------------------------------------------------
    # Static / customer profile fields
    # --------------------------------------------------------

    customer_df = grouped.agg(
        age_group=("age_group", "first"),
        gender=("gender", mode_or_nan),
        country=("country", mode_or_nan),
        region=("region", mode_or_nan),

        subscription_plan=(
            "subscription_plan",
            mode_or_nan
        ),

        monthly_fee=(
            "monthly_fee",
            "median"
        ),

        subscription_start_date=(
            "subscription_start_date",
            "min"
        ),

        payment_method=(
            "payment_method",
            mode_or_nan
        ),

        discount_applied=(
            "discount_applied",
            mode_or_nan
        ),

        avg_weekly_watch_time=(
            "avg_weekly_watch_time",
            "mean"
        ),

        days_since_last_watch=(
            "days_since_last_watch",
            "min"
        ),

        content_diversity_score=(
            "content_diversity_score",
            "mean"
        ),

        churn_status=(
            TARGET_COLUMN,
            "first"
        ),
    ).reset_index()

    # --------------------------------------------------------
    # Viewing / engagement aggregates
    # --------------------------------------------------------

    viewing_features = grouped.agg(
        total_watch_time_minutes=(
            "watch_time_minutes",
            "sum"
        ),

        avg_watch_time_minutes=(
            "watch_time_minutes",
            "mean"
        ),

        max_watch_time_minutes=(
            "watch_time_minutes",
            "max"
        ),

        total_sessions=(
            "session_count",
            "sum"
        ),

        avg_session_count=(
            "session_count",
            "mean"
        ),

        avg_completion_percentage=(
            "completion_percentage",
            "mean"
        ),

        max_completion_percentage=(
            "completion_percentage",
            "max"
        ),

        avg_rating=(
            "rating",
            "mean"
        ),

        max_rating=(
            "rating",
            "max"
        ),

        unique_titles_watched=(
            "title",
            safe_nunique
        ),

        unique_content_types=(
            "content_type",
            safe_nunique
        ),

        unique_genres=(
            "genre",
            safe_nunique
        ),

        unique_languages=(
            "language",
            safe_nunique
        ),

        unique_devices=(
            "device_type",
            safe_nunique
        ),

        unique_recommendation_sources=(
            "recommendation_source",
            safe_nunique
        ),

        unique_time_periods=(
            "time_of_day",
            safe_nunique
        ),
    ).reset_index()

    customer_df = customer_df.merge(
        viewing_features,
        on=ID_COLUMN,
        how="left",
    )

    # --------------------------------------------------------
    # Like rate
    # --------------------------------------------------------

    like_rate = (
        grouped["liked"]
        .apply(binary_rate)
        .rename("like_rate")
        .reset_index()
    )

    customer_df = customer_df.merge(
        like_rate,
        on=ID_COLUMN,
        how="left",
    )

    # --------------------------------------------------------
    # Discount rate
    # --------------------------------------------------------

    discount_rate = (
        grouped["discount_applied"]
        .apply(binary_rate)
        .rename("discount_rate")
        .reset_index()
    )

    customer_df = customer_df.merge(
        discount_rate,
        on=ID_COLUMN,
        how="left",
    )

    # --------------------------------------------------------
    # Subscription tenure
    # --------------------------------------------------------

    # We calculate tenure using the customer's observed
    # latest watched date rather than subscription_end_date.
    #
    # IMPORTANT:
    # This is a feature-engineering choice, not an ML leakage
    # guarantee. Later, before training, we should define a
    # formal prediction cutoff date/time.
    latest_watch = (
        grouped["date_watched"]
        .max()
        .rename("last_observed_watch_date")
        .reset_index()
    )

    customer_df = customer_df.merge(
        latest_watch,
        on=ID_COLUMN,
        how="left",
    )

    customer_df["subscription_tenure_days"] = (
        customer_df["last_observed_watch_date"]
        - customer_df["subscription_start_date"]
    ).dt.days

    customer_df["subscription_tenure_months"] = (
        customer_df["subscription_tenure_days"] / 30.44
    )

    # --------------------------------------------------------
    # Activity duration
    # --------------------------------------------------------

    first_watch = (
        grouped["date_watched"]
        .min()
        .rename("first_observed_watch_date")
        .reset_index()
    )

    customer_df = customer_df.merge(
        first_watch,
        on=ID_COLUMN,
        how="left",
    )

    customer_df["observed_activity_span_days"] = (
        customer_df["last_observed_watch_date"]
        - customer_df["first_observed_watch_date"]
    ).dt.days

    # --------------------------------------------------------
    # Session / viewing intensity
    # --------------------------------------------------------

    customer_df["minutes_per_session"] = np.where(
        customer_df["total_sessions"] > 0,
        customer_df["total_watch_time_minutes"]
        / customer_df["total_sessions"],
        0,
    )

    customer_df["sessions_per_active_observation"] = np.where(
        customer_df["observed_activity_span_days"] > 0,
        customer_df["total_sessions"]
        / customer_df["observed_activity_span_days"],
        customer_df["total_sessions"],
    )

    # --------------------------------------------------------
    # Content engagement
    # --------------------------------------------------------

    customer_df["liked_and_watched_intensity"] = (
        customer_df["like_rate"]
        * customer_df["avg_completion_percentage"]
        / 100.0
    )

    customer_df["content_engagement_score"] = (
        customer_df["avg_completion_percentage"] * 0.5
        + customer_df["like_rate"] * 100.0 * 0.3
        + customer_df["content_diversity_score"] * 0.2
    )

    # --------------------------------------------------------
    # Drop raw fields that are no longer needed
    # --------------------------------------------------------

    columns_to_drop = [
        "subscription_end_date",
        "first_observed_watch_date",
        "last_observed_watch_date",
    ]

    # subscription_start_date is also removed after tenure creation.
    columns_to_drop.append("subscription_start_date")

    customer_df = customer_df.drop(
        columns=[
            col
            for col in columns_to_drop
            if col in customer_df.columns
        ]
    )

    # --------------------------------------------------------
    # Remove identifier from modeling table
    # --------------------------------------------------------

    # Keep user_id for traceability, but it will NOT be a model
    # feature later.
    #
    # We keep it in this preprocessing output so we can identify
    # customers when analyzing predictions.
    #
    # ML stage will explicitly exclude user_id.

    return customer_df


# ============================================================
# FINAL CLEANUP
# ============================================================

def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Replace infinite values generated during calculations.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Report newly created missing values.
    print("\n========== FEATURE-ENGINEERING MISSING VALUES ==========")

    missing = df.isna().sum()

    print(
        missing[missing > 0]
        .sort_values(ascending=False)
    )

    # Numeric rounding for readability.
    float_columns = df.select_dtypes(
        include=["float64", "float32"]
    ).columns

    for col in float_columns:
        df[col] = df[col].round(3)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("==============================================")
    print(" OTT CUSTOMER DATA PROCESSING")
    print("==============================================")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE.resolve()}"
        )

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded: {len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    # --------------------------------------------------------
    # 2. BASIC QUALITY AUDIT
    # --------------------------------------------------------

    report_dataset_quality(df)

    # --------------------------------------------------------
    # 3. CLEAN STRINGS
    # --------------------------------------------------------

    df = clean_string_columns(df)

    # --------------------------------------------------------
    # 4. CONVERT DATES
    # --------------------------------------------------------

    df = convert_dates(df)

    report_date_conversion(df)

    # --------------------------------------------------------
    # 5. NUMERIC VALIDATION
    # --------------------------------------------------------

    validate_numeric_ranges(df)

    # --------------------------------------------------------
    # 6. TARGET VALIDATION
    # --------------------------------------------------------

    validate_target(df)

    # --------------------------------------------------------
    # 7. IDENTIFY DATA GRAIN
    # --------------------------------------------------------

    print("\n========== DATA GRAIN ==========")

    unique_users = df[ID_COLUMN].nunique()
    rows = len(df)

    print(f"Rows: {rows:,}")
    print(f"Unique users: {unique_users:,}")

    if unique_users < rows:

        print(
            "Result: Multiple rows per user detected."
        )

        print(
            "The dataset appears to contain "
            "customer/event-level records."
        )

        print(
            "Creating one customer-level row per user..."
        )

    else:

        print(
            "Result: One row per user detected."
        )

    # --------------------------------------------------------
    # 8. CUSTOMER FEATURE ENGINEERING
    # --------------------------------------------------------

    customer_df = create_customer_features(df)

    # --------------------------------------------------------
    # 9. FINAL CLEANUP
    # --------------------------------------------------------

    customer_df = final_cleanup(customer_df)

    # --------------------------------------------------------
    # 10. FINAL AUDIT
    # --------------------------------------------------------

    print("\n========== FINAL DATASET ==========")

    print(
        f"Rows: {len(customer_df):,}"
    )

    print(
        f"Columns: {len(customer_df.columns)}"
    )

    print("\nColumns:")

    for i, col in enumerate(
        customer_df.columns,
        start=1
    ):
        print(f"{i}. {col}")

    print("\nDuplicate rows:")

    print(
        customer_df.duplicated().sum()
    )

    print("\nMissing values:")

    print(
        customer_df.isna()
        .sum()
        .loc[lambda x: x > 0]
    )

    print("\nTarget distribution:")

    print(
        customer_df[TARGET_COLUMN]
        .value_counts(dropna=False)
    )

    print("\nTarget percentage:")

    print(
        customer_df[TARGET_COLUMN]
        .value_counts(
            normalize=True,
            dropna=False
        )
        .mul(100)
        .round(2)
    )

    # --------------------------------------------------------
    # 11. SAVE
    # --------------------------------------------------------

    customer_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n==============================================")
    print("Processing completed successfully.")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")
    print("==============================================")


if __name__ == "__main__":
    main()



                now real work starting

 RAW NETFLIX CSV
       ↓
1. UNDERSTAND DATA
       ↓
2. CHECK DATA QUALITY
       ↓
3. CHECK TARGET
       ↓
4. CHECK LEAKAGE
       ↓
5. SELECT FEATURES
       ↓
6. CONVERT DATATYPES
       ↓
7. FEATURE ENGINEERING
       ↓
8. FINAL CLEAN DATASET
       ↓
9. TRAIN/TEST SPLIT
       ↓
10. ENCODING + SCALING
       ↓
11. MODEL TRAINING
       ↓
12. MODEL COMPARISON
       ↓
13. EVALUATION
       ↓
14. INTERPRETATION
       ↓
15. DEPLOYMENT

                the steps we have done 

Basic Dataset Understanding ✅
        ↓
Duplicate/User Check ✅
        ↓
Target Check ✅
        ↓
Categorical Inspection ✅
        ↓
Leakage Investigation ✅
        ↓
👉 Date Conversion + Validation  ← WE ARE HERE