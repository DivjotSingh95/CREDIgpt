from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_KEY = "SK_ID_CURR"
DAYS_EMPLOYED_ANOMALY = 365243

TIER_1_COLUMNS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_ID_PUBLISH",
    "DAYS_REGISTRATION",
    "DAYS_LAST_PHONE_CHANGE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "ORGANIZATION_TYPE",
]

TIER_2_COLUMNS = [
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
]

ENGINEERED_COLUMNS = [
    "EXT_SOURCES_MEAN",
    "EXT_SOURCES_MAX",
    "EXT_SOURCES_MIN",
    "EXT_SOURCES_VARIANCE",
    "ANNUITY_TO_INCOME_RATIO",
    "CREDIT_TO_INCOME_RATIO",
    "GOODS_PRICE_TO_CREDIT_RATIO",
    "EMPLOYMENT_TO_AGE_RATIO",
    "INST_DPD_MAX",
    "INST_DPD_MEAN",
    "PAYMENT_FRACTION_MEAN",
    "PAYMENT_FRACTION_MAX",
    "PAYMENT_FRACTION_LT1_COUNT",
    "ACTIVE_DEBT_TO_CREDIT_RATIO",
    "ACTIVE_LOAN_COUNT",
    "OVERDUE_DEBT_SUM",
    "PREV_REJECTION_RATE",
    "PREV_APPROVAL_RATE",
]

BASE_COLUMNS = [PRIMARY_KEY] + TIER_1_COLUMNS + TIER_2_COLUMNS
SELECTED_COLUMNS = BASE_COLUMNS + ENGINEERED_COLUMNS

DATA_DIR = Path(__file__).parent
OUTPUT_FILE = DATA_DIR / "selected_features.csv"
SOURCE_FILE = DATA_DIR / "application_train.csv"


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def build_ext_source_features(df: pd.DataFrame) -> pd.DataFrame:
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ext_values = df[ext_cols]

    df["EXT_SOURCES_MEAN"] = ext_values.mean(axis=1)
    df["EXT_SOURCES_MAX"] = ext_values.max(axis=1)
    df["EXT_SOURCES_MIN"] = ext_values.min(axis=1)
    df["EXT_SOURCES_VARIANCE"] = ext_values.var(axis=1)
    return df


def build_application_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ANNUITY_TO_INCOME_RATIO"] = safe_ratio(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_TO_INCOME_RATIO"] = safe_ratio(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["GOODS_PRICE_TO_CREDIT_RATIO"] = safe_ratio(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])

    days_employed = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)
    df["EMPLOYMENT_TO_AGE_RATIO"] = safe_ratio(days_employed, df["DAYS_BIRTH"])
    return df


def build_installment_features(data_dir: Path) -> pd.DataFrame:
    installments = pd.read_csv(data_dir / "installments_payments.csv", encoding="latin-1")

    installments["DPD"] = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    installments["PAYMENT_FRACTION"] = safe_ratio(
        installments["AMT_PAYMENT"],
        installments["AMT_INSTALMENT"],
    )

    return (
        installments.groupby(PRIMARY_KEY)
        .agg(
            INST_DPD_MAX=("DPD", "max"),
            INST_DPD_MEAN=("DPD", "mean"),
            PAYMENT_FRACTION_MEAN=("PAYMENT_FRACTION", "mean"),
            PAYMENT_FRACTION_MAX=("PAYMENT_FRACTION", "max"),
            PAYMENT_FRACTION_LT1_COUNT=("PAYMENT_FRACTION", lambda s: (s < 1.0).sum()),
        )
        .reset_index()
    )


def build_bureau_features(data_dir: Path) -> pd.DataFrame:
    bureau = pd.read_csv(data_dir / "bureau.csv", encoding="latin-1")

    active = bureau[bureau["CREDIT_ACTIVE"] == "Active"].copy()
    active["ACTIVE_DEBT_TO_CREDIT"] = safe_ratio(
        active["AMT_CREDIT_SUM_DEBT"],
        active["AMT_CREDIT_SUM"],
    )

    active_features = (
        active.groupby(PRIMARY_KEY)
        .agg(
            ACTIVE_DEBT_TO_CREDIT_RATIO=("ACTIVE_DEBT_TO_CREDIT", "mean"),
            ACTIVE_LOAN_COUNT=(PRIMARY_KEY, "count"),
        )
        .reset_index()
    )

    overdue_features = (
        bureau.groupby(PRIMARY_KEY)
        .agg(OVERDUE_DEBT_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"))
        .reset_index()
    )

    return active_features.merge(overdue_features, on=PRIMARY_KEY, how="outer")


def build_previous_application_features(data_dir: Path) -> pd.DataFrame:
    previous = pd.read_csv(data_dir / "previous_application.csv", encoding="latin-1")

    grouped = (
        previous.groupby(PRIMARY_KEY)
        .agg(
            total_apps=("SK_ID_PREV", "count"),
            refused_apps=("NAME_CONTRACT_STATUS", lambda s: (s == "Refused").sum()),
            approved_apps=("NAME_CONTRACT_STATUS", lambda s: (s == "Approved").sum()),
        )
        .reset_index()
    )

    grouped["PREV_REJECTION_RATE"] = safe_ratio(
        grouped["refused_apps"].astype(float),
        grouped["total_apps"].astype(float),
    )
    grouped["PREV_APPROVAL_RATE"] = safe_ratio(
        grouped["approved_apps"].astype(float),
        grouped["total_apps"].astype(float),
    )

    return grouped[[PRIMARY_KEY, "PREV_REJECTION_RATE", "PREV_APPROVAL_RATE"]]


def build_engineered_features(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    df = build_ext_source_features(df)
    df = build_application_ratio_features(df)

    installment_features = build_installment_features(data_dir)
    bureau_features = build_bureau_features(data_dir)
    previous_features = build_previous_application_features(data_dir)

    df = df.merge(installment_features, on=PRIMARY_KEY, how="left")
    df = df.merge(bureau_features, on=PRIMARY_KEY, how="left")
    df = df.merge(previous_features, on=PRIMARY_KEY, how="left")
    return df


def detect_columns() -> dict[str, list[str]]:
    """Scan all CSV files and report which base columns each file contains."""
    matches: dict[str, list[str]] = {}

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        if csv_path.name == OUTPUT_FILE.name:
            continue

        file_columns = pd.read_csv(csv_path, nrows=0, encoding="latin-1").columns.tolist()
        found = [col for col in BASE_COLUMNS if col in file_columns]

        if found:
            matches[csv_path.name] = found

    return matches


def extract_features(source_file: Path, output_file: Path) -> pd.DataFrame:
    """Extract base columns, engineer new features, and save to CSV."""
    df = pd.read_csv(source_file, encoding="latin-1")

    available = [col for col in BASE_COLUMNS if col in df.columns]
    missing = [col for col in BASE_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in {source_file.name}: {missing}")

    df = df[available]
    df = build_engineered_features(df, DATA_DIR)
    df = df[SELECTED_COLUMNS]
    df.to_csv(output_file, index=False)
    return df


def main() -> None:
    print("Scanning CSV files for base columns...\n")

    matches = detect_columns()
    for filename, found in matches.items():
        print(f"{filename}: {len(found)}/{len(BASE_COLUMNS)} base columns found")

    print(f"\nBuilding {len(ENGINEERED_COLUMNS)} engineered features...")
    print(f"Extracting from {SOURCE_FILE.name}...")
    selected = extract_features(SOURCE_FILE, OUTPUT_FILE)

    print(f"Saved {len(selected):,} rows x {len(selected.columns)} columns to {OUTPUT_FILE.name}")
    print(f"  Base columns:       {len(BASE_COLUMNS)}")
    print(f"  Engineered columns: {len(ENGINEERED_COLUMNS)}")
    print("\nFirst 5 rows (selected engineered columns):")
    print(selected[[PRIMARY_KEY] + ENGINEERED_COLUMNS[:8]].head())


if __name__ == "__main__":
    main()
