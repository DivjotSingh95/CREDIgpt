from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent


@dataclass
class ValidationResult:
    is_valid: bool
    records_uploaded: int
    missing_values: int
    duplicate_customers: int
    invalid_numeric_fields: int
    missing_columns: list[str]
    cleaned_df: pd.DataFrame
    errors: list[str]

    @property
    def status_label(self) -> str:
        return "PASSED" if self.is_valid else "FAILED"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    return normalized


def validate_dataset(df: pd.DataFrame) -> ValidationResult:
    """Basic structural validation to check if dataset is valid (i.e. not empty)."""
    errors: list[str] = []
    if df.empty:
        return ValidationResult(
            is_valid=False,
            records_uploaded=0,
            missing_values=0,
            duplicate_customers=0,
            invalid_numeric_fields=0,
            missing_columns=[],
            cleaned_df=df,
            errors=["Uploaded file is empty."],
        )

    # Let's count duplicate SK_ID_CURR or customer_id if present
    cleaned_df = normalize_columns(df)
    duplicate_customers = 0
    id_cols = [c for c in ["sk_id_curr", "customer_id", "id"] if c in cleaned_df.columns]
    if id_cols:
        duplicate_customers = int(cleaned_df[id_cols[0]].duplicated().sum())
        if duplicate_customers > 0:
            errors.append(f"Found {duplicate_customers} duplicate customer IDs.")

    # Find invalid numeric fields
    NUMERIC_KEYWORDS = ["income", "earnings", "credit", "loan", "annuity", "days", "age", "years", "id", "curr", "amt", "cnt", "rate", "score", "pop", "ratio"]
    invalid_numeric_fields = 0
    for col in cleaned_df.columns:
        if any(kw in col for kw in NUMERIC_KEYWORDS):
            # Try converting to numeric; non-null values that can't be converted are invalid
            non_null = cleaned_df[col].dropna()
            # If the column dtype is already numeric, it contains no invalid strings
            if not pd.api.types.is_numeric_dtype(non_null):
                coerced = pd.to_numeric(non_null, errors='coerce')
                invalid_count = int(coerced.isna().sum())
                invalid_numeric_fields += invalid_count

    if invalid_numeric_fields > 0:
        errors.append(f"Found {invalid_numeric_fields} invalid numeric values.")

    missing_values = int(cleaned_df.isna().sum().sum())
    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        records_uploaded=len(cleaned_df),
        missing_values=missing_values,
        duplicate_customers=duplicate_customers,
        invalid_numeric_fields=invalid_numeric_fields,
        missing_columns=[],
        cleaned_df=cleaned_df,
        errors=errors,
    )


def run_feature_engineering_pipeline(df: pd.DataFrame, confirmed_mappings: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rename columns, calculate engineered features, join precomputed bureau/installments/prev aggregates,
    pad missing columns, and save the datasets to the database directory.
    """
    from utils.mapping import get_target_columns, ENGINEERED_COLUMNS

    # 1. Original Dataset
    original_df = df.copy()

    # 2. Mapped Dataset (rename based on confirmed mappings)
    mapped_df = df.copy()
    rename_dict = {orig: target for orig, target in confirmed_mappings.items() if target}
    mapped_df = mapped_df.rename(columns=rename_dict)

    # 3. Engineered Dataset
    engineered_df = mapped_df.copy()
    target_all = get_target_columns()

    # Ensure all base features exist in df (pad with NaN if missing)
    for col in target_all:
        if col not in ENGINEERED_COLUMNS and col not in engineered_df.columns:
            engineered_df[col] = np.nan

    # Coerce categoricals to string for safety
    CATEGORICAL_COLUMNS = [
        "CODE_GENDER",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "ORGANIZATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
    ]
    for col in CATEGORICAL_COLUMNS:
        if col in engineered_df.columns:
            engineered_df[col] = engineered_df[col].astype(str).replace("nan", "missing").fillna("missing")

    # Compute EXT_SOURCE features
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    for col in ext_cols:
        if col in engineered_df.columns:
            engineered_df[col] = pd.to_numeric(engineered_df[col], errors="coerce")
    ext_values = engineered_df[ext_cols]
    engineered_df["EXT_SOURCES_MEAN"] = ext_values.mean(axis=1)
    engineered_df["EXT_SOURCES_MAX"] = ext_values.max(axis=1)
    engineered_df["EXT_SOURCES_MIN"] = ext_values.min(axis=1)
    engineered_df["EXT_SOURCES_VARIANCE"] = ext_values.var(axis=1)

    # Compute application ratios
    ratio_cols = ["AMT_ANNUITY", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_GOODS_PRICE", "DAYS_EMPLOYED", "DAYS_BIRTH"]
    for col in ratio_cols:
        if col in engineered_df.columns:
            engineered_df[col] = pd.to_numeric(engineered_df[col], errors="coerce")

    def safe_ratio(num, denom):
        return num / denom.replace(0, np.nan)

    engineered_df["ANNUITY_TO_INCOME_RATIO"] = safe_ratio(engineered_df["AMT_ANNUITY"], engineered_df["AMT_INCOME_TOTAL"])
    engineered_df["CREDIT_TO_INCOME_RATIO"] = safe_ratio(engineered_df["AMT_CREDIT"], engineered_df["AMT_INCOME_TOTAL"])
    engineered_df["GOODS_PRICE_TO_CREDIT_RATIO"] = safe_ratio(engineered_df["AMT_GOODS_PRICE"], engineered_df["AMT_CREDIT"])

    DAYS_EMPLOYED_ANOMALY = 365243
    days_employed_clean = engineered_df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_ANOMALY, np.nan)
    engineered_df["EMPLOYMENT_TO_AGE_RATIO"] = safe_ratio(days_employed_clean, engineered_df["DAYS_BIRTH"])

    # Compute missing indicators
    engineered_df["EXT_SOURCE_1_missing"] = engineered_df["EXT_SOURCE_1"].isna().astype(int)
    engineered_df["EXT_SOURCE_3_missing"] = engineered_df["EXT_SOURCE_3"].isna().astype(int)

    # Join with precomputed auxiliary features on SK_ID_CURR
    if "SK_ID_CURR" in engineered_df.columns:
        engineered_df["SK_ID_CURR"] = pd.to_numeric(engineered_df["SK_ID_CURR"], errors="coerce")
        aux_path = DATA_DIR / "database" / "precomputed_features.csv"
        if aux_path.exists():
            try:
                aux_df = pd.read_csv(aux_path, encoding="latin-1")
                aux_df["SK_ID_CURR"] = pd.to_numeric(aux_df["SK_ID_CURR"], errors="coerce")
                
                # Exclude columns that are already present in engineered_df to prevent suffix duplication
                cols_to_drop = [c for c in aux_df.columns if c in engineered_df.columns and c != "SK_ID_CURR"]
                aux_df_clean = aux_df.drop(columns=cols_to_drop)
                
                engineered_df = engineered_df.merge(aux_df_clean, on="SK_ID_CURR", how="left")
            except Exception as exc:
                print("Failed to join precomputed features:", exc)
        else:
            print("database/precomputed_features.csv not found, skipping merge.")

    # Guarantee all engineered columns are initialized in the dataframe
    for col in ENGINEERED_COLUMNS:
        if col not in engineered_df.columns:
            engineered_df[col] = np.nan

    # Guarantee all columns from selected_features_processed.csv exist
    for col in target_all:
        if col not in engineered_df.columns:
            engineered_df[col] = np.nan

    # Reorder engineered_df columns to match selected_features_processed.csv exactly
    engineered_df = engineered_df[target_all]

    # Save all datasets to the database directory
    db_dir = DATA_DIR / "database"
    os.makedirs(db_dir, exist_ok=True)
    original_df.to_csv(db_dir / "original_uploaded.csv", index=False)
    mapped_df.to_csv(db_dir / "mapped_dataset.csv", index=False)
    engineered_df.to_csv(db_dir / "engineered_features.csv", index=False)

    return original_df, mapped_df, engineered_df

