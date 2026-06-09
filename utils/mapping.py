import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent
PROCESSED_FILE = DATA_DIR / "selected_features_processed.csv"

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
    "EXT_SOURCE_1_missing",
    "EXT_SOURCE_3_missing",
]

ALIASES = {
    "SK_ID_CURR": ["customer_id", "sk_id_curr", "id", "cust_id", "applicant_id", "client_id", "id_curr"],
    "AMT_INCOME_TOTAL": ["income", "annual_salary", "monthly_income", "salary", "earnings", "amt_income_total", "total_income"],
    "AMT_CREDIT": ["loan_amount", "amt_credit", "loan_val", "loan_value", "credit_amount", "credit_val", "amount_credit"],
    "DAYS_BIRTH": ["age", "days_birth", "customer_age", "client_age", "dob", "date_of_birth", "age_in_days", "age_days", "birth_days", "days_of_birth"],
    "DAYS_EMPLOYED": ["years_employed", "days_employed", "employment_years", "employment_length", "tenure", "employment_days", "work_days", "days_of_employment", "days_worked", "employment_duration"],
    "AMT_ANNUITY": ["annuity", "amt_annuity", "monthly_payment", "installment_amount", "amount_annuity"],
    "AMT_GOODS_PRICE": ["goods_price", "amt_goods_price", "purchase_price", "value_of_goods"],
    "CODE_GENDER": ["gender", "code_gender", "sex"],
    "NAME_INCOME_TYPE": ["income_type", "name_income_type", "employment_status", "occupation_type"],
    "NAME_EDUCATION_TYPE": ["education", "name_education_type", "education_level", "academic_degree"],
    "ORGANIZATION_TYPE": ["organization", "organization_type", "employer_industry", "company_type", "industry"],
    "NAME_FAMILY_STATUS": ["family_status", "name_family_status", "marital_status"],
    "NAME_HOUSING_TYPE": ["housing_type", "name_housing_type", "residential_status"],
    "CNT_CHILDREN": ["children", "cnt_children", "number_of_children", "dependents"],
    "CNT_FAM_MEMBERS": ["family_members", "cnt_fam_members", "household_size"]
}


def get_target_columns() -> list[str]:
    """Load the target schema from selected_features_processed.csv columns."""
    if PROCESSED_FILE.exists():
        try:
            df = pd.read_csv(PROCESSED_FILE, nrows=0)
            return list(df.columns)
        except Exception:
            pass
    # Hardcoded fallback of expected training columns
    return [
        "SK_ID_CURR", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", 
        "DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL", "AMT_CREDIT", 
        "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION", 
        "DAYS_LAST_PHONE_CHANGE", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", 
        "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE", 
        "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", 
        "DEF_60_CNT_SOCIAL_CIRCLE", "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", 
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", 
        "AMT_REQ_CREDIT_BUREAU_YEAR", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS", 
        "NAME_HOUSING_TYPE", "EXT_SOURCES_MEAN", "EXT_SOURCES_MAX", "EXT_SOURCES_MIN", 
        "EXT_SOURCES_VARIANCE", "ANNUITY_TO_INCOME_RATIO", "CREDIT_TO_INCOME_RATIO", 
        "GOODS_PRICE_TO_CREDIT_RATIO", "EMPLOYMENT_TO_AGE_RATIO", "INST_DPD_MAX", 
        "INST_DPD_MEAN", "PAYMENT_FRACTION_MEAN", "PAYMENT_FRACTION_MAX", 
        "PAYMENT_FRACTION_LT1_COUNT", "ACTIVE_DEBT_TO_CREDIT_RATIO", "ACTIVE_LOAN_COUNT", 
        "OVERDUE_DEBT_SUM", "PREV_REJECTION_RATE", "PREV_APPROVAL_RATE", 
        "EXT_SOURCE_1_missing", "EXT_SOURCE_3_missing"
    ]


def get_base_target_columns() -> list[str]:
    """Get columns that need direct mapping (base inputs)."""
    target_all = get_target_columns()
    return [col for col in target_all if col not in ENGINEERED_COLUMNS]


def normalize_string(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())


def get_heuristic_mappings(uploaded_cols: list[str], target_cols: list[str], sample_dict: dict) -> dict:
    """Fallback mapping logic using string similarities and aliases."""
    mappings = {}
    normalized_targets = {normalize_string(col): col for col in target_cols}
    
    # Pre-build reverse lookup for ALIASES
    alias_lookup = {}
    for target_col, aliases in ALIASES.items():
        for alias in aliases:
            alias_lookup[normalize_string(alias)] = target_col

    mapped_targets = set()
    
    for col in uploaded_cols:
        norm_col = normalize_string(col)
        matched_col = None
        confidence = 0.0
        explanation = "No matching feature detected."
        
        # 1. Exact or normalized exact match
        if col in target_cols:
            matched_col = col
            confidence = 1.0
            explanation = f"Exact match with model feature '{col}'."
        elif norm_col in normalized_targets:
            matched_col = normalized_targets[norm_col]
            confidence = 0.95
            explanation = f"Exact match ignoring case/spaces (mapped to '{matched_col}')."
            
        # 2. Alias match
        elif norm_col in alias_lookup:
            matched_col = alias_lookup[norm_col]
            confidence = 0.90
            explanation = f"Matched via semantic alias list to model feature '{matched_col}'."
            
        # 3. Substring alias match
        else:
            for target_col, aliases in ALIASES.items():
                for alias in aliases:
                    norm_alias = normalize_string(alias)
                    if len(norm_alias) > 3 and (norm_alias in norm_col or norm_col in norm_alias):
                        matched_col = target_col
                        confidence = 0.80
                        explanation = f"Matched by similarity to alias '{alias}' for '{target_col}'."
                        break
                if matched_col:
                    break
        
        # Avoid duplicate mapping to the same target column
        if matched_col and matched_col not in mapped_targets:
            mapped_targets.add(matched_col)
            mappings[col] = {
                "model_feature": matched_col,
                "confidence": confidence,
                "explanation": explanation
            }
        else:
            mappings[col] = {
                "model_feature": None,
                "confidence": 0.0,
                "explanation": "No matching base feature detected."
            }
            
    return mappings


def get_gemini_mappings(uploaded_cols: list[str], target_cols: list[str], sample_dict: dict, api_key: str) -> dict:
    """Map columns using the Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        print("google-generativeai is not installed, falling back to heuristics.")
        return get_heuristic_mappings(uploaded_cols, target_cols, sample_dict)

    if not api_key:
        print("No API key provided, falling back to heuristics.")
        return get_heuristic_mappings(uploaded_cols, target_cols, sample_dict)

    genai.configure(api_key=api_key)
    
    prompt = f"""
    You are an AI credit risk data engineering expert. Your job is to match columns from an uploaded financial dataset to the target schema of a credit default prediction model.
    
    Uploaded Column Names and 3 sample values for each (in JSON format):
    {json.dumps(sample_dict, indent=2)}
    
    Target Schema Features (available base columns to map to):
    {json.dumps(target_cols, indent=2)}
    
    Match each uploaded column to the single most appropriate target feature. 
    Guidelines:
    - An uploaded column can match at most one target column.
    - Multiple uploaded columns must NOT map to the same target feature.
    - If there is no clear semantic match, set "model_feature" to null.
    - Provide a confidence score between 0.0 and 1.0 (float).
    - Provide a clear, concise explanation of the semantic match.
    
    You must output a valid JSON object only. The keys of the JSON object must be the original uploaded column names. The values must be JSON objects with exactly these keys:
    - "model_feature": the matched target feature name (or null)
    - "confidence": confidence score (float, between 0.0 and 1.0)
    - "explanation": a short human-readable explanation of why you matched them
    
    Example Output Format:
    {{
      "Annual Salary": {{
        "model_feature": "AMT_INCOME_TOTAL",
        "confidence": 0.98,
        "explanation": "Semantically identical to AMT_INCOME_TOTAL representing annual earnings."
      }},
      "User Name": {{
        "model_feature": null,
        "confidence": 0.0,
        "explanation": "No matching feature in target schema."
      }}
    }}
    
    Do not output markdown block formatting (e.g. ```json) around the JSON, just the raw JSON text.
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
            request_options={"timeout": 6.0}
        )
        data = json.loads(response.text.strip())
        
        # Validate output schema
        validated = {}
        for col in uploaded_cols:
            if col in data and isinstance(data[col], dict):
                model_feature = data[col].get("model_feature")
                if model_feature not in target_cols:
                    model_feature = None
                validated[col] = {
                    "model_feature": model_feature,
                    "confidence": float(data[col].get("confidence", 0.0)),
                    "explanation": str(data[col].get("explanation", ""))
                }
            else:
                validated[col] = {
                    "model_feature": None,
                    "confidence": 0.0,
                    "explanation": "Mapping skipped by AI."
                }
        return validated
    except Exception as exc:
        print("Gemini API call failed, falling back to heuristics:", exc)
        return get_heuristic_mappings(uploaded_cols, target_cols, sample_dict)
