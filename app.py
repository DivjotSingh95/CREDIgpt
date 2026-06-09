"""
RiskIntel – main Streamlit application.
Upload ANY CSV / XLSX → XGBoost runs predictions → dashboard KPIs update live.
No column requirements on the uploaded file.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "xgboost_default_model.json"

CATEGORICAL_COLUMNS = [
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
]

st.set_page_config(
    page_title="RiskIntel | Credit Risk Intelligence",
    page_icon="RI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── model helpers ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_model() -> xgb.XGBClassifier | None:
    if MODEL_FILE.exists():
        m = xgb.XGBClassifier()
        m.load_model(str(MODEL_FILE))
        return m
    return None


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str).fillna("missing"))
    return out


def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val) or np.isnan(val) or np.isinf(val):
            return default
        return float(val)
    except Exception:
        return default


def run_predictions(df: pd.DataFrame, model: xgb.XGBClassifier) -> dict:
    """
    Run model on the dataframe. Works even if some expected features are missing
    – missing columns are filled with 0.5 for EXT_SOURCE features and 0.0 for others.
    """
    expected: list[str] = model.get_booster().feature_names  # type: ignore[union-attr]

    # Drop columns not in model; add missing ones
    encoded = encode_categoricals(df)
    for col in expected:
        if col not in encoded.columns:
            encoded[col] = 0.5 if "EXT_SOURCE" in col else 0.0

    X = encoded[expected].copy()

    # Convert any remaining objects to numeric
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.5 if "EXT_SOURCE" in col else 0.0)

    # Fill remaining NaNs: 0.5 for EXT_SOURCE features, 0.0 for others
    for col in X.columns:
        if "EXT_SOURCE" in col or "ext" in col.lower():
            X[col] = X[col].fillna(0.5)
        else:
            X[col] = X[col].fillna(0.0)

    proba = model.predict_proba(X)[:, 1]
    
    # Calibrate probability based on class imbalance correction:
    # XGBoost was trained on 50/50 downsampled data, true prior is ~8%.
    p_true = 0.08
    p_train = 0.5
    proba_cal = (proba * p_true / p_train) / (proba * p_true / p_train + (1 - proba) * (1 - p_true) / (1 - p_train))

    # Determine default prediction at High Risk threshold (>= 15% probability of default)
    pred = (proba_cal >= 0.15).astype(int)

    n = len(df)
    low_risk_mask = proba_cal < 0.07
    medium_risk_mask = (proba_cal >= 0.07) & (proba_cal < 0.15)
    high_risk_mask = proba_cal >= 0.15
    critical_mask = proba_cal >= 0.25

    total_credit = safe_float(df["AMT_CREDIT"].sum()) if "AMT_CREDIT" in df.columns else 0.0
    avg_income = safe_float(df["AMT_INCOME_TOTAL"].mean()) if "AMT_INCOME_TOTAL" in df.columns else 0.0
    avg_credit = safe_float(df["AMT_CREDIT"].mean()) if "AMT_CREDIT" in df.columns else 0.0

    # Construct customer profiles for the UI dropdown selection
    customer_profiles = []
    n_samples = min(1000, len(df))
    for i in range(n_samples):
        row_id = int(df.iloc[i]["SK_ID_CURR"]) if "SK_ID_CURR" in df.columns else (100000 + i)
        row_income = safe_float(df.iloc[i]["AMT_INCOME_TOTAL"]) if "AMT_INCOME_TOTAL" in df.columns else 0.0
        row_credit = safe_float(df.iloc[i]["AMT_CREDIT"]) if "AMT_CREDIT" in df.columns else 0.0
        row_annuity = safe_float(df.iloc[i]["AMT_ANNUITY"]) if "AMT_ANNUITY" in df.columns else 0.0
        
        row_age_days = safe_float(df.iloc[i]["DAYS_BIRTH"]) if "DAYS_BIRTH" in df.columns else np.nan
        row_age = safe_float(abs(row_age_days) / 365.25) if not np.isnan(row_age_days) else 0.0
        
        row_emp_days = safe_float(df.iloc[i]["DAYS_EMPLOYED"]) if "DAYS_EMPLOYED" in df.columns else np.nan
        row_emp = safe_float(abs(row_emp_days) / 365.25) if (not np.isnan(row_emp_days) and row_emp_days < 365243) else 0.0
        
        row_pd = safe_float(proba_cal[i])
        row_gender = str(df.iloc[i]["CODE_GENDER"]) if "CODE_GENDER" in df.columns else "Unknown"
        row_income_type = str(df.iloc[i]["NAME_INCOME_TYPE"]) if "NAME_INCOME_TYPE" in df.columns else "Unknown"
        row_education = str(df.iloc[i]["NAME_EDUCATION_TYPE"]) if "NAME_EDUCATION_TYPE" in df.columns else "Unknown"
        
        row_ext1 = safe_float(df.iloc[i]["EXT_SOURCE_1"], 0.5) if "EXT_SOURCE_1" in df.columns else 0.5
        row_ext2 = safe_float(df.iloc[i]["EXT_SOURCE_2"], 0.5) if "EXT_SOURCE_2" in df.columns else 0.5
        row_ext3 = safe_float(df.iloc[i]["EXT_SOURCE_3"], 0.5) if "EXT_SOURCE_3" in df.columns else 0.5
        
        # If actual values are missing, default display to 0.5 midpoint
        if np.isnan(row_ext1):
            row_ext1 = 0.5
        if np.isnan(row_ext2):
            row_ext2 = 0.5
        if np.isnan(row_ext3):
            row_ext3 = 0.5
            
        risk_lvl = "High" if row_pd >= 0.15 else "Medium" if row_pd >= 0.07 else "Low"
        
        customer_profiles.append({
            "id": row_id,
            "income": row_income,
            "credit": row_credit,
            "annuity": row_annuity,
            "age": row_age,
            "employment": row_emp,
            "pd": row_pd,
            "risk_level": risk_lvl,
            "gender": row_gender,
            "income_type": row_income_type,
            "education": row_education,
            "ext1": row_ext1,
            "ext2": row_ext2,
            "ext3": row_ext3
        })

    return {
        "total_customers": n,
        "default_rate": safe_float(proba_cal.mean()),
        "avg_risk_score": safe_float(proba_cal.mean()),
        "low_risk_count": int(low_risk_mask.sum()),
        "medium_risk_count": int(medium_risk_mask.sum()),
        "high_risk_count": int(high_risk_mask.sum()),
        "critical_count": int(critical_mask.sum()),
        "total_portfolio": safe_float(total_credit),
        "avg_income": safe_float(avg_income),
        "avg_credit": safe_float(avg_credit),
        "proba": [safe_float(p) for p in proba_cal],
        "pred": [int(p) for p in pred],
        "customer_profiles": customer_profiles
    }

# ── KPI helpers ────────────────────────────────────────────────────────────────

def fmt_money(v: float) -> str:
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    if v >= 1e6:
        return f"${v/1e6:.1f}M"
    if v >= 1e3:
        return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def fmt_num(v: int) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}k"
    return str(v)


def build_kpis(stats: dict) -> dict[str, str]:
    n = stats.get("total_customers") or 1
    return {
        "TOTAL_CUSTOMERS": fmt_num(stats["total_customers"]),
        "ACTIVE_PORTFOLIO": fmt_money(stats["total_portfolio"]),
        "AVG_RISK_SCORE": f"{stats['avg_risk_score']:.0%}",
        "HIGH_RISK_CUSTOMERS": fmt_num(stats["high_risk_count"]),
        "DEFAULT_RATE": f"{stats['default_rate']:.2%}",
        "CRITICAL_EXPOSURE": fmt_num(stats["critical_count"]),
        "LOW_RISK_PCT": f"{stats.get('low_risk_count', 0) / n:.0%}",
        "MEDIUM_RISK_PCT": f"{stats.get('medium_risk_count', 0) / n:.0%}",
        "HIGH_RISK_PCT": f"{stats.get('high_risk_count', 0) / n:.0%}",
        "LOW_RISK_COUNT": fmt_num(stats.get('low_risk_count', 0)),
        "MEDIUM_RISK_COUNT": fmt_num(stats.get('medium_risk_count', 0)),
        "HIGH_RISK_COUNT": fmt_num(stats.get('high_risk_count', 0)),
    }


def build_dynamic_insights(stats: dict) -> list[dict]:
    profiles = stats.get("customer_profiles", [])
    if not profiles:
        return []
        
    df = pd.DataFrame(profiles)
    insights = []
    
    # 1. Gender risk insight
    if "gender" in df.columns and "pd" in df.columns:
        gender_stats = df.groupby("gender")["pd"].mean()
        if len(gender_stats) >= 2:
            high_gender = gender_stats.idxmax()
            high_gender_val = gender_stats.max()
            low_gender = gender_stats.idxmin()
            low_gender_val = gender_stats.min()
            insights.append({
                "title": "Gender Risk Disparity",
                "desc": f"Male default risk averages {high_gender_val:.1%} vs Female default risk at {low_gender_val:.1%}.",
                "icon": "wc",
                "color": "var(--secondary)"
            })
        
    # 2. Education driver
    if "education" in df.columns and "pd" in df.columns:
        edu_stats = df.groupby("education")["pd"].mean()
        if not edu_stats.empty:
            high_edu = edu_stats.idxmax()
            high_edu_val = edu_stats.max()
            insights.append({
                "title": "Education Risk Factor",
                "desc": f"Customers with '{high_edu}' have the highest average default risk of {high_edu_val:.1%}.",
                "icon": "school",
                "color": "var(--amber)"
            })
        
    # 3. Credit Leverage
    if "credit" in df.columns and "income" in df.columns and "pd" in df.columns:
        df["cir"] = df["credit"] / df["income"].replace(0, 1)
        high_risk_cir = df[df["pd"] >= 0.15]["cir"].mean()
        low_risk_cir = df[df["pd"] < 0.07]["cir"].mean()
        if not np.isnan(high_risk_cir) and not np.isnan(low_risk_cir):
            insights.append({
                "title": "Credit Leverage Ratio",
                "desc": f"High risk accounts hold a leverage ratio of {high_risk_cir:.1f}x income vs {low_risk_cir:.1f}x for low risk accounts.",
                "icon": "account_balance_wallet",
                "color": "var(--success)"
            })
            
    # 4. Age and Employment
    if "age" in df.columns and "employment" in df.columns and "pd" in df.columns:
        young_risk = df[df["age"] < 35]["pd"].mean()
        old_risk = df[df["age"] >= 55]["pd"].mean()
        if not np.isnan(young_risk) and not np.isnan(old_risk):
            insights.append({
                "title": "Age Demographics",
                "desc": f"Younger borrowers (<35 yrs) default probability averages {young_risk:.1%} vs {old_risk:.1%} for mature borrowers.",
                "icon": "calendar_month",
                "color": "var(--teal)"
            })
            
    return insights


def build_insights_html(insights: list[dict]) -> str:
    if not insights:
        return """
        <li><span class="material-symbols-outlined" style="color:var(--secondary);">neurology</span><div><strong>EXT_SOURCE Signals</strong><p class="card-subtitle">External credit bureau scores are the strongest predictor in the model.</p></div></li>
        <li><span class="material-symbols-outlined" style="color:var(--amber);">pattern</span><div><strong>Days Employed Ratio</strong><p class="card-subtitle">Employment-to-age ratio shows high predictive weight for default risk.</p></div></li>
        <li><span class="material-symbols-outlined" style="color:var(--success);">verified_user</span><div><strong>Payment History</strong><p class="card-subtitle">DPD metrics and payment fraction strongly separate low vs high risk.</p></div></li>
        """
    html = ""
    for ins in insights:
        html += f"""
        <li>
          <span class="material-symbols-outlined" style="color:{ins['color']};">{ins['icon']}</span>
          <div>
            <strong>{ins['title']}</strong>
            <p class="card-subtitle">{ins['desc']}</p>
          </div>
        </li>"""
    return html


def build_dist_bars(proba: list[float]) -> str:
    """Build HTML for score distribution chart from probability array."""
    arr = np.array(proba)
    buckets = [
        ("0–5%", (arr < 0.05).sum(), False),
        ("5–10%", ((arr >= 0.05) & (arr < 0.10)).sum(), False),
        ("10–15%", ((arr >= 0.10) & (arr < 0.15)).sum(), True),
        ("15–20%", ((arr >= 0.15) & (arr < 0.20)).sum(), True),
        (">20%", (arr >= 0.20).sum(), True),
    ]
    max_count = max(c for _, c, _ in buckets) or 1
    items = ""
    for label, count, hot in buckets:
        pct = int(count / max_count * 100)
        cls = "dist-bar hot" if hot else "dist-bar"
        items += f"""
        <div class="dist-item">
          <div class="{cls}" style="--h:{pct}%;"></div>
          <small>{label}</small>
        </div>"""
    return items


# ── HTML shell ─────────────────────────────────────────────────────────────────

def app_html(kpis: dict[str, str] | None = None, dist_bars: str = "", has_data: bool = False, customer_profiles_json: str = "[]", gemini_api_key: str = "", dataset_summary_json: str = "{}", dist_bars_overview: str = "", insights_html: str = "") -> str:  # noqa: FBT001
    # Defaults (shown when no data uploaded yet)
    k = {
        "TOTAL_CUSTOMERS": "—",
        "ACTIVE_PORTFOLIO": "—",
        "AVG_RISK_SCORE": "—",
        "HIGH_RISK_CUSTOMERS": "—",
        "DEFAULT_RATE": "—",
        "CRITICAL_EXPOSURE": "—",
        "LOW_RISK_PCT": "—",
        "MEDIUM_RISK_PCT": "—",
        "HIGH_RISK_PCT": "—",
        "LOW_RISK_COUNT": "—",
        "MEDIUM_RISK_COUNT": "—",
        "HIGH_RISK_COUNT": "—",
    }
    if kpis:
        k.update(kpis)

    upload_badge = (
        '<span class="badge" style="color:var(--success);background:rgba(5,150,105,.12);">'
        '✓ Data Loaded</span>'
        if has_data else
        '<span class="badge" style="color:var(--amber);background:rgba(217,119,6,.12);">No Data</span>'
    )

    if not dist_bars:
        dist_bars = """
        <div class="dist-item"><div class="dist-bar" style="--h:20%;"></div><small>0–20%</small></div>
        <div class="dist-item"><div class="dist-bar" style="--h:45%;"></div><small>20–40%</small></div>
        <div class="dist-item"><div class="dist-bar" style="--h:85%;background:rgba(75,65,225,.42);"></div><small>40–60%</small></div>
        <div class="dist-item"><div class="dist-bar" style="--h:30%;"></div><small>60–80%</small></div>
        <div class="dist-item"><div class="dist-bar hot" style="--h:15%;"></div><small>80–100%</small></div>
        """

    if not dist_bars_overview:
        dist_bars_overview = """
        <div class="chart-bar" style="--h:38%;--fill:50%;"></div>
        <div class="chart-bar" style="--h:52%;--fill:50%;"></div>
        <div class="chart-bar" style="--h:76%;--fill:58%;"></div>
        <div class="chart-bar" style="--h:62%;--fill:50%;"></div>
        <div class="chart-bar" style="--h:92%;--fill:78%;"></div>
        <div class="chart-bar" style="--h:72%;--fill:74%;"></div>
        <div class="chart-bar" style="--h:52%;--fill:56%;"></div>
        """

    if not insights_html:
        insights_html = """
        <li><span class="material-symbols-outlined" style="color:var(--secondary);">neurology</span><div><strong>EXT_SOURCE Signals</strong><p class="card-subtitle">External credit bureau scores are the strongest predictor in the model.</p></div></li>
        <li><span class="material-symbols-outlined" style="color:var(--amber);">pattern</span><div><strong>Days Employed Ratio</strong><p class="card-subtitle">Employment-to-age ratio shows high predictive weight for default risk.</p></div></li>
        <li><span class="material-symbols-outlined" style="color:var(--success);">verified_user</span><div><strong>Payment History</strong><p class="card-subtitle">DPD metrics and payment fraction strongly separate low vs high risk.</p></div></li>
        """

    return rf"""<!DOCTYPE html>
<html lang="en" class="light">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --on-primary-fixed: #131b2e;
      --surface-bright: #f7f9fb;
      --primary-fixed-dim: #bec6e0;
      --outline: #76777d;
      --primary-fixed: #dae2fd;
      --error: #ba1a1a;
      --surface-container-lowest: #ffffff;
      --background: #f7f9fb;
      --on-primary-container: #7c839b;
      --on-error: #ffffff;
      --surface-container-high: #e6e8ea;
      --surface: #f7f9fb;
      --outline-variant: #c6c6cd;
      --on-background: #191c1e;
      --error-container: #ffdad6;
      --on-primary: #ffffff;
      --on-error-container: #93000a;
      --on-surface: #191c1e;
      --surface-container: #eceef0;
      --primary: #000000;
      --secondary-container: #645efb;
      --surface-variant: #e0e3e5;
      --on-surface-variant: #45464d;
      --secondary-fixed: #e2dfff;
      --primary-container: #131b2e;
      --surface-container-low: #f2f4f6;
      --secondary: #4b41e1;
      --on-secondary: #ffffff;
      --surface-container-highest: #e0e3e5;
      --teal: #009485;
      --success: #059669;
      --amber: #d97706;
      --shadow: 0 1px 3px rgba(15,23,42,.04), 0 18px 45px rgba(15,23,42,.055);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; background: var(--surface); color: var(--on-surface); font-family: Inter, sans-serif; }}
    body {{ overflow-x: hidden; }}
    button, input, select, textarea {{ font-family: Inter, sans-serif; }}
    .material-symbols-outlined {{ font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 22; line-height: 1; }}

    .app-shell {{ min-height: 100vh; background: radial-gradient(circle at 78% 4%, rgba(226,223,255,.55), transparent 28%), var(--surface); }}
    .sidebar {{
      position: fixed; left: 0; top: 0; bottom: 0; width: 280px; z-index: 100;
      display: flex; flex-direction: column; background: rgba(247,249,251,.88);
      backdrop-filter: blur(18px); border-right: 1px solid rgba(198,198,205,.42);
      box-shadow: 0 12px 35px rgba(15,23,42,.035);
      transform: translateX(-280px);
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .sidebar::after {{
      content: '';
      position: absolute; right: -24px; top: 0; bottom: 0; width: 24px;
      background: transparent;
      z-index: 99;
    }}
    .sidebar:hover {{
      transform: translateX(0);
      box-shadow: 12px 0 50px rgba(15,23,42,.08);
    }}
    .brand {{ padding: 24px; }}
    .brand-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 28px; }}
    .brand-mark {{ width: 40px; height: 40px; border-radius: 10px; background: var(--primary-container); display: grid; place-items: center; color: white; }}
    .brand-title {{ font-family: Manrope, sans-serif; font-size: 24px; line-height: 1; font-weight: 800; color: var(--on-primary-fixed); }}
    .brand-tier {{ display: block; margin-top: 3px; font-size: 12px; line-height: 16px; font-weight: 600; color: var(--on-primary-container); }}
    .nav {{ display: flex; flex-direction: column; gap: 4px; }}
    .nav-btn {{
      appearance: none; border: 0; background: transparent; color: var(--on-surface-variant);
      display: flex; align-items: center; gap: 16px; width: 100%; min-height: 45px;
      padding: 0 16px; border-radius: 0 9px 9px 0; border-left: 4px solid transparent;
      font-weight: 700; font-size: 14px; cursor: pointer; transition: .18s ease; text-decoration: none;
    }}
    .nav-btn:hover {{ transform: translateX(4px); background: var(--surface-container-high); color: var(--on-surface); }}
    .nav-btn.active {{ transform: scale(.98); border-left-color: var(--secondary); background: rgba(100,94,251,.10); color: var(--secondary); font-weight: 800; }}
    .sidebar-footer {{ margin-top: auto; padding: 18px 24px; border-top: 1px solid rgba(198,198,205,.3); }}

    .topbar {{
      position: fixed; left: 0; right: 0; top: 0; height: 64px; z-index: 15;
      display: flex; align-items: center; justify-content: space-between; padding: 0 24px 0 32px;
      background: rgba(247,249,251,.84); backdrop-filter: blur(14px); box-shadow: 0 1px 8px rgba(15,23,42,.03);
      transition: left 0.3s ease;
    }}
    .search {{ width: min(380px, 35vw); height: 40px; display: flex; align-items: center; gap: 10px; background: var(--surface-container-low); border: 1px solid rgba(198,198,205,.4); border-radius: 999px; padding: 0 14px; }}
    .search input {{ flex: 1; border: 0; outline: 0; background: transparent; font-size: 14px; color: var(--on-surface); }}
    .kbd {{ font-size: 11px; color: var(--outline); background: var(--surface-container-high); border-radius: 5px; padding: 2px 7px; font-weight: 700; }}
    .top-links {{ display: flex; align-items: center; gap: 24px; }}
    .top-links a {{ color: var(--on-surface-variant); text-decoration: none; font-weight: 700; font-size: 14px; }}
    .top-actions {{ display: flex; align-items: center; gap: 12px; }}
    .icon-btn {{ width: 40px; height: 40px; display: grid; place-items: center; border: 0; border-radius: 999px; color: var(--on-surface-variant); background: transparent; cursor: pointer; }}
    .icon-btn:hover {{ background: var(--surface-container-high); }}
    .avatar {{ width: 36px; height: 36px; border-radius: 999px; background: linear-gradient(135deg,#dae2fd,#4b41e1); }}

    main {{ margin-left: 0; padding: 88px 32px 48px; transition: margin-left 0.3s ease; }}
    .page {{ display: none; }}
    .page.active {{ display: block; }}
    .section {{ margin-top: 32px; }}
    .card {{ background: var(--surface-container-lowest); border-radius: 14px; padding: 24px; box-shadow: var(--shadow); border: 1px solid rgba(198,198,205,.32); }}
    .glass-card {{ background: rgba(255,255,255,.7); backdrop-filter: blur(12px); border-radius: 16px; padding: 24px; border: 1px solid rgba(198,198,205,.42); box-shadow: var(--shadow); }}
    .hover-lift {{ transition: transform .2s ease, box-shadow .2s ease; }}
    .hover-lift:hover {{ transform: translateY(-4px); box-shadow: 0 24px 56px rgba(15,23,42,.08); }}
    .grid-12 {{ display: grid; grid-template-columns: repeat(12,minmax(0,1fr)); gap: 24px; }}
    .span-12 {{ grid-column: span 12; }} .span-8 {{ grid-column: span 8; }} .span-7 {{ grid-column: span 7; }}
    .span-6 {{ grid-column: span 6; }} .span-5 {{ grid-column: span 5; }} .span-4 {{ grid-column: span 4; }}
    .span-3 {{ grid-column: span 3; }}
    .hero-copy {{ grid-column: span 7; padding: 20px 0; }}
    .hero-visual {{ grid-column: span 5; display: flex; align-items: center; }}
    .eyebrow {{ display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; color: var(--secondary); margin-bottom: 16px; }}
    .display {{ font-family: Manrope, sans-serif; font-size: 56px; line-height: 64px; font-weight: 800; color: var(--on-primary-fixed); margin: 0 0 18px; }}
    .lead {{ font-size: 18px; line-height: 28px; color: var(--on-surface-variant); margin: 0 0 28px; }}
    .actions {{ display: flex; gap: 14px; flex-wrap: wrap; }}
    .btn {{ border: 0; border-radius: 10px; padding: 0 22px; min-height: 44px; font-size: 14px; font-weight: 700; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; transition: .18s ease; }}
    .btn.primary {{ background: var(--primary-container); color: white; }}
    .btn.primary:hover {{ background: #1e2d4c; }}
    .btn.secondary {{ background: var(--surface-container-high); color: var(--on-surface); }}
    .btn.secondary:hover {{ background: var(--surface-container-highest); }}
    .badge {{ display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 900; }}
    .badge.good {{ color: var(--success); background: rgba(5,150,105,.12); }}
    .badge.bad {{ color: var(--error); background: rgba(186,26,26,.1); }}
    .badge.warn {{ color: var(--amber); background: rgba(217,119,6,.12); }}

    .stats {{ display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 24px; margin-top: 32px; }}
    .stat-card {{ display: flex; flex-direction: column; gap: 14px; }}
    .stat-top {{ display: flex; align-items: center; justify-content: space-between; }}
    .stat-icon {{ width: 44px; height: 44px; display: grid; place-items: center; border-radius: 12px; background: rgba(75,65,225,.1); color: var(--secondary); }}
    .metric-label {{ margin: 0; font-size: 13px; color: var(--on-primary-container); font-weight: 700; }}
    .metric-value {{ margin: 0; font-family: Manrope, sans-serif; font-size: 32px; font-weight: 800; line-height: 1; }}

    .heatmap-grid {{ display: grid; grid-template-columns: repeat(12,1fr); gap: 3px; padding: 16px 0; }}
    .heatmap-cell {{ aspect-ratio: 1; border-radius: 3px; }}
    .mini-chart {{ position: relative; height: 80px; margin-top: 12px; background: var(--surface-container-low); border-radius: 10px; overflow: hidden; }}
    .bars {{ display: flex; align-items: flex-end; gap: 3px; height: 100%; padding: 8px; }}
    .bars span {{ flex: 1; border-radius: 4px 4px 0 0; background: var(--secondary); height: var(--h); opacity: var(--o); }}

    .card-head {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; gap: 16px; }}
    .card-title {{ margin: 0; font-family: Manrope, sans-serif; font-size: 16px; font-weight: 800; color: var(--on-primary-fixed); }}
    .card-subtitle {{ margin: 4px 0 0; font-size: 13px; color: var(--on-primary-container); }}
    .chart-box {{ display: flex; align-items: flex-end; gap: 8px; height: 220px; }}
    .chart-bar {{ flex: 1; border-radius: 8px 8px 0 0; background: var(--surface-container-high); position: relative; height: var(--h); }}
    .chart-bar::after {{ content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: var(--fill); border-radius: inherit; background: linear-gradient(180deg, var(--secondary), rgba(75,65,225,.55)); }}
    .mini-kpis {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-top: 20px; }}
    .tiny-title {{ margin: 0 0 8px; font-size: 12px; font-weight: 900; text-transform: uppercase; letter-spacing: .1em; color: var(--on-primary-container); }}
    .tiny-value {{ margin: 0; font-family: Manrope, sans-serif; font-size: 24px; line-height: 32px; font-weight: 800; }}

    .insight-list {{ list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 12px; }}
    .insight-list li {{ display: flex; align-items: flex-start; gap: 16px; padding: 16px; border-radius: 12px; transition: .18s ease; }}
    .insight-list li:hover {{ background: var(--surface-container-low); }}
    .ai-box {{ margin-top: 24px; background: var(--primary-container); color: var(--on-primary); border-radius: 12px; padding: 24px; overflow: hidden; position: relative; }}
    .ai-input {{ display: flex; gap: 8px; align-items: center; padding: 8px; border: 1px solid rgba(255,255,255,.2); border-radius: 10px; background: rgba(255,255,255,.10); }}
    .ai-input input, .ai-input textarea {{ flex: 1; border: 0; outline: 0; background: transparent; color: inherit; resize: none; }}
    .ai-input input::placeholder, .ai-input textarea::placeholder {{ color: rgba(255,255,255,.55); }}

    .node-section {{ overflow: hidden; display: grid; grid-template-columns: 1fr 1fr; }}
    .node-art {{ min-height: 320px; position: relative; background: linear-gradient(135deg,#131b2e,#4b41e1 55%,#71f8e4); }}
    .node-art::before {{ content: ""; position: absolute; inset: 36px; border: 1px solid rgba(255,255,255,.25); border-radius: 16px; }}
    .node {{ position: absolute; width: 16px; height: 16px; border-radius: 50%; background: white; box-shadow: 0 0 30px rgba(255,255,255,.75); }}
    .node.n1 {{ left: 18%; top: 28%; }} .node.n2 {{ left: 70%; top: 38%; width: 22px; height: 22px; }} .node.n3 {{ left: 48%; top: 70%; }}
    .pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 999px; background: var(--surface-container-high); font-weight: 800; font-size: 14px; margin: 4px 8px 4px 0; }}
    .dot {{ width: 8px; height: 8px; border-radius: 999px; background: var(--secondary); display: inline-block; }}

    /* Portfolio page */
    .page-title-row {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin-bottom: 28px; }}
    .page-eyebrow {{ color: var(--secondary); text-transform: uppercase; letter-spacing: .18em; font-size: 13px; font-weight: 900; }}
    .page-title {{ margin: 5px 0 0; font-family: Manrope, sans-serif; font-size: 48px; line-height: 56px; font-weight: 800; }}
    .filters {{ display: flex; align-items: center; gap: 14px; padding: 8px; border-radius: 14px; border: 1px solid rgba(198,198,205,.4); background: white; box-shadow: var(--shadow); }}
    .filter-group {{ padding: 0 12px; border-right: 1px solid rgba(198,198,205,.35); }}
    .filter-group:last-child {{ border-right: 0; }}
    .filter-label {{ display: block; color: var(--outline); font-size: 10px; font-weight: 900; text-transform: uppercase; }}
    select {{ border: 0; outline: 0; background: transparent; font-weight: 800; color: var(--on-surface); }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(6,minmax(0,1fr)); gap: 24px; margin-bottom: 24px; }}
    .kpi {{ min-height: 150px; display: flex; flex-direction: column; justify-content: space-between; overflow: hidden; position: relative; }}
    .kpi::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 4px; background: rgba(75,65,225,.14); transition: .18s ease; }}
    .kpi.bad::after {{ background: rgba(186,26,26,.14); }}
    .kpi:hover::after {{ height: 7px; }}
    .map {{ min-height: 430px; border-radius: 12px; background: radial-gradient(circle at 34% 34%,rgba(186,26,26,.5),transparent 9%), radial-gradient(circle at 74% 48%,rgba(75,65,225,.55),transparent 9%), linear-gradient(135deg,#dfe4ec,#f7f9fb); position: relative; overflow: hidden; }}
    .map svg {{ position: absolute; inset: 0; width: 100%; height: 100%; opacity: .7; }}
    .map-point {{ position: absolute; width: 16px; height: 16px; border-radius: 999px; box-shadow: 0 0 0 8px rgba(186,26,26,.12); }}
    .map-point.red {{ background: var(--error); left: 32%; top: 28%; animation: pulse 1.8s infinite; }}
    .map-point.blue {{ background: var(--secondary); left: 73%; top: 49%; box-shadow: 0 0 0 8px rgba(75,65,225,.13); }}
    @keyframes pulse {{ 50% {{ transform: scale(1.18); }} }}
    .dist-chart {{ height: 280px; display: flex; align-items: flex-end; gap: 16px; }}
    .dist-item {{ flex: 1; display: flex; flex-direction: column; align-items: center; gap: 9px; height: 100%; justify-content: flex-end; }}
    .dist-bar {{ width: 100%; border-radius: 10px 10px 0 0; background: rgba(75,65,225,.16); height: var(--h); }}
    .dist-bar.hot {{ background: rgba(186,26,26,.22); }}

    /* Upload page */
    .upload-hero {{ max-width: 960px; margin: 0 auto 40px; text-align: center; }}
    .upload-zone {{ max-width: 640px; margin: 0 auto; }}
    .upload-notice {{ max-width: 640px; margin: 24px auto 0; background: rgba(75,65,225,.06); border: 1px solid rgba(75,65,225,.18); border-radius: 12px; padding: 18px 24px; font-size: 14px; color: var(--on-surface-variant); }}

    /* Customer page */
    .profile-grid {{ display: grid; grid-template-columns: 4fr 8fr; gap: 24px; }}
    .customer-avatar {{ width: 64px; height: 64px; border-radius: 999px; background: linear-gradient(135deg,#dae2fd,#4b41e1); border: 2px solid var(--surface); }}
    .risk-chip {{ display: inline-flex; align-items: center; padding: 5px 11px; border-radius: 999px; background: rgba(255,218,214,.85); color: var(--error); font-size: 12px; font-weight: 900; }}
    .driver {{ padding: 16px; border-radius: 10px; border: 1px solid rgba(198,198,205,.42); background: var(--surface-bright); margin-top: 14px; }}
    .gauge {{ min-height: 300px; display: grid; place-items: center; text-align: center; }}
    .score {{ font-family: Manrope, sans-serif; font-size: 72px; line-height: 1; font-weight: 800; color: var(--error); }}
    .timeline {{ position: relative; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 46px 14px 24px; }}
    .timeline::before {{ content: ""; position: absolute; left: 16px; right: 16px; top: 53px; height: 2px; background: rgba(198,198,205,.35); }}
    .time-node {{ position: relative; z-index: 1; text-align: center; font-size: 11px; font-weight: 900; color: var(--outline); text-transform: uppercase; }}
    .time-node::before {{ content: ""; display: block; width: 16px; height: 16px; border-radius: 999px; background: #14b8a6; border: 4px solid white; box-shadow: 0 3px 8px rgba(15,23,42,.12); margin: 0 auto 12px; }}
    .time-node.bad {{ color: var(--error); }}
    .time-node.bad::before {{ width: 20px; height: 20px; background: var(--error); }}

    /* Chat page */
    .chat-layout {{ display: grid; grid-template-columns: 1fr 1fr; height: 720px; border: 1px solid rgba(198,198,205,.32); border-radius: 12px; overflow: hidden; background: white; }}
    .chat-left {{ display: flex; flex-direction: column; border-right: 1px solid rgba(198,198,205,.28); background: var(--surface-container-lowest); }}
    .chat-head {{ padding: 20px 24px; border-bottom: 1px solid rgba(198,198,205,.18); display: flex; justify-content: space-between; align-items: center; }}
    .chat-messages {{ flex: 1; padding: 24px; overflow: auto; }}
    .chat-row {{ display: flex; align-items: flex-start; gap: 14px; margin-bottom: 22px; }}
    .chat-row.user {{ justify-content: flex-end; }}
    .bubble {{ max-width: 86%; padding: 14px 16px; border-radius: 18px 18px 18px 4px; background: var(--surface-container); box-shadow: 0 4px 12px rgba(15,23,42,.04); line-height: 1.5; }}
    .bubble.user {{ border-radius: 18px 18px 4px 18px; background: var(--secondary); color: white; }}
    .chat-input-wrap {{ padding: 20px 24px; border-top: 1px solid rgba(198,198,205,.34); background: var(--surface); }}
    .suggestions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
    .suggestions button {{ border: 1px solid var(--outline-variant); background: white; color: var(--on-surface); border-radius: 999px; padding: 8px 12px; font-weight: 600; cursor: pointer; }}
    .chat-right {{ padding: 24px; background: var(--surface-bright); overflow: auto; }}
    .table {{ width: 100%; border-collapse: collapse; }}
    .table th {{ padding: 12px 8px; border-bottom: 1px solid rgba(198,198,205,.35); text-align: left; color: var(--outline); text-transform: uppercase; font-size: 12px; font-weight: 900; }}
    .table td {{ padding: 14px 8px; border-bottom: 1px solid rgba(198,198,205,.18); font-weight: 700; }}

    /* Settings */
    .settings-grid {{ display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 24px; }}
    .toggle-line {{ display: flex; align-items: center; justify-content: space-between; padding: 16px 0; border-bottom: 1px solid rgba(198,198,205,.25); }}
    .toggle {{ width: 42px; height: 24px; border-radius: 999px; background: var(--secondary); position: relative; }}
    .toggle::after {{ content: ""; position: absolute; right: 3px; top: 3px; width: 18px; height: 18px; border-radius: 999px; background: white; }}

    .progress {{ height: 8px; border-radius: 999px; background: var(--surface-container); overflow: hidden; }}
    .progress span {{ display: block; height: 100%; border-radius: inherit; background: var(--secondary); width: var(--w); }}

    @media (max-width: 1100px) {{
      .sidebar {{ position: fixed; transform: translateX(-280px); z-index: 100; }}
      .sidebar:hover {{ transform: translateX(0); }}
      .topbar {{ position: fixed; left: 0; width: 100%; }}
      main {{ margin-left: 0; padding-top: 88px; }}
      .grid-12, .upload-layout, .profile-grid, .chat-layout, .node-section {{ grid-template-columns: 1fr; }}
      .hero-copy, .hero-visual, .span-8, .span-7, .span-6, .span-5, .span-4, .span-3, .span-12 {{ grid-column: span 1; }}
      .stats, .kpi-grid, .mini-kpis, .settings-grid {{ grid-template-columns: 1fr; }}
      .top-links {{ display: none; }}
      .search {{ width: 52vw; }}
    }}
  </style>
</head>
<body>
<div class="app-shell">
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-row">
        <div class="brand-mark"><span class="material-symbols-outlined" style="font-variation-settings:'FILL' 1;">security</span></div>
        <div><div class="brand-title">RiskIntel</div><span class="brand-tier">Enterprise Tier</span></div>
      </div>
      <nav class="nav" id="nav">
        <button class="nav-btn active" data-page="overview"><span class="material-symbols-outlined">dashboard</span>Overview</button>
        <a class="nav-btn" href="/?page=upload" target="_top" data-route="upload"><span class="material-symbols-outlined">cloud_upload</span>Upload Dataset</a>
        <button class="nav-btn" data-page="portfolio"><span class="material-symbols-outlined">pie_chart</span>Portfolio Dashboard</button>
        <button class="nav-btn" data-page="customer"><span class="material-symbols-outlined">analytics</span>Customer Risk Analytics</button>
        <button class="nav-btn" data-page="xai"><span class="material-symbols-outlined">psychology</span>Explainable AI</button>
        <button class="nav-btn" data-page="ask"><span class="material-symbols-outlined">chat_paste_go</span>Ask Your Data</button>
        <button class="nav-btn" data-page="settings"><span class="material-symbols-outlined">settings</span>Settings</button>
      </nav>
    </div>
    <div class="sidebar-footer">
      <button class="nav-btn"><span class="material-symbols-outlined">help</span>Support</button>
      <button class="nav-btn"><span class="material-symbols-outlined">account_circle</span>Profile</button>
    </div>
  </aside>

  <header class="topbar">
    <div class="search"><span class="material-symbols-outlined" style="cursor:pointer;" onclick="handleGlobalSearch(event)">search</span><input id="global-search" placeholder="Search Risk Profiles (Enter ID)..." onkeydown="handleGlobalSearch(event)" /><span class="kbd">Enter</span></div>
    <div class="top-links"><a>Market Data</a><a>Liquidity</a><a>Regulatory</a></div>
    <div class="top-actions">
      <span style="font-size:13px;font-weight:700;color:var(--on-surface-variant);">{upload_badge}</span>
      <button class="icon-btn"><span class="material-symbols-outlined">notifications</span></button>
      <div style="height:32px;width:1px;background:rgba(198,198,205,.5);"></div>
      <div class="avatar"></div>
    </div>
  </header>

  <main>
    <!-- OVERVIEW PAGE -->
    <section class="page active" id="overview">
      <div class="grid-12">
        <div class="hero-copy">
          <div class="eyebrow"><span class="material-symbols-outlined" style="font-variation-settings:'FILL' 1;">auto_awesome</span>Enterprise Risk Control</div>
          <h1 class="display">Credit Risk<br />Intelligence Platform</h1>
          <p class="lead">AI-powered risk analytics, customer scoring, portfolio monitoring, and explainable decision intelligence for high-stakes fintech operations.</p>
          <div class="actions">
            <a class="btn primary" href="/?page=upload" data-route="upload">Upload Dataset</a>
            <button class="btn secondary">View Documentation</button>
          </div>
        </div>
        <div class="hero-visual">
          <div class="glass-card floating" style="width:100%;">
            <div class="card-head"><strong>Global Risk Concentration</strong><div style="display:flex;gap:5px;"><span class="dot" style="background:var(--error)"></span><span class="dot"></span><span class="dot" style="background:var(--outline-variant)"></span></div></div>
            <div class="heatmap-grid" id="heatmap-overview"></div>
            <div class="mini-chart">
              <span style="position:absolute;right:16px;top:12px;font-size:10px;color:var(--outline);font-weight:800;">Live Updates 2ms</span>
              <div class="bars"><span style="--h:28%;--o:.2"></span><span style="--h:44%;--o:.3"></span><span style="--h:62%;--o:.4"></span><span style="--h:94%;--o:.65"></span><span style="--h:78%;--o:1"></span><span style="--h:55%;--o:.5"></span><span style="--h:86%;--o:.8"></span></div>
            </div>
          </div>
        </div>
      </div>

      <!-- LIVE KPI CARDS -->
      <div class="stats">
        <div class="card stat-card">
          <div class="stat-top"><div class="stat-icon"><span class="material-symbols-outlined">group</span></div>{upload_badge}</div>
          <p class="metric-label">Total Customers</p>
          <p class="metric-value">{k["TOTAL_CUSTOMERS"]}</p>
        </div>
        <div class="card stat-card">
          <div class="stat-top"><div class="stat-icon"><span class="material-symbols-outlined">account_balance_wallet</span></div><span class="badge" style="color:var(--secondary);background:rgba(226,223,255,.55);">Portfolio</span></div>
          <p class="metric-label">Total Credit Portfolio</p>
          <p class="metric-value">{k["ACTIVE_PORTFOLIO"]}</p>
        </div>
        <div class="card stat-card">
          <div class="stat-top"><div class="stat-icon"><span class="material-symbols-outlined">speed</span></div><span class="badge warn">AI Score</span></div>
          <p class="metric-label">Avg Default Probability</p>
          <p class="metric-value">{k["AVG_RISK_SCORE"]}</p>
        </div>
        <div class="card stat-card">
          <div class="stat-top"><div class="stat-icon" style="color:var(--error);background:rgba(255,218,214,.35);"><span class="material-symbols-outlined">warning</span></div><span class="badge bad">Risk</span></div>
          <p class="metric-label">High Risk Customers</p>
          <p class="metric-value">{k["HIGH_RISK_CUSTOMERS"]}</p>
        </div>
      </div>

      <div class="section grid-12">
        <div class="glass-card span-8 hover-lift">
          <div class="card-head"><div><h3 class="card-title">Predictive Risk Modeling</h3><p class="card-subtitle">XGBoost default probability distribution</p></div><div><button class="btn secondary" style="min-height:34px;padding:0 12px;">Daily</button><button class="btn primary" style="min-height:34px;padding:0 12px;">Weekly</button></div></div>
          <div style="height: 220px; position: relative;"><canvas id="overviewDistChart"></canvas></div>
          <div class="mini-kpis">
            <div class="card"><p class="tiny-title">Default Rate</p><p class="tiny-value">{k["DEFAULT_RATE"]}</p></div>
            <div class="card"><p class="tiny-title">Critical Cases</p><p class="tiny-value">{k["CRITICAL_EXPOSURE"]}</p></div>
            <div class="card"><p class="tiny-title">AI Confidence</p><p class="tiny-value" style="color:var(--success);">XGBoost</p></div>
          </div>
        </div>
        <div class="span-4">
          <div class="glass-card hover-lift">
            <h3 class="card-title" style="margin-bottom:16px;">Explainable AI Insights</h3>
            <ul class="insight-list">
              {insights_html}
            </ul>
          </div>
          <div class="ai-box"><h4 style="margin:0 0 8px;color:var(--primary-fixed);font-size:14px;">Risk Assessment AI</h4><p style="font-family:Manrope,sans-serif;font-size:20px;font-weight:800;margin:0 0 16px;">Upload data to activate live scoring</p><div class="ai-input"><input placeholder="Explain the spike in defaults..." /><button class="icon-btn" style="background:white;color:var(--primary);width:32px;height:32px;"><span class="material-symbols-outlined">send</span></button></div></div>
        </div>
      </div>

      <div class="section glass-card node-section">
        <div style="padding:24px;"><h2 class="card-title" style="font-size:32px;line-height:40px;">Decision Node Intelligence</h2><p class="lead" style="font-size:18px;margin-bottom:24px;">Our XGBoost model maps 52 engineered features into a default probability score, providing interpretable outputs for every credit decision.</p><span class="pill"><span class="dot"></span>XGBoost</span><span class="pill"><span class="dot" style="background:var(--success);"></span>SHAP Values</span><span class="pill"><span class="dot" style="background:var(--amber);"></span>Counterfactuals</span></div>
        <div class="node-art"><div class="node n1"></div><div class="node n2"></div><div class="node n3"></div></div>
      </div>
    </section>

    <!-- PORTFOLIO PAGE -->
    <section class="page" id="portfolio">
      <div class="page-title-row"><div><div class="page-eyebrow">Portfolio Analysis</div><h1 class="page-title">Risk Overview</h1></div><div class="filters"><div class="filter-group"><span class="filter-label">Industry</span><select><option>All Industries</option><option>Financial Services</option></select></div><div class="filter-group"><span class="filter-label">Risk Level</span><select><option>All</option><option>High Risk</option></select></div>{upload_badge}</div></div>
      <div class="kpi-grid">
        <div class="card kpi"><span class="tiny-title">Total Customers</span><p class="tiny-value">{k["TOTAL_CUSTOMERS"]}</p><span class="badge good">Uploaded</span></div>
        <div class="card kpi"><span class="tiny-title">Total Exposure</span><p class="tiny-value">{k["ACTIVE_PORTFOLIO"]}</p><span class="badge good">Portfolio</span></div>
        <div class="card kpi bad"><span class="tiny-title">Default Rate</span><p class="tiny-value">{k["DEFAULT_RATE"]}</p><span class="badge bad">Model</span></div>
        <div class="card kpi"><span class="tiny-title">Avg Risk Score</span><p class="tiny-value">{k["AVG_RISK_SCORE"]}</p><span class="badge good">Stable</span></div>
        <div class="card kpi bad"><span class="tiny-title">High Risk Count</span><p class="tiny-value">{k["HIGH_RISK_CUSTOMERS"]}</p><span class="badge bad">≥12% PD</span></div>
        <div class="card kpi bad"><span class="tiny-title">Critical Cases</span><p class="tiny-value">{k["CRITICAL_EXPOSURE"]}</p><span class="badge">≥20% PD</span></div>
      </div>
      <div class="grid-12">
        <div class="glass-card span-8"><div class="card-head"><h3 class="card-title">Global Risk Concentration</h3><div><span class="badge bad">High Risk</span><span class="badge" style="color:var(--secondary);background:rgba(226,223,255,.45);margin-left:8px;">Primary Focus</span></div></div><div class="map"><svg viewBox="0 0 800 430"><path d="M90,220 C210,90 350,110 425,175 S600,325 730,145" fill="none" stroke="#4b41e1" stroke-width="4" stroke-dasharray="10 12"/><path d="M70,320 C210,245 340,345 500,255 S650,105 760,270" fill="none" stroke="#131b2e" stroke-width="2" stroke-dasharray="7 12"/></svg><span class="map-point red"></span><span class="map-point blue"></span></div></div>
        <div class="glass-card span-4"><h3 class="card-title">Default Probability Distribution</h3><div style="height: 280px; position: relative;"><canvas id="distChart"></canvas></div></div>
        <div class="glass-card span-6"><div class="card-head"><h3 class="card-title">Credit Exposure by Risk</h3><span class="badge" style="color:white;background:var(--secondary);">Exposure</span></div><div style="height: 260px; position: relative;"><canvas id="exposureChart"></canvas></div></div>
        <div class="glass-card span-3"><h3 class="card-title">Risk Segmentation</h3><div style="height: 250px; position: relative; display: flex; align-items: center; justify-content: center;"><canvas id="segmentChart"></canvas></div></div>
        <div class="glass-card span-3"><h3 class="card-title">Risk by Education Level</h3><div style="height: 250px; position: relative;"><canvas id="educationRiskChart"></canvas></div></div>
      </div>
    </section>

    <!-- CUSTOMER PAGE -->
    <section class="page" id="customer">
      <div class="page-title-row">
        <div>
          <div style="color:var(--on-surface-variant);font-size:13px;">Analytics / Customer Profile</div>
          <h1 class="page-title" style="font-size:32px;line-height:40px;">Risk Profile Analysis</h1>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
          <div class="search" style="position:static; width:220px; background:white; border:1px solid rgba(198,198,205,.6); height:38px;">
            <span class="material-symbols-outlined" style="cursor:pointer;" onclick="filterCustomers()">search</span>
            <input id="cust-search" placeholder="Search ID, Risk, etc..." oninput="filterCustomers()" onkeydown="if(event.key==='Enter') {{ event.preventDefault(); filterCustomers(); }}" style="width:100%; border:0; outline:0; background:transparent;" />
          </div>
          <select id="cust-select" onchange="selectCustomer(this.value)" style="padding:8px 12px; border-radius:10px; border:1px solid rgba(198,198,205,.6); font-weight:800; background:white; color:var(--on-surface); height:38px;">
            <option value="">No Data Uploaded</option>
          </select>
        </div>
      </div>
      
      <div class="profile-grid" id="customer-profile-section" style="display:none;">
        <div>
          <div class="card">
            <div class="card-head">
              <div class="customer-avatar"></div>
              <span id="cust-profile-risk-chip" class="risk-chip">HIGH RISK</span>
            </div>
            <h3 class="card-title" id="cust-profile-id">Customer Profile</h3>
            <p class="card-subtitle" id="cust-profile-demographics">Demographics Details</p>
            <hr style="border:0;border-top:1px solid rgba(198,198,205,.3);margin:24px 0;">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;">
              <div><p class="tiny-title">Income</p><p class="tiny-value" id="cust-profile-income">$0</p></div>
              <div><p class="tiny-title">Loan Amount</p><p class="tiny-value" id="cust-profile-credit">$0</p></div>
              <div><p class="tiny-title">Age</p><p class="tiny-value" id="cust-profile-age">0 years</p></div>
              <div><p class="tiny-title">Employment</p><p class="tiny-value" id="cust-profile-employment">0 years</p></div>
            </div>
          </div>
          
          <div class="card" style="margin-top:24px;">
            <h3 class="card-title"><span class="material-symbols-outlined" style="color:var(--secondary);vertical-align:-4px;">insights</span> Key Risk Drivers</h3>
            
            <div class="driver">
              <span>External Source Rank (EXT_SOURCE)</span>
              <strong style="float:right;color:var(--secondary);" id="driver-ext-label">Moderate</strong>
              <div class="progress" style="margin-top:12px;"><span id="driver-ext-pct" style="--w:50%;background:var(--secondary);"></span></div>
            </div>
            
            <div class="driver">
              <span>Credit-to-Income Exposure</span>
              <strong style="float:right;color:var(--amber);" id="driver-cir-label">High</strong>
              <div class="progress" style="margin-top:12px;"><span id="driver-cir-pct" style="--w:70%;background:var(--amber);"></span></div>
            </div>
            
            <div class="driver">
              <span>Model Probability of Default</span>
              <strong style="float:right;color:var(--error);" id="driver-pd-label">Critical</strong>
              <div class="progress" style="margin-top:12px;"><span id="driver-pd-pct" style="--w:90%;background:var(--error);"></span></div>
            </div>
          </div>
        </div>
        
        <div>
          <div class="grid-12">
            <div class="card gauge span-6">
              <div>
                <p class="tiny-title">Default Probability Score</p>
                <div class="score" id="cust-profile-pd" style="font-size:80px;">0.0%</div>
                <span id="cust-profile-risk-level" class="risk-chip">HIGH RISK</span>
                <p class="card-subtitle" style="max-width:300px; margin-top:10px;">Calculated dynamically by the XGBoost RiskIntel default prediction model.</p>
              </div>
            </div>
            
            <div class="card span-6">
              <h3 class="card-title">Socio-Economic Profile</h3>
              <ul class="insight-list" style="margin-top:15px;">
                <li><span class="material-symbols-outlined" style="color:var(--secondary);">work</span><div><strong>Income Type</strong><p class="card-subtitle" id="cust-profile-income-type">Working</p></div></li>
                <li><span class="material-symbols-outlined" style="color:var(--secondary);">school</span><div><strong>Education</strong><p class="card-subtitle" id="cust-profile-education">Secondary / secondary special</p></div></li>
                <li><span class="material-symbols-outlined" style="color:var(--secondary);">wc</span><div><strong>Gender</strong><p class="card-subtitle" id="cust-profile-gender">Female</p></div></li>
              </ul>
            </div>
          </div>
          
          <div class="card" style="margin-top:24px;">
            <div class="card-head"><h3 class="card-title">Behavioral Timeline & Risk Profile</h3><span class="badge">Historical Profile</span></div>
            <div class="timeline">
              <span class="time-node">Ingestion</span>
              <span class="time-node">Mapping</span>
              <span class="time-node">Engineering</span>
              <span class="time-node">SHAP Check</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px;">
              <div class="card" style="background:#ecfdf5; border-color:rgba(5,150,105,.18);">
                <strong>Mitigating Strengths</strong>
                <p class="card-subtitle" id="cust-profile-strength">High external source rating and stable job tenure reduce overall score.</p>
              </div>
              <div class="card" style="background:var(--error-container); border-color:rgba(186,26,26,.15);">
                <strong style="color:var(--error);">Risk Amplifiers</strong>
                <p class="card-subtitle" id="cust-profile-amplifier">High debt ratio or missing bureau history increases default risk.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Placeholder if no data -->
      <div class="card" id="customer-no-data-placeholder" style="text-align:center; padding:80px 0;">
        <span class="material-symbols-outlined" style="font-size:60px; color:var(--outline); margin-bottom:16px;">contact_page</span>
        <h3 class="card-title">No Customer Data Loaded</h3>
        <p class="muted">Upload a dataset and confirm mappings to view detailed customer-wise risk profiles.</p>
      </div>
    </section>

    <!-- XAI PAGE -->
    <section class="page" id="xai">
      <div class="page-title-row"><div><div class="page-eyebrow">Model Governance</div><h1 class="page-title">Explainable AI</h1></div></div>
      <div class="grid-12"><div class="glass-card span-7"><h3 class="card-title">Feature Contribution</h3><p class="card-subtitle">Top predictors in XGBoost model (approximate gain)</p><div class="chart-box" style="height:360px;"><div class="chart-bar" style="--h:92%;--fill:100%;"></div><div class="chart-bar" style="--h:72%;--fill:100%;"></div><div class="chart-bar" style="--h:56%;--fill:100%;"></div><div class="chart-bar" style="--h:31%;--fill:100%;"></div><div class="chart-bar" style="--h:22%;--fill:100%;"></div></div><div style="display:flex;justify-content:space-between;color:var(--outline);font-size:12px;font-weight:800;"><span>EXT_SOURCE</span><span>Days Empl.</span><span>AMT_Credit</span><span>DPD Max</span><span>Annuity</span></div></div><div class="glass-card span-5"><h3 class="card-title">Counterfactuals</h3><div class="driver"><strong>Improve external bureau score</strong><p class="card-subtitle">EXT_SOURCE is the highest weight feature; a 0.1 lift reduces default probability significantly.</p></div><div class="driver"><strong>Stable employment history</strong><p class="card-subtitle">Longer employment tenure lowers EMPLOYMENT_TO_AGE_RATIO risk.</p></div><div class="driver"><strong>Reduce overdue debt</strong><p class="card-subtitle">OVERDUE_DEBT_SUM reduction shifts profile toward low risk band.</p></div><div class="ai-box"><h3 style="margin:0 0 8px;">Decision Narrative</h3><p style="color:rgba(255,255,255,.8);">External credit bureau signals and employment stability dominate the XGBoost output. Payment history metrics provide the second tier of discriminatory power.</p></div></div></div>
    </section>

    <!-- ASK PAGE -->
    <section class="page" id="ask">
      <div class="chat-layout"><div class="chat-left"><div class="chat-head"><h2 class="card-title"><span class="material-symbols-outlined" style="color:var(--secondary);vertical-align:-4px;font-variation-settings:'FILL' 1;">psychology</span> Ask Your Data</h2><button class="btn secondary" style="min-height:32px;padding:0 12px;" onclick="clearChatHistory()">Clear History</button></div><div class="chat-messages"><div class="chat-row"><div class="stat-icon" style="background:var(--secondary);color:white;"><span class="material-symbols-outlined">auto_awesome</span></div><div class="bubble">Hello. I have the RiskIntel XGBoost model loaded. Upload a dataset via the sidebar to activate live scoring and KPI updates.</div></div><div class="chat-row user"><div class="bubble user">What is the default rate in the uploaded data?</div><div class="avatar"></div></div><div class="chat-row"><div class="stat-icon" style="background:var(--secondary);color:white;"><span class="material-symbols-outlined">auto_awesome</span></div><div class="bubble">The model predicts a default rate of <strong>{k["DEFAULT_RATE"]}</strong> across {k["TOTAL_CUSTOMERS"]} customers. {k["HIGH_RISK_CUSTOMERS"]} customers score above 12% probability of default (High Risk threshold).</div></div></div><div class="chat-input-wrap"><div class="suggestions"><button onclick="useSuggestion(this.textContent)">What is the average income of high-risk customers?</button><button onclick="useSuggestion(this.textContent)">Show me the highest risk customers in the portfolio</button><button onclick="useSuggestion(this.textContent)">Explain the demographics of the dataset</button></div><div class="ai-input" style="background:white;border-color:rgba(198,198,205,.5);"><textarea id="chat-input" rows="2" placeholder="Ask questions about the uploaded dataset..." style="color:var(--on-surface);" onkeydown="if(event.key==='Enter' &amp;&amp; !event.shiftKey) {{ event.preventDefault(); sendChatMessage(); }}"></textarea><button id="chat-send" class="icon-btn" style="background:var(--secondary);color:white;" onclick="sendChatMessage()"><span class="material-symbols-outlined">send</span></button></div><p class="tiny-title" style="text-align:center;margin-top:12px;">RiskIntel AI can hallucinate financial projections. Verify critical outputs.</p></div></div>
      <div class="chat-right"><div class="card-head"><div><span class="badge" style="color:#0f0069;background:var(--secondary-fixed);">RESULTS GENERATED</span></div><div><button class="icon-btn"><span class="material-symbols-outlined">download</span></button></div></div><section class="card"><h3 class="card-title">Portfolio Summary</h3><p class="card-subtitle" style="font-size:16px;">XGBoost model scores <strong style="color:var(--primary);">{k["TOTAL_CUSTOMERS"]}</strong> customers. Default rate: <strong style="color:var(--error);">{k["DEFAULT_RATE"]}</strong>. Average probability of default: <strong>{k["AVG_RISK_SCORE"]}</strong>.</p><div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px;"><div class="card" style="background:rgba(255,218,214,.35);"><p class="tiny-title">High Risk</p><p class="tiny-value" style="color:var(--error);">{k["HIGH_RISK_CUSTOMERS"]}</p></div><div class="card" style="background:rgba(226,223,255,.35);"><p class="tiny-title">Critical</p><p class="tiny-value" style="color:var(--secondary);">{k["CRITICAL_EXPOSURE"]}</p></div></div></section><section class="card" style="margin-top:18px;"><div class="card-head"><h3 class="card-title">Default Probability</h3></div><div class="chart-box"><div class="chart-bar" style="--h:40%;--fill:100%;"></div><div class="chart-bar" style="--h:65%;--fill:100%;"></div><div class="chart-bar" style="--h:85%;--fill:100%;"></div><div class="chart-bar" style="--h:45%;--fill:100%;"></div><div class="chart-bar" style="--h:95%;--fill:100%;"></div><div class="chart-bar" style="--h:30%;--fill:100%;"></div><div class="chart-bar" style="--h:75%;--fill:100%;"></div></div></section></div></div>
    </section>

    <!-- SETTINGS PAGE -->
    <section class="page" id="settings">
      <div class="page-title-row"><div><div class="page-eyebrow">Workspace</div><h1 class="page-title">Settings</h1></div></div>
      <div class="settings-grid"><div class="card"><h3 class="card-title">Controls</h3><div class="toggle-line"><span>Enterprise Sync</span><span class="toggle"></span></div><div class="toggle-line"><span>Analyst approval for high-risk decisions</span><span class="toggle"></span></div><div class="toggle-line"><span>Regulatory mode</span><span class="toggle"></span></div></div><div class="card"><h3 class="card-title">Risk Threshold</h3><p class="score" style="font-size:58px;color:var(--secondary);">12%</p><div class="progress"><span style="--w:12%;"></span></div><p class="card-subtitle">Default alert trigger for portfolio monitoring.</p></div><div class="card"><h3 class="card-title">Model Display</h3><div class="driver">Always show confidence</div><div class="driver">Show feature drivers</div><div class="driver">Enable audit trail</div></div></div>
    </section>
  </main>
</div>

<script>
  // Injected JSON data
  const customers = {customer_profiles_json};
  const geminiApiKey = "{gemini_api_key}";
  const datasetSummary = {dataset_summary_json};
  
  function performGlobalSearch(query) {{
    if (!query) return;
    
    // Switch to Customer Risk Analytics page
    document.querySelectorAll('.nav-btn[data-page]').forEach(b => {{
      b.classList.toggle('active', b.dataset.page === 'customer');
    }});
    document.querySelectorAll('.page').forEach(p => {{
      p.classList.toggle('active', p.id === 'customer');
    }});
    
    // Fill the customer search input and trigger filter
    const custSearch = document.getElementById('cust-search');
    if (custSearch) {{
      custSearch.value = query;
      filterCustomers();
    }}
    
    window.scrollTo({{ top: 0, behavior: 'smooth' }});
  }}

  function handleGlobalSearch(event) {{
    if (event.type === 'keydown' && event.key !== 'Enter') return;
    if (event) event.preventDefault();
    const query = document.getElementById('global-search').value.trim();
    performGlobalSearch(query);
  }}



  function filterCustomers() {{
    const q = document.getElementById('cust-search').value.toLowerCase().trim();
    const select = document.getElementById('cust-select');
    select.innerHTML = '';
    
    if (customers.length === 0) {{
      const opt = document.createElement('option');
      opt.textContent = "No Data Uploaded";
      select.appendChild(opt);
      document.getElementById('customer-profile-section').style.display = 'none';
      document.getElementById('customer-no-data-placeholder').style.display = 'block';
      const placeholderTitle = document.querySelector('#customer-no-data-placeholder h3');
      const placeholderText = document.querySelector('#customer-no-data-placeholder p');
      if (placeholderTitle) placeholderTitle.textContent = 'No Customer Data Loaded';
      if (placeholderText) placeholderText.textContent = 'Upload a dataset and confirm mappings to view detailed customer-wise risk profiles.';
      return;
    }}
    
    const filtered = customers.filter(c => 
      c.id.toString().includes(q) || 
      c.risk_level.toLowerCase().includes(q) || 
      (c.gender && c.gender.toLowerCase().includes(q)) ||
      (c.income_type && c.income_type.toLowerCase().includes(q)) ||
      (c.education && c.education.toLowerCase().includes(q))
    );
    
    filtered.forEach(c => {{
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = "ID: " + c.id + " (PD: " + (c.pd*100).toFixed(1) + "% - " + c.risk_level + ")";
      select.appendChild(opt);
    }});
    
    const placeholder = document.getElementById('customer-no-data-placeholder');
    const profileSec = document.getElementById('customer-profile-section');
    
    if (filtered.length > 0) {{
      profileSec.style.display = 'grid';
      placeholder.style.display = 'none';
      selectCustomer(filtered[0].id);
    }} else {{
      const opt = document.createElement('option');
      opt.textContent = "No matches";
      select.appendChild(opt);
      profileSec.style.display = 'none';
      placeholder.style.display = 'block';
      const placeholderTitle = placeholder.querySelector('h3');
      const placeholderText = placeholder.querySelector('p');
      if (placeholderTitle) placeholderTitle.textContent = 'No Matches Found';
      if (placeholderText) placeholderText.textContent = 'No customers matched your search query: "' + q + '". Try searching for ID, risk stage (High/Medium/Low), gender, or education level.';
    }}
  }}
  
  function selectCustomer(id) {{
    const c = customers.find(x => x.id == id);
    if (!c) return;
    
    // Update profile layout visibility
    document.getElementById('customer-profile-section').style.display = 'grid';
    document.getElementById('customer-no-data-placeholder').style.display = 'none';
    
    // Update texts
    document.getElementById('cust-profile-id').textContent = 'Customer #' + c.id;
    document.getElementById('cust-profile-demographics').textContent = 'Gender: ' + c.gender + ' | Education: ' + c.education;
    document.getElementById('cust-profile-income').textContent = '$' + Number(c.income).toLocaleString(undefined, {{maximumFractionDigits:0}});
    document.getElementById('cust-profile-credit').textContent = '$' + Number(c.credit).toLocaleString(undefined, {{maximumFractionDigits:0}});
    document.getElementById('cust-profile-age').textContent = Math.round(c.age) + ' years';
    document.getElementById('cust-profile-employment').textContent = c.employment.toFixed(1) + ' years';
    
    const pdVal = c.pd * 100;
    document.getElementById('cust-profile-pd').textContent = pdVal.toFixed(1) + '%';
    document.getElementById('cust-profile-risk-level').textContent = c.risk_level.toUpperCase() + ' RISK';
    
    // Update risk chip
    const chip = document.getElementById('cust-profile-risk-chip');
    chip.textContent = c.risk_level.toUpperCase() + ' RISK';
    
    // Remove old styles and set risk colors
    chip.style.background = c.risk_level === 'High' ? 'var(--error-container)' : c.risk_level === 'Medium' ? 'rgba(217,119,6,0.15)' : 'rgba(5,150,105,0.15)';
    chip.style.color = c.risk_level === 'High' ? 'var(--error)' : c.risk_level === 'Medium' ? 'var(--amber)' : 'var(--success)';
    
    // Set risk score text color
    const scoreText = document.getElementById('cust-profile-pd');
    scoreText.style.color = c.risk_level === 'High' ? 'var(--error)' : c.risk_level === 'Medium' ? 'var(--amber)' : 'var(--success)';
    
    // Update Socio-Economic Details
    document.getElementById('cust-profile-income-type').textContent = c.income_type;
    document.getElementById('cust-profile-education').textContent = c.education;
    document.getElementById('cust-profile-gender').textContent = c.gender;
    
    // Calculate and update risk drivers
    // 1. EXT Source (lower mean is higher risk)
    const extMean = ((c.ext1 || 0.5) + (c.ext2 || 0.5) + (c.ext3 || 0.5)) / 3;
    const extPct = Math.round((1 - extMean) * 100);
    const extLabel = extMean > 0.6 ? 'Low Risk' : extMean > 0.35 ? 'Moderate' : 'Critical';
    document.getElementById('driver-ext-pct').style.setProperty('--w', extPct + '%');
    document.getElementById('driver-ext-pct').style.background = extMean > 0.6 ? 'var(--success)' : extMean > 0.35 ? 'var(--amber)' : 'var(--error)';
    document.getElementById('driver-ext-label').textContent = extLabel;
    document.getElementById('driver-ext-label').style.color = extMean > 0.6 ? 'var(--success)' : extMean > 0.35 ? 'var(--amber)' : 'var(--error)';
    
    // 2. Credit-to-Income (higher is higher risk)
    const cir = c.credit / (c.income || 1);
    const cirPct = Math.min(100, Math.round(cir * 15));
    const cirLabel = cir < 2.0 ? 'Low Risk' : cir < 4.5 ? 'Moderate' : 'Critical';
    document.getElementById('driver-cir-pct').style.setProperty('--w', cirPct + '%');
    document.getElementById('driver-cir-pct').style.background = cir < 2.0 ? 'var(--success)' : cir < 4.5 ? 'var(--amber)' : 'var(--error)';
    document.getElementById('driver-cir-label').textContent = cirLabel;
    document.getElementById('driver-cir-label').style.color = cir < 2.0 ? 'var(--success)' : cir < 4.5 ? 'var(--amber)' : 'var(--error)';
    
    // 3. Model PD
    document.getElementById('driver-pd-pct').style.setProperty('--w', pdVal.toFixed(0) + '%');
    document.getElementById('driver-pd-pct').style.background = pdVal >= 12 ? 'var(--error)' : pdVal >= 5 ? 'var(--amber)' : 'var(--success)';
    document.getElementById('driver-pd-label').textContent = c.risk_level + ' Risk';
    document.getElementById('driver-pd-label').style.color = pdVal >= 12 ? 'var(--error)' : pdVal >= 5 ? 'var(--amber)' : 'var(--success)';
    
    // Strengths and Amplifiers
    if (c.risk_level === 'High') {{
      document.getElementById('cust-profile-strength').textContent = "Moderate income and active loan repayments are positive factors.";
      document.getElementById('cust-profile-amplifier').textContent = "Critical default probability (" + pdVal.toFixed(1) + "%) driven by low external bureau ratings and high credit utilization.";
    }} else if (c.risk_level === 'Medium') {{
      document.getElementById('cust-profile-strength').textContent = "Customer age and stable employment duration provide positive default resistance.";
      document.getElementById('cust-profile-amplifier').textContent = "Elevated risk index due to moderate external credit checks and loan size.";
    }} else {{
      document.getElementById('cust-profile-strength').textContent = "Excellent external credit rating (Mean score: " + extMean.toFixed(2) + ") and stable background ensure a very safe risk profile.";
      document.getElementById('cust-profile-amplifier').textContent = "No significant default risk signals found in behavioral or demographic records.";
    }}
  }}

  const colors = [
    'rgba(5,150,105,.10)','rgba(5,150,105,.22)','rgba(5,150,105,.42)',
    'rgba(217,119,6,.32)','rgba(217,119,6,.52)',
    'rgba(186,26,26,.40)','rgba(186,26,26,.62)','rgba(186,26,26,.82)'
  ];
  function fillHeatmap(id) {{
    const grid = document.getElementById(id);
    if (!grid) return;
    for (let i = 0; i < 84; i++) {{
      const cell = document.createElement('div');
      cell.className = 'heatmap-cell';
      cell.style.background = colors[Math.floor(Math.random() * colors.length)];
      grid.appendChild(cell);
    }}
  }}
  fillHeatmap('heatmap-overview');

  document.addEventListener('click', (event) => {{
    const route = event.target.closest('[data-route]');
    if (route) {{
      event.preventDefault();
      const target = '/?page=' + encodeURIComponent(route.dataset.route);
      try {{ window.top.location.assign(target); }} catch(e) {{ window.location.href = target; }}
      return;
    }}
    const btn = event.target.closest('[data-page]');
    if (!btn) return;
    const pageId = btn.dataset.page;
    document.querySelectorAll('.nav-btn[data-page]').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === pageId));
    window.scrollTo({{ top: 0, behavior: 'smooth' }});
  }});

  // Initialize dropdown if data is present
  if (customers && customers.length > 0) {{
    const select = document.getElementById('cust-select');
    select.innerHTML = '';
    customers.forEach(c => {{
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = "ID: " + c.id + " (PD: " + (c.pd*100).toFixed(1) + "% - " + c.risk_level + ")";
      select.appendChild(opt);
    }});
    
    document.getElementById('customer-profile-section').style.display = 'grid';
    document.getElementById('customer-no-data-placeholder').style.display = 'none';
    selectCustomer(customers[0].id);
  }} else {{
    document.getElementById('customer-profile-section').style.display = 'none';
    document.getElementById('customer-no-data-placeholder').style.display = 'block';
  }}

  function useSuggestion(text) {{
    document.getElementById('chat-input').value = text;
    sendChatMessage();
  }}

  function clearChatHistory() {{
    const messages = document.querySelector('.chat-messages');
    messages.innerHTML = `
      <div class="chat-row">
        <div class="stat-icon" style="background:var(--secondary);color:white;"><span class="material-symbols-outlined">auto_awesome</span></div>
        <div class="bubble">Chat history cleared. How can I help you analyze the credit portfolio risk today?</div>
      </div>
    `;
  }}

  async function sendChatMessage() {{
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    
    input.value = '';
    
    const messages = document.querySelector('.chat-messages');
    
    const userRow = document.createElement('div');
    userRow.className = 'chat-row user';
    userRow.innerHTML = '<div class="bubble user">' + msg.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;") + '</div><div class="avatar"></div>';
    messages.appendChild(userRow);
    messages.scrollTop = messages.scrollHeight;
    
    const loadingRow = document.createElement('div');
    loadingRow.className = 'chat-row';
    loadingRow.id = 'chat-loading-bubble';
    loadingRow.innerHTML = `
      <div class="stat-icon" style="background:var(--secondary);color:white;"><span class="material-symbols-outlined">auto_awesome</span></div>
      <div class="bubble">Thinking...</div>
    `;
    messages.appendChild(loadingRow);
    messages.scrollTop = messages.scrollHeight;
    
    const apiKey = geminiApiKey;
    if (!apiKey || apiKey === "") {{
      removeLoadingAndReply("Error: Gemini API Key is missing. Please make sure to save it in Streamlit Secrets or upload environment variables to enable AI chat operations.");
      return;
    }}
    
    const prompt = "You are RiskIntel AI, an advanced credit risk assistant. You have access to a summary of the uploaded credit portfolio default risk dataset.\\n\\nDataset Summary:\\n" + JSON.stringify(datasetSummary, null, 2) + "\\n\\nAnswer the user's question contextually using the summary metrics, statistics, and risk driver demographics.\\nGuidelines:\\n- Be direct, professional, and clear.\\n- Use markdown (e.g. bold, bullet points, tables) to present numeric data or comparisons.\\n- If asked about specific customers, refer to the 'highest_risk_customers' list provided in the summary.\\n- If they ask general risk questions, use the summaries to explain patterns.\\n- Do not hallucinate or make up facts that are not present or inferred from the dataset.\\n\\nUser Question:\\n\\\"" + msg.replace(/\\"/g, '\\\\\\"') + "\\\"";
    
    try {{
      const url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + apiKey;
      const res = await fetch(url, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{
          contents: [{{ parts: [{{ text: prompt }}] }}]
        }})
      }});
      const data = await res.json();
      if (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts[0]) {{
        const reply = data.candidates[0].content.parts[0].text;
        removeLoadingAndReply(reply);
      }} else if (data.error) {{
        removeLoadingAndReply("Error from Gemini API: " + data.error.message);
      }} else {{
        removeLoadingAndReply("Error: Failed to fetch valid response from Gemini API. Structure received: " + JSON.stringify(data));
      }}
    }} catch(e) {{
      removeLoadingAndReply("Error querying Gemini API: " + e.message + ". Please verify your internet connection and API key.");
    }}
  }}

  function removeLoadingAndReply(text) {{
    const loading = document.getElementById('chat-loading-bubble');
    if (loading) loading.remove();
    
    let htmlContent = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>")
      .replace(/\\*(.*?)\\*/g, "<em>$1</em>")
      .replace(/\\n/g, "<br>");
      
    const messages = document.querySelector('.chat-messages');
    const replyRow = document.createElement('div');
    replyRow.className = 'chat-row';
    replyRow.innerHTML = `
      <div class="stat-icon" style="background:var(--secondary);color:white;"><span class="material-symbols-outlined">auto_awesome</span></div>
      <div class="bubble">${{htmlContent}}</div>
    `;
    messages.appendChild(replyRow);
    messages.scrollTop = messages.scrollHeight;
  }}

  function initCharts() {{
    let dataSrc = customers;
    if (!dataSrc || dataSrc.length === 0) {{
      dataSrc = [];
      // Generate some nice dummy data
      for (let i = 0; i < 120; i++) {{
        const pd = Math.random() * 0.25;
        const rl = pd >= 0.12 ? 'High' : pd >= 0.05 ? 'Medium' : 'Low';
        dataSrc.push({{
          id: 100000 + i,
          pd: pd,
          risk_level: rl,
          credit: 25000 + Math.random() * 95000,
          education: ['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary'][Math.floor(Math.random() * 4)]
        }});
      }}
    }}

    // 1. Distribution Bins
    const bins = [0, 0, 0, 0, 0];
    dataSrc.forEach(c => {{
      const pd = c.pd || 0;
      if (pd < 0.05) bins[0]++;
      else if (pd < 0.10) bins[1]++;
      else if (pd < 0.15) bins[2]++;
      else if (pd < 0.20) bins[3]++;
      else bins[4]++;
    }});

    // 2. Risk Segmentation Count
    let lowCount = 0, medCount = 0, highCount = 0;
    dataSrc.forEach(c => {{
      if (c.risk_level === 'Low') lowCount++;
      else if (c.risk_level === 'Medium') medCount++;
      else if (c.risk_level === 'High') highCount++;
    }});

    // 3. Credit Exposure by Segment
    let lowCredit = 0, medCredit = 0, highCredit = 0;
    dataSrc.forEach(c => {{
      const cr = c.credit || 0;
      if (c.risk_level === 'Low') lowCredit += cr;
      else if (c.risk_level === 'Medium') medCredit += cr;
      else if (c.risk_level === 'High') highCredit += cr;
    }});

    // 4. Avg Risk by Education
    const eduMap = {{}};
    dataSrc.forEach(c => {{
      const edu = c.education || 'Unknown';
      if (!eduMap[edu]) eduMap[edu] = {{ sum: 0, count: 0 }};
      eduMap[edu].sum += (c.pd || 0);
      eduMap[edu].count++;
    }});
    const eduLabels = Object.keys(eduMap);
    const eduAvgRisk = eduLabels.map(label => {{
      const entry = eduMap[label];
      return (entry.sum / entry.count) * 100;
    }});

    // Draw Chart 1: Distribution Bar Chart (Portfolio Page)
    const ctx1 = document.getElementById('distChart');
    if (ctx1) {{
      new Chart(ctx1.getContext('2d'), {{
        type: 'bar',
        data: {{
          labels: ['0–5%', '5–10%', '10–15%', '15–20%', '>20%'],
          datasets: [{{
            label: 'Customer Count',
            data: bins,
            backgroundColor: [
              'rgba(5, 150, 105, 0.6)',
              'rgba(5, 150, 105, 0.85)',
              'rgba(217, 119, 6, 0.75)',
              'rgba(186, 26, 26, 0.75)',
              'rgba(186, 26, 26, 0.95)'
            ],
            borderColor: [
              'var(--success)',
              'var(--success)',
              'var(--amber)',
              'var(--error)',
              'var(--error)'
            ],
            borderWidth: 1.5,
            borderRadius: 6
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            y: {{ beginAtZero: true, grid: {{ color: 'rgba(0,0,0,0.04)' }} }},
            x: {{ grid: {{ display: false }} }}
          }}
        }}
      }});
    }}

    // Draw Chart 2: Segment Doughnut (Portfolio Page)
    const ctx2 = document.getElementById('segmentChart');
    if (ctx2) {{
      new Chart(ctx2.getContext('2d'), {{
        type: 'doughnut',
        data: {{
          labels: ['Low Risk', 'Med Risk', 'High Risk'],
          datasets: [{{
            data: [lowCount, medCount, highCount],
            backgroundColor: ['#009485', '#d97706', '#ba1a1a'],
            borderWidth: 2,
            hoverOffset: 4
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ position: 'bottom', labels: {{ boxWidth: 12, font: {{ weight: 600 }} }} }}
          }},
          cutout: '65%'
        }}
      }});
    }}

    // Draw Chart 3: Exposure by Risk Segment (Portfolio Page)
    const ctx3 = document.getElementById('exposureChart');
    if (ctx3) {{
      new Chart(ctx3.getContext('2d'), {{
        type: 'bar',
        data: {{
          labels: ['Low Risk', 'Med Risk', 'High Risk'],
          datasets: [{{
            label: 'Total Exposure ($)',
            data: [lowCredit, medCredit, highCredit],
            backgroundColor: [
              'rgba(0, 148, 133, 0.8)',
              'rgba(217, 119, 6, 0.8)',
              'rgba(186, 26, 26, 0.8)'
            ],
            borderRadius: 8
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            y: {{
              grid: {{ color: 'rgba(0,0,0,0.04)' }},
              ticks: {{
                callback: function(value) {{
                  if (value >= 1e6) return '$' + (value/1e6).toFixed(1) + 'M';
                  if (value >= 1e3) return '$' + (value/1e3).toFixed(0) + 'K';
                  return '$' + value;
                }}
              }}
            }},
            x: {{ grid: {{ display: false }} }}
          }}
        }}
      }});
    }}

    // Draw Chart 4: Risk by Education Horizontal Bar (Portfolio Page)
    const ctx4 = document.getElementById('educationRiskChart');
    if (ctx4) {{
      new Chart(ctx4.getContext('2d'), {{
        type: 'bar',
        data: {{
          labels: eduLabels.map(l => l.length > 18 ? l.substring(0, 15) + '...' : l),
          datasets: [{{
            label: 'Avg Default PD (%)',
            data: eduAvgRisk,
            backgroundColor: 'rgba(75, 65, 225, 0.8)',
            borderRadius: 6
          }}]
        }},
        options: {{
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ beginAtZero: true, max: 100, grid: {{ color: 'rgba(0,0,0,0.04)' }} }},
            y: {{ grid: {{ display: false }} }}
          }}
        }}
      }});
    }}

    // Draw Chart 5: Overview Page Line Chart
    const ctx5 = document.getElementById('overviewDistChart');
    if (ctx5) {{
      new Chart(ctx5.getContext('2d'), {{
        type: 'line',
        data: {{
          labels: ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%'],
          datasets: [{{
            label: 'Default Risk Profile',
            data: bins,
            fill: true,
            backgroundColor: 'rgba(75, 65, 225, 0.12)',
            borderColor: 'var(--secondary)',
            borderWidth: 3,
            tension: 0.35,
            pointBackgroundColor: 'var(--secondary)',
            pointRadius: 4
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            y: {{ beginAtZero: true, grid: {{ color: 'rgba(0,0,0,0.04)' }} }},
            x: {{ grid: {{ display: false }} }}
          }}
        }}
      }});
    }}
  }}

  // Call initialization after rendering
  setTimeout(initCharts, 100);
</script>
</body>
</html>"""


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    query_page = st.query_params.get("page", "")
    if query_page == "upload":
        st.session_state["active_page"] = "upload"
    else:
        st.session_state["active_page"] = "overview"

    if st.session_state.get("active_page") == "upload":
        import pages.upload_dataset as ud
        ud.main(configure_page=False, show_sidebar=True)
        return

    # ── build KPIs from session state or disk fallback ──
    import json
    import os
    stats: dict | None = st.session_state.get("dashboard_stats")
    
    stats_path = os.path.join("database", "dashboard_stats.json")
    if stats is None and os.path.exists(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            st.session_state["dashboard_stats"] = stats
            st.session_state["has_data"] = True
        except Exception:
            pass

    has_data: bool = st.session_state.get("has_data", False)
    kpis: dict | None = build_kpis(stats) if stats else None
    dist_bars = ""
    customer_profiles_json = "[]"
    dataset_summary_json = "{}"
    insights_html = ""
    if stats:
        if stats.get("proba"):
            dist_bars = build_dist_bars(stats["proba"])
        if stats.get("customer_profiles"):
            customer_profiles_json = json.dumps(stats["customer_profiles"])
        
        # Build dynamic insights
        insights_html = build_insights_html(build_dynamic_insights(stats))
        
        # Build compact summary of data for Ask Your Data AI context
        dataset_summary = {
            "total_customers": stats.get("total_customers", 0),
            "default_rate": f"{stats.get('default_rate', 0.0):.2%}" if "default_rate" in stats else "0.0%",
            "avg_risk_score": f"{stats.get('avg_risk_score', 0.0):.2%}" if "avg_risk_score" in stats else "0.0%",
            "low_risk_count": stats.get("low_risk_count", 0),
            "medium_risk_count": stats.get("medium_risk_count", 0),
            "high_risk_count": stats.get("high_risk_count", 0),
            "critical_count": stats.get("critical_count", 0),
            "total_portfolio": f"${stats.get('total_portfolio', 0.0):,.0f}" if "total_portfolio" in stats else "$0",
            "avg_income": f"${stats.get('avg_income', 0.0):,.0f}" if "avg_income" in stats else "$0",
            "avg_credit": f"${stats.get('avg_credit', 0.0):,.0f}" if "avg_credit" in stats else "$0",
        }
        # Add up to 100 customer records for concrete contextual queries
        if "customer_profiles" in stats and stats["customer_profiles"]:
            # Sort descending by PD to extract top 30 highest risk profiles
            sorted_profiles = sorted(stats["customer_profiles"], key=lambda x: x.get("pd", 0.0), reverse=True)
            dataset_summary["highest_risk_customers"] = sorted_profiles[:30]
            dataset_summary["customer_profiles_sample"] = stats["customer_profiles"][:70]
        dataset_summary_json = json.dumps(dataset_summary)
    else:
        # Default insights fallback
        insights_html = build_insights_html([])

    # Determine Gemini API key for chat operations
    api_key = ""
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

    st.markdown(
        """
        <style>
        #MainMenu, header, footer, [data-testid="stSidebar"] { display: none !important; }
        .stApp { background: #f7f9fb; }
        .block-container { padding: 0 !important; max-width: none !important; }
        iframe { display: block; width: 100%; border: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.components.v1.html(app_html(kpis=kpis, dist_bars=dist_bars, has_data=has_data, customer_profiles_json=customer_profiles_json, gemini_api_key=api_key, dataset_summary_json=dataset_summary_json, insights_html=insights_html), height=2200, scrolling=True)


if __name__ == "__main__":
    main()
