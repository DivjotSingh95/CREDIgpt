from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

import os
from database.db_connection import TABLE_NAME, process_upload_to_database
from utils.validation import ValidationResult, validate_dataset, run_feature_engineering_pipeline
from utils.mapping import get_base_target_columns, get_gemini_mappings


def inject_upload_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

        :root {
            --surface: #f7f9fb;
            --surface-card: #ffffff;
            --surface-soft: #f2f4f6;
            --surface-high: #e6e8ea;
            --outline: #c6c6cd;
            --text: #191c1e;
            --muted: #45464d;
            --secondary: #4b41e1;
            --secondary-soft: #e2dfff;
            --error: #ba1a1a;
            --error-soft: #ffdad6;
            --success: #059669;
            --warning: #d97706;
            --primary-container: #131b2e;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(226, 223, 255, .72), transparent 30%),
                var(--surface);
            color: var(--text);
            font-family: Inter, sans-serif;
        }

        h1, h2, h3 {
            font-family: Manrope, sans-serif;
            letter-spacing: 0;
        }

        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebarNav"] {
            display: none !important;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, .88);
            border-right: 1px solid rgba(198, 198, 205, .5);
            position: fixed !important;
            z-index: 999999 !important;
            transform: translateX(-100%) !important;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        [data-testid="stSidebar"]::after {
            content: '';
            position: absolute;
            right: -24px;
            top: 0;
            bottom: 0;
            width: 24px;
            background: transparent;
        }

        [data-testid="stSidebar"]:hover {
            transform: translateX(0) !important;
        }

        [data-testid="stAppViewContainer"] {
            padding-left: 0 !important;
        }

        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }

        .risk-shell {
            display: flex;
            flex-direction: column;
            gap: 24px;
        }

        .page-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 24px;
            margin-bottom: 10px;
        }

        .eyebrow {
            color: var(--secondary);
            font-size: 13px;
            font-weight: 900;
            letter-spacing: .18em;
            text-transform: uppercase;
        }

        .page-title {
            margin: 4px 0 0;
            font-size: 46px;
            line-height: 54px;
            font-weight: 800;
        }

        .page-subtitle {
            max-width: 760px;
            color: var(--muted);
            font-size: 16px;
            line-height: 26px;
            margin-top: 10px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 8px 13px;
            background: rgba(226, 223, 255, .48);
            color: var(--secondary);
            font-size: 13px;
            font-weight: 900;
        }

        .card {
            background: rgba(255, 255, 255, .84);
            border: 1px solid rgba(198, 198, 205, .52);
            border-radius: 14px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, .04), 0 18px 45px rgba(15, 23, 42, .055);
            backdrop-filter: blur(12px);
            padding: 22px;
        }

        .upload-card {
            min-height: 360px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 2px dashed rgba(118, 119, 125, .42);
            background:
                radial-gradient(circle at top left, rgba(75, 65, 225, .07), transparent 48%),
                rgba(255, 255, 255, .72);
            animation: floatIn .35s ease both;
        }

        .upload-icon {
            width: 82px;
            height: 82px;
            margin: 0 auto 20px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            background: rgba(75, 65, 225, .10);
            color: var(--secondary);
            font-size: 32px;
            font-weight: 900;
        }

        .upload-title {
            font-family: Manrope, sans-serif;
            font-size: 24px;
            line-height: 32px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .muted {
            color: var(--muted);
            font-size: 14px;
            line-height: 22px;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 18px;
            margin-top: 16px;
        }

        .kpi {
            background: var(--surface-card);
            border: 1px solid rgba(198, 198, 205, .52);
            border-radius: 12px;
            padding: 18px;
            min-height: 124px;
            position: relative;
            overflow: hidden;
        }

        .kpi::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 4px;
            background: rgba(75, 65, 225, .16);
        }

        .kpi.success::after { background: rgba(5, 150, 105, .22); }
        .kpi.error::after { background: rgba(186, 26, 26, .22); }
        .kpi.warning::after { background: rgba(217, 119, 6, .22); }

        .kpi-label {
            color: #76777d;
            font-size: 11px;
            font-weight: 900;
            letter-spacing: .05em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .kpi-value {
            font-family: Manrope, sans-serif;
            color: var(--text);
            font-size: 30px;
            line-height: 36px;
            font-weight: 800;
        }

        .kpi-status-pass { color: var(--success); }
        .kpi-status-fail { color: var(--error); }

        .file-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            margin-top: 18px;
        }

        .file-meta {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .file-badge {
            width: 44px;
            height: 44px;
            border-radius: 11px;
            display: grid;
            place-items: center;
            background: var(--surface-soft);
            color: var(--secondary);
            font-weight: 900;
        }

        .progress-track {
            width: 100%;
            height: 8px;
            border-radius: 999px;
            background: var(--surface-high);
            overflow: hidden;
            margin-top: 14px;
        }

        .progress-fill {
            height: 100%;
            width: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--secondary), #7c3aed);
            animation: fillBar .55s ease both;
        }

        .error-card {
            border-color: rgba(186, 26, 26, .24);
            background: rgba(255, 218, 214, .42);
        }

        .success-card {
            border-color: rgba(5, 150, 105, .22);
            background: rgba(236, 253, 245, .72);
        }

        .section-title {
            margin: 0 0 4px;
            font-family: Manrope, sans-serif;
            font-size: 22px;
            line-height: 30px;
            font-weight: 800;
        }

        div[data-testid="stFileUploader"] section {
            border-radius: 14px;
            border: 1px solid rgba(198, 198, 205, .55);
            background: rgba(255, 255, 255, .84);
        }

        /* Style the actual upload button inside the file uploader */
        div[data-testid="stFileUploader"] button {
            background-color: var(--secondary) !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            font-weight: 700 !important;
            padding: 8px 16px !important;
            transition: background-color 0.18s ease !important;
        }
        div[data-testid="stFileUploader"] button:hover {
            background-color: #3b31c1 !important;
        }

        /* Style Confirm Mapping buttons */
        div.stButton button[kind="primary"] {
            background-color: var(--secondary) !important;
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(75, 65, 225, 0.2) !important;
            transition: transform 0.18s ease, background-color 0.18s ease !important;
        }
        div.stButton button[kind="primary"]:hover {
            background-color: #3b31c1 !important;
            transform: translateY(-1px) !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        @keyframes fillBar {
            from { width: 12%; }
            to { width: 100%; }
        }

        @keyframes floatIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 1050px) {
            .page-header { align-items: flex-start; flex-direction: column; }
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 680px) {
            .page-title { font-size: 34px; line-height: 42px; }
            .kpi-grid { grid-template-columns: 1fr; }
            .file-card { align-items: flex-start; flex-direction: column; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;margin:10px 0 26px;">
                <div style="width:40px;height:40px;border-radius:10px;background:#131b2e;color:white;display:flex;align-items:center;justify-content:center;font-weight:900;">RI</div>
                <div>
                    <div style="font-family:Manrope,sans-serif;font-weight:900;font-size:1.35rem;line-height:1;">RiskIntel</div>
                    <div style="font-size:.76rem;color:#7c839b;font-weight:800;">Enterprise Tier</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <a href="/" target="_self" style="display:flex;align-items:center;gap:12px;padding:12px 14px;border-radius:10px;color:#45464d;text-decoration:none;font-weight:800;">
                <span style="font-size:18px;">▦</span> Overview
            </a>
            <a href="/?page=upload" target="_self" style="display:flex;align-items:center;gap:12px;padding:12px 14px;border-left:4px solid #4b41e1;border-radius:0 10px 10px 0;background:rgba(100,94,251,.10);color:#4b41e1;text-decoration:none;font-weight:900;">
                <span style="font-size:18px;">⬆</span> Upload Dataset
            </a>
            <div style="margin-top:14px;padding:12px;border-radius:12px;background:rgba(226,223,255,.45);border:1px solid rgba(75,65,225,.12);">
                <div style="font-weight:900;color:#4b41e1;font-size:13px;">Upload enabled</div>
                <div style="font-size:12px;color:#45464d;margin-top:4px;">CSV and XLSX files accepted.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def bytes_to_mb(size: int) -> str:
    return f"{size / (1024 * 1024):.2f} MB"


def render_header() -> None:
    st.markdown(
        """
        <div class="page-header">
            <div>
                <div class="eyebrow">RiskIntel Data Operations</div>
                <h1 class="page-title">Upload Dataset</h1>
                <div class="page-subtitle">
                    Ingest customer credit data, validate features flexibly, store the dataset in PostgreSQL,
                    and prepare it for Portfolio Dashboard and Risk Analytics modules.
                </div>
            </div>
            <div class="status-pill">Secure Ingestion Pipeline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_shell() -> None:
    st.markdown(
        """
        <div class="card upload-card">
            <div>
                <div class="upload-icon">CSV</div>
                <div class="upload-title">Drop dataset here</div>
                <div class="muted">
                    Upload CSV or Excel files with any schema.
                    Maximum practical size depends on your Streamlit server configuration.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_file_card(file_name: str, file_size: int, status: str, is_success: bool = True) -> None:
    status_class = "success-card" if is_success else "error-card"
    status_color = "#059669" if is_success else "#ba1a1a"
    st.markdown(
        f"""
        <div class="card file-card {status_class}">
            <div class="file-meta">
                <div class="file-badge">FILE</div>
                <div>
                    <div style="font-weight:900;">{file_name}</div>
                    <div class="muted">{bytes_to_mb(file_size)}</div>
                </div>
            </div>
            <div style="color:{status_color};font-weight:900;">{status}</div>
        </div>
        <div class="progress-track"><div class="progress-fill"></div></div>
        """,
        unsafe_allow_html=True,
    )


def render_error_card(title: str, messages: list[str]) -> None:
    list_items = "".join(f"<li>{message}</li>" for message in messages)
    st.markdown(
        f"""
        <div class="card error-card">
            <h3 class="section-title" style="color:#ba1a1a;">{title}</h3>
            <ul class="muted" style="margin-bottom:0;">{list_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_success_card(rows_stored: int, columns_count: int, upload_timestamp: str) -> None:
    st.markdown(
        f"""
        <div class="card success-card">
            <h3 class="section-title" style="color:#059669;">Dataset Successfully Processed</h3>
            <div class="kpi-grid">
                <div class="kpi success"><div class="kpi-label">Database Status</div><div class="kpi-value kpi-status-pass">Connected</div></div>
                <div class="kpi success"><div class="kpi-label">Rows Stored</div><div class="kpi-value">{rows_stored:,}</div></div>
                <div class="kpi"><div class="kpi-label">Total Columns</div><div class="kpi-value">{columns_count}</div></div>
                <div class="kpi"><div class="kpi-label">Upload Timestamp</div><div class="kpi-value" style="font-size:20px;line-height:28px;">{upload_timestamp}</div></div>
            </div>
            <div class="muted" style="margin-top:14px;">Table: <strong>{TABLE_NAME}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_validation_summary(result: ValidationResult) -> None:
    status_class = "kpi-status-pass" if result.is_valid else "kpi-status-fail"
    card_class = "success" if result.is_valid else "error"
    st.markdown(
        f"""
        <div class="card">
            <h3 class="section-title">Dataset Validation</h3>
            <div class="kpi-grid">
                <div class="kpi"><div class="kpi-label">Records Uploaded</div><div class="kpi-value">{result.records_uploaded:,}</div></div>
                <div class="kpi warning"><div class="kpi-label">Missing Values</div><div class="kpi-value">{result.missing_values:,}</div></div>
                <div class="kpi warning"><div class="kpi-label">Duplicate Customers</div><div class="kpi-value">{result.duplicate_customers:,}</div></div>
                <div class="kpi {card_class}"><div class="kpi-label">Validation Status</div><div class="kpi-value {status_class}">{result.status_label}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.errors:
        render_error_card("Validation Issues", result.errors)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")

    buffer = BytesIO(raw_bytes)
    if suffix == "csv":
        return pd.read_csv(buffer)
    if suffix == "xlsx":
        return pd.read_excel(buffer)
    raise ValueError("Invalid file type. Upload a CSV or XLSX file.")


def render_column_information(df: pd.DataFrame) -> None:
    column_info = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_values": [int(df[column].isna().sum()) for column in df.columns],
            "non_null_values": [int(df[column].notna().sum()) for column in df.columns],
        }
    )
    st.markdown('<div class="card"><h3 class="section-title">Column Information</h3></div>', unsafe_allow_html=True)
    st.dataframe(column_info, use_container_width=True, hide_index=True)


def render_preview(df: pd.DataFrame) -> None:
    st.markdown('<div class="card"><h3 class="section-title">Dataset Preview</h3><div class="muted">Showing first 20 matching rows.</div></div>', unsafe_allow_html=True)
    search_query = st.text_input("Search preview", placeholder="Search across visible dataset values")
    preview_df = df.copy()
    if search_query:
        mask = preview_df.astype(str).apply(
            lambda row: row.str.contains(search_query, case=False, na=False).any(),
            axis=1,
        )
        preview_df = preview_df[mask]
    st.dataframe(preview_df.head(20), use_container_width=True, height=420)
    render_column_information(df)


def render_database_analytics(analytics: dict) -> None:
    total_customers = int(analytics.get("total_customers") or 0)
    average_income = float(analytics.get("average_income") or 0)
    average_loan_amount = float(analytics.get("average_loan_amount") or 0)
    st.markdown(
        f"""
        <div class="card">
            <h3 class="section-title">Database Analytics</h3>
            <div class="kpi-grid">
                <div class="kpi success"><div class="kpi-label">Total Customers</div><div class="kpi-value">{total_customers:,}</div></div>
                <div class="kpi"><div class="kpi-label">Average Income</div><div class="kpi-value">${average_income:,.0f}</div></div>
                <div class="kpi"><div class="kpi-label">Average Loan Amount</div><div class="kpi-value">${average_loan_amount:,.0f}</div></div>
                <div class="kpi success"><div class="kpi-label">Source Table</div><div class="kpi-value" style="font-size:22px;">{TABLE_NAME}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main(configure_page: bool = True, show_sidebar: bool = True) -> None:
    if configure_page:
        st.set_page_config(
            page_title="Upload Dataset | RiskIntel",
            page_icon="RI",
            layout="wide",
        )
    inject_upload_css()
    if show_sidebar:
        render_sidebar()
    render_header()

    # Load target columns
    target_cols = get_base_target_columns()

    # Determine if API Key is available
    env_api_key = ""
    if "GEMINI_API_KEY" in st.secrets:
        env_api_key = st.secrets["GEMINI_API_KEY"]
    elif "GOOGLE_API_KEY" in st.secrets:
        env_api_key = st.secrets["GOOGLE_API_KEY"]
    
    if not env_api_key:
        env_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

    left, right = st.columns([0.64, 0.36], gap="large")
    with left:
        render_upload_shell()
        uploaded_file = st.file_uploader(
            "Upload customer risk dataset",
            type=["csv", "xlsx"],
            label_visibility="collapsed",
        )
        
        # API Key input if not in environment or secrets
        if not env_api_key:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Enter Gemini API key to activate AI mapping...",
                help="If left blank, local fuzzy matching heuristics will be used to map columns automatically."
            )
        else:
            api_key = env_api_key

    with right:
        # AI Schema Detection Card
        st.markdown(
            """
            <div class="card">
                <h3 class="section-title">AI Schema Detection</h3>
                <div class="muted">Intelligent Column Mapping Pipeline</div>
                <p class="muted" style="margin-top: 10px; font-size: 13.5px; line-height: 20px;">
                    Upload any credit-risk related dataset. Our AI automatically identifies, maps, and transforms your columns into the model schema before prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Processing Flow Card
        st.markdown(
            """
            <div class="card" style="margin-top: 20px;">
                <h3 class="section-title" style="margin-bottom: 15px;">Processing Flow</h3>
                <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
                    <div style="padding:8px 12px; background:rgba(75, 65, 225, 0.08); color:var(--secondary); font-weight:800; border-radius:8px; width:100%; text-align:center; font-size:13px;">Upload Dataset</div>
                    <div style="font-size:12px; color:var(--outline); line-height:1;">↓</div>
                    <div style="padding:8px 12px; background:rgba(75, 65, 225, 0.12); color:var(--secondary); font-weight:800; border-radius:8px; width:100%; text-align:center; font-size:13px;">Gemini Column Detection</div>
                    <div style="font-size:12px; color:var(--outline); line-height:1;">↓</div>
                    <div style="padding:8px 12px; background:rgba(75, 65, 225, 0.18); color:var(--secondary); font-weight:800; border-radius:8px; width:100%; text-align:center; font-size:13px;">Feature Mapping</div>
                    <div style="font-size:12px; color:var(--outline); line-height:1;">↓</div>
                    <div style="padding:8px 12px; background:rgba(75, 65, 225, 0.24); color:var(--secondary); font-weight:800; border-radius:8px; width:100%; text-align:center; font-size:13px;">Feature Engineering</div>
                    <div style="font-size:12px; color:var(--outline); line-height:1;">↓</div>
                    <div style="padding:8px 12px; background:var(--primary-container); color:white; font-weight:800; border-radius:8px; width:100%; text-align:center; font-size:13px;">Model Prediction</div>
                    <div style="font-size:12px; color:var(--outline); line-height:1;">↓</div>
                    <div style="padding:8px 12px; background:rgba(5, 150, 105, 0.08); color:var(--success); font-weight:800; border-radius:8px; width:100%; text-align:center; font-size:13px;">SHAP Explainability</div>
                    <div style="font-size:12px; color:var(--outline); line-height:1;">↓</div>
                    <div style="padding:8px 12px; background:rgba(5, 150, 105, 0.16); color:var(--success); font-weight:800; border-radius:8px; width:100%; text-align:center; font-size:13px;">Risk Report</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="card">
                <h3 class="section-title">Waiting for Dataset</h3>
                <div class="muted">Upload a CSV or XLSX file to begin validation and database ingestion.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        df = read_uploaded_file(uploaded_file)
    except Exception as exc:
        render_file_card(uploaded_file.name, uploaded_file.size, "Upload Failed", is_success=False)
        render_error_card("File Processing Error", [str(exc)])
        return

    render_file_card(uploaded_file.name, uploaded_file.size, "Upload Complete")
    
    # Structural validation
    validation = validate_dataset(df)
    render_validation_summary(validation)

    # API Key notification
    if api_key:
        st.success("✨ Gemini AI Schema Detection is active.", icon="🤖")
    else:
        st.warning("⚠️ Using local heuristic mapping fallback. Enter Gemini API key to activate AI Schema Detection.", icon="⚠️")

    # Extract column names and sample values
    sample_dict = {}
    for col in df.columns:
        samples = [str(x) for x in df[col].dropna().head(3).tolist()]
        sample_dict[col] = samples

    # Run schema mapping
    mapping_state_key = f"detected_mappings_{uploaded_file.name}"
    if mapping_state_key not in st.session_state:
        with st.spinner("AI is analyzing dataset schema..."):
            detected = get_gemini_mappings(list(df.columns), target_cols, sample_dict, api_key)
            st.session_state[mapping_state_key] = detected

    # Render mapping UI table
    st.markdown('<div class="card"><h3 class="section-title">Detected Feature Mapping</h3></div>', unsafe_allow_html=True)
    
    target_options = ["None"] + target_cols
    user_mappings = {}

    head_cols = st.columns([0.25, 0.25, 0.15, 0.35])
    head_cols[0].markdown("**Uploaded Column**")
    head_cols[1].markdown("**Model Feature**")
    head_cols[2].markdown("**Confidence**")
    head_cols[3].markdown("**Explanation**")
    st.markdown("<hr style='margin: 8px 0;'>", unsafe_allow_html=True)

    detected_map = st.session_state[mapping_state_key]

    for i, col in enumerate(df.columns):
        mapping = detected_map.get(col, {"model_feature": None, "confidence": 0.0, "explanation": "No match detected."})
        detected_feat = mapping.get("model_feature") or "None"
        confidence = mapping.get("confidence", 0.0)
        explanation = mapping.get("explanation", "")

        try:
            default_idx = target_options.index(detected_feat)
        except ValueError:
            default_idx = 0

        row_cols = st.columns([0.25, 0.25, 0.15, 0.35])
        row_cols[0].write(f"`{col}`")
        
        selected_feat = row_cols[1].selectbox(
            f"Map {col}",
            options=target_options,
            index=default_idx,
            key=f"map_sel_{col}_{uploaded_file.name}_{i}",
            label_visibility="collapsed"
        )
        
        conf_pct = int(confidence * 100)
        if conf_pct >= 90:
            badge_style = "background:rgba(5, 150, 105, 0.12); color:var(--success);"
        elif conf_pct >= 75:
            badge_style = "background:rgba(217, 119, 6, 0.12); color:var(--warning);"
        else:
            badge_style = "background:rgba(118, 119, 125, 0.12); color:var(--muted);"
            
        row_cols[2].markdown(f'<span class="badge" style="{badge_style}">{conf_pct}%</span>', unsafe_allow_html=True)
        row_cols[3].write(explanation)

        user_mappings[col] = selected_feat if selected_feat != "None" else None

    # Render Preview
    render_preview(df)

    # Confirm & Process Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Confirm Mappings & Process Dataset", type="primary", use_container_width=True):
        # 1. Run Feature Engineering
        with st.spinner("Engineering features and preparing dataset..."):
            try:
                original_df, mapped_df, engineered_df = run_feature_engineering_pipeline(df, user_mappings)
            except Exception as exc:
                st.error(f"Feature engineering failed: {exc}")
                return

        # 2. Run Predictions using XGBoost
        from app import load_model, run_predictions
        model = load_model()
        if model is None:
            st.error("XGBoost model file not found. Place 'xgboost_default_model.json' in the app directory.")
            return

        with st.spinner("Running XGBoost predictions..."):
            try:
                # Compare processed columns with model expected features
                expected_features = model.get_booster().feature_names
                missing_features = [f for f in expected_features if f not in engineered_df.columns or engineered_df[f].isna().all()]
                if missing_features:
                    st.warning(f"⚠️ Warning: The following critical features are unavailable in the uploaded dataset and will be filled with defaults: {', '.join(missing_features)}")

                stats = run_predictions(engineered_df, model)
            except Exception as exc:
                st.error(f"XGBoost predictions failed: {exc}")
                return

        # 3. Cache and persist results
        st.session_state["dashboard_stats"] = stats
        st.session_state["has_data"] = True
        
        # Persist stats to disk so they survive page reloads/session resets
        db_dir = os.path.join("database")
        os.makedirs(db_dir, exist_ok=True)
        stats_path = os.path.join(db_dir, "dashboard_stats.json")
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        except Exception as exc:
            st.warning(f"Failed to persist stats to disk: {exc}")

        # 4. Save to Database
        upload_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with st.spinner("Connecting to PostgreSQL and saving dataset..."):
            try:
                rows_stored, db_analytics = process_upload_to_database(engineered_df)
                render_success_card(rows_stored, len(engineered_df.columns), upload_timestamp)
                render_database_analytics(db_analytics)
                st.balloons()
            except Exception as exc:
                st.warning(f"PostgreSQL storage skipped or failed: {exc}")
                st.success("✅ Dataset successfully processed! Dashboard KPIs updated in session memory.")
                st.balloons()

        # 5. Redirect back to Dashboard Overview
        import time
        st.success("🔄 Redirecting back to Dashboard Overview in 2 seconds...")
        st.components.v1.html(
            """
            <script>
                setTimeout(function() {
                    try {
                        window.top.location.assign('/');
                    } catch(e) {
                        window.location.href = '/';
                    }
                }, 2000);
            </script>
            """,
            height=0
        )
        time.sleep(2.5)
        st.query_params.clear()
        if "active_page" in st.session_state:
            st.session_state["active_page"] = "overview"
        st.rerun()


if __name__ == "__main__":
    main()
