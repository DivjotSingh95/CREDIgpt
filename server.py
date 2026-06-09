import os
import sys
import json
import shutil
from io import BytesIO
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project directory to sys.path
sys.path.insert(0, os.getcwd())

from app import load_model, run_predictions
from utils.mapping import get_base_target_columns, get_heuristic_mappings, get_gemini_mappings
from utils.validation import run_feature_engineering_pipeline
from database.db_connection import process_upload_to_database

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CREDIgpt Web API",
    description="Backend services for the CREDIgpt Credit Risk Intelligence Platform"
)

# Enable CORS for cross-domain requests (Vercel frontend calling Render backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files folder
os.makedirs("static", exist_ok=True)
os.makedirs("database", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class MappingConfirmation(BaseModel):
    mappings: dict

def get_api_key() -> str:
    # Check Streamlit secrets first for compatibility
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r", encoding="utf-8") as f:
                content = f.read()
                for line in content.splitlines():
                    if "GEMINI_API_KEY" in line or "GOOGLE_API_KEY" in line:
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip().replace('"', '').replace("'", "")
        except Exception:
            pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

def generate_fallback_stats() -> dict:
    print("Generating fallback dashboard stats from test sample...")
    test_path = os.path.join(os.getcwd(), "application_test.csv")
    if os.path.exists(test_path):
        try:
            df = pd.read_csv(test_path, nrows=50)
            base_targets = get_base_target_columns()
            full_mappings = get_heuristic_mappings(list(df.columns), base_targets, {})
            confirmed_mappings = {col: full_mappings[col]["model_feature"] for col in df.columns}
            original, mapped, engineered = run_feature_engineering_pipeline(df, confirmed_mappings)
            model = load_model()
            if model is not None:
                stats = run_predictions(engineered, model)
                stats_path = os.path.join("database", "dashboard_stats.json")
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                return stats
        except Exception as e:
            print(f"Fallback generation failed: {e}")
    return {}

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/config")
def get_config():
    return {"gemini_api_key": get_api_key()}

@app.get("/api/stats")
def get_stats():
    # Return empty response so app starts fresh. Individual user sessions use sessionStorage.
    return {}


@app.get("/api/load-sample")
def load_sample_dataset():
    sample_file_path = os.path.join("scratch", "sample_upload.csv")
    
    # Generate the sample file if it doesn't exist
    if not os.path.exists(sample_file_path):
        os.makedirs("scratch", exist_ok=True)
        # Try selected_features_processed.csv first
        processed_csv_path = "selected_features_processed.csv"
        test_csv_path = "application_test.csv"
        train_csv_path = "application_train.csv"
        
        df = None
        if os.path.exists(processed_csv_path):
            try:
                print("Reading selected_features_processed.csv to build sample...")
                df = pd.read_csv(processed_csv_path, nrows=50)
                # Keep only the base target columns to simulate a raw upload
                base_targets = get_base_target_columns()
                existing_base = [col for col in base_targets if col in df.columns]
                df = df[existing_base]
            except Exception as e:
                print(f"Failed to read selected_features_processed.csv: {e}")
                df = None
                
        if df is None and os.path.exists(test_csv_path):
            try:
                df = pd.read_csv(test_csv_path, nrows=50)
            except Exception:
                pass
        if df is None and os.path.exists(train_csv_path):
            try:
                df = pd.read_csv(train_csv_path, nrows=50)
            except Exception:
                pass
                
        if df is None:
            # Generate dummy dataframe with essential columns
            print("Generating fallback mock sample CSV...")
            dummy_data = []
            for i in range(50):
                dummy_data.append({
                    "Applicant ID": 100001 + i,
                    "CODE_GENDER": "M" if i % 2 == 0 else "F",
                    "CNT_CHILDREN": i % 3,
                    "Annual Earnings": 120000.0 + (i * 3500.0),
                    "Loan Value": 300000.0 + (i * 12000.0),
                    "AMT_ANNUITY": 15000.0 + (i * 450.0),
                    "AMT_GOODS_PRICE": 280000.0 + (i * 12000.0),
                    "NAME_INCOME_TYPE": "Working" if i % 3 != 0 else "Commercial associate",
                    "NAME_EDUCATION_TYPE": "Secondary / secondary special" if i % 4 != 0 else "Higher education",
                    "NAME_FAMILY_STATUS": "Married" if i % 2 == 0 else "Single",
                    "NAME_HOUSING_TYPE": "House / apartment",
                    "Age in Days": -12000 - (i * 120),
                    "Employment Days": -1500 - (i * 50),
                    "DAYS_REGISTRATION": -4000.0,
                    "DAYS_ID_PUBLISH": -2500,
                    "CNT_FAM_MEMBERS": (i % 3) + 2.0,
                    "REGION_RATING_CLIENT": 2,
                    "REGION_RATING_CLIENT_W_CITY": 2,
                    "ORGANIZATION_TYPE": "Business Entity Type 3",
                    "EXT_SOURCE_1": 0.45 + (i * 0.005),
                    "EXT_SOURCE_2": 0.55 + (i * 0.002),
                    "EXT_SOURCE_3": 0.35 + (i * 0.006),
                    "OBS_30_CNT_SOCIAL_CIRCLE": 1.0,
                    "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
                    "OBS_60_CNT_SOCIAL_CIRCLE": 1.0,
                    "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
                    "DAYS_LAST_PHONE_CHANGE": -100.0,
                    "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
                    "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
                    "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
                    "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
                    "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
                    "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0
                })
            df = pd.DataFrame(dummy_data)
        else:
            # Rename columns to simulate real-world uploads
            df = df.rename(columns={
                "SK_ID_CURR": "Applicant ID",
                "AMT_INCOME_TOTAL": "Annual Earnings",
                "AMT_CREDIT": "Loan Value",
                "DAYS_BIRTH": "Age in Days",
                "DAYS_EMPLOYED": "Employment Days",
                "CODE_GENDER": "Gender Code"
            })
            
        df.to_csv(sample_file_path, index=False)

    try:
        df = pd.read_csv(sample_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read sample dataset: {e}")

    # Standardize cache as CSV format
    temp_path = os.path.join("database", "temp_upload.csv")
    orig_path = os.path.join("database", "original_uploaded.csv")
    df.to_csv(temp_path, index=False)
    df.to_csv(orig_path, index=False)

    # Extract column names and 3 samples for each
    sample_dict = {}
    for col in df.columns:
        sample_dict[col] = [
            x if (pd.notna(x) and not (isinstance(x, float) and np.isnan(x))) else None 
            for x in df[col].head(3).tolist()
        ]

    # Generate mappings using Heuristics or Gemini API
    api_key = get_api_key()
    base_targets = get_base_target_columns()
    
    if api_key:
        print("Using Gemini API for schema mapping on sample data...")
        mappings = get_gemini_mappings(list(df.columns), base_targets, sample_dict, api_key)
    else:
        print("Using local heuristics for schema mapping on sample data...")
        mappings = get_heuristic_mappings(list(df.columns), base_targets, sample_dict)

    return {
        "file_name": "sample_portfolio.csv",
        "file_size": os.path.getsize(sample_file_path),
        "columns": list(df.columns),
        "mappings": mappings,
        "target_columns": base_targets,
        "sample_data": sample_dict
    }

@app.post("/api/upload-raw")
async def upload_raw(file: UploadFile = File(...)):
    filename = file.filename
    filename_lower = filename.lower()
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".xlsx")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx) files are supported.")
        
    contents = await file.read()
    buffer = BytesIO(contents)
    
    try:
        if filename_lower.endswith(".csv"):
            df = pd.read_csv(buffer)
        else:
            df = pd.read_excel(buffer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse the file: {str(e)}")

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")

    # Save original raw upload for persistence
    orig_path = os.path.join("database", "original_uploaded" + ("_excel.xlsx" if filename_lower.endswith(".xlsx") else ".csv"))
    with open(orig_path, "wb") as f:
        f.write(contents)
        
    # Standardize cache as CSV format for processing
    temp_path = os.path.join("database", "temp_upload.csv")
    df.to_csv(temp_path, index=False)

    # Extract column names and 3 samples for each
    sample_dict = {}
    for col in df.columns:
        # Fill NaN values to avoid JSON serialization issues
        sample_dict[col] = [
            x if (pd.notna(x) and not (isinstance(x, float) and np.isnan(x))) else None 
            for x in df[col].head(3).tolist()
        ]

    # Generate mappings using Heuristics or Gemini API
    api_key = get_api_key()
    base_targets = get_base_target_columns()
    
    if api_key:
        print("Using Gemini API for schema mapping...")
        mappings = get_gemini_mappings(list(df.columns), base_targets, sample_dict, api_key)
    else:
        print("Using local heuristics for schema mapping...")
        mappings = get_heuristic_mappings(list(df.columns), base_targets, sample_dict)

    return {
        "file_name": filename,
        "file_size": len(contents),
        "columns": list(df.columns),
        "mappings": mappings,
        "target_columns": base_targets,
        "sample_data": sample_dict
    }

@app.post("/api/process")
def process_mappings(confirmation: MappingConfirmation):
    temp_path = os.path.join("database", "temp_upload.csv")
    if not os.path.exists(temp_path):
        raise HTTPException(status_code=400, detail="No uploaded dataset found. Please upload a file first.")

    try:
        if temp_path.endswith(".csv") or True: # assume CSV/Excel parsed previously
            df = pd.read_csv(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read cached dataset: {e}")

    # 1. Run Feature Engineering
    try:
        original_df, mapped_df, engineered_df = run_feature_engineering_pipeline(df, confirmation.mappings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature engineering pipeline failed: {str(e)}")

    # 2. Run Predictions using XGBoost
    model = load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="XGBoost model file 'xgboost_default_model.json' not found.")

    try:
        stats = run_predictions(engineered_df, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predictions failed: {str(e)}")

    # Cache and persist stats to disk
    stats_path = os.path.join("database", "dashboard_stats.json")
    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to persist stats to JSON: {e}")

    # 3. Save to database (PostgreSQL)
    rows_stored = 0
    db_success = False
    db_error = None
    db_analytics = {}
    try:
        rows_stored, db_analytics = process_upload_to_database(engineered_df)
        db_success = True
    except Exception as e:
        db_error = str(e)
        print(f"Database insertion skipped: {e}")

    return {
        "success": True,
        "stats": stats,
        "db_success": db_success,
        "db_error": db_error,
        "db_analytics": db_analytics,
        "rows_stored": rows_stored if db_success else len(df),
        "columns_count": len(engineered_df.columns),
        "upload_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

class ChatRequest(BaseModel):
    message: str
    stats: dict = None

def get_local_fallback_chat_response(message: str, stats: dict) -> str:
    message_lower = message.lower()
    
    total_cust = stats.get("total_customers", 0)
    def_rate = stats.get("default_rate", 0.0)
    avg_pd = stats.get("avg_risk_score", 0.0)
    high_risk = stats.get("high_risk_count", 0)
    critical = stats.get("critical_count", 0)
    portfolio = stats.get("total_portfolio", 0.0)
    avg_income = stats.get("avg_income", 0.0)
    avg_credit = stats.get("avg_credit", 0.0)
    
    # 1. Default rate or probability
    if "default rate" in message_lower or "default ratio" in message_lower or "default percentage" in message_lower:
        return f"Based on the local analysis of the uploaded dataset, the predicted default rate across the portfolio is **{def_rate*100:.2f}%**. The average probability of default calculated by our XGBoost model is **{avg_pd*100:.2f}%**."
        
    # 2. Total customers
    if "how many customer" in message_lower or "total customer" in message_lower or "number of customer" in message_lower or "count of customer" in message_lower or "dataset size" in message_lower:
        return f"There are a total of **{total_cust:,}** customer records in the uploaded dataset."
        
    # 3. High risk
    if "high risk" in message_lower or "high-risk" in message_lower:
        pct = (high_risk / total_cust * 100) if total_cust else 0
        return f"There are **{high_risk:,}** customers classified as High Risk (probability of default &ge; 15%). This represents **{pct:.1f}%** of the uploaded portfolio."
        
    # 4. Critical cases
    if "critical" in message_lower or "critical cases" in message_lower:
        return f"There are **{critical:,}** critical risk cases flagged (probability of default &ge; 25%). These require immediate risk mitigation action."
        
    # 5. Portfolio or exposure
    if "portfolio" in message_lower or "exposure" in message_lower or "total credit" in message_lower or "loan value" in message_lower:
        return f"The total credit portfolio exposure scored is **${portfolio:,.2f}** with an average credit limit of **${avg_credit:,.2f}** per customer."
        
    # 6. Average income
    if "average income" in message_lower or "avg income" in message_lower or "salary" in message_lower or "earnings" in message_lower or "pay" in message_lower:
        return f"The average annual income for customers in this dataset is **${avg_income:,.2f}**."
        
    # 7. Highest risk customers list
    if "highest risk" in message_lower or "worst customer" in message_lower or "top risk" in message_lower or "specific customer" in message_lower or "show me the" in message_lower or "flagged" in message_lower:
        profiles = stats.get("customer_profiles", [])
        if not profiles:
            return "No customer profiles are currently loaded in the database stats."
        # Sort by pd descending
        sorted_p = sorted(profiles, key=lambda x: x.get("pd", 0.0), reverse=True)
        top_5 = sorted_p[:5]
        resp = "Here are the top 5 highest-risk customer profiles predicted by the model:\n\n"
        resp += "| Customer ID | Prob. of Default | Risk Level | Income | Credit | Gender | Education |\n"
        resp += "| --- | --- | --- | --- | --- | --- | --- |\n"
        for p in top_5:
            resp += f"| {p['id']} | {p['pd']*100:.1f}% | **{p['risk_level']}** | ${p['income']:,.0f} | ${p['credit']:,.0f} | {p['gender']} | {p['education']} |\n"
        return resp

    # 8. Demographics or general summary
    if "demographic" in message_lower or "summary" in message_lower or "explain the data" in message_lower or "about the data" in message_lower:
        profiles = stats.get("customer_profiles", [])
        m_count = sum(1 for p in profiles if p.get("gender") == "M")
        f_count = sum(1 for p in profiles if p.get("gender") == "F")
        m_pct = (m_count / total_cust * 100) if total_cust else 0
        f_pct = (f_count / total_cust * 100) if total_cust else 0
        return f"**Dataset Demographics & Summary**:\n" \
               f"- **Total Customers**: {total_cust}\n" \
               f"- **Gender Split**: Male: {m_count} ({m_pct:.1f}%) | Female: {f_count} ({f_pct:.1f}%)\n" \
               f"- **Average Income**: ${avg_income:,.2f}\n" \
               f"- **Average Loan Credit**: ${avg_credit:,.2f}\n" \
               f"- **Default Rate**: {def_rate*100:.2f}%\n" \
               f"- **High Risk Count (PD &ge; 15%)**: {high_risk} ({high_risk/total_cust*100:.1f}%)"

    # Default general response
    return f"I am CREDIgpt. Gemini API is currently offline or rate-limited, but here is the **Portfolio Status Summary** of the uploaded dataset:\n\n" \
           f"- **Total Scored Customers**: {total_cust}\n" \
           f"- **Average Default Probability**: {avg_pd*100:.2f}%\n" \
           f"- **Overall Predicted Default Rate**: {def_rate*100:.2f}%\n" \
           f"- **Total Portfolio Exposure**: ${portfolio:,.2f}\n" \
           f"- **High Risk Accounts (PD &ge; 15%)**: {high_risk} (Critical: {critical})\n\n" \
           f"You can ask about specific metrics like 'average income', 'highest risk customers', 'default rate', or 'demographics' for direct answers."

@app.post("/api/chat")
def chat_with_data(req: ChatRequest):
    # Use stats passed in the request (supports frontend session isolation)
    stats = req.stats
    
    if not stats or not stats.get("total_customers"):
        # No stats provided or empty stats in the session
        msg = ("Welcome! I am CREDIgpt, your credit risk intelligence assistant. "
               "Currently, no active portfolio dataset is loaded in your session. "
               "Please navigate to the 'Upload Dataset' tab to upload a portfolio or load the sample dataset. "
               "Once loaded, I will be able to answer questions about default rates, risk distributions, "
               "high-risk customers, and specific customer profiles in your portfolio.")
        return {"response": msg}

        
    api_key = get_api_key()
    if not api_key:
        print("No Gemini API key found. Using local rule-based response engine.")
        return {"response": get_local_fallback_chat_response(req.message, stats)}
        
    try:
        import google.generativeai as genai
        
        # Build dataset summary
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
        
        # Add top 20 highest risk customer profiles
        profiles = stats.get("customer_profiles", [])
        if profiles:
            sorted_profiles = sorted(profiles, key=lambda x: x.get("pd", 0.0), reverse=True)
            dataset_summary["highest_risk_customers"] = sorted_profiles[:20]
            dataset_summary["customer_profiles_sample"] = profiles[:40]
            
        genai.configure(api_key=api_key)
        prompt = f"""You are CREDIgpt, an advanced credit risk assistant. You have access to a summary of the uploaded credit portfolio default risk dataset.

Dataset Summary:
{json.dumps(dataset_summary, indent=2)}

Answer the user's question contextually using the summary metrics, statistics, and risk driver demographics.
Guidelines:
- Be direct, professional, and clear.
- Use markdown (e.g. bold, bullet points, tables) to present numeric data or comparisons.
- If asked about specific customers, refer to the 'highest_risk_customers' list provided in the summary.
- If they ask general risk questions, use the summaries to explain patterns.
- Do not hallucinate or make up facts that are not present or inferred from the dataset.

User Question:
"{req.message}"
"""
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return {"response": response.text.strip()}
    except Exception as exc:
        print(f"Gemini chat failed: {exc}. Falling back to local rule-based response engine.")
        return {"response": get_local_fallback_chat_response(req.message, stats)}

if __name__ == "__main__":
    import uvicorn
    # If run directly, launch uvicorn on port 8000
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
