import os
import sys
from pathlib import Path

# Add project directory to sys.path
project_dir = Path("d:/data science proj/home-credit-default-risk")
sys.path.insert(0, str(project_dir))

from app import app_html

def main():
    print("Extracting HTML shell...")
    
    # We call app_html with specific placeholder strings
    html = app_html(
        kpis=None,
        dist_bars="DIST_BARS_PLACEHOLDER",
        has_data=False,
        customer_profiles_json="CUSTOMER_PROFILES_PLACEHOLDER",
        gemini_api_key="GEMINI_API_KEY_PLACEHOLDER",
        dataset_summary_json="DATASET_SUMMARY_PLACEHOLDER",
        dist_bars_overview="DIST_BARS_OVERVIEW_PLACEHOLDER",
        insights_html="INSIGHTS_HTML_PLACEHOLDER"
    )
    
    # Now we write it to a temporary file
    out_dir = project_dir / "static"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "index.html"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"HTML shell extracted to {out_path}")

if __name__ == "__main__":
    main()
