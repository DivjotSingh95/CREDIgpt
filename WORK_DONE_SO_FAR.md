# CREDIgpt credit risk dashboard website
# Work Done So Far - Handoff & Architecture Notes

We have successfully migrated the RiskIntel Credit Risk Intelligence Platform from a Streamlit-wrapped app to a standard website architecture. The app now runs as a high-performance **FastAPI backend** and a pure **HTML5/CSS3/JavaScript Single-Page Application (SPA)** frontend, preserving the entire high-fidelity design system, graphics, predictive logic, and features.

## Files Created & Updated

- `server.py` [NEW/UPDATED] - The FastAPI backend server. Handles routing, schema auto-mapping (via Gemini/heuristics), feature engineering, XGBoost predictions, database logging, and exposes the new `/api/chat` endpoint.
- `static/index.html` [NEW/UPDATED] - The single-page website frontend. Built with pure HTML/CSS/JS (vanilla glassmorphic shell layout, Chart.js, fuzzy customer search, drag-and-drop uploader, schema mapping table, and results logs).
- `app.py` [PRESERVED] - Streamlit implementation kept as a reference/fallback.
- `utils/mapping.py` [UPDATED] - Schema mapping utilities. Expanded base column alias lists for `DAYS_BIRTH` (matching "Age in Days") and `DAYS_EMPLOYED` (matching "Employment Days") and added a 6-second timeout block for the Gemini API call to prevent server hangs during rate-limiting.
- `utils/validation.py` [PRESERVED] - Validation pipeline for schema and quality checking.
- `requirements.txt` [UPDATED] - Includes the dependencies `fastapi`, `uvicorn`, and `python-multipart`.
- `README.md` [UPDATED] - Command-line run guide for the FastAPI server.

---

## 1. Migration to Standard Website Architecture (FastAPI + SPA)

Instead of running inside a Streamlit iframe, the app is now served natively:
- **FastAPI Backend**: Exposes clean endpoints:
  - `GET /` -> Serves `static/index.html`
  - `GET /api/config` -> Serves current configuration settings (like Gemini API keys safely).
  - `GET /api/stats` -> Serves cached database risk metrics from `database/dashboard_stats.json`. Generates fallback stats from `application_test.csv` if missing.
  - `POST /api/upload-raw` -> Receives CSV/XLSX uploads and returns column mapping predictions.
  - `POST /api/process` -> Validates confirmed mappings, runs the feature engineering pipeline, executes the XGBoost model, persists metrics, and saves data.
  - `POST /api/chat` -> Serves the chat requests.
- **Vanilla Frontend SPA**: Employs CSS transitions for the sidebar collapsible menu, switches tabs instantly in-memory, loads statistics dynamically, and initializes Chart.js canvas contexts smoothly.

---

## 2. Refined Calibration, Thresholds, and Expected Default Rate

- **Expected Default Rate correction**: Changed the `default_rate` calculation from the binary classification rate (`pred.mean()`) to the average Probability of Default (`proba_cal.mean()`). This resolves the unrealistic "49% default rate" display and correctly shows the true expected portfolio default rate of **4.29%** (based on the market prior).
- **Calibrated Risk Bands**: To resolve the over-inflated count of High Risk clients, we set industry-standard credit risk boundaries aligned with calibrated probabilities:
  - **Low Risk**: $p_{cal} < 7\%$ (below/near market prior)
  - **Medium Risk**: $7\% \le p_{cal} < 15\%$ (moderate warning)
  - **High Risk**: $p_{cal} \ge 15\%$ (alerts & manual review trigger)
  - **Critical Risk**: $p_{cal} \ge 25\%$ (severe default probability)
- **Feature Imputation**: Missing external bureau features are filled with a conservative `0.5` midpoint to prevent the XGBoost model from treating missing ratings as high-risk outliers (`0.0`).

---

## 3. Persistent Global Search & ID Redirect

- The global search box in the persistent header has been fully wired up. Typing a customer ID (e.g. `100001` or `100015`) and pressing Enter or clicking search redirects the user to the **Customer Risk Analytics** page.
- It copies the query into the local search input, live-filters the customer drop-down list to matches, and selects the matching customer, immediately updating the risk details cards, timeline, strengths, and risk amplifiers.

---

## 4. Ask Your Data Chatbot with Gemini / Local Fallback

- The chat box was fully connected to the backend `/api/chat` POST endpoint.
- **Server-Side Integration**: Chat queries are sent to the server where they are processed. The server reads the API key securely from the environment or `.streamlit/secrets.toml`.
- **Offline Fallback Engine**: If the Gemini API key is missing or rate-limited (Quota exceeded), the server catches the exception and falls back to a **local rule-based risk intelligence engine**!
- The local fallback engine parses queries (e.g. "default rate", "highest risk", "average income", "demographics") and returns precise markdown tables, KPIs, and metrics directly from the current dataset stats. The user gets a working chatbot experience even without internet or API keys!
- **Markdown Table and Typography Rendering**: Frontend Javascript was updated to parse markdown bold (`**`), italics (`*`), newlines (`\n`), and markdown tables (`| cell |`) into native HTML, rendering results logs beautifully.

---

## 5. Collapsible Collated Sidebar Navigation

- The left-hand navigation bar has been styled with hover-collapsible behaviors.
- The sidebar collapses offscreen by default (`transform: translateX(-280px)`) and exposes an invisible `::after` hover target on the left edge.
- Moving the cursor to the left edge slides open the drawer menu smoothly, overlapping the dashboard without shifting elements or resizing layouts.

---

## How to Verify and Run

1. Run the local uvicorn server:
   ```powershell
   python server.py
   ```

2. Run the endpoint validation suite:
   ```powershell
   python scratch/test_upload_api.py
   ```
   This verifies column mapping aliases, schema engineering pipelines, default rate averages, and XGBoost predictions.

3. Run the chat API test:
   ```powershell
   python scratch/test_chat_api.py
   ```
   This validates Gemini integration and the rule-based local fallback engine.
