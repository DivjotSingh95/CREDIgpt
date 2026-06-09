# RiskIntel Credit Risk Intelligence Platform

A high-performance FastAPI backend + vanilla HTML/CSS/JS frontend single-page application (SPA) for the RiskIntel Credit Risk Intelligence Platform.

## Architecture

- **Backend**: FastAPI web server (`server.py`) serving static files, performing predictive risk modeling via XGBoost, handling schema mapping, and exposing API endpoints for operations.
- **Frontend**: A premium, responsive vanilla HTML5/CSS3/JavaScript SPA (`static/index.html`) featuring glassmorphism aesthetics, interactive Chart.js visualizations, fuzzy customer search, and real-time AI-powered portfolio chat.

## Run Instructions

To run the application locally:

1. Install the dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

2. Start the FastAPI web server:
   ```powershell
   python server.py
   ```

3. Open your browser and navigate to:
   ```text
   http://127.0.0.1:8000
   ```

## Production Data Ingestion & PostgreSQL

Set these database environment variables before uploading a dataset to persist predictions to PostgreSQL:

```powershell
$env:DB_HOST="localhost"
$env:DB_PORT="5432"
$env:DB_NAME="riskintel_db"
$env:DB_USER="postgres"
$env:DB_PASSWORD="your_password"
```

If these environment variables are missing, the server will skip PostgreSQL writes and run fully in local mode (saving results in `database/dashboard_stats.json`), allowing seamless offline testing.
