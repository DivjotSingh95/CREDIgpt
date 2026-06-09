import os
import re
from pathlib import Path

project_dir = Path("d:/data science proj/home-credit-default-risk")
html_path = project_dir / "static" / "index.html"

def main():
    print("Modifying static/index.html...")
    if not html_path.exists():
        print("Error: index.html not found. Run extract_html.py first.")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Add spinner animation style and other styling tweaks
    spinner_style = """
    @keyframes spin {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    .mapping-select {
      width: 100%;
      padding: 8px 12px;
      border: 1px solid rgba(198,198,205,.6);
      border-radius: 8px;
      font-weight: 700;
      background: white;
      color: var(--on-surface);
      outline: 0;
      cursor: pointer;
    }
    .mapping-select:focus {
      border-color: var(--secondary);
    }
    """
    html = html.replace("  </style>", spinner_style + "\n  </style>")

    # 2. Update navigation sidebar "Upload Dataset" link to be a button in index.html
    # Look for: <a class="nav-btn" href="/?page=upload" target="_top" data-route="upload"><span class="material-symbols-outlined">cloud_upload</span>Upload Dataset</a>
    html = re.sub(
        r'<a class="nav-btn"[^>]*data-route="upload"[^>]*>.*?cloud_upload.*?Upload Dataset.*?</a>',
        '<button class="nav-btn" data-page="upload"><span class="material-symbols-outlined">cloud_upload</span>Upload Dataset</button>',
        html,
        flags=re.DOTALL
    )

    # 3. Update Overview page primary action button to point to the data-page upload tab
    # Look for: <a class="btn primary" href="/?page=upload" data-route="upload">Upload Dataset</a>
    html = html.replace(
        '<a class="btn primary" href="/?page=upload" data-route="upload">Upload Dataset</a>',
        '<button class="btn primary" data-page="upload">Upload Dataset</button>'
    )

    # 4. Insert native HTML Upload Section right after Overview section
    upload_section_html = """
    <!-- UPLOAD PAGE -->
    <section class="page" id="upload">
      <div class="page-title-row">
        <div>
          <div class="page-eyebrow">Data Operations</div>
          <h1 class="page-title">Upload Dataset</h1>
          <p class="card-subtitle" style="font-size: 16px;">Ingest customer credit data, validate features, and run predictive risk scoring.</p>
        </div>
        <div class="status-pill"><span class="material-symbols-outlined" style="vertical-align:-3px;font-size:16px;font-variation-settings:'FILL' 1;">security</span> Secure Ingestion Pipeline</div>
      </div>
      
      <div class="grid-12" style="margin-top:20px;">
        <div class="card span-8" style="display:flex; flex-direction:column; gap:20px;">
          <h3 class="card-title">Select Dataset File</h3>
          
          <div class="card upload-card" id="drop-zone" onclick="document.getElementById('file-uploader').click()" style="min-height:260px; display:flex; align-items:center; justify-content:center; text-align:center; border:2px dashed rgba(118, 119, 125, .42); background:radial-gradient(circle at top left, rgba(75, 65, 225, .07), transparent 48%), rgba(255, 255, 255, .72); cursor:pointer;">
            <input type="file" id="file-uploader" accept=".csv, .xlsx" style="display:none;" />
            <div>
              <div class="upload-icon" style="width:72px; height:72px; margin:0 auto 16px; border-radius:50%; display:grid; place-items:center; background:rgba(75, 65, 225, .1); color:var(--secondary); font-size:28px;"><span class="material-symbols-outlined" style="font-size:36px; line-height:72px;">cloud_upload</span></div>
              <div class="upload-title" style="font-size:20px; font-weight:800; margin-bottom:6px;">Drag & drop your CSV or Excel file here</div>
              <div class="muted">or <span style="color:var(--secondary); text-decoration:underline; font-weight:800;">browse files</span> from your computer</div>
            </div>
          </div>
          
          <!-- File status card -->
          <div id="file-status-container" style="display:none;">
            <div class="card file-card success-card" style="display:flex; align-items:center; justify-content:space-between; padding:16px; border-left:4px solid var(--success); background:rgba(236, 253, 245, .5); gap:12px;">
              <div class="file-meta" style="display:flex; align-items:center; gap:12px;">
                <div class="file-badge" style="width:36px; height:36px; border-radius:8px; display:grid; place-items:center; background:var(--surface-container-high); color:var(--secondary); font-weight:900; font-size:12px;">FILE</div>
                <div>
                  <div style="font-weight:800;" id="uploaded-file-name">filename.csv</div>
                  <div class="muted" id="uploaded-file-size" style="font-size:12px;">1.2 MB</div>
                </div>
              </div>
              <div style="color:var(--success); font-weight:800; font-size:14px; margin-left:auto;" id="upload-status-label">Ready</div>
            </div>
            <div class="progress-track" style="height:6px; border-radius:99px; background:var(--surface-container-high); overflow:hidden; margin-top:8px;">
              <div class="progress-fill" id="upload-progress-fill" style="width:0%; height:100%; background:var(--secondary); transition:width 0.2s ease;"></div>
            </div>
          </div>
        </div>

        <div class="card span-4" style="background:var(--primary-container); color:white;">
          <h3 class="card-title" style="color:white; font-size:18px; margin-bottom:12px;">Data Schema Specs</h3>
          <p style="font-size:13px; line-height:20px; color:rgba(255,255,255,.75); margin-bottom:18px;">Our AI mapper connects your dataset to model features. We recommend matching these base columns for best performance:</p>
          <div style="display:flex; flex-direction:column; gap:10px; font-size:12px;">
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,.1); padding-bottom:6px;"><strong>Customer ID</strong><span>SK_ID_CURR</span></div>
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,.1); padding-bottom:6px;"><strong>Annual Income</strong><span>AMT_INCOME_TOTAL</span></div>
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,.1); padding-bottom:6px;"><strong>Loan Amount</strong><span>AMT_CREDIT</span></div>
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,.1); padding-bottom:6px;"><strong>Monthly Payment</strong><span>AMT_ANNUITY</span></div>
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,.1); padding-bottom:6px;"><strong>Customer Age (Days)</strong><span>DAYS_BIRTH</span></div>
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,.1); padding-bottom:6px;"><strong>Years Employed (Days)</strong><span>DAYS_EMPLOYED</span></div>
          </div>
        </div>
      </div>
      
      <!-- Schema mapping section -->
      <div id="mapping-container" style="display:none; margin-top:24px;">
        <div class="card">
          <h3 class="section-title">AI Schema Detection & Mapping</h3>
          <p class="muted">We analyzed your columns using semantic similarity. Review and customize mappings before processing.</p>
          
          <div style="margin-top:18px; overflow-x:auto;">
            <table class="table" style="width:100%; border-collapse:collapse;">
              <thead>
                <tr style="border-bottom:1px solid var(--outline-variant);">
                  <th style="width:25%; padding:12px 8px; text-align:left; font-size:12px; text-transform:uppercase; color:var(--outline);">Uploaded Column</th>
                  <th style="width:25%; padding:12px 8px; text-align:left; font-size:12px; text-transform:uppercase; color:var(--outline);">Model Target Feature</th>
                  <th style="width:15%; padding:12px 8px; text-align:left; font-size:12px; text-transform:uppercase; color:var(--outline);">Confidence</th>
                  <th style="width:35%; padding:12px 8px; text-align:left; font-size:12px; text-transform:uppercase; color:var(--outline);">Match Explanation</th>
                </tr>
              </thead>
              <tbody id="mapping-table-body">
                <!-- Dynamically populated rows -->
              </tbody>
            </table>
          </div>
          
          <button id="btn-confirm-mappings" class="btn primary" style="margin-top:24px; width:100%; justify-content:center; min-height:48px; font-size:15px;">Confirm Mappings & Process Dataset</button>
        </div>
      </div>

      <!-- Processing overlay loader -->
      <div id="processing-loader" style="display:none; margin-top:24px;">
        <div class="card success-card" style="text-align:center; padding:48px; border-color:var(--secondary); background:rgba(100,94,251,.05);">
          <span class="material-symbols-outlined" style="font-size:48px; color:var(--secondary); animation:spin 2s infinite linear; display:inline-block;">sync</span>
          <h3 class="section-title" style="margin-top:16px;" id="loader-title">Engineering Features...</h3>
          <p class="muted" id="loader-subtitle">Running pipeline: mapping columns, calculating ratios, and executing default risk predictions.</p>
        </div>
      </div>
      
      <!-- Success/Error alert box -->
      <div id="upload-result-container" style="display:none; margin-top:24px;">
        <!-- Dynamically injected success/error card -->
      </div>
    </section>
    """
    html = html.replace("    <!-- PORTFOLIO PAGE -->", upload_section_html + "\n\n    <!-- PORTFOLIO PAGE -->")

    # 5. Add element identifiers to Overview page metrics so they can be written dynamically
    html = html.replace(
        '<p class="metric-label">Total Customers</p>\n          <p class="metric-value">—</p>',
        '<p class="metric-label">Total Customers</p>\n          <p class="metric-value" id="kpi-total-customers">—</p>'
    )
    html = html.replace(
        '<p class="metric-label">Total Credit Portfolio</p>\n          <p class="metric-value">—</p>',
        '<p class="metric-label">Total Credit Portfolio</p>\n          <p class="metric-value" id="kpi-active-portfolio">—</p>'
    )
    html = html.replace(
        '<p class="metric-label">Avg Default Probability</p>\n          <p class="metric-value">—</p>',
        '<p class="metric-label">Avg Default Probability</p>\n          <p class="metric-value" id="kpi-avg-risk-score">—</p>'
    )
    html = html.replace(
        '<p class="metric-label">High Risk Customers</p>\n          <p class="metric-value">—</p>',
        '<p class="metric-label">High Risk Customers</p>\n          <p class="metric-value" id="kpi-high-risk-customers">—</p>'
    )

    # In overview: Default Rate and Critical Cases in mini-kpis
    html = html.replace(
        '<div class="card"><p class="tiny-title">Default Rate</p><p class="tiny-value">—</p></div>',
        '<div class="card"><p class="tiny-title">Default Rate</p><p class="tiny-value" id="kpi-default-rate">—</p></div>'
    )
    html = html.replace(
        '<div class="card"><p class="tiny-title">Critical Cases</p><p class="tiny-value">—</p></div>',
        '<div class="card"><p class="tiny-title">Critical Cases</p><p class="tiny-value" id="kpi-critical-exposure">—</p></div>'
    )

    # In top actions header status badge
    html = html.replace(
        '<span class="badge" style="color:var(--amber);background:rgba(217,119,6,.12);">No Data</span>',
        '<span id="header-status-badge"><span class="badge" style="color:var(--amber);background:rgba(217,119,6,.12);">No Data</span></span>'
    )

    # Overview page hero actions: remove obsolete data-route and redirect
    html = html.replace(
        '<a class="btn primary" href="/?page=upload" data-route="upload">Upload Dataset</a>',
        '<button class="btn primary" data-page="upload">Upload Dataset</button>'
    )

    # 6. Add ChartJS instances and dynamic API load handling in Javascript
    # Look for:
    #   const customers = CUSTOMER_PROFILES_PLACEHOLDER;
    #   const geminiApiKey = "GEMINI_API_KEY_PLACEHOLDER";
    #   const datasetSummary = DATASET_SUMMARY_PLACEHOLDER;
    
    js_init_vars = """
  let customers = [];
  let geminiApiKey = "";
  let datasetSummary = {};
  let hasData = false;
  
  let distChartInstance = null;
  let segmentChartInstance = null;
  let exposureChartInstance = null;
  let educationRiskChartInstance = null;
  let overviewDistChartInstance = null;
  
  let tempUploadedFileInfo = null;
    """
    
    html = html.replace('  const customers = CUSTOMER_PROFILES_PLACEHOLDER;', '')
    html = html.replace('  const geminiApiKey = "GEMINI_API_KEY_PLACEHOLDER";', '')
    html = html.replace('  const datasetSummary = DATASET_SUMMARY_PLACEHOLDER;', js_init_vars)

    # Update initCharts() to destroy previous chart instances
    chart_destroyer_code = """
  function initCharts() {
    if (distChartInstance) distChartInstance.destroy();
    if (segmentChartInstance) segmentChartInstance.destroy();
    if (exposureChartInstance) exposureChartInstance.destroy();
    if (educationRiskChartInstance) educationRiskChartInstance.destroy();
    if (overviewDistChartInstance) overviewDistChartInstance.destroy();

    let dataSrc = customers;
    """
    html = html.replace("  function initCharts() {\n    let dataSrc = customers;", chart_destroyer_code)

    # Update Chart definitions to store reference to instances
    html = html.replace("new Chart(ctx1.getContext('2d'), {", "distChartInstance = new Chart(ctx1.getContext('2d'), {")
    html = html.replace("new Chart(ctx2.getContext('2d'), {", "segmentChartInstance = new Chart(ctx2.getContext('2d'), {")
    html = html.replace("new Chart(ctx3.getContext('2d'), {", "exposureChartInstance = new Chart(ctx3.getContext('2d'), {")
    html = html.replace("new Chart(ctx4.getContext('2d'), {", "educationRiskChartInstance = new Chart(ctx4.getContext('2d'), {")
    html = html.replace("new Chart(ctx5.getContext('2d'), {", "overviewDistChartInstance = new Chart(ctx5.getContext('2d'), {")

    # Add dynamic load script and AJAX upload handling script
    dynamic_ajax_scripts = """
  function fmtMoney(v) {
    if (v >= 1e9) return "$" + (v / 1e9).toFixed(1) + "B";
    if (v >= 1e6) return "$" + (v / 1e6).toFixed(1) + "M";
    if (v >= 1e3) return "$" + (v / 1e3).toFixed(0) + "K";
    return "$" + v.toFixed(0);
  }

  function buildInsightsHTML(stats) {
    const profiles = stats.customer_profiles || [];
    if (profiles.length === 0) return "";
    
    let html = "";
    
    // 1. Gender Disparity
    const malePDs = [];
    const femalePDs = [];
    profiles.forEach(p => {
      if (p.gender === 'M') malePDs.push(p.pd);
      else if (p.gender === 'F') femalePDs.push(p.pd);
    });
    if (malePDs.length && femalePDs.length) {
      const maleAvg = malePDs.reduce((a,b)=>a+b, 0) / malePDs.length;
      const femaleAvg = femalePDs.reduce((a,b)=>a+b, 0) / femalePDs.length;
      html += `<li><span class="material-symbols-outlined" style="color:var(--secondary);">wc</span><div><strong>Gender Risk Disparity</strong><p class="card-subtitle">Male average default risk is ${(maleAvg*100).toFixed(1)}% vs Female average at ${(femaleAvg*100).toFixed(1)}%.</p></div></li>`;
    }
    
    // 2. Education Factor
    const eduMap = {};
    profiles.forEach(p => {
      if (p.education) {
        if (!eduMap[p.education]) eduMap[p.education] = [];
        eduMap[p.education].push(p.pd);
      }
    });
    let highestEdu = "";
    let highestEduVal = -1;
    for (let edu in eduMap) {
      const avg = eduMap[edu].reduce((a,b)=>a+b, 0) / eduMap[edu].length;
      if (avg > highestEduVal) {
        highestEduVal = avg;
        highestEdu = edu;
      }
    }
    if (highestEdu) {
      html += `<li><span class="material-symbols-outlined" style="color:var(--amber);">school</span><div><strong>Education Risk Factor</strong><p class="card-subtitle">Borrowers with '${highestEdu}' have the highest average default risk at ${(highestEduVal*100).toFixed(1)}%.</p></div></li>`;
    }
    
    // 3. Credit Leverage
    const highRiskLeverages = [];
    const lowRiskLeverages = [];
    profiles.forEach(p => {
      const cir = p.credit / (p.income || 1);
      if (p.risk_level === 'High') highRiskLeverages.push(cir);
      else if (p.risk_level === 'Low') lowRiskLeverages.push(cir);
    });
    if (highRiskLeverages.length && lowRiskLeverages.length) {
      const highAvg = highRiskLeverages.reduce((a,b)=>a+b, 0) / highRiskLeverages.length;
      const lowAvg = lowRiskLeverages.reduce((a,b)=>a+b, 0) / lowRiskLeverages.length;
      html += `<li><span class="material-symbols-outlined" style="color:var(--success);">account_balance_wallet</span><div><strong>Credit Leverage Ratio</strong><p class="card-subtitle">High risk accounts carry a leverage ratio of ${highAvg.toFixed(1)}x income vs ${lowAvg.toFixed(1)}x for low risk.</p></div></li>`;
    }
    
    // 4. Age Demographics
    const youngPDs = [];
    const oldPDs = [];
    profiles.forEach(p => {
      if (p.age < 35) youngPDs.push(p.pd);
      else if (p.age >= 55) oldPDs.push(p.pd);
    });
    if (youngPDs.length && oldPDs.length) {
      const youngAvg = youngPDs.reduce((a,b)=>a+b, 0) / youngPDs.length;
      const oldAvg = oldPDs.reduce((a,b)=>a+b, 0) / oldPDs.length;
      html += `<li><span class="material-symbols-outlined" style="color:var(--teal);">calendar_month</span><div><strong>Age Demographics</strong><p class="card-subtitle">Younger borrowers (&lt;35 yrs) average risk is ${(youngAvg*100).toFixed(1)}% vs ${(oldAvg*100).toFixed(1)}% for mature borrowers (&gt;=55 yrs).</p></div></li>`;
    }
    
    return html;
  }

  function buildDistBarsHTML(proba) {
    if (proba.length === 0) return "";
    
    const counts = [0, 0, 0, 0, 0];
    proba.forEach(p => {
      if (p < 0.05) counts[0]++;
      else if (p < 0.10) counts[1]++;
      else if (p < 0.15) counts[2]++;
      else if (p < 0.20) counts[3]++;
      else counts[4]++;
    });
    
    const labels = ["0–5%", "5–10%", "10–15%", "15–20%", ">20%"];
    const maxCount = Math.max(...counts) || 1;
    let html = "";
    
    counts.forEach((c, idx) => {
      const pct = Math.round(c / maxCount * 100);
      const isHot = idx >= 2;
      const cls = isHot ? "dist-bar hot" : "dist-bar";
      html += `
        <div class="dist-item">
          <div class="${cls}" style="--h:${pct}%;"></div>
          <small>${labels[idx]}</small>
        </div>`;
    });
    
    return html;
  }

  function buildOverviewBarsHTML(proba) {
    if (proba.length === 0) return "";
    
    const counts = [0, 0, 0, 0, 0, 0, 0];
    proba.forEach(p => {
      const idx = Math.min(6, Math.floor(p * 7 / 0.25));
      counts[idx]++;
    });
    
    const maxCount = Math.max(...counts) || 1;
    let html = "";
    counts.forEach(c => {
      const pct = Math.round(c / maxCount * 92);
      const fill = Math.round(pct * 0.85);
      html += `<div class="chart-bar" style="--h:${pct}%;--fill:${fill}%;"></div>`;
    });
    return html;
  }

  async function loadDashboardData() {
    try {
      const statsRes = await fetch('/api/stats');
      const configRes = await fetch('/api/config');
      const stats = await statsRes.json();
      const config = await configRes.json();

      if (config && config.gemini_api_key) {
        window.geminiApiKey = config.gemini_api_key;
      }

      if (stats && stats.total_customers) {
        window.customers = stats.customer_profiles || [];
        window.datasetSummary = stats;
        window.hasData = true;

        // Populate KPIs
        const kTotal = document.getElementById('kpi-total-customers');
        const kPortfolio = document.getElementById('kpi-active-portfolio');
        const kAvgPD = document.getElementById('kpi-avg-risk-score');
        const kHighRisk = document.getElementById('kpi-high-risk-customers');
        
        if (kTotal) kTotal.textContent = stats.total_customers.toLocaleString();
        if (kPortfolio) kPortfolio.textContent = fmtMoney(stats.total_portfolio);
        if (kAvgPD) kAvgPD.textContent = (stats.avg_risk_score * 100).toFixed(2) + "%";
        if (kHighRisk) kHighRisk.textContent = stats.high_risk_count.toLocaleString();
        
        // Portfolio KPIs: Total Customers (idx 0), Total Exposure (idx 1), Default Rate (idx 2), Avg Risk Score (idx 3), High Risk Count (idx 4), Critical Cases (idx 5)
        const portfolioKPIs = document.querySelectorAll('#portfolio .kpi p.tiny-value');
        if (portfolioKPIs.length >= 6) {
          portfolioKPIs[0].textContent = stats.total_customers.toLocaleString();
          portfolioKPIs[1].textContent = fmtMoney(stats.total_portfolio);
          portfolioKPIs[2].textContent = (stats.default_rate * 100).toFixed(2) + "%";
          portfolioKPIs[3].textContent = (stats.avg_risk_score * 100).toFixed(2) + "%";
          portfolioKPIs[4].textContent = stats.high_risk_count.toLocaleString();
          portfolioKPIs[5].textContent = stats.critical_count.toLocaleString();
        }

        // In overview: Default Rate and Critical Cases
        const miniKPIs = document.querySelectorAll('.mini-kpis p.tiny-value');
        if (miniKPIs.length >= 2) {
          miniKPIs[0].textContent = (stats.default_rate * 100).toFixed(2) + "%";
          miniKPIs[1].textContent = stats.critical_count.toLocaleString();
        }

        // Update top bar status badge and other badges to "✓ Data Loaded"
        document.querySelectorAll('.badge').forEach(el => {
          if (el.textContent.includes("No Data")) {
            el.innerHTML = '✓ Data Loaded';
            el.style.color = 'var(--success)';
            el.style.background = 'rgba(5,150,105,.12)';
          }
        });
        const headerBadge = document.getElementById('header-status-badge');
        if (headerBadge) {
          headerBadge.innerHTML = '<span class="badge" style="color:var(--success);background:rgba(5,150,105,.12);">✓ Data Loaded</span>';
        }

        // Dynamic Insights
        const insightsList = document.getElementById('insights-list');
        if (insightsList) {
          insightsList.innerHTML = buildInsightsHTML(stats);
        }

        // Distribution Bars
        const distBarsContainer = document.getElementById('dist-bars-container');
        if (distBarsContainer) {
          distBarsContainer.innerHTML = buildDistBarsHTML(stats.proba || []);
        }
        const distBarsOverview = document.getElementById('dist-bars-overview-container');
        if (distBarsOverview) {
          distBarsOverview.innerHTML = buildOverviewBarsHTML(stats.proba || []);
        }

        // Dynamic Overview AI box assessment status text
        const assessTitle = document.querySelector('.ai-box p');
        if (assessTitle) {
          assessTitle.textContent = "AI recommends: " + stats.high_risk_count + " accounts flag default alerts (PD >= 12%).";
        }

        // Update Ask Your Data page portfolio summary card
        const askSummaryLabel = document.querySelector('#ask .chat-right section.card p.card-subtitle');
        if (askSummaryLabel) {
          askSummaryLabel.innerHTML = `XGBoost model scores <strong style="color:var(--primary);">${stats.total_customers.toLocaleString()}</strong> customers. Default rate: <strong style="color:var(--error);">${(stats.default_rate * 100).toFixed(2)}%</strong>. Average probability of default: <strong>${(stats.avg_risk_score * 100).toFixed(2)}%</strong>.`;
        }
        const askHighRiskVal = document.querySelector('#ask .chat-right section.card div p.tiny-value');
        if (askHighRiskVal) {
          askHighRiskVal.textContent = stats.high_risk_count.toLocaleString();
          const askCriticalVal = askHighRiskVal.closest('div').nextElementSibling.querySelector('p.tiny-value');
          if (askCriticalVal) askCriticalVal.textContent = stats.critical_count.toLocaleString();
        }

        // Customer profile selection dropdown
        const select = document.getElementById('cust-select');
        if (select) {
          select.innerHTML = '';
          window.customers.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.id;
            opt.textContent = "ID: " + c.id + " (PD: " + (c.pd*100).toFixed(1) + "% - " + c.risk_level + ")";
            select.appendChild(opt);
          });
          document.getElementById('customer-profile-section').style.display = 'grid';
          document.getElementById('customer-no-data-placeholder').style.display = 'none';
          selectCustomer(window.customers[0].id);
        }

        initCharts();
      }
    } catch (e) {
      console.error("Failed to load dashboard data:", e);
    }
  }

  // File Uploader Event Listeners
  const dropZone = document.getElementById('drop-zone');
  const fileUploader = document.getElementById('file-uploader');
  
  if (dropZone && fileUploader) {
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'var(--secondary)';
      dropZone.style.background = 'rgba(75, 65, 225, .1)';
    });
    
    dropZone.addEventListener('dragleave', () => {
      dropZone.style.borderColor = 'rgba(118, 119, 125, .42)';
      dropZone.style.background = 'radial-gradient(circle at top left, rgba(75, 65, 225, .07), transparent 48%), rgba(255, 255, 255, .72)';
    });
    
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'rgba(118, 119, 125, .42)';
      dropZone.style.background = 'radial-gradient(circle at top left, rgba(75, 65, 225, .07), transparent 48%), rgba(255, 255, 255, .72)';
      if (e.dataTransfer.files.length) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });
    
    fileUploader.addEventListener('change', () => {
      if (fileUploader.files.length) {
        handleFileUpload(fileUploader.files[0]);
      }
    });
  }

  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = 2;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  async function handleFileUpload(file) {
    const container = document.getElementById('file-status-container');
    const nameLabel = document.getElementById('uploaded-file-name');
    const sizeLabel = document.getElementById('uploaded-file-size');
    const statusLabel = document.getElementById('upload-status-label');
    const progressFill = document.getElementById('upload-progress-fill');
    
    nameLabel.textContent = file.name;
    sizeLabel.textContent = formatBytes(file.size);
    statusLabel.textContent = "Uploading file to server...";
    progressFill.style.width = '30%';
    container.style.display = 'block';

    // Hide previous screens
    document.getElementById('mapping-container').style.display = 'none';
    document.getElementById('upload-result-container').style.display = 'none';
    document.getElementById('processing-loader').style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload-raw', {
        method: 'POST',
        body: formData
      });
      progressFill.style.width = '70%';
      
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Server upload failed.");
      }
      
      const data = await res.json();
      tempUploadedFileInfo = data;
      
      progressFill.style.width = '100%';
      statusLabel.textContent = "AI Schema Mapped Successfully!";
      
      // Render mapping UI
      renderMappingTable(data);
    } catch (e) {
      progressFill.style.width = '100%';
      progressFill.style.background = 'var(--error)';
      statusLabel.textContent = "Upload Failed";
      statusLabel.style.color = 'var(--error)';
      
      const resContainer = document.getElementById('upload-result-container');
      resContainer.innerHTML = `
        <div class="card error-card">
          <h3 class="section-title" style="color:var(--error);">Failed to Parse Dataset</h3>
          <p class="muted">${e.message}</p>
        </div>
      `;
      resContainer.style.display = 'block';
    }
  }

  function renderMappingTable(data) {
    const tbody = document.getElementById('mapping-table-body');
    tbody.innerHTML = '';
    
    data.columns.forEach(col => {
      const detected = data.mappings[col] || { model_feature: null, confidence: 0, explanation: "Unmapped" };
      const row = document.createElement('tr');
      row.style.borderBottom = '1px solid rgba(198,198,205,.18)';
      
      // Generate options for dropdown
      let selectOptions = '<option value="None">-- Ignore / Unmapped --</option>';
      data.target_columns.forEach(tgt => {
        const selected = (tgt === detected.model_feature) ? 'selected' : '';
        selectOptions += `<option value="${tgt}" ${selected}>${tgt}</option>`;
      });
      
      const confPct = Math.round(detected.confidence * 100);
      let badgeStyle = 'color:#6b7280; background:#f3f4f6;';
      if (detected.confidence >= 0.85) {
        badgeStyle = 'color:var(--success); background:rgba(5,150,105,.12);';
      } else if (detected.confidence >= 0.5) {
        badgeStyle = 'color:var(--amber); background:rgba(217,119,6,.12);';
      }
      
      row.innerHTML = `
        <td style="padding:14px 8px; font-weight:700;">${col}</td>
        <td style="padding:14px 8px;">
          <select class="mapping-select" data-uploaded-col="${col.replace(/"/g, '&quot;')}">
            ${selectOptions}
          </select>
        </td>
        <td style="padding:14px 8px;">
          <span class="badge" style="${badgeStyle}">${confPct}%</span>
        </td>
        <td style="padding:14px 8px; color:var(--on-surface-variant); font-size:13px;">${detected.explanation}</td>
      `;
      tbody.appendChild(row);
    });
    
    document.getElementById('mapping-container').style.display = 'block';
  }

  const btnConfirm = document.getElementById('btn-confirm-mappings');
  if (btnConfirm) {
    btnConfirm.addEventListener('click', async () => {
      const finalMappings = {};
      document.querySelectorAll('.mapping-select').forEach(sel => {
        const colName = sel.dataset.uploadedCol;
        const targetFeature = sel.value;
        finalMappings[colName] = targetFeature !== 'None' ? targetFeature : null;
      });

      document.getElementById('mapping-container').style.display = 'none';
      const loader = document.getElementById('processing-loader');
      const loaderTitle = document.getElementById('loader-title');
      const loaderSubtitle = document.getElementById('loader-subtitle');
      
      loaderTitle.textContent = "Engineering Features...";
      loaderSubtitle.textContent = "Mapping raw variables, computing leverage ratios, and pulling bureau records...";
      loader.style.display = 'block';

      try {
        const res = await fetch('/api/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mappings: finalMappings })
        });
        
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Processing failed.");
        }
        
        const result = await res.json();
        
        loaderSubtitle.textContent = "Saving portfolio metrics to PostgreSQL...";
        
        // Show success alert
        const resContainer = document.getElementById('upload-result-container');
        let dbStatus = '<div class="kpi success"><div class="kpi-label">Database Status</div><div class="kpi-value kpi-status-pass">Connected</div></div>';
        let dbNote = `Table: <strong>customer_data</strong>`;
        
        if (!result.db_success) {
          dbStatus = '<div class="kpi error"><div class="kpi-label">Database Status</div><div class="kpi-value kpi-status-fail">Disconnected</div></div>';
          dbNote = `<strong style="color:var(--error);">Notice:</strong> Database insertion failed (saved locally only): <em>${result.db_error}</em>`;
        }

        resContainer.innerHTML = `
          <div class="card success-card" style="border-left:4px solid var(--success);">
            <h3 class="section-title" style="color:var(--success);">Dataset Successfully Processed</h3>
            <div class="kpi-grid">
              ${dbStatus}
              <div class="kpi success"><div class="kpi-label">Rows Stored</div><div class="kpi-value">${result.rows_stored.toLocaleString()}</div></div>
              <div class="kpi"><div class="kpi-label">Total Columns</div><div class="kpi-value">${result.columns_count}</div></div>
              <div class="kpi"><div class="kpi-label">Upload Timestamp</div><div class="kpi-value" style="font-size:18px;line-height:28px;">${result.upload_timestamp}</div></div>
            </div>
            <div class="muted" style="margin-top:14px;">${dbNote}</div>
          </div>
        `;
        
        loader.style.display = 'none';
        resContainer.style.display = 'block';
        
        // Load the new metrics and refresh the entire UI!
        await loadDashboardData();
        
        // Automatically route back to the Overview tab after 2.5 seconds
        setTimeout(() => {
          const overviewBtn = document.querySelector('.nav-btn[data-page="overview"]');
          if (overviewBtn) {
            overviewBtn.click();
          }
          // Clear status container
          document.getElementById('file-status-container').style.display = 'none';
          document.getElementById('upload-result-container').style.display = 'none';
        }, 2500);

      } catch (e) {
        loader.style.display = 'none';
        const resContainer = document.getElementById('upload-result-container');
        resContainer.innerHTML = `
          <div class="card error-card">
            <h3 class="section-title" style="color:var(--error);">Feature Engineering Failed</h3>
            <p class="muted">${e.message}</p>
          </div>
        `;
        resContainer.style.display = 'block';
      }
    });
  }

  // Load configuration and data on boot
  loadDashboardData();
    """
    
    html = html.replace('  // Call initialization after rendering\n  setTimeout(initCharts, 100);', dynamic_ajax_scripts)

    # Save modified file
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    print("static/index.html updated successfully!")

if __name__ == "__main__":
    main()
