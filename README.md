# Greyhound Analysis System (V44/V45 Production Release)

**Status:** Live Production  
**Strategies:** V44 (Back) & V45 (Lay)  
**Last Updated:** January 10, 2026

## 🚀 Quick Start in 60 Seconds

1.  **Install Requirements** (First time only):
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Bot:**
    ```bash
    python run.py
    ```

3.  The GUI will launch. It automatically starts scanning markets and persisting data.

---

## 🏛 System Overview

This system is an advanced algorithmic trading bot for Greyhound Racing, utilizing XGBoost machine learning models to identify value bets in real-time.

### Core Strategies
*   **BACK Strategy (V44 Steamer):** Identifies dogs that are "steaming" (price drop) with high model confidence (>35%).
*   **LAY Strategy (V45 Drifter):** Identifies favorites that are "drifting" (price rise) with weak model ratings (>60% Drift Prob).
*   **COVER Strategy:** A hedge mechanic that activates when high risk is detected (2+ Lays in one race), automatically backing the strongest steamer.

### Architecture
*   **`app.py`**: The Central Command GUI. Handles user interaction, **Persistent Session Management**, and displays live signals.
*   **`MarketAlphaEngine`**: The brain. Loads V44/V45 models, computes rolling features (using 365 days of history), and generating predictions.
*   **`LivePriceScraper`**: Background service that continuously monitors Betfair markets (T-60m to T-1m) and saves price movements to the database.
*   **`Automated Reporting`**: `reconcile_live_bets.py` triggers `generate_pl_report.py` to keep P/L summaries in sync with CSV data.

---

## 📂 Directory Structure

| Folder | Contents |
| :--- | :--- |
| **`root`** | `run.py` (Entry point), `requirements.txt`, `SYSTEM_MANUAL.md` |
| **`src/`** | Core application logic (GUI, Feature Engineering, Betting Manager) |
| **`scripts/`** | Critical scripts: `predict_v44_prod.py`, `reconcile_live_bets.py`, `generate_pl_report.py` |
| **`models/`** | Active production models (`xgb_v44*.pkl`, `xgb_v45*.pkl`) |
| **`archive/`** | Legacy files (Older versions V28-V43) can be found here |
