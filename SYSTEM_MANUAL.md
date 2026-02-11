# 📘 System Manual (V44/V45 Production)

**Version:** 2026-01-10  
**Engine:** Market Alpha V44/V45

---

## 1. Betting Strategies

### A. BACK Strategy (V44 Steamer)
**Goal:** Profit from high-confidence runners attracting market support.
*   **Model:** `xgb_v44_steamer.pkl` (Production)
*   **Trigger Condition:**
    *   **Probability >= 0.35** (35%)
    *   **Price < $15.00**
    *   **Track != Tasmania** (Launceston, Hobart, Devonport)
*   **Execution:** 
    *   Place at Current Market Price.
    *   Updates continuously until jump.

### B. LAY Strategy (V45 Drifter)
**Goal:** Profit from weak favorites that the market is abandoning.
*   **Model:** `xgb_v45_production.pkl` (Production)
*   **Trigger Condition:**
    *   **Drift Probability >= 0.60** (60%)
    *   **Price < $15.00**
    *   **Track != Tasmania**
    *   **Safety Exclusion:** Do **not** place LAY bets when `Steam_Prob` > **0.20**
*   **Execution:**
    *   Place Lay bet at Current Market Price.
    *   **Volume Control:** Max 2 Lays per race (top 2 by confidence).

### C. COVER Strategy (Risk Management)
**Goal:** Offset potential losses in volatile races.
*   **Trigger:** Active ONLY when a race has **2 or more LAY bets**.
*   **Action:** Places a BACK bet on the "Cover Dog".
*   **Selection:** The highest rated steamer in the race with **Prob > 0.42** and **Price < $10**.

---

## 2. System Architecture

### The "Live Engine"
The system runs as a unified application started by `run.py`.

1.  **Data Ingestion (`LivePriceScraper`)**:
    *   Runs in a background thread.
    *   Detects races 60 mins out.
    *   **Self-Healing:** If a race is missing from your local DB, the scraper *automatically creates it* (Track, Race, Dogs).
    *   **Price Capture:** Saves prices at T-60, 30, 15, 10, 5, 2, 1 minutes.

2.  **Prediction Engine (`MarketAlphaEngine`)**:
    *   Located in `scripts/predict_v44_prod.py`.
    *   Loads **365 days** of historical data on startup.
    *   Computes **Rolling Features** (Steam/Drift history for Dogs and Trainers).
    *   **Robustness:** If `Price5Min` is missing (common in new races), it falls back to `LivePrice` or `Back` from the API.

3.  **User Interface (`app.py`)**:
    *   Displays the "Live Alpha Radar" (Top 5 prospects).
    *   Manages "Active Bets" and "Settled Bets".
    *   **Persistent Session:** Logs in once and maintains session via Keep-Alive (every 15 mins) to prevent API errors.
    *   Calculates P/L and ROI in real-time.

4.  **Reporting**:
    *   Bets are reconciled via `scripts/reconcile_live_bets.py`.
    *   Report is **automatically generated** (`live_bets_summary.xlsx`) with Daily/Weekly stats split by Back/Lay.

---

## 5. Daily Operations

### Start-Up
1.  Open Terminal.
2.  Run `python run.py`.
3.  Wait for **"Live price scraper started"** log.

### Routine Checks
*   **"No races found"**: Ensure the `greyhound_racing.db` has been updated or wait for the Scraper to auto-populate upcoming races (T-60m).
*   **Database Updates**: The scraper handles live races. For historical analysis, run `scripts/scraper_v2.py` weekly.

---

## 4. Troubleshooting

| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| **0 Signals on Radar** | Prices missing or Targets missed | Check Scraper logs. Wait for races to get within 5 mins. |
| **Trainer Stats Mismatch** | Old Verification Script | Use the updated `verify_cache_integrity.py` which handles Unknown Trainers correctly. |
| **"Ghost Bets"** | DB Sync Issue | Use **"Clear Active Bets"** button in Debug menu. |

---

## 5. File Maintenance
*   **Logs**: Cleared automatically or archived.
*   **Models**: Located in `models/`. Do not delete `xgb_v44*` or `xgb_v45*`.
*   **Archive**: Old files (V33, etc.) are in `archive/` and can be safely ignored.
