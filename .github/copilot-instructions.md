# Greyhound Racing Database - AI Coding Agent Instructions

## Project Overview

This is a **greyhound racing prediction and betting system** (V44/V45 Production).
*   **Goal**: Live automated betting on Betfair.
*   **Stack**: Python, SQLite, XGBoost, Tkinter (GUI), Betfair API.

---

## 🏗 System Architecture

### 1. Data Flow
```
Betfair API (LivePriceScraper) -> SQLite DB (greyhound_racing.db) -> MarketAlphaEngine -> GUI (app.py)
```

### 2. Core Components
| Component | File | Purpose |
| :--- | :--- | :--- |
| **GUI** | `src/gui/app.py` | Main user interface. Displays "Live Alpha Radar". |
| **Engine** | `scripts/predict_v44_prod.py` | Loads models, computes rolling features, generates signals. |
| **Scraper** | `scripts/live_price_scraper.py` | Background service. Captures prices (T-60m to T-1m) & auto-creates missing races. |
| **Runner** | `run.py` | Single entry point. Starts GUI + Scraper. |

---

## 🤖 Production Models

| Model | Type | File | Strategy |
| :--- | :--- | :--- | :--- |
| **V44** | XGBoost | `models/xgb_v44_steamer.pkl` | **BACK** (Value Steamers) |
| **V45** | XGBoost | `models/xgb_v45_production.pkl` | **LAY** (Weak Favorites/Drifters) |

---

## 🎯 Betting Strategies

### A. BACK Strategy (V44)
*   **Target**: High-confidence steamers.
*   **Logic**:
    *   `Steam_Prob` >= **0.35**
    *   `Price` < **$15.00**
    *   **Track** != Tasmania
*   **Execution**: Place at LTP.

### B. LAY Strategy (V45)
*   **Target**: Drifting favorites.
*   **Logic**:
    *   `Drift_Prob` >= **0.60**
    *   `Price` < **$15.00**
    *   **Track** != Tasmania
    *   **Safety Exclusion:** Do NOT place LAY when `Steam_Prob` > **0.20**
*   **Execution**: Place Lay at LTP. Max 2 per race.

### C. COVER Strategy
*   **Logic**: If 2+ Lays in a race -> Back the strong steamer (`Prob > 0.42`, `Price < $10`).

---

## 🛠 Key Developer Tasks

### 1. Feature Engineering (`MarketAlphaEngine`)
Features are computed **on-the-fly** inside `predict_v44_prod.py` using cached history.
*   **Rolling Steam/Drift**: Calculated from `Is_Steamer_Hist` / `Is_Drifter_Hist` over last 10 (Dog) or 50 (Trainer) races.
*   **History Loading**: Loads 365 days of data on startup to populate the cache.

### 2. Live Price Injection
*   The `LivePriceScraper` continuously updates `Price5Min`, `Price60Min` etc. in the DB.
*   **Fallback**: If `Price5Min` is missing, the Engine falls back to `LivePrice` or `Back` from the API.

### 3. Database (`greyhound_racing.db`)
*   **GreyhoundEntries**: Main data table.
*   **LiveBets**: Stores active/settled betas.

---

## ⚠️ Important Rules for AI Agents

1.  **Do NOT create new "vXX" scripts** without explicit instruction. Use existing V44/V45 structure.
2.  **Persistent Session**: ALWAYS use `self.fetcher` or `BetfairOddsFetcher()` with persistence. NEVER call `logout()` inside a loop.
3.  **App Structure**: `app.py` is the controller. It polls `alpha_engine.predict()` every 30s.
4.  **Reporting**: All reconciliation logic must trigger `generate_pl_report.py` to ensure Excel sync.
5.  **Cleanliness**: Keep the root directory clean. Move old scripts to `archive/`.
6.  **Validation**: Use `scripts/verify_cache_integrity.py` to check that new code changes don't break predictions.
