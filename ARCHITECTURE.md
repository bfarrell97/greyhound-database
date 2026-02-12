# Greyhound Racing Analysis System - Architecture

**System Version:** V44/V45 Production  
**Last Updated:** 2026-02-12  
**Status:** Live Production

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [ML Models](#ml-models)
6. [Database Schema](#database-schema)
7. [API Integrations](#api-integrations)
8. [Threading Model](#threading-model)
9. [Feature Engineering](#feature-engineering)
10. [Betting Strategies](#betting-strategies)

---

## System Overview

### Purpose

Algorithmic trading system for greyhound racing on Betfair Exchange, utilizing XGBoost machine learning models to identify value bets in real-time.

### Key Features

- **Real-time price scraping** (T-60min to T-1min)
- **ML-based predictions** (V44 Steamer BACK, V45 Drifter LAY)
- **Automated bet placement** via Betfair API
- **Live P/L tracking** and reporting
- **Discord notifications** for trade signals
- **Persistent session management** (auto-reconnect)
- **Paper trading mode** for strategy testing

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11+ |
| **GUI** | Tkinter + tksheet |
| **ML Framework** | XGBoost 2.x |
| **Database** | SQLite 3 |
| **APIs** | Betfair REST API, Topaz API |
| **Data Processing** | pandas, numpy |
| **Notifications** | Discord webhooks |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         GUI APPLICATION                          │
│                     (src/gui/app.py - 5,748 lines)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Live Alpha   │  │ Active Bets  │  │ Settled Bets │        │
│  │ Radar        │  │ Panel        │  │ Panel        │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CORE SERVICES                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌────────────────────┐                  │
│  │ Market Alpha     │  │ Live Betting       │                  │
│  │ Engine (V44/V45) │  │ Manager            │                  │
│  │ predict_v44_prod │  │ live_betting.py    │                  │
│  └────────┬─────────┘  └─────────┬──────────┘                  │
│           │                       │                             │
│           │  ┌────────────────────┴───────────┐                │
│           │  │                                  │               │
│           ▼  ▼                                  ▼               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐│
│  │ Feature          │  │ Bet Scheduler    │  │ Result       ││
│  │ Engineering      │  │ bet_scheduler.py │  │ Tracker      ││
│  │ (Rolling Stats)  │  └──────────────────┘  └──────────────┘│
│  └──────────────────┘                                          │
└─────────────────────────────────────────────────────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│  │ SQLite         │  │ Live Price     │  │ Models           │ │
│  │ Database       │  │ Scraper        │  │ (XGBoost .pkl)   │ │
│  │ (365 days)     │  │ Background     │  │ V44/V45 Prod     │ │
│  └────────┬───────┘  └────────┬───────┘  └─────────┬────────┘ │
└───────────┼──────────────────────┼───────────────────┼──────────┘
            │                      │                   │
            ▼                      ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL APIs                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│  │ Betfair REST   │  │ Topaz Form API │  │ Discord Webhook  │ │
│  │ (Odds + Orders)│  │ (Historical)   │  │ (Notifications)  │ │
│  └────────────────┘  └────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. GUI Application (`src/gui/app.py`)

**Purpose:** Main user interface for viewing races, signals, and managing bets.

**Key Features:**
- **Live Alpha Radar:** Top 5 prospects ranked by confidence
- **Active Bets Panel:** Real-time bet tracking with P/L
- **Settled Bets Panel:** Historical bet results
- **Manual Override:** Place/cancel bets manually
- **Session Management:** Persistent Betfair login with keep-alive

**Threading:**
- Main thread: UI rendering and user input
- Background threads: Price scraping, prediction updates, bet monitoring

**Lines of Code:** 5,748 (largest file)

### 2. Market Alpha Engine (`scripts/predict_v44_prod.py`)

**Purpose:** Core prediction engine using V44 (BACK Steamer) and V45 (LAY Drifter) models.

**Workflow:**
1. Load 365 days of historical data on startup
2. Calculate rolling features (steam/drift statistics)
3. Generate predictions for upcoming races (T-60min to jump)
4. Emit signals when confidence thresholds met
5. Handle robustness (missing prices, new races)

**Models Used:**
- `models/xgb_v44_steamer.pkl` - BACK strategy
- `models/xgb_v45_production.pkl` - LAY strategy

**Key Methods:**
- `load_historical_data()` - Loads 365-day lookback
- `calculate_rolling_features()` - Compute steam/drift stats
- `predict_race()` - Generate predictions for single race
- `get_top_prospects()` - Return ranked betting opportunities

### 3. Live Price Scraper (`scripts/live_price_scraper.py`)

**Purpose:** Background service continuously monitoring Betfair markets.

**Features:**
- **Self-healing:** Auto-creates missing races in database
- **Price capture:** Saves prices at T-60, 30, 15, 10, 5, 2, 1 minutes
- **Market detection:** Finds greyhound races 60 minutes before jump
- **Robust:** Handles API failures, retries, and rate limiting

**Threading:** Runs in dedicated background thread

**Data Saved:**
- Live prices (BackPrice, LayPrice)
- Matched volume (TotalMatched)
- Best available prices (BestBack, BestLay)
- Starting price (BSP) after race

### 4. Live Betting Manager (`src/core/live_betting.py`)

**Purpose:** Manages bet placement, tracking, and reconciliation.

**Responsibilities:**
- Place BACK and LAY bets via Betfair API
- Track unmatched bets (pending orders)
- Monitor matched bets (filled orders)
- Calculate P/L in real-time
- Handle bet cancellation
- Reconcile with Betfair current orders API

**Bet States:**
- `PLACED` - Order submitted to Betfair
- `UNMATCHED` - Order on book, not filled
- `MATCHED` - Order filled (SETTLED)
- `TIMEOUT` - Order expired at T-2min
- `CANCELLED` - Manually cancelled

### 5. Feature Engineering (`src/features/feature_engineering.py`)

**Purpose:** Compute predictive features from raw race data.

**Feature Categories:**

**A. Price Features:**
- `Price60Min`, `Price30Min`, `Price15Min`, `Price10Min`, `Price5Min`, `Price2Min`
- `PriceChange_60_30`, `PriceChange_30_15`, `PriceChange_15_5`
- `LivePrice` (current price)

**B. Volume Features:**
- `TotalMatched` (market liquidity)
- `BackVolume`, `LayVolume` (depth of market)
- `VolumeRatio` (back/lay balance)

**C. Rolling Statistics (365-day lookback):**
- `SteamRate_Last10` (% of recent races where dog steamed)
- `DriftRate_Last10` (% of recent races where dog drifted)
- `TrainerSteamRate` (trainer's historical steam tendency)
- `AvgPriceMove` (typical price movement for dog)

**D. Form Features:**
- `Last3Starts` (recent placings)
- `WinRate` (career win %)
- `BoxWinRate` (box position win %)
- `TrackWinRate` (track-specific win %)

**E. Race Context:**
- `FavoriteStatus` (is favorite, 2nd fav, etc.)
- `FieldSize` (number of runners)
- `TrackType` (straight, turning)

**Feature Engineering Versions:**
- `feature_engineering.py` - Current (V44/V45)
- `feature_engineering_v38-v41.py` - Legacy versions (may be unused)

### 6. Database (`src/core/database.py`)

**Purpose:** SQLite database storing all race and dog data.

**Key Tables:**

| Table | Records | Purpose |
|-------|---------|---------|
| **Greyhounds** | ~50,000+ | Dog profiles, career stats |
| **Trainers** | ~5,000+ | Trainer profiles |
| **Tracks** | ~50 | Australian/NZ tracks |
| **RaceMeetings** | ~365 days | Daily race meetings |
| **Races** | ~100,000+ | Individual race records |
| **RaceResults** | ~800,000+ | Dog results with prices |
| **LivePrices** | Millions | Price snapshots (T-60 to jump) |
| **LiveBets** | ~1,500+ | Bet history |

**Performance:**
- Historical data: 365 days loaded on startup
- Price lookback: T-60min to jump (8-10 snapshots per race)
- Total DB size: ~500MB - 2GB (depends on history)

### 7. Bet Scheduler (`src/integration/bet_scheduler.py`)

**Purpose:** Schedule bets to be placed at optimal times (e.g., T-5min).

**Features:**
- Queue bets for future placement
- Monitor race start times
- Execute bets at scheduled time
- Handle race delays/abandonments
- Cancel queued bets if conditions change

**Use Case:** User queues bet at T-15min, but wants execution at T-5min when prices are more stable.

### 8. API Integrations

#### Betfair API (`src/integration/betfair_api.py`)

**Endpoints Used:**
- `listMarketCatalogue` - Find upcoming races
- `listMarketBook` - Get live prices and volumes
- `placeOrders` - Submit BACK/LAY bets
- `listCurrentOrders` - Check active bets
- `cancelOrders` - Cancel pending bets

**Authentication:**
- Certificate-based (non-interactive login)
- Session token stored, refreshed every 8 hours
- Keep-alive pings every 15 minutes

#### Topaz API (`src/integration/topaz_api.py`)

**Purpose:** Historical form data provider (race results, dog stats).

**Data Provided:**
- Past race results
- Sectional times
- Track conditions
- Historical odds

**Usage:** Import historical data for model training and feature engineering.

#### Discord Webhook (`src/utils/discord_notifier.py`)

**Purpose:** Send notifications to Discord channel.

**Notifications:**
- High-confidence signals
- Bet placement confirmations
- Win/loss results
- System errors/warnings

---

## Data Flow

### 1. Startup Sequence

```
1. GUI launches (run.py)
   ↓
2. Load ML models (V44/V45 XGBoost)
   ↓
3. Connect to database (greyhound_racing.db)
   ↓
4. Login to Betfair (certificate auth)
   ↓
5. Start background threads:
   - Live price scraper
   - Market alpha monitor
   - Result tracker
   ↓
6. Load 365-day historical data
   ↓
7. Calculate rolling features
   ↓
8. Display GUI (ready for trading)
```

### 2. Price Scraping Loop (Every 60 seconds)

```
1. Query Betfair for races starting in next 60 minutes
   ↓
2. For each race:
   - Fetch current prices (listMarketBook)
   - Calculate time to jump
   - Determine price snapshot time (T-60, T-30, T-15, etc.)
   ↓
3. Save prices to database (LivePrices table)
   ↓
4. Auto-create missing races if needed (self-healing)
   ↓
5. Sleep 60 seconds, repeat
```

### 3. Prediction Generation (Every 30 seconds)

```
1. Market Alpha Engine wakes up
   ↓
2. Fetch races within betting window (T-30min to jump)
   ↓
3. For each race:
   - Load price history (T-60, T-30, T-15, T-10, T-5)
   - Calculate features (price moves, volume, rolling stats)
   - Run XGBoost prediction (V44 for BACK, V45 for LAY)
   - Check thresholds (Prob >= 0.35 for BACK, >= 0.60 for LAY)
   ↓
4. Emit signals to GUI (Live Alpha Radar)
   ↓
5. Auto-place bets if enabled
   ↓
6. Send Discord notifications for high-confidence signals
   ↓
7. Sleep 30 seconds, repeat
```

### 4. Bet Placement Flow

```
User clicks "Place Bet" (or auto-execution triggers)
   ↓
1. Validate bet parameters (stake, price, selection)
   ↓
2. Calculate liability (for LAY bets)
   ↓
3. Check bankroll/limits
   ↓
4. Submit order to Betfair (placeOrders API)
   ↓
5. Receive BetID from Betfair
   ↓
6. Save bet to database (LiveBets table)
   ↓
7. Add to Active Bets panel in GUI
   ↓
8. Monitor bet status:
   - UNMATCHED → Keep monitoring
   - MATCHED → Move to Settled Bets
   - TIMEOUT (T-2min) → Cancel if unmatched
   ↓
9. Calculate P/L after race result
   ↓
10. Update reports (live_bets_summary.csv)
```

### 5. Result Processing

```
Race finishes
   ↓
1. Fetch result from Betfair (marketBook with status CLOSED)
   ↓
2. Identify winning dog
   ↓
3. For each bet in this race:
   - BACK bet: Win if dog won, else loss
   - LAY bet: Win if dog lost, else loss
   - Calculate P/L: (Stake * (Odds - 1)) for win, -Stake for loss
   ↓
4. Update LiveBets table with Result and Profit
   ↓
5. Move from Active Bets to Settled Bets in GUI
   ↓
6. Trigger P/L report generation (generate_pl_report.py)
   ↓
7. Send Discord notification (win/loss summary)
```

---

## ML Models

### V44: Steamer BACK Strategy

**Model File:** `models/xgb_v44_steamer.pkl`

**Purpose:** Identify dogs experiencing price shortening (steaming) with high win probability.

**Target Variable:** Binary (1 = dog will win race, 0 = dog will not win)

**Key Features (Top 10 by importance):**
1. `PriceChange_15_5` (recent price drop)
2. `SteamRate_Last10` (historical steam tendency)
3. `LivePrice` (current odds)
4. `TotalMatched` (market liquidity)
5. `TrainerSteamRate` (trainer tendency)
6. `WinRate` (career win %)
7. `BoxWinRate` (box position advantage)
8. `AvgPriceMove` (typical volatility)
9. `FavoriteStatus` (market rank)
10. `Last3Starts` (recent form)

**Prediction Output:** Probability (0-1) that dog will win

**Threshold:** Prob >= 0.35 (35%) + Price < $15.00 → Place BACK bet

**Training:**
- Dataset: Last 12 months of race results (~30,000 races)
- Positive class: Dogs that won (steamers only)
- Negative class: Dogs that didn't win
- Class balance: SMOTE oversampling of winners (minority class)
- XGBoost params: `max_depth=5`, `learning_rate=0.05`, `n_estimators=200`

**Performance (Backtest):**
- Strike rate: ~35-40% (profitable at average odds of 5.0-8.0)
- ROI: Target +3% to +8%
- Sample size: 2,000+ bets over 6 months

### V45: Drifter LAY Strategy

**Model File:** `models/xgb_v45_production.pkl`

**Purpose:** Identify weak favorites that are drifting in price (losing market support).

**Target Variable:** Binary (1 = dog will drift significantly, 0 = dog will not drift)

**Key Features (Top 10 by importance):**
1. `PriceChange_30_15` (early drift signal)
2. `DriftRate_Last10` (historical drift tendency)
3. `LivePrice` (current odds)
4. `BackVolume` (lack of support)
5. `TrainerDriftRate` (trainer weakness)
6. `PlaceRate` (inconsistency)
7. `FieldSize` (competitive race)
8. `FavoriteStatus` (market favorite = target)
9. `TrackWinRate` (track-specific weakness)
10. `PriceChange_15_5` (continued drift)

**Prediction Output:** Probability (0-1) that dog will continue drifting and likely lose

**Threshold:** Drift Prob >= 0.60 (60%) + Price < $15.00 + Steam Prob < 0.20 → Place LAY bet

**Safety Check:** Do NOT lay if `SteamProb > 0.20` (dog may recover)

**Training:**
- Dataset: Last 12 months of favorites/2nd favorites (~20,000 dogs)
- Positive class: Dogs that drifted and lost
- Negative class: Dogs that held or won
- Class balance: Natural (drifters are ~40% of favorites)
- XGBoost params: `max_depth=4`, `learning_rate=0.03`, `n_estimators=250`

**Performance (Backtest):**
- Strike rate: ~60-65% (LAY bets win when dog loses)
- ROI: Target +2% to +5%
- Sample size: 3,000+ bets over 6 months

### COVER Strategy (Risk Management)

**Trigger:** Active when race has 2+ LAY bets placed

**Purpose:** Hedge losses in volatile races by backing strongest steamer

**Selection:** Highest `SteamProb` dog with Prob > 0.42 and Price < $10

**Stake:** Proportional to LAY liability (cover ~50-70% of potential loss)

**Status:** Experimental (adds complexity, marginal benefit observed)

---

## Database Schema

### Key Tables

#### LivePrices (Price Snapshots)
```sql
CREATE TABLE LivePrices (
    PriceID INTEGER PRIMARY KEY,
    RaceID INTEGER,
    SelectionID INTEGER,
    Timestamp TEXT,
    TimeToJump INTEGER,        -- Minutes until race
    BackPrice REAL,
    LayPrice REAL,
    TotalMatched REAL,
    BestBack REAL,
    BestLay REAL,
    FOREIGN KEY (RaceID) REFERENCES Races(RaceID)
);
```

#### LiveBets (Bet Tracking)
```sql
CREATE TABLE LiveBets (
    BetID TEXT PRIMARY KEY,    -- Betfair BetID
    MarketID TEXT,
    SelectionID INTEGER,
    Date TEXT,
    Time TEXT,
    Track TEXT,
    Race INTEGER,
    Dog TEXT,
    BetType TEXT,             -- 'BACK' or 'LAY'
    Status TEXT,              -- 'PLACED', 'MATCHED', 'UNMATCHED', 'TIMEOUT'
    Stake REAL,
    Price REAL,
    BSP REAL,                 -- Betfair Starting Price
    Result TEXT,              -- 'WIN' or 'LOSS'
    Profit REAL
);
```

#### Greyhounds (Dog Profiles)
```sql
CREATE TABLE Greyhounds (
    GreyhoundID INTEGER PRIMARY KEY,
    GreyhoundName TEXT UNIQUE,
    Starts INTEGER,
    Wins INTEGER,
    WinPercentage REAL,
    BestTime REAL,
    TrainerID INTEGER,
    FOREIGN KEY (TrainerID) REFERENCES Trainers(TrainerID)
);
```

---

## API Integrations

### Betfair Exchange API

**Base URL:** `https://api.betfair.com/exchange/betting/rest/v1.0/`

**Authentication:** Certificate-based (X-Application, X-Authentication headers)

**Rate Limits:**
- 5 requests/second per account
- 200 requests/minute for heavy operations
- Persistent session (8-hour token expiry)

**Key Endpoints:**

| Endpoint | Purpose | Frequency |
|----------|---------|-----------|
| `listMarketCatalogue` | Find upcoming races | Every 5 minutes |
| `listMarketBook` | Get live prices | Every 60 seconds |
| `placeOrders` | Submit bets | On demand |
| `listCurrentOrders` | Check active bets | Every 2 minutes |
| `cancelOrders` | Cancel bets | On demand |

**Error Handling:**
- Retry on timeout (3 attempts)
- Exponential backoff on rate limit
- Session refresh on `INVALID_SESSION_INFORMATION`
- Fallback to cached prices on API failure

### Topaz Form Data API

**Base URL:** `https://api.topaz.horse/`  
**API Key:** Stored in `src/core/config.py`

**Endpoints:**
- `GET /races` - List upcoming races
- `GET /race/{race_id}/results` - Historical results
- `GET /dog/{dog_id}/form` - Dog form guide

**Usage:**
- Bulk historical import (weekly)
- On-demand form lookups
- Backup data source when Betfair slow

---

## Threading Model

### Main Thread (GUI)
- Tkinter event loop
- User interactions
- Display updates
- Non-blocking operations only

### Background Threads

| Thread | Purpose | Interval | Priority |
|--------|---------|----------|----------|
| **Price Scraper** | Fetch live prices | 60s | High |
| **Market Alpha Monitor** | Generate predictions | 30s | High |
| **Result Tracker** | Process results | 60s | Medium |
| **Keep-Alive** | Maintain Betfair session | 900s (15min) | Low |
| **Bet Reconciliation** | Sync with Betfair orders | 120s | Medium |

**Thread Safety:**
- Database: SQLite handles locking automatically
- Shared state: Protected by `threading.Lock()`
- GUI updates: Use `root.after()` for thread-safe callbacks

---

## Feature Engineering

### Rolling Statistics (365-Day Lookback)

**Purpose:** Capture long-term tendencies of dogs and trainers.

**Calculation:**

```python
# For each dog, calculate over last 365 days:
SteamRate_Last10 = (Races where Price60 > Price5) / Total Races
DriftRate_Last10 = (Races where Price60 < Price5) / Total Races
AvgPriceMove = Mean(|Price60 - Price5|) across all races
WinRate = Wins / Starts
```

**Trainer Stats:**
```python
# For each trainer, aggregate across all dogs:
TrainerSteamRate = Sum(Dog SteamRates) / Num Dogs
TrainerWinRate = Total Wins / Total Starts
```

**Performance:**
- Initial load: ~30-60 seconds (365 days of data)
- Incremental update: <1 second per race
- Cached in memory, refreshed daily

### Price Movement Features

**Capture momentum and volatility:**

```python
PriceChange_60_30 = (Price60Min - Price30Min) / Price60Min
PriceChange_30_15 = (Price30Min - Price15Min) / Price30Min
PriceChange_15_5 = (Price15Min - Price5Min) / Price15Min
PriceVolatility = StdDev([Price60, Price30, Price15, Price10, Price5])
```

**Interpretation:**
- Negative change = Steaming (price shortening, dog popular)
- Positive change = Drifting (price lengthening, dog unpopular)

---

## Betting Strategies

### BACK Strategy (V44 Steamer)

**Entry Criteria:**
1. Steamer probability >= 35%
2. Current price < $15.00
3. Track != Tasmania (excluded)
4. Time window: T-30min to T-5min

**Stake:** Fixed (e.g., $10) or Kelly Criterion

**Exit:** Race jump (bet settles automatically)

**Expected Outcome:**
- Win: Stake × (Odds - 1)
- Loss: -Stake

### LAY Strategy (V45 Drifter)

**Entry Criteria:**
1. Drift probability >= 60%
2. Current price < $15.00
3. Steam probability < 20% (safety check)
4. Track != Tasmania
5. Max 2 LAYs per race (volume control)
6. Time window: T-30min to T-5min

**Stake:** Fixed or Kelly Criterion

**Exit:** Race jump

**Expected Outcome:**
- Win: +Stake (dog loses)
- Loss: -Stake × (Odds - 1) (dog wins = liability)

### COVER Strategy (Hedge)

**Trigger:** Race has 2+ active LAY bets

**Entry Criteria:**
1. Highest SteamProb dog in race
2. SteamProb >= 42%
3. Price < $10.00

**Stake:** Proportional to LAY liability (aim to break even if cover wins)

**Purpose:** Reduce downside in high-exposure races

---

## File Structure Summary

```
greyhound-database/
├── run.py                      # Entry point
├── README.md                   # User documentation
├── SYSTEM_MANUAL.md           # Operations manual
├── ARCHITECTURE.md            # This file
├── requirements.txt           # Python dependencies
│
├── src/                       # Production code (28 files)
│   ├── core/                  # Core services (6 files)
│   │   ├── config.py          # API keys, settings
│   │   ├── database.py        # SQLite wrapper
│   │   ├── live_betting.py    # Bet management
│   │   ├── paper_trading.py   # Simulation mode
│   │   └── predictor.py       # Prediction wrapper
│   │
│   ├── features/              # Feature engineering (5 files)
│   │   ├── feature_engineering.py      # Current (V44/V45)
│   │   └── feature_engineering_v38-v41.py  # Legacy
│   │
│   ├── gui/                   # User interface (3 files)
│   │   └── app.py             # Main GUI (5,748 lines)
│   │
│   ├── integration/           # External APIs (5 files)
│   │   ├── bet_scheduler.py   # Bet timing
│   │   ├── betfair_api.py     # Betfair REST
│   │   ├── betfair_fetcher.py # Odds fetching
│   │   └── topaz_api.py       # Form data
│   │
│   ├── models/                # ML models (7 files)
│   │   ├── benchmark_cmp.py   # Benchmark comparison
│   │   ├── ml_model.py        # Model wrapper
│   │   ├── pace_strategy.py   # Pace analysis
│   │   └── pir_evaluator.py   # PIR evaluation
│   │
│   └── utils/                 # Utilities (2 files)
│       ├── discord_notifier.py
│       └── result_tracker.py
│
├── scripts/                   # Production scripts (~17 files)
│   ├── predict_v44_prod.py    # Market Alpha Engine
│   ├── live_price_scraper.py  # Price capture
│   ├── reconcile_live_bets.py # Bet sync
│   ├── generate_pl_report.py  # Reporting
│   ├── train_v44_*.py         # V44 training (3 files)
│   ├── train_v45_*.py         # V45 training (2 files)
│   └── import_*.py            # Data import (4 files)
│
├── models/                    # XGBoost models
│   ├── xgb_v44_steamer.pkl    # BACK strategy
│   └── xgb_v45_production.pkl # LAY strategy
│
├── archive/                   # Old versions (482 files)
│   └── temp_scripts_2026/     # Archived temp scripts
│
├── outputs/                   # Reports and logs
│   ├── logs/
│   └── reports/
│
└── tests/                     # Test suite (4 files)
```

---

## System Requirements

### Hardware
- CPU: 4+ cores recommended (threading)
- RAM: 4GB minimum, 8GB recommended
- Disk: 5GB free space (database + models)
- Network: Stable internet (API calls every 30-60s)

### Software
- Python 3.11+
- SQLite 3.35+
- Operating System: Windows 10/11, macOS, Linux

### Dependencies (requirements.txt)
```
pandas>=2.0.0
numpy>=1.24.0
xgboost>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
tksheet>=6.0.0
requests>=2.31.0
betfairlightweight>=2.18.0 (optional)
```

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Startup Time** | 30-60s | Loading 365 days of data |
| **Prediction Latency** | <1s per race | Feature calc + XGBoost |
| **Database Queries** | <100ms | Indexed lookups |
| **API Response Time** | 200-500ms | Betfair REST calls |
| **Memory Usage** | 500MB-1GB | Historical data in memory |
| **CPU Usage** | 5-15% idle, 40% during predictions | Multi-threaded |

---

## Future Improvements

### Planned Features
1. **Live streaming integration** (Betfair Stream API)
2. **Advanced position sizing** (Kelly Criterion)
3. **Multi-model ensemble** (V44 + V45 + V46)
4. **Track bias analysis** (inside/outside box advantage)
5. **Pace scenario modeling** (early speed vs. run-on)

### Technical Debt
1. **Large GUI file** (5,748 lines - should be split)
2. **Legacy feature files** (v38-v41 - remove if unused)
3. **Minimal test coverage** (4 test files - expand to 40%+)
4. **Missing docstrings** (add to all 28 production files)
5. **No type hints** (add throughout codebase)

---

## Monitoring and Observability

### Logs
- Location: `outputs/logs/`
- Format: `YYYY-MM-DD_greyhound.log`
- Levels: DEBUG, INFO, WARNING, ERROR
- Rotation: Daily

### Metrics
- Total bets placed
- Strike rate (win %)
- ROI (return on investment)
- Average odds
- P/L by strategy (BACK vs. LAY)
- Matched rate (% of bets that fill)

### Alerts (Discord)
- High-confidence signals (Prob > 50%)
- Large wins/losses (>$50)
- API failures
- Database errors
- Session expiry warnings

---

**End of Architecture Document**

For detailed operational procedures, see `SYSTEM_MANUAL.md`.  
For development setup, see `DEVELOPMENT.md` (to be created).
