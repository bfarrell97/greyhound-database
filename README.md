# Greyhound Racing Analysis System

**Status:** 🟢 Live Production  
**Version:** V44/V45  
**Last Updated:** 2026-02-12

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🚀 Quick Start in 60 Seconds

```bash
# 1. Install requirements (first time only)
pip install -r requirements.txt

# 2. Configure API credentials
# Edit src/core/config.py with your Betfair API key

# 3. Place Betfair SSL certificates
mkdir -p certs
# Copy client-2048.crt and client-2048.key to certs/

# 4. Run the application
python run.py
```

The GUI will launch and automatically start:
- 🔍 Scanning upcoming greyhound races
- 📊 Generating ML predictions (V44/V45)
- 💾 Persisting price data
- 📈 Displaying live betting opportunities

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Strategies](#strategies)
- [Documentation](#documentation)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Advanced algorithmic trading bot for greyhound racing** on Betfair Exchange, powered by XGBoost machine learning models to identify value bets in real-time.

### What It Does

1. **Monitors** Betfair markets for upcoming greyhound races (T-60 minutes to jump)
2. **Analyzes** price movements and historical patterns using 365 days of data
3. **Predicts** profitable betting opportunities using V44 (BACK) and V45 (LAY) ML models
4. **Executes** automated bet placement with risk management
5. **Tracks** live P/L and generates performance reports

### Who It's For

- **Algorithmic traders** looking to automate greyhound racing bets
- **Data scientists** interested in sports betting ML applications
- **Python developers** wanting to learn betting automation
- **Betfair users** seeking an edge in greyhound markets

---

## Features

### 🤖 ML-Powered Predictions

- **V44 Steamer Model** - Identifies dogs experiencing price shortening (35%+ win probability)
- **V45 Drifter Model** - Spots weak favorites drifting in price (60%+ drift probability)
- **365-day historical lookback** for robust feature engineering
- **XGBoost ensemble models** with >1000 training samples per model

### 📊 Live Price Scraping

- **Continuous monitoring** of Betfair markets (60-second intervals)
- **Price snapshots** at T-60, 30, 15, 10, 5, 2, 1 minutes before jump
- **Self-healing** - Automatically creates missing races in database
- **Robust** - Handles API failures, retries, and rate limiting

### 💰 Automated Trading

- **Real-time bet placement** via Betfair REST API
- **Position tracking** - Monitor unmatched and matched bets
- **Live P/L calculation** - See profits/losses update in real-time
- **Bet scheduling** - Queue bets for execution at optimal times (e.g., T-5min)
- **Risk management** - Stake limits, price caps, volume controls

### 📈 Reporting & Notifications

- **Discord notifications** for high-confidence signals and results
- **CSV reports** - `live_bets.csv` with full bet history
- **Excel summaries** - `live_bets_summary.xlsx` with daily/weekly P/L
- **Live GUI dashboard** - Active bets, settled bets, and top prospects

### 🛡️ Production-Ready

- **Persistent Betfair session** with automatic keep-alive (15-minute pings)
- **Multi-threaded** - Background services don't block GUI
- **Database-backed** - SQLite storage for all historical data
- **Logging** - Comprehensive logs for debugging and auditing
- **Error handling** - Graceful recovery from API failures

---

## System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     GUI Application                        │
│              (Live Alpha Radar + Bet Panels)              │
└─────────────────┬─────────────────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────────────────┐
│                   Core Services                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ Market Alpha│  │ Live Betting │  │ Price Scraper   │ │
│  │ Engine      │  │ Manager      │  │ (Background)    │ │
│  │ (V44/V45)   │  │              │  │                 │ │
│  └─────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────┬─────────────────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────────────────────┐
│              Data Layer & APIs                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│  │ SQLite   │  │ Betfair  │  │ Topaz    │  │ Discord  ││
│  │ Database │  │ REST API │  │ Form API │  │ Webhook  ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘│
└───────────────────────────────────────────────────────────┘
```

**For detailed architecture, see [ARCHITECTURE.md](ARCHITECTURE.md)**

---

## Installation

### Prerequisites

- **Python 3.11+** (required)
- **Git** (for cloning repository)
- **Betfair account** with API access
- **Betfair SSL certificates** (for non-interactive login)
- **Topaz API key** (optional, for historical data)

### Step-by-Step Setup

**1. Clone Repository**
```bash
git clone https://github.com/bfarrell97/greyhound-database.git
cd greyhound-database
```

**2. Create Virtual Environment**
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure API Credentials**

Edit `src/core/config.py`:
```python
# Betfair API Credentials
BETFAIR_APP_KEY = "your_app_key_here"
BETFAIR_USERNAME = "your_username@email.com"
BETFAIR_PASSWORD = "your_password"

# Topaz API Key (optional)
TOPAZ_API_KEY = "your_topaz_key_here"

# Discord Webhook (optional)
DISCORD_WEBHOOK_URL = "your_webhook_url_here"
```

**5. Place Betfair Certificates**
```bash
mkdir -p certs
# Copy your Betfair SSL certificates:
#   - client-2048.crt → certs/client-2048.crt
#   - client-2048.key → certs/client-2048.key
```

**6. Initialize Database**
```bash
python -c "from src.core.database import GreyhoundDatabase; GreyhoundDatabase()"
```

**7. Import Historical Data (Recommended)**
```bash
# Import last 90 days for model training
python scripts/import_topaz_history.py --days 90
```

**8. Verify Setup**
```bash
# Test Betfair connection
python -c "from src.integration.betfair_api import BetfairAPI; print('Connection OK')"
```

**9. Run Application**
```bash
python run.py
```

---

## Configuration

### API Keys Setup

**Betfair API:**
1. Register at https://www.betfair.com/developer
2. Create an application to get your App Key
3. Generate SSL certificates for non-interactive login
4. Download certificates and place in `certs/` directory

**Topaz API (Optional):**
1. Sign up at https://api.topaz.horse
2. Get API key from dashboard
3. Add to `src/core/config.py`

**Discord Notifications (Optional):**
1. Create Discord webhook in your server settings
2. Copy webhook URL
3. Add to `src/core/config.py`

### Strategy Configuration

**Edit thresholds in `scripts/predict_v44_prod.py`:**

```python
# V44 BACK Strategy
BACK_THRESHOLD = 0.35  # 35% win probability
BACK_MAX_PRICE = 15.0  # Maximum odds

# V45 LAY Strategy
LAY_THRESHOLD = 0.60   # 60% drift probability
LAY_MAX_PRICE = 15.0   # Maximum odds
LAY_STEAM_SAFETY = 0.20  # Don't lay if steam prob > 20%
MAX_LAYS_PER_RACE = 2  # Volume control
```

### Stake Configuration

**Fixed stake mode** (in `src/core/live_betting.py`):
```python
DEFAULT_STAKE = 10.0  # $10 per bet
```

**Kelly Criterion** (advanced - not yet implemented):
```python
# Stake = (Edge × Odds) / (Odds - 1) × Fraction of bankroll
```

---

## Usage

### Starting the Application

**GUI Mode (default):**
```bash
python run.py
```

**Paper Trading Mode (simulation):**
```bash
python run.py --paper-trade
```

**Background Mode (headless):**
```bash
python run.py --no-gui
```

### GUI Overview

**Live Alpha Radar (Top Panel):**
- Shows top 5 betting prospects ranked by confidence
- Updates every 30 seconds
- Click to view details or place bet

**Active Bets (Left Panel):**
- Displays bets currently on book (unmatched)
- Shows bets awaiting settlement (matched, race not finished)
- Real-time P/L tracking

**Settled Bets (Right Panel):**
- Historical bet results
- Win/loss breakdown
- Profit/loss per bet

**Manual Override (Bottom):**
- Place custom bets
- Cancel active bets
- Adjust stake amounts

### Command-Line Scripts

**Generate P/L Report:**
```bash
python scripts/generate_pl_report.py
```

**Reconcile Bets (sync with Betfair):**
```bash
python scripts/reconcile_live_bets.py
```

**Train Models:**
```bash
# Train V44 Steamer model
python scripts/train_v44_production.py

# Train V45 Drifter model
python scripts/train_v45_drifter.py
```

**Import Historical Data:**
```bash
python scripts/import_topaz_history.py --days 365
```

---

## Strategies

### V44: Steamer BACK Strategy

**Goal:** Profit from high-confidence runners attracting market support

**Entry Criteria:**
- ✅ Model win probability ≥ 35%
- ✅ Current price < $15.00
- ✅ Track ≠ Tasmania (excluded)
- ✅ Time window: T-30min to T-5min

**Execution:**
- Place BACK bet at current market price
- Bet settles when race jumps
- Win if dog wins race

**Expected Performance:**
- Strike rate: 35-40%
- Average odds: 5.0-8.0
- Target ROI: +3% to +8%

**Example:**
```
Dog: "Fast Freddy"
Price: $6.00
Model Prob: 38%
Stake: $10
→ WIN: +$50 profit (if dog wins)
→ LOSS: -$10 (if dog loses)
```

### V45: Drifter LAY Strategy

**Goal:** Profit from weak favorites that the market is abandoning

**Entry Criteria:**
- ✅ Drift probability ≥ 60%
- ✅ Current price < $15.00
- ✅ Steam probability < 20% (safety check)
- ✅ Track ≠ Tasmania
- ✅ Max 2 LAYs per race (volume control)
- ✅ Time window: T-30min to T-5min

**Execution:**
- Place LAY bet at current market price
- Bet settles when race jumps
- Win if dog loses race

**Expected Performance:**
- Strike rate: 60-65%
- Average odds: 3.0-7.0
- Target ROI: +2% to +5%

**Example:**
```
Dog: "Slow Sally"
Price: $5.00
Model Drift Prob: 65%
Stake: $10
→ WIN: +$10 profit (if dog loses)
→ LOSS: -$40 liability (if dog wins)
```

### COVER Strategy (Risk Management)

**Trigger:** Activated when race has 2+ LAY bets

**Selection:** Highest steam probability dog with:
- Steam prob ≥ 42%
- Price < $10.00

**Purpose:** Hedge losses in volatile races

**Status:** Experimental (marginal benefit observed)

---

## Documentation

| Document | Description |
|----------|-------------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Complete system design and architecture |
| **[DEVELOPMENT.md](DEVELOPMENT.md)** | Developer setup and contribution guide |
| **[TESTING.md](TESTING.md)** | Testing procedures and coverage |
| **[SYSTEM_MANUAL.md](SYSTEM_MANUAL.md)** | Operational procedures and user manual |
| **[FILE_AUDIT.md](FILE_AUDIT.md)** | File inventory and cleanup notes |

---

## Performance

### System Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Startup Time** | 30-60s | Loading 365 days of data |
| **Prediction Latency** | <1s | Per race prediction |
| **API Response** | 200-500ms | Betfair REST calls |
| **Memory Usage** | 500MB-1GB | Historical data cached |
| **CPU Usage** | 5-15% idle | Multi-threaded |

### Trading Performance (Historical)

**Current Status (Live Production):**
- Total bets: 1,438
- P/L: -$137.75
- Strike rate: 33.4%
- Period: Dec 2025 - Feb 2026

**Best Week:**
- Week of Jan 5-11: +$15.10 (284 bets)

**Worst Week:**
- Week of Jan 19-25: -$87.90 (299 bets)

**Note:** System currently unprofitable. Model improvements and parameter tuning in progress.

---

## Troubleshooting

### Common Issues

**"No races found" in GUI**
- **Cause:** Database empty or races not within 60-minute window
- **Fix:** Wait for price scraper to detect upcoming races, or import historical data

**"API timeout" errors**
- **Cause:** Betfair rate limiting or network issues
- **Fix:** Check internet connection, reduce API call frequency in config

**"Session expired" warnings**
- **Cause:** Betfair session token expired (8-hour timeout)
- **Fix:** System auto-refreshes; if persists, check credentials in config.py

**"Database locked" errors**
- **Cause:** SQLite accessed by multiple threads simultaneously
- **Fix:** Restart application; if persists, check for zombie processes

**Bets not matching**
- **Cause:** Prices not competitive or low liquidity
- **Fix:** Adjust price limits, bet earlier (T-10min instead of T-2min)

### Debug Mode

**Enable verbose logging:**
```python
# Edit src/core/config.py
LOG_LEVEL = 'DEBUG'
```

**Check logs:**
```bash
tail -f outputs/logs/$(date +%Y-%m-%d)_greyhound.log
```

### Support

- **Issues:** [GitHub Issues](https://github.com/bfarrell97/greyhound-database/issues)
- **Documentation:** See `docs/` directory
- **Email:** Contact repository owner

---

## Project Structure

```
greyhound-database/
├── run.py                      # Application entry point ⭐
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Production source code (28 files)
│   ├── core/                   # Core services (6 files)
│   ├── features/               # Feature engineering (5 files)
│   ├── gui/                    # User interface (3 files)
│   ├── integration/            # External APIs (5 files)
│   ├── models/                 # ML model wrappers (7 files)
│   └── utils/                  # Utilities (2 files)
│
├── scripts/                    # Production scripts (15 files)
│   ├── predict_v44_prod.py     # Market Alpha Engine ⭐
│   ├── live_price_scraper.py   # Price capture ⭐
│   ├── reconcile_live_bets.py  # Bet sync
│   ├── generate_pl_report.py   # P/L reporting
│   └── train_*.py              # Model training
│
├── models/                     # Trained XGBoost models
│   ├── xgb_v44_steamer.pkl     # BACK strategy
│   └── xgb_v45_production.pkl  # LAY strategy
│
├── tests/                      # Test suite
├── outputs/                    # Logs and reports
├── certs/                      # Betfair SSL certificates
├── archive/                    # Old versions (V28-V43)
└── docs/                       # Documentation
    ├── ARCHITECTURE.md
    ├── DEVELOPMENT.md
    ├── TESTING.md
    └── SYSTEM_MANUAL.md
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.11+ |
| **ML Framework** | XGBoost 2.x |
| **GUI** | Tkinter + tksheet |
| **Database** | SQLite 3 |
| **APIs** | Betfair REST API, Topaz API |
| **Data** | pandas, numpy |
| **Notifications** | Discord webhooks |

---

## Contributing

Contributions welcome! Please see [DEVELOPMENT.md](DEVELOPMENT.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

**Quick contribution guide:**
1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Disclaimer

**⚠️ Trading Warning**

This software is for educational and research purposes. Gambling involves financial risk:

- **No guarantees** - Past performance doesn't guarantee future results
- **Lose money** - You can lose more than your initial stake (LAY bets have unlimited liability)
- **Use at your own risk** - Developer not responsible for financial losses
- **Test first** - Always use paper trading mode before risking real money
- **Gamble responsibly** - Only bet what you can afford to lose

By using this software, you acknowledge these risks and agree to use it responsibly.

---

## Acknowledgments

- **Betfair** - For providing the Exchange API
- **Topaz** - For historical form data
- **XGBoost** - For the ML framework
- **Python community** - For excellent data science libraries

---

## Changelog

**v44/v45 (2026-02-12) - Current**
- Enhanced documentation (ARCHITECTURE, DEVELOPMENT, TESTING)
- Cleaned up scripts directory (162 → 15 files)
- Improved README with comprehensive setup guide

**v44/v45 (2026-01-10)**
- V44 Steamer production model
- V45 Drifter production model
- Live price scraper improvements
- Persistent session management

**v43 and earlier**
- See `archive/` directory for legacy versions

---

**Made with 🦾 by Brad**

For questions or support, open an issue on GitHub.

Happy trading! 🐕💰
