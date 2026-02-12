# Development Guide - Greyhound Racing Analysis System

**System Version:** V44/V45 Production  
**Last Updated:** 2026-02-12

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Project Structure](#project-structure)
4. [Development Workflow](#development-workflow)
5. [Model Training](#model-training)
6. [Adding Features](#adding-features)
7. [Testing](#testing)
8. [Debugging](#debugging)
9. [Deployment](#deployment)
10. [Contributing](#contributing)

---

## Getting Started

### Prerequisites

**Required:**
- Python 3.11 or higher
- Git
- SQLite 3.35+
- Betfair account with API access
- Betfair SSL certificates (for non-interactive login)

**Optional:**
- Topaz API key (for historical data)
- Discord webhook (for notifications)

### Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/bfarrell97/greyhound-database.git
cd greyhound-database

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API credentials
# Edit src/core/config.py with your API keys

# 5. Place Betfair certificates
mkdir -p certs
# Copy client-2048.crt and client-2048.key to certs/

# 6. Initialize database
python -c "from src.core.database import GreyhoundDatabase; GreyhoundDatabase()"

# 7. Run application
python run.py
```

### First-Time Setup

**1. Betfair API Access:**
- Register at https://www.betfair.com/developer
- Generate API key (App Key)
- Download SSL certificates for non-interactive login
- Place certificates in `certs/` directory

**2. Import Historical Data (Optional but Recommended):**
```bash
# Import last 90 days of data for training
python scripts/import_topaz_history.py --days 90
```

**3. Train Models (if not using pre-trained):**
```bash
# Train V44 Steamer model
python scripts/train_v44_production.py

# Train V45 Drifter model
python scripts/train_v45_drifter.py
```

**4. Verify Setup:**
```bash
# Test Betfair connection
python -c "from src.integration.betfair_api import BetfairAPI; BetfairAPI().test_connection()"

# Check database
python -c "from src.core.database import GreyhoundDatabase; db = GreyhoundDatabase(); print(f'Tracks: {len(db.get_all_tracks())}')"
```

---

## Development Environment

### Recommended IDE Setup

**VS Code:**
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true
}
```

**PyCharm:**
- Set Python interpreter to `venv/bin/python`
- Enable type checking
- Set code style to Black
- Enable pytest for testing

### Development Dependencies

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Or manually:
pip install pytest pytest-cov black pylint mypy jupyter ipython
```

### Environment Variables

Create `.env` file (optional, for sensitive config):
```bash
BETFAIR_APP_KEY=your_app_key
BETFAIR_USERNAME=your_username
BETFAIR_PASSWORD=your_password
TOPAZ_API_KEY=your_topaz_key
DISCORD_WEBHOOK_URL=your_webhook_url
```

---

## Project Structure

```
greyhound-database/
│
├── run.py                      # Application entry point
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
│
├── src/                        # Production source code
│   ├── core/                   # Core services
│   │   ├── config.py           # Configuration (API keys)
│   │   ├── database.py         # Database layer
│   │   ├── live_betting.py     # Bet management
│   │   ├── paper_trading.py    # Simulation mode
│   │   └── predictor.py        # Prediction wrapper
│   │
│   ├── features/               # Feature engineering
│   │   └── feature_engineering.py  # Current V44/V45 features
│   │
│   ├── gui/                    # User interface
│   │   └── app.py              # Main GUI (5,748 lines)
│   │
│   ├── integration/            # External API clients
│   │   ├── bet_scheduler.py    # Bet timing
│   │   ├── betfair_api.py      # Betfair REST client
│   │   ├── betfair_fetcher.py  # Odds fetching
│   │   └── topaz_api.py        # Form data API
│   │
│   ├── models/                 # ML model wrappers
│   │   ├── benchmark_cmp.py    # Benchmark comparison
│   │   ├── ml_model.py         # Model loader
│   │   ├── pace_strategy.py    # Pace analysis
│   │   └── pir_evaluator.py    # PIR evaluation
│   │
│   └── utils/                  # Utilities
│       ├── discord_notifier.py # Discord notifications
│       └── result_tracker.py   # Result tracking
│
├── scripts/                    # Production scripts (15 files)
│   ├── predict_v44_prod.py     # Market Alpha Engine
│   ├── live_price_scraper.py   # Price capture service
│   ├── reconcile_live_bets.py  # Bet reconciliation
│   ├── generate_pl_report.py   # P/L reporting
│   ├── train_v44_*.py          # V44 model training
│   ├── train_v45_*.py          # V45 model training
│   └── import_*.py             # Data import utilities
│
├── models/                     # Trained XGBoost models
│   ├── xgb_v44_steamer.pkl
│   └── xgb_v45_production.pkl
│
├── tests/                      # Test suite
│   ├── test_database.py
│   ├── test_features.py
│   └── test_predictions.py
│
├── archive/                    # Archived files
│   ├── temp_scripts_2026/      # Recently archived scripts
│   └── [legacy versions]       # Old V28-V43 code
│
├── outputs/                    # Generated files
│   ├── logs/                   # Application logs
│   └── reports/                # P/L reports
│
├── certs/                      # Betfair SSL certificates
│   ├── client-2048.crt
│   └── client-2048.key
│
└── docs/                       # Documentation
    ├── ARCHITECTURE.md         # System design
    ├── DEVELOPMENT.md          # This file
    ├── TESTING.md              # Test guide
    └── SYSTEM_MANUAL.md        # User manual
```

---

## Development Workflow

### 1. Feature Development

**Branching Strategy:**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes, commit frequently
git commit -m "feat: add new feature"

# Push to remote
git push origin feature/your-feature-name

# Create pull request (or merge directly if solo)
```

**Commit Message Convention:**
```
feat: Add new feature
fix: Bug fix
docs: Documentation changes
refactor: Code refactoring
test: Add/update tests
chore: Maintenance tasks
```

### 2. Code Style

**Follow PEP 8 with Black formatter:**
```bash
# Format code
black src/ scripts/

# Check style
pylint src/

# Type checking
mypy src/ --ignore-missing-imports
```

**Example:**
```python
"""Module docstring.

Detailed description of module purpose.
"""

from typing import List, Dict, Optional
import pandas as pd


def calculate_features(
    data: pd.DataFrame,
    lookback_days: int = 365
) -> pd.DataFrame:
    """Calculate rolling features from historical data.
    
    Args:
        data: Historical race data
        lookback_days: Days of history to use
    
    Returns:
        DataFrame with calculated features
    
    Example:
        >>> df = load_data()
        >>> features = calculate_features(df, lookback_days=90)
    """
    # Implementation
    pass
```

### 3. Testing Workflow

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_database.py::test_insert_race

# Run in watch mode (requires pytest-watch)
ptw tests/
```

### 4. Debugging

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Use IPython for interactive debugging:**
```python
# Add breakpoint in code
import IPython; IPython.embed()

# Or use pdb
import pdb; pdb.set_trace()
```

**Check database state:**
```bash
sqlite3 greyhound_racing.db
> SELECT COUNT(*) FROM Races;
> SELECT * FROM LiveBets ORDER BY Date DESC LIMIT 10;
```

---

## Model Training

### V44 Steamer Model (BACK Strategy)

**1. Prepare Training Data:**
```bash
# Import 12 months of historical data
python scripts/import_topaz_history.py --days 365
```

**2. Train Model:**
```python
# scripts/train_v44_production.py

from src.core.database import GreyhoundDatabase
from src.features.feature_engineering import FeatureEngineer
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load data
db = GreyhoundDatabase()
data = db.get_historical_races(days=365)

# Engineer features
fe = FeatureEngineer()
features = fe.calculate_features(data, feature_set='v44_steamer')

# Prepare X, y
X = features.drop(['Win', 'GreyhoundID'], axis=1)
y = features['Win']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost
model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=200,
    objective='binary:logistic',
    eval_metric='logloss',
    scale_pos_weight=10  # Handle class imbalance
)

model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import roc_auc_score
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.3f}")

# Save model
import joblib
joblib.dump(model, 'models/xgb_v44_steamer.pkl')
```

**3. Backtest:**
```bash
# Test on out-of-sample data
python scripts/backtest_v44_steamer.py --start-date 2025-06-01 --end-date 2025-12-31
```

**4. Deploy:**
- Replace `models/xgb_v44_steamer.pkl` with new model
- Restart application

### V45 Drifter Model (LAY Strategy)

**Similar workflow, different feature set:**
```python
# Target: Dogs that drift and lose
features = fe.calculate_features(data, feature_set='v45_drifter')

# Train with different params (lower learning rate for LAY)
model = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.03,
    n_estimators=250,
    objective='binary:logistic',
    scale_pos_weight=1.5  # Drifters are ~40% of data
)
```

### Hyperparameter Tuning

**Grid Search:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05],
    'n_estimators': [150, 200, 250],
    'min_child_weight': [1, 3, 5]
}

grid = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")
```

### Model Versioning

**Naming convention:**
```
models/
├── xgb_v44_steamer_YYYYMMDD.pkl      # Dated backups
├── xgb_v44_steamer.pkl               # Production (latest)
├── xgb_v45_production_YYYYMMDD.pkl
└── xgb_v45_production.pkl
```

---

## Adding Features

### 1. Add Feature to Feature Engineering

**Edit `src/features/feature_engineering.py`:**
```python
def calculate_pace_rating(self, race_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate early speed rating for each dog.
    
    Args:
        race_data: Race data with Last3Starts column
    
    Returns:
        DataFrame with PaceRating column added
    """
    # Implementation: Count recent box 1-2 starts
    race_data['PaceRating'] = race_data.apply(
        lambda row: self._calc_pace(row['Last3Starts']),
        axis=1
    )
    return race_data
```

### 2. Update Feature List

**Add to feature set definition:**
```python
FEATURE_SETS = {
    'v44_steamer': [
        'PriceChange_15_5',
        'SteamRate_Last10',
        'LivePrice',
        'PaceRating',  # NEW FEATURE
        # ... existing features
    ]
}
```

### 3. Retrain Models

```bash
# Retrain with new feature
python scripts/train_v44_production.py

# Backtest to verify improvement
python scripts/backtest_v44_steamer.py
```

### 4. Update Documentation

- Add feature description to ARCHITECTURE.md
- Document feature in code docstring
- Update training scripts if needed

---

## Testing

### Unit Tests

**Test database operations:**
```python
# tests/test_database.py

import pytest
from src.core.database import GreyhoundDatabase

def test_insert_and_retrieve_race():
    db = GreyhoundDatabase(':memory:')  # In-memory DB
    
    # Insert test race
    race_id = db.insert_race(
        meeting_id=1,
        race_number=1,
        distance=500,
        track_id=1
    )
    
    # Retrieve
    race = db.get_race(race_id)
    assert race['Distance'] == 500
```

**Test feature engineering:**
```python
# tests/test_features.py

from src.features.feature_engineering import FeatureEngineer
import pandas as pd

def test_steam_rate_calculation():
    fe = FeatureEngineer()
    
    # Create test data
    data = pd.DataFrame({
        'Price60Min': [5.0, 6.0, 7.0],
        'Price5Min': [4.0, 5.5, 8.0]
    })
    
    # Calculate
    result = fe.calculate_steam_rate(data)
    
    # Verify
    assert result.iloc[0]['SteamRate'] > 0  # Price dropped = steam
    assert result.iloc[2]['SteamRate'] == 0  # Price rose = no steam
```

### Integration Tests

**Test Betfair API:**
```python
# tests/test_betfair_integration.py

from src.integration.betfair_api import BetfairAPI

def test_betfair_connection():
    api = BetfairAPI()
    markets = api.list_markets()
    assert len(markets) > 0
```

### Manual Testing

**Test GUI locally:**
1. Run `python run.py`
2. Check "Live Alpha Radar" updates
3. Manually place paper trade
4. Verify bet appears in Active Bets
5. Check logs for errors

---

## Debugging

### Common Issues

**1. "No races found" in GUI**
- **Cause:** Database empty or price scraper not running
- **Fix:** Import historical data, wait for scraper to detect races

**2. "API timeout" errors**
- **Cause:** Betfair rate limiting or slow connection
- **Fix:** Reduce API call frequency, check network

**3. "Model prediction failed"**
- **Cause:** Missing features or data format mismatch
- **Fix:** Check feature engineering output, verify column names

**4. Database locked errors**
- **Cause:** Multiple threads accessing SQLite simultaneously
- **Fix:** Use connection pooling or retry logic

### Debug Tools

**1. Interactive Python Shell:**
```bash
ipython
from src.core.database import GreyhoundDatabase
db = GreyhoundDatabase()
races = db.get_upcoming_races()
```

**2. Database Browser:**
```bash
# GUI tool
sqlitebrowser greyhound_racing.db

# CLI tool
sqlite3 greyhound_racing.db
```

**3. Network Monitoring:**
```bash
# Monitor Betfair API calls
tcpdump -i any -A host api.betfair.com
```

---

## Deployment

### Production Checklist

- [ ] Models trained on full dataset
- [ ] Backtests pass with positive ROI
- [ ] All tests pass (`pytest`)
- [ ] Logs rotating correctly
- [ ] Discord notifications working
- [ ] Betfair credentials valid
- [ ] Database backed up
- [ ] System monitoring enabled

### Running in Production

**1. Screen/tmux session (Linux/Mac):**
```bash
screen -S greyhound
python run.py
# Ctrl+A, D to detach
```

**2. Windows Service:**
```bash
# Use NSSM (Non-Sucking Service Manager)
nssm install GreyhoundBot "C:\path\to\python.exe" "C:\path\to\run.py"
nssm start GreyhoundBot
```

**3. Docker (optional):**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run.py"]
```

### Monitoring

**Check logs:**
```bash
tail -f outputs/logs/$(date +%Y-%m-%d)_greyhound.log
```

**Monitor P/L:**
```bash
# Daily summary
python scripts/generate_pl_report.py --today
```

**System health:**
- Discord: Check for error notifications
- Database: Monitor size growth
- CPU/Memory: Should be <50% utilization

---

## Contributing

### Pull Request Process

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes
4. Add tests
5. Run test suite (`pytest`)
6. Format code (`black src/`)
7. Commit changes (`git commit -m 'feat: add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open Pull Request

### Code Review Checklist

- [ ] Code follows project style (Black + PEP 8)
- [ ] All tests pass
- [ ] New code has tests (>80% coverage)
- [ ] Docstrings added for new functions/classes
- [ ] Type hints added
- [ ] No secrets/credentials committed
- [ ] Documentation updated
- [ ] Breaking changes documented

### Coding Standards

**Do:**
- ✅ Write clear, self-documenting code
- ✅ Add docstrings to all public functions
- ✅ Use type hints
- ✅ Handle errors gracefully
- ✅ Log important events
- ✅ Write tests for new features

**Don't:**
- ❌ Commit credentials or API keys
- ❌ Leave commented-out code
- ❌ Use global variables
- ❌ Ignore type hints
- ❌ Skip tests
- ❌ Push directly to main (unless solo)

---

## Getting Help

### Resources

- **Documentation:** See `docs/` directory
- **Architecture:** `docs/ARCHITECTURE.md`
- **User Manual:** `docs/SYSTEM_MANUAL.md`
- **Testing Guide:** `docs/TESTING.md`

### Support Channels

- **Issues:** GitHub Issues (bug reports, feature requests)
- **Discord:** Community discussion
- **Email:** Contact repository owner

### FAQ

**Q: How often should I retrain models?**  
A: Monthly or when performance degrades. Monitor ROI weekly.

**Q: Can I run multiple strategies simultaneously?**  
A: Yes, but manage bankroll carefully. Separate tracking recommended.

**Q: What if Betfair API goes down?**  
A: System will retry automatically. Check logs. Manual intervention if >1 hour outage.

**Q: How do I test new features without risking money?**  
A: Use paper trading mode (`--paper-trade` flag) for simulation.

---

**End of Development Guide**

For system architecture, see `ARCHITECTURE.md`.  
For operational procedures, see `SYSTEM_MANUAL.md`.  
For testing procedures, see `TESTING.md`.
