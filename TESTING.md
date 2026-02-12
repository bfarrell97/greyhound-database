# Testing Guide - Greyhound Racing Analysis System

**Last Updated:** 2026-02-12  
**Version:** V44/V45 Production

---

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Unit Tests](#unit-tests)
5. [Integration Tests](#integration-tests)
6. [Manual Testing](#manual-testing)
7. [Test Coverage](#test-coverage)
8. [Continuous Testing](#continuous-testing)
9. [Troubleshooting Tests](#troubleshooting-tests)

---

## Overview

### Testing Philosophy

**Quality over quantity** - Focus on testing critical paths and business logic rather than achieving 100% coverage.

**Test pyramid approach:**
```
        /\
       /  \
      / UI \          ← Few (manual tests)
     /------\
    /  Integ \        ← Some (API, database)
   /----------\
  /    Unit    \      ← Many (functions, classes)
 /--------------\
```

### Current State

**Existing Tests:** 4 test files (minimal coverage)  
**Target Coverage:** 40-60% (focus on core logic)  
**Priority Areas:**
1. Feature engineering (business logic)
2. Database operations (data integrity)
3. API integrations (external dependencies)
4. Bet placement (financial risk)

---

## Test Structure

### Directory Layout

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures
├── test_database.py            # Database tests
├── test_features.py            # Feature engineering tests
├── test_predictions.py         # Model prediction tests
├── test_betfair_api.py         # Betfair API tests
├── test_live_betting.py        # Bet management tests
├── test_price_scraper.py       # Price scraping tests
└── fixtures/                   # Test data
    ├── sample_races.json
    ├── sample_prices.json
    └── sample_bets.json
```

### Test Categories

| Category | Purpose | Examples |
|----------|---------|----------|
| **Unit** | Test individual functions | Feature calculations, data transforms |
| **Integration** | Test component interactions | Database + API, Model + Features |
| **Manual** | GUI and end-to-end testing | Bet placement, P/L display |
| **Performance** | Load and stress testing | Large dataset processing, API limits |

---

## Running Tests

### Basic Usage

**Run all tests:**
```bash
pytest tests/
```

**Run with verbose output:**
```bash
pytest tests/ -v
```

**Run specific test file:**
```bash
pytest tests/test_database.py
```

**Run specific test function:**
```bash
pytest tests/test_database.py::test_insert_race
```

**Run with coverage:**
```bash
pytest --cov=src tests/
```

**Generate HTML coverage report:**
```bash
pytest --cov=src --cov-report=html tests/
# Open htmlcov/index.html in browser
```

### Watch Mode

**Auto-run tests on file changes:**
```bash
# Install pytest-watch
pip install pytest-watch

# Run in watch mode
ptw tests/
```

### Parallel Execution

**Speed up test suite with parallel execution:**
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest tests/ -n 4
```

---

## Unit Tests

### Database Tests

**File:** `tests/test_database.py`

**Purpose:** Verify database operations (CRUD) work correctly

**Example:**
```python
import pytest
from src.core.database import GreyhoundDatabase

@pytest.fixture
def test_db():
    """Create in-memory test database."""
    db = GreyhoundDatabase(':memory:')
    yield db
    # Cleanup happens automatically (in-memory)

def test_insert_and_retrieve_race(test_db):
    """Test inserting and retrieving a race."""
    # Insert test race
    race_id = test_db.insert_race(
        meeting_id=1,
        race_number=5,
        distance=500,
        track_id=1,
        race_time='2026-02-12 14:30:00'
    )
    
    # Retrieve
    race = test_db.get_race(race_id)
    
    # Assert
    assert race is not None
    assert race['RaceNumber'] == 5
    assert race['Distance'] == 500

def test_insert_duplicate_greyhound_fails(test_db):
    """Test that duplicate greyhound names are rejected."""
    # Insert first dog
    test_db.insert_greyhound('Fast Freddy')
    
    # Attempt duplicate
    with pytest.raises(Exception):
        test_db.insert_greyhound('Fast Freddy')

def test_get_upcoming_races(test_db):
    """Test fetching races within time window."""
    # Insert races at different times
    # ... setup code ...
    
    # Fetch races in next 60 minutes
    upcoming = test_db.get_upcoming_races(minutes=60)
    
    # Assert correct filtering
    assert len(upcoming) == 3
    assert all(race['TimeToJump'] <= 60 for race in upcoming)
```

### Feature Engineering Tests

**File:** `tests/test_features.py`

**Purpose:** Verify feature calculations are correct

**Example:**
```python
import pytest
import pandas as pd
from src.features.feature_engineering import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create sample race data for testing."""
    return pd.DataFrame({
        'GreyhoundID': [1, 2, 3],
        'Price60Min': [5.0, 6.0, 7.0],
        'Price30Min': [4.5, 6.5, 7.5],
        'Price15Min': [4.0, 7.0, 8.0],
        'Price5Min': [3.5, 7.5, 8.5],
        'TotalMatched': [10000, 15000, 8000]
    })

def test_steam_rate_calculation(sample_data):
    """Test steam rate feature calculation."""
    fe = FeatureEngineer()
    
    # Calculate features
    result = fe.calculate_steam_rate(sample_data)
    
    # Verify steam rate logic
    # Dog 1: Price dropped from 5.0 → 3.5 = STEAMER (positive rate)
    assert result.iloc[0]['SteamRate'] > 0
    
    # Dog 2: Price rose from 6.0 → 7.5 = DRIFTER (zero steam rate)
    assert result.iloc[1]['SteamRate'] == 0
    
    # Dog 3: Price rose = DRIFTER
    assert result.iloc[2]['SteamRate'] == 0

def test_price_change_features(sample_data):
    """Test price change percentage calculations."""
    fe = FeatureEngineer()
    
    result = fe.calculate_price_changes(sample_data)
    
    # Verify calculations
    # Price 60→30: (5.0 - 4.5) / 5.0 = 10% drop
    assert abs(result.iloc[0]['PriceChange_60_30'] - 0.10) < 0.01
    
    # Price 15→5: (4.0 - 3.5) / 4.0 = 12.5% drop
    assert abs(result.iloc[0]['PriceChange_15_5'] - 0.125) < 0.01

def test_rolling_statistics(test_db, sample_data):
    """Test rolling statistics calculation."""
    fe = FeatureEngineer(db=test_db)
    
    # Setup: Insert historical data (365 days)
    # ... insert test data ...
    
    # Calculate rolling stats
    result = fe.calculate_rolling_features(sample_data, lookback_days=365)
    
    # Verify features exist
    assert 'SteamRate_Last10' in result.columns
    assert 'DriftRate_Last10' in result.columns
    assert 'AvgPriceMove' in result.columns
    
    # Verify ranges (probabilities should be 0-1)
    assert result['SteamRate_Last10'].between(0, 1).all()
    assert result['DriftRate_Last10'].between(0, 1).all()
```

### Model Prediction Tests

**File:** `tests/test_predictions.py`

**Purpose:** Verify ML model predictions work correctly

**Example:**
```python
import pytest
import joblib
import pandas as pd
from scripts.predict_v44_prod import MarketAlphaEngine

@pytest.fixture
def test_engine():
    """Create test engine with loaded models."""
    engine = MarketAlphaEngine(db_path=':memory:')
    return engine

def test_v44_prediction(test_engine, sample_data):
    """Test V44 Steamer model prediction."""
    # Make prediction
    predictions = test_engine.predict_steamers(sample_data)
    
    # Verify output format
    assert 'SteamProb' in predictions.columns
    assert len(predictions) == len(sample_data)
    
    # Verify probability range
    assert predictions['SteamProb'].between(0, 1).all()

def test_v45_prediction(test_engine, sample_data):
    """Test V45 Drifter model prediction."""
    predictions = test_engine.predict_drifters(sample_data)
    
    assert 'DriftProb' in predictions.columns
    assert predictions['DriftProb'].between(0, 1).all()

def test_threshold_filtering(test_engine):
    """Test that only high-confidence predictions pass threshold."""
    # Create test data with varying probabilities
    test_data = pd.DataFrame({
        'SteamProb': [0.10, 0.35, 0.45, 0.60],
        'Price': [5.0, 7.0, 10.0, 12.0]
    })
    
    # Apply V44 threshold (prob >= 0.35, price < 15)
    signals = test_engine.filter_back_signals(test_data)
    
    # Should keep rows 2, 3, 4 (prob >= 0.35)
    assert len(signals) == 3
    assert signals['SteamProb'].min() >= 0.35

def test_edge_cases(test_engine):
    """Test handling of edge cases."""
    # Empty data
    empty = pd.DataFrame()
    result = test_engine.predict_steamers(empty)
    assert len(result) == 0
    
    # Missing columns
    incomplete = pd.DataFrame({'Price5Min': [5.0]})
    with pytest.raises(KeyError):
        test_engine.predict_steamers(incomplete)
    
    # NaN values
    with_nans = sample_data.copy()
    with_nans.loc[0, 'Price5Min'] = None
    # Should handle gracefully (skip or fill)
    result = test_engine.predict_steamers(with_nans)
    assert len(result) > 0
```

---

## Integration Tests

### Betfair API Tests

**File:** `tests/test_betfair_api.py`

**Purpose:** Verify Betfair API integration works

**Note:** These tests hit real Betfair API (or mock)

**Example:**
```python
import pytest
from src.integration.betfair_api import BetfairAPI
from unittest.mock import Mock, patch

@pytest.fixture
def mock_betfair():
    """Mock Betfair API for testing without hitting real API."""
    with patch('src.integration.betfair_api.requests') as mock_requests:
        yield mock_requests

def test_login_success(mock_betfair):
    """Test successful Betfair login."""
    # Mock successful response
    mock_betfair.post.return_value.json.return_value = {
        'sessionToken': 'test_token_123',
        'loginStatus': 'SUCCESS'
    }
    
    api = BetfairAPI()
    result = api.login()
    
    assert result is True
    assert api.session_token == 'test_token_123'

def test_login_failure(mock_betfair):
    """Test failed Betfair login."""
    mock_betfair.post.return_value.json.return_value = {
        'loginStatus': 'INVALID_CREDENTIALS'
    }
    
    api = BetfairAPI()
    result = api.login()
    
    assert result is False
    assert api.session_token is None

def test_list_markets(mock_betfair):
    """Test fetching market list."""
    # Mock market response
    mock_betfair.post.return_value.json.return_value = [
        {
            'marketId': '1.12345',
            'marketName': 'Test Race',
            'eventName': 'Test Track'
        }
    ]
    
    api = BetfairAPI()
    api.session_token = 'test_token'
    
    markets = api.list_markets(event_type='greyhound')
    
    assert len(markets) == 1
    assert markets[0]['marketId'] == '1.12345'

@pytest.mark.integration  # Mark as integration test
def test_real_betfair_connection():
    """Test connection to real Betfair API (requires credentials)."""
    api = BetfairAPI()
    
    # This hits real API - skip in CI
    result = api.test_connection()
    assert result is True
```

### Database Integration Tests

**File:** `tests/test_database_integration.py`

**Purpose:** Test database operations with real SQLite

**Example:**
```python
import pytest
import tempfile
import os
from src.core.database import GreyhoundDatabase

@pytest.fixture
def temp_db():
    """Create temporary database file for testing."""
    # Create temp file
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    # Create database
    db = GreyhoundDatabase(path)
    
    yield db
    
    # Cleanup
    db.close()
    os.unlink(path)

def test_concurrent_access(temp_db):
    """Test multiple threads accessing database."""
    import threading
    
    def insert_race(race_num):
        temp_db.insert_race(
            meeting_id=1,
            race_number=race_num,
            distance=500,
            track_id=1
        )
    
    # Create 10 threads
    threads = [
        threading.Thread(target=insert_race, args=(i,))
        for i in range(10)
    ]
    
    # Start all
    for t in threads:
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Verify all inserted
    races = temp_db.get_all_races()
    assert len(races) == 10
```

---

## Manual Testing

### GUI Testing Checklist

**Startup:**
- [ ] Application launches without errors
- [ ] GUI window maximizes correctly
- [ ] All panels render (Live Alpha, Active Bets, Settled Bets)
- [ ] Status bar shows "Logged in to Betfair"

**Price Scraping:**
- [ ] "Live price scraper started" message in logs
- [ ] Races appear in database within 60 seconds
- [ ] Price updates every 60 seconds

**Predictions:**
- [ ] Live Alpha Radar populates within 5 minutes
- [ ] Top prospects show confidence scores
- [ ] Predictions update every 30 seconds

**Bet Placement (Paper Trade Mode):**
- [ ] Click "Place Bet" opens confirmation dialog
- [ ] Bet appears in Active Bets after placement
- [ ] Unmatched bets show correct price and stake
- [ ] Matched bets move to correct section
- [ ] P/L calculates correctly after race

**Bet Management:**
- [ ] "Cancel All" button cancels all active bets
- [ ] Individual bet cancellation works
- [ ] Settled bets show WIN/LOSS correctly
- [ ] P/L summary updates in real-time

**Error Handling:**
- [ ] API timeout shows error message (not crash)
- [ ] Database lock retries automatically
- [ ] Session expiry triggers re-login
- [ ] Invalid bet rejected with clear message

### Manual Test Scenarios

**Scenario 1: Normal Operation**
1. Start application
2. Wait for races to load (T-60min window)
3. Observe predictions appearing in Live Alpha Radar
4. Place paper trade bet on top prospect
5. Wait for race to jump
6. Verify result and P/L update

**Scenario 2: API Failure**
1. Disconnect internet mid-session
2. Observe error messages in GUI
3. Reconnect internet
4. Verify automatic recovery (scraper resumes)

**Scenario 3: High Volume**
1. Run during peak racing hours (Saturday afternoon)
2. Monitor 10+ races simultaneously
3. Place multiple bets across different races
4. Verify no lag or crashes
5. Check all bets tracked correctly

---

## Test Coverage

### Current Coverage

**Run coverage report:**
```bash
pytest --cov=src --cov-report=term-missing tests/
```

**Example output:**
```
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
src/core/config.py                   15      0   100%
src/core/database.py                250     50    80%   123-145, 200-210
src/features/feature_engineering.py 180     90    50%   90-120, 150-180
src/integration/betfair_api.py      120     60    50%   45-70, 100-115
src/models/ml_model.py               60     40    33%   30-60
---------------------------------------------------------------
TOTAL                               625    240    62%
```

### Coverage Goals

| Module | Target | Priority |
|--------|--------|----------|
| **core/database.py** | 80%+ | High (data integrity) |
| **features/** | 70%+ | High (business logic) |
| **integration/betfair_api.py** | 60%+ | Medium (external API) |
| **core/live_betting.py** | 75%+ | High (financial risk) |
| **gui/app.py** | 20%+ | Low (manual testing) |

### Improving Coverage

**1. Identify untested code:**
```bash
pytest --cov=src --cov-report=html tests/
# Open htmlcov/index.html
# Red lines = untested code
```

**2. Write tests for critical paths:**
- Focus on red lines in high-priority modules
- Prioritize happy path (normal operation)
- Add edge case tests (errors, empty data)

**3. Mock external dependencies:**
```python
# Mock Betfair API to test bet placement logic
@patch('src.integration.betfair_api.BetfairAPI.place_order')
def test_bet_placement(mock_place_order):
    mock_place_order.return_value = {'betId': '12345', 'status': 'SUCCESS'}
    
    manager = LiveBettingManager()
    result = manager.place_bet(...)
    
    assert result['success'] is True
```

---

## Continuous Testing

### Pre-Commit Hooks

**Setup git hooks to run tests before commit:**
```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/
        language: system
        pass_filenames: false
        always_run: true
EOF

# Install hooks
pre-commit install
```

**Now tests run automatically on `git commit`**

### CI/CD Integration

**Example GitHub Actions workflow:**
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ --cov=src
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Troubleshooting Tests

### Common Issues

**1. Import errors**
```
ModuleNotFoundError: No module named 'src'
```
**Fix:** Add src to PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

**2. Database locked**
```
sqlite3.OperationalError: database is locked
```
**Fix:** Use in-memory database for tests
```python
db = GreyhoundDatabase(':memory:')
```

**3. Slow tests**
```
Tests taking >30 seconds
```
**Fix:** Use pytest-xdist for parallel execution
```bash
pytest tests/ -n auto
```

**4. Flaky tests (intermittent failures)**
```
Test passes sometimes, fails others
```
**Fix:** 
- Remove time-dependent logic
- Mock external APIs
- Use fixed random seeds
- Increase timeouts for async operations

---

## Best Practices

### Writing Good Tests

**✅ Do:**
- Write descriptive test names (`test_steam_rate_with_price_drop`)
- Use fixtures for setup/teardown
- Test one thing per test
- Mock external dependencies
- Use parametrized tests for multiple scenarios
- Keep tests fast (<1s per test)

**❌ Don't:**
- Test implementation details
- Rely on external services (use mocks)
- Share state between tests
- Skip cleanup
- Write brittle tests (sensitive to minor changes)

### Test-Driven Development (TDD)

**Red-Green-Refactor cycle:**

1. **Red** - Write failing test first
```python
def test_new_feature():
    result = my_new_function(input_data)
    assert result == expected_output
```

2. **Green** - Write minimal code to pass
```python
def my_new_function(data):
    return expected_output  # Hardcoded for now
```

3. **Refactor** - Improve implementation
```python
def my_new_function(data):
    # Proper implementation
    return calculated_result
```

---

**End of Testing Guide**

For development setup, see [DEVELOPMENT.md](DEVELOPMENT.md).  
For system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).
