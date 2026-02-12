# Greyhound Database - File Audit

**Date:** 2026-02-12  
**Total Files:** 676 Python files

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| **Production (src/)** | 28 | ✅ Active |
| **Scripts** | 162 | ⚠️ Mix of active/temp |
| **Archive** | 482 | ❌ Old versions |
| **Tests** | 4 | ✅ Active |
| **Root** | 1 (run.py) | ✅ Entry point |

**Total Production:** ~35-50 files  
**Cleanup Target:** ~620+ files to review

---

## Production Files (src/ - 28 files)

### Core (4 files)
- `src/core/__init__.py`
- `src/core/config.py` - Configuration
- `src/core/database.py` - Database layer
- `src/core/live_betting.py` - Live betting manager
- `src/core/paper_trading.py` - Paper trading simulator
- `src/core/predictor.py` - Prediction wrapper

### Features (5 files)
- `src/features/feature_engineering.py` - Current features (V44/V45)
- `src/features/feature_engineering_v38.py` - ⚠️ Legacy?
- `src/features/feature_engineering_v39.py` - ⚠️ Legacy?
- `src/features/feature_engineering_v40.py` - ⚠️ Legacy?
- `src/features/feature_engineering_v41.py` - ⚠️ Legacy?

**Action:** Check if v38-v41 are still used or can be archived.

### GUI (3 files)
- `src/gui/__init__.py`
- `src/gui/app.py` - Main application (5,748 lines!)
- `src/gui/temp_virtual_func.py` - ⚠️ Temp file?

**Action:** Review temp_virtual_func.py for removal.

### Integration (4 files)
- `src/integration/__init__.py`
- `src/integration/bet_scheduler.py` - Bet scheduling
- `src/integration/betfair_api.py` - Betfair REST client
- `src/integration/betfair_fetcher.py` - Odds fetching
- `src/integration/topaz_api.py` - Topaz (form) API

### Models (6 files)
- `src/models/__init__.py`
- `src/models/benchmark_cmp.py` - Benchmark comparison
- `src/models/benchmark_fast_updater.py` - Fast benchmark updates
- `src/models/ml_model.py` - ML model wrapper
- `src/models/pace_strategy.py` - Pace analysis
- `src/models/pir_evaluator.py` - PIR model evaluation

### Utils (2 files)
- `src/utils/discord_notifier.py` - Discord notifications
- `src/utils/result_tracker.py` - Result tracking

---

## Scripts (162 files) - Needs Audit

### Production Scripts (Keep - ~10-15 files)

**Critical:**
- `scripts/predict_v44_prod.py` - **V44/V45 Production predictor** ✅
- `scripts/reconcile_live_bets.py` - Bet reconciliation ✅
- `scripts/generate_pl_report.py` - P/L reporting ✅
- `scripts/live_price_scraper.py` - Price scraping ✅

**Training:**
- `scripts/train_v44_production.py` - V44 model training
- `scripts/train_v44_steamer.py` - V44 steamer training
- `scripts/train_v45_drifter.py` - V45 drifter training

**Data Management:**
- `scripts/import_topaz_history.py` - Historical data import
- `scripts/import_bsp.py` - BSP data import

### Debug/Analysis Scripts (Archive or Remove - ~150 files)

**Prefix patterns suggesting temp/debug:**
- `check_*.py` (34 files) - Temporary data checks
- `debug_*.py` (14 files) - Debug scripts
- `test_*.py` (6 files) - Manual tests
- `temp_*.py` (1 file) - Temporary
- `backtest_*.py` (34 files) - Backtest experiments
- `analyze_*.py` (12 files) - One-off analysis
- `validate_*.py` (10 files) - Validation scripts
- `verify_*.py` (10 files) - Verification scripts
- `diagnose_*.py` (5 files) - Diagnostic scripts
- `compare_*.py` (4 files) - Comparison scripts

**Old version scripts:**
- `predict_v41_tips.py` - Legacy V41 (current is V44/V45)
- `train_v41_*.py` (3 files) - Legacy V41 training
- `validate_v41_*.py` (6 files) - Legacy V41 validation
- `optimize_v41_*.py` (2 files) - Legacy V41 optimization
- `backtest_v41_*.py` (2 files) - Legacy V41 backtests

**Suggested Archive List (150+ files):**

```
archive_candidates/
├── analysis/ (30+ files)
│   ├── analyze_*.py
│   ├── diagnose_*.py
│   ├── profile_*.py
│   └── examine_*.py
├── backtests/ (34 files)
│   └── backtest_*.py
├── checks/ (34 files)
│   ├── check_*.py
│   ├── verify_*.py
│   └── validate_*.py
├── debug/ (20 files)
│   ├── debug_*.py
│   ├── test_*.py
│   └── temp_*.py
├── legacy_v41/ (15+ files)
│   ├── predict_v41_*.py
│   ├── train_v41_*.py
│   ├── validate_v41_*.py
│   └── optimize_v41_*.py
├── experiments/ (20+ files)
│   ├── experiment_*.py
│   ├── simulate_*.py
│   ├── sweep_*.py
│   └── grid_*.py
└── maintenance/ (15+ files)
    ├── fix_*.py
    ├── cleanup_*.py
    ├── repair_*.py
    ├── migrate_*.py
    └── rebuild_*.py
```

---

## Archive Directory (482 files) - Already Archived

**Good!** Already has archive/ directory with:
- Old analysis scripts
- Legacy model versions (V28-V43)
- Deprecated experiments

**Action:** Review if any archive/ files should be permanently deleted.

---

## Tests (4 files)

Need to locate and audit test files.

---

## Root Files

**Active:**
- `run.py` - Entry point ✅
- `README.md` - Documentation ✅
- `SYSTEM_MANUAL.md` - User manual ✅
- `requirements.txt` - Dependencies (need to verify exists)
- `greydb.bat` - Windows launcher ✅

**Data files:**
- `live_bets.csv` - Live bet data ✅
- `live_bets_summary.csv` - Summary report ✅
- `live_bets_summary.xlsx` - Excel report ✅

**Created by analysis:**
- `MODEL_ANALYSIS.md` - Analysis doc (can remove if unwanted)

---

## Proposed Cleanup Plan

### Phase 1: Archive Temp Scripts (~150 files)
Move to `archive/temp_scripts/`:
- All `check_*.py`, `debug_*.py`, `test_*.py`, `temp_*.py`
- All `backtest_*.py` (experiments, not production)
- All `analyze_*.py`, `diagnose_*.py`
- All `validate_*.py`, `verify_*.py` (one-off validations)
- All legacy V41 scripts
- All experiment scripts

### Phase 2: Audit Legacy Feature Files
Review `src/features/feature_engineering_v38-v41.py`:
- If unused → archive
- If used → document why

### Phase 3: Remove Obvious Temp Files
- `src/gui/temp_virtual_func.py` (if not used)
- Any other `temp_*.py` files in src/

### Phase 4: Documentation
Create comprehensive docs (like Horse_Ladder_Trading):
- ARCHITECTURE.md
- DEVELOPMENT.md
- FILE_AUDIT.md (this document)
- Improve README.md

### Phase 5: Code Quality
Add docstrings and type hints to all 28 production files in src/.

---

## Questions for Brad

1. **Legacy feature files:** Are v38-v41 feature engineering files still used?
2. **Temp GUI file:** Is `src/gui/temp_virtual_func.py` still needed?
3. **Script retention:** Any specific analysis/debug scripts you want to keep accessible?
4. **Archive policy:** Delete old archive/ files or keep everything?

---

## Next Steps

1. **Confirm cleanup approach** with Brad
2. **Archive temp scripts** to `archive/temp_scripts_2026/`
3. **Remove unused files** from src/
4. **Create documentation** (Architecture, Development, Testing)
5. **Add docstrings/type hints** to production files
6. **Git commit** cleanup changes

**Estimated cleanup:** 150+ files to archive, ~30 production files to document.
