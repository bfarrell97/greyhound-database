# Greyhound Database Restructuring Plan

**Created:** 2026-02-12  
**Status:** In Progress  
**Model:** Horse_Ladder_Trading comprehensive improvement approach

---

## Overview

Complete restructuring and documentation of the Greyhound Racing Analysis System (V44/V45 Production), following the same comprehensive approach used for Horse_Ladder_Trading.

**Current State:**
- 676 total Python files (28 production, 162 scripts, 482 archived)
- Minimal documentation
- No docstrings or type hints
- Cluttered scripts directory

**Target State:**
- Clean, well-documented codebase
- Comprehensive architecture documentation
- 100% docstring coverage on production files
- 100% type hint coverage
- Clear separation of production vs. archived code

---

## Phase 1: Documentation Foundation

**Status:** Starting  
**Time Estimate:** 3-4 hours

### Deliverables

1. **ARCHITECTURE.md** (Complete system design)
   - System overview and components
   - Data flow diagrams
   - Model architecture (V44/V45)
   - Database schema
   - API integrations (Betfair, Topaz)
   - Threading model
   - Feature engineering pipeline

2. **README.md** (Enhanced user guide)
   - Quick start (60 seconds)
   - Installation instructions
   - System requirements
   - Configuration guide
   - Strategy overview (BACK/LAY/COVER)
   - Troubleshooting

3. **DEVELOPMENT.md** (Developer guide)
   - Development setup
   - Code structure
   - Contribution guidelines
   - Model training workflow
   - Adding new features
   - Debugging guide

4. **TESTING.md** (Test documentation)
   - Test structure
   - Running tests
   - Coverage reports
   - Integration testing
   - Manual testing procedures

5. **FILE_AUDIT.md** (Inventory - already created ✅)
   - Active vs. archived files
   - Production file list
   - Cleanup recommendations

---

## Phase 2: File Cleanup

**Status:** Partially Complete (31/133 scripts archived)  
**Time Estimate:** 1-2 hours

### Actions

1. ✅ Created archive structure (`archive/temp_scripts_2026/`)
2. ⏳ Archive remaining ~100-110 temp scripts
3. Keep only ~15-20 production scripts
4. Remove unused files from src/ if any
5. Update .gitignore

### Production Scripts to Retain
- predict_v44_prod.py (V44/V45 engine)
- reconcile_live_bets.py (bet sync)
- generate_pl_report.py (reporting)
- live_price_scraper.py (price data)
- train_v44_*.py (3 files - training)
- train_v45_*.py (2 files - training)
- train_pace_model.py (pace analysis)
- import_*.py (4 files - data import)
- predict_v41_tips.py (legacy, still imported)
- predict_lay_strategy_betfair.py (still imported)
- predict_back_strategy.py (BACK strategy)

**Total: ~17 production scripts**

---

## Phase 3: Code Quality Improvements

**Status:** Not Started  
**Time Estimate:** 4-6 hours

### 3.1 Add Docstrings (Google Style)

**Target:** All 28 production files in `src/`

**Format:**
```python
"""Module description.

Detailed explanation of module purpose, features, and usage.

Example:
    >>> from src.core import GreyhoundDatabase
    >>> db = GreyhoundDatabase()
    >>> db.connect()

Attributes:
    MODULE_CONSTANT: Description

See Also:
    related_module: Related functionality
"""
```

**Files to Document:**
- src/core/ (6 files: config, database, live_betting, paper_trading, predictor)
- src/features/ (5 files: feature_engineering*.py)
- src/gui/ (3 files: app, temp_virtual_func, __init__)
- src/integration/ (5 files: bet_scheduler, betfair_api, betfair_fetcher, topaz_api)
- src/models/ (7 files: benchmark_cmp, ml_model, pace_strategy, pir_evaluator, etc.)
- src/utils/ (2 files: discord_notifier, result_tracker)

### 3.2 Add Type Hints

**Target:** 100% coverage on all function signatures

**Example:**
```python
def calculate_edge(
    model_prob: float,
    market_odds: float
) -> float:
    """Calculate betting edge.
    
    Args:
        model_prob: Model probability (0-1)
        market_odds: Market decimal odds
    
    Returns:
        Edge value (positive = value bet)
    """
    market_prob = 1 / market_odds
    return model_prob - market_prob
```

### 3.3 Clean Up Imports

- Remove unused imports
- Organize import order (stdlib, third-party, local)
- Add missing `__all__` exports to `__init__.py` files

### 3.4 Code Formatting

- Consistent style throughout
- Fix long lines (>100 chars)
- Improve variable naming where needed

---

## Phase 4: Testing (Optional)

**Status:** Not Started  
**Time Estimate:** 3-4 hours  
**Priority:** Lower (only if requested)

### Expand Test Coverage

Currently only 4 test files. Add:
- Unit tests for core modules
- Integration tests for Betfair API
- Feature engineering tests
- Model prediction tests
- Database operation tests

**Target:** 40-60% code coverage

---

## Implementation Order

### Session 1: Documentation (NOW)
1. Create ARCHITECTURE.md ← Start here
2. Enhance README.md
3. Create DEVELOPMENT.md
4. Create TESTING.md

### Session 2: Cleanup
1. Complete script archival
2. Remove unused src/ files
3. Update .gitignore
4. Git commit cleanup

### Session 3: Code Quality
1. Add docstrings to all src/ files (28 files)
2. Add type hints
3. Clean imports
4. Git commit improvements

### Session 4: Testing (if requested)
1. Review existing tests
2. Add missing test coverage
3. Create test documentation

---

## Time Estimates

| Phase | Time | Status |
|-------|------|--------|
| Documentation | 3-4h | Starting |
| Cleanup | 1-2h | 25% done |
| Code Quality | 4-6h | Not started |
| Testing | 3-4h | Optional |
| **Total** | **11-16h** | **~10% complete** |

---

## Current Session Plan

**Focus:** Phase 1 - Documentation Foundation

**Next Steps:**
1. ✅ Create RESTRUCTURING_PLAN.md (this file)
2. → Create ARCHITECTURE.md (comprehensive system design)
3. → Enhance README.md
4. → Create DEVELOPMENT.md
5. → Create TESTING.md

After documentation is complete, proceed to cleanup and code quality improvements.

---

## Success Criteria

✅ **Documentation**
- ARCHITECTURE.md covers all major components
- README.md provides clear quick-start
- DEVELOPMENT.md enables new contributors
- TESTING.md documents test approach

✅ **Cleanup**
- scripts/ directory has <20 files (production only)
- All temp/debug scripts archived
- No unused files in src/

✅ **Code Quality**
- 100% docstring coverage (all modules, classes, public methods)
- 100% type hint coverage (all function signatures)
- Consistent code style
- Clear, readable code

✅ **Testing** (optional)
- 40-60% code coverage
- Key workflows tested
- Integration tests pass

---

## Notes

- Following exact same approach as Horse_Ladder_Trading Phase 1+2
- System performance issues with large directory operations (676 files)
- May need to complete cleanup in smaller batches
- Documentation first ensures value even if cleanup is slow

**Ready to proceed with Phase 1: Documentation!**
