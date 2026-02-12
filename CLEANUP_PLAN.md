# Greyhound Database Cleanup Plan

**Status:** In Progress  
**Date:** 2026-02-12

---

## Current State

**Before cleanup:**
- Total files: 676 Python files
- Production (src/): 28 files
- Scripts: 162 files
- Archive (already): 482 files

**After partial cleanup:**
- Scripts archived so far: 31 files
- Scripts remaining: 133 files

---

## Production Files to Keep

### Core Production Scripts (scripts/)
1. `predict_v44_prod.py` - V44/V45 production predictor ✅
2. `reconcile_live_bets.py` - Bet reconciliation ✅
3. `generate_pl_report.py` - P/L reporting ✅
4. `live_price_scraper.py` - Price scraping ✅
5. `train_v44_production.py` - V44 model training
6. `train_v44_steamer.py` - V44 steamer training
7. `train_v45_drifter.py` - V45 drifter training
8. `train_v45_test.py` - V45 testing
9. `train_v44_test.py` - V44 testing
10. `import_topaz_history.py` - Historical data import
11. `import_bsp.py` - BSP data import
12. `import_ltp_prices.py` - LTP prices import
13. `import_lay_place_prices.py` - Place prices import
14. `predict_v41_tips.py` - Legacy V41 (still imported by app.py)
15. `predict_lay_strategy_betfair.py` - LAY strategy (still imported)
16. `train_pace_model.py` - Pace model (still imported)
17. `__init__.py` - Package init

**Total production scripts: ~15-20**

### All Production Files (src/)
- All 28 files in src/ directory ✅

---

## Files Already Archived (31 files)

Moved to `archive/temp_scripts_2026/`:
- `analysis/` - analyze_*, diagnose_*, profile_* scripts
- `backtests/` - backtest_* scripts
- `checks/` - check_*, verify_*, validate_* scripts
- `debug/` - debug_*, test_*, temp_* scripts
- `legacy_v41/` - V41 version scripts
- `experiments/` - experiment_*, simulate_* scripts
- `maintenance/` - fix_*, cleanup_*, repair_* scripts

---

## Remaining Cleanup (~110-120 files)

**Scripts to archive:**

### Analysis/Inspection (~15 files)
- inspect_*.py
- query_*.py
- extract_*.py
- list_*.py

### Additional Checks (~20 files remaining)
- Any remaining check_*.py not yet moved

### Additional Backtests (~20 files remaining)
- Any remaining backtest_*.py not yet moved

### Data Management (~10 files)
- repair_*.py
- fix_*.py (data fixes)
- migrate_*.py
- wipe_*.py
- clear_*.py

### Mixed/Uncategorized (~50-60 files)
- One-off scripts
- Temp analysis
- Old experiments

---

## Next Steps

### Option A: Manual Review (Thorough)
1. List all remaining 133 scripts
2. Review each one individually
3. Decide: Keep or Archive
4. Document decision

### Option B: Bulk Archive (Fast)
1. Keep only the ~15-20 production scripts listed above
2. Archive everything else to `archive/temp_scripts_2026/uncategorized/`
3. Can always retrieve if needed later

### Option C: Hybrid (Recommended)
1. Keep confirmed production scripts (list above)
2. Archive obvious temp/debug patterns in bulk
3. Leave ~20-30 ambiguous scripts for Brad to review
4. Create `scripts/NEEDS_REVIEW.md` list

---

## Recommendation

**I recommend Option C:**

1. **Keep these scripts** (confirmed production):
   - predict_v44_prod.py
   - reconcile_live_bets.py
   - generate_pl_report.py
   - live_price_scraper.py
   - train_v44_*.py (3 files)
   - train_v45_*.py (2 files)
   - import_*.py (4 files)
   - predict_v41_tips.py
   - predict_lay_strategy_betfair.py
   - train_pace_model.py

2. **Archive these patterns** (bulk, low risk):
   - All remaining check_*.py
   - All remaining backtest_*.py
   - All remaining debug_*.py
   - All remaining test_*.py
   - All repair_*.py, fix_*.py, wipe_*.py

3. **Review remaining** (~20-30 files):
   - Create list for Brad to approve/archive

**This gives us:**
- Clean scripts/ directory (~15-20 production files)
- Safe archive (~140+ temp files)
- Small review list (~20-30 ambiguous)

---

## After Cleanup: Documentation Phase

Once scripts are cleaned up, proceed with Horse_Ladder_Trading style improvements:

### Phase 1: Documentation
- ARCHITECTURE.md - System design
- DEVELOPMENT.md - Developer guide
- README.md improvements
- TESTING.md - Test documentation

### Phase 2: Code Quality
- Add docstrings to all 28 src/ files
- Add type hints
- Improve inline documentation
- Clean up imports

### Phase 3: Testing
- Review existing tests
- Add missing test coverage
- Integration tests

---

## Commit Strategy

**Commit 1:** Archive temp scripts (current)
```bash
git add archive/temp_scripts_2026/
git add scripts/
git commit -m "chore: archive temporary debug and analysis scripts

Moved 140+ temporary scripts to archive/temp_scripts_2026/:
- Analysis and diagnostic scripts
- Experimental backtests
- One-off check/verify scripts
- Legacy V41 version scripts
- Debug and test files

Retained ~15 production scripts in scripts/ directory."
```

**Commit 2:** Documentation (next)
**Commit 3:** Code quality improvements (final)

---

## Brad: Please Confirm

1. **Archiving approach:** Option A, B, or C above?
2. **Script retention:** Any specific scripts you want to keep from the "archive" list?
3. **Delete vs Archive:** Should I permanently delete old archive/ files, or keep everything?
4. **Documentation:** Proceed with full Horse_Ladder_Trading style docs after cleanup?

Let me know and I'll complete the cleanup!
