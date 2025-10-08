# Cleanup and Fixes Needed

## 1. Missing Figures

### Production Map
- **Status**: Generating but not saving
- **Issue**: Need to check why it's not being exported

### H-P Envelope
- **Status**: FAILING
- **Error**: `operands could not be broadcast together with shapes (981508,) (961965,)`
- **Issue**: Shape mismatch between production and harvest area data
- **Fix Needed**: Debug envelope calculation in `agririchter/analysis/envelope.py`

## 2. Repository Cleanup

### Files to Remove (Development/Testing)

**Task Implementation Summaries** (move to docs/development/):
- TASK_*_IMPLEMENTATION_SUMMARY.md (15 files)
- TASK_*_VERIFICATION.md (4 files)
- TASK_*_COMPLETE.md (2 files)

**Demo Scripts** (keep in examples/ or remove):
- demo_agririchter_figures.py
- demo_agririchter_scale.py
- demo_data_validator.py
- demo_event_calculator.py
- demo_events_pipeline.py
- demo_grid_manager.py
- demo_hp_envelope_real_events.py
- demo_performance_optimizations.py
- demo_run_pipeline.sh
- demo_spatial_mapper.py

**Test/Verification Scripts** (keep in tests/ or remove):
- debug_pipeline.py
- test_agririchter_scale_severity.* (3 files)
- test_hp_envelope_*.* (9 files)
- test_newer_countries_states.py
- test_severity_*.* (9 files)
- test_severity_classification.py
- test_state_filtering.py
- verify_spam2020_structure.py
- verify_state_level_events.py
- verify_validation_module.py

**Temporary Output Files**:
- *.eps, *.svg, *.png, *.jpg in root directory (test outputs)
- FINAL_COMPREHENSIVE_TEST_RESULTS.md
- NEWER_COUNTRIES_STATE_FILTERING_VERIFICATION.md
- HP_ENVELOPE_AXIS_VERIFICATION.md
- COUNTRY_STATE_FILTERING_FIX.md
- MIGRATION_SUMMARY.txt
- SPAM_2020_STRUCTURE.md
- SPATIAL_MAPPER_VERIFICATION.md
- STATE_LEVEL_VERIFICATION_GUIDE.md

**Old Python Files**:
- AgriRichterv2.py
- BalanceAnalysis.py
- figplot_StocksSampler.py
- Stocks_IC.py
- stocks_ton_kcalconvert.py
- StocksSampler.py

## 3. Overlapping Event Labels

### Issue
Laki1783 and Drought18761878 have nearly identical magnitudes:
- Laki1783: M = 5.8609
- Drought18761878: M = 5.8634

### Solutions

**Option 1: Improve adjustText parameters**
```python
adjust_text(texts, ax=ax,
           expand_points=(1.5, 1.5),  # More space around points
           expand_text=(1.2, 1.2),    # More space around text
           force_points=0.5,          # Stronger repulsion from points
           force_text=0.5,            # Stronger repulsion between labels
           arrowprops=dict(arrowstyle='->', color='red', lw=0.5, alpha=0.6))
```

**Option 2: Smart label placement**
- Detect overlapping events (magnitude difference < 0.05)
- Offset labels vertically for close events
- Use different arrow styles

**Option 3: Selective labeling**
- Only label events above certain magnitude threshold
- Add legend with all events
- Interactive plot with hover labels

## 4. Recommended Actions

### Immediate (Before Commit)

1. **Create development docs folder**
   ```bash
   mkdir -p docs/development
   mv TASK_*.md docs/development/
   mv *_VERIFICATION.md docs/development/
   ```

2. **Create examples folder**
   ```bash
   mkdir -p examples
   mv demo_*.py examples/
   ```

3. **Remove temporary files**
   ```bash
   rm -f test_*.png test_*.eps test_*.svg test_*.jpg
   rm -f *_VERIFICATION.md *_FIX.md MIGRATION_SUMMARY.txt
   ```

4. **Fix overlapping labels**
   - Update adjustText parameters in agririchter_scale.py

5. **Fix H-P Envelope**
   - Debug shape mismatch in envelope.py
   - Ensure production and harvest arrays align

6. **Fix Production Map**
   - Ensure map is being saved to figures directory

### After Initial Commit

1. Complete H-P Envelope visualization
2. Complete global production/harvest maps
3. Add interactive plots (plotly)
4. Add MATLAB validation comparison
5. Performance optimization

## 5. Priority Order

1. ✅ Fix overlapping labels (HIGH - affects main figure)
2. ⏳ Clean up repository (HIGH - before commit)
3. ⏳ Fix H-P Envelope (MEDIUM - secondary figure)
4. ⏳ Fix Production Map (MEDIUM - secondary figure)
5. ⏳ Organize documentation (LOW - can be done later)
