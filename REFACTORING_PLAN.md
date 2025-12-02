# Self-Contained Publication Figures - Refactoring Plan

## Goal
Create a single `publication_figures/` folder that contains everything needed to generate the 4 figures, with no dependencies on the main `agririchter` package.

## Structure

```
publication_figures/
├── fig1_global_maps.py          # Main scripts (updated imports)
├── fig2_agrichter_scale.py
├── fig3_hp_envelopes.py
├── fig4_country_envelopes.py
├── run_all.py                   # Master script
├── README.md                    # Instructions
│
└── lib/                         # Self-contained library
    ├── __init__.py
    ├── config.py                # From agririchter.core.config
    ├── constants.py             # From agririchter.core.constants
    ├── grid_manager.py          # From agririchter.data.grid_manager
    ├── spatial_mapper.py        # From agririchter.data.spatial_mapper
    ├── events.py                # From agririchter.data.events
    ├── event_calculator.py      # From agririchter.analysis.event_calculator
    ├── envelope_calculator.py   # From agririchter.analysis.envelope_v2
    ├── envelope_builder.py      # From agririchter.analysis.envelope_builder
    ├── agririchter_viz.py       # From agririchter.visualization.agririchter_scale
    ├── envelope_viz.py          # From agririchter.visualization.hp_envelope
    └── map_generator.py         # From agririchter.visualization.global_map_generator
```

## Steps

1. ✅ Create `publication_figures/` directory
2. ✅ Create `publication_figures/lib/` directory
3. Copy needed modules from `agririchter/` to `lib/`
4. Update imports in copied modules (agririchter.X → lib.X)
5. Copy and update the 4 figure scripts
6. Create run_all.py master script
7. Create README.md with instructions
8. Test that it works independently

## Modules to Copy

### Core (2 files)
- agririchter/core/config.py → lib/config.py
- agririchter/core/constants.py → lib/constants.py

### Data (3 files)
- agririchter/data/grid_manager.py → lib/grid_manager.py
- agririchter/data/spatial_mapper.py → lib/spatial_mapper.py
- agririchter/data/events.py → lib/events.py

### Analysis (3 files)
- agririchter/analysis/event_calculator.py → lib/event_calculator.py
- agririchter/analysis/envelope_v2.py → lib/envelope_calculator.py
- agririchter/analysis/envelope_builder.py → lib/envelope_builder.py

### Visualization (3 files)
- agririchter/visualization/agririchter_scale.py → lib/agririchter_viz.py
- agririchter/visualization/hp_envelope.py → lib/envelope_viz.py
- agririchter/visualization/global_map_generator.py → lib/map_generator.py

**Total: ~11 Python files in lib/**

## Import Changes

All scripts will change from:
```python
from agririchter.core.config import Config
```

To:
```python
from lib.config import Config
```

## Benefits

✅ Self-contained - one folder has everything
✅ No dependency on main agririchter package
✅ Can be moved/shared independently
✅ Scripts stay small (~200 lines)
✅ Shared code in lib/ (no duplication)

## Risks

⚠️ Need to update all internal imports in copied modules
⚠️ Need to test thoroughly
⚠️ Will be out of sync with main package if that changes

## Testing

After refactoring:
1. Run each fig script individually
2. Verify outputs match original
3. Check that no agririchter imports remain
4. Verify it works from a different directory
