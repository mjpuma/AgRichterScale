# Project Structure - Why It's Complex

## The Situation

You have an **existing research codebase** (`agririchter` package) that was built over time with many modules. This is NOT something I created - it's your existing project structure.

## Current Structure

```
AgRichter2025/
├── agririchter/              # Existing package (NOT created by me)
│   ├── core/                 # Configuration, constants
│   ├── data/                 # Data loading, events, spatial mapping
│   ├── analysis/             # Calculations (envelopes, events)
│   └── visualization/        # Plotting modules
│
├── ancillary/                # Data files (Excel, CSV)
├── spam2020V2r0_*/          # SPAM data
├── USDAdata/                # USDA data
│
└── fig*.py                  # NEW: Simple scripts I created
```

## The Problem

The `agririchter` package has:
- **23 Python files** across multiple subdirectories
- **Complex dependencies** between modules
- **Helper functions** that create FAKE data (like `create_sample_events_data()`)
- **Real functions** buried in the package structure

## Why Debugging Is Hard

1. **Hidden synthetic data** - Functions like `create_sample_events_data()` create fake data
2. **Deep module nesting** - Real code is in `agririchter/analysis/event_calculator.py`
3. **Multiple similar scripts** - You have 12+ `generate_*.py` scripts from previous attempts

## What I Did

I created 4 **simple standalone scripts** (`fig1.py`, `fig2.py`, etc.) that:
- Import from the existing `agririchter` package
- Use REAL data (not synthetic)
- Are easy to run individually

**I did NOT create the complex package structure - that already existed.**

## Options to Simplify

### Option 1: Keep Current Structure (Recommended)
- Keep the `agririchter` package as-is (it works)
- Use the 4 simple `fig*.py` scripts I created
- Delete all the old `generate_*.py` scripts
- **Pros**: Nothing breaks, clean scripts
- **Cons**: Still have package complexity underneath

### Option 2: Flatten Everything (Risky)
- Copy all needed code from `agririchter/` into single files
- Put everything in one folder
- **Pros**: Simpler structure
- **Cons**: High risk of breaking things, hard to maintain

### Option 3: Hybrid Approach
- Keep core `agririchter` package
- Create a `scripts/` folder with just the 4 figure scripts
- Delete all old generation scripts
- **Pros**: Clean separation, nothing breaks
- **Cons**: Still have package dependency

## My Recommendation

**Option 1 + Cleanup:**

1. **Keep**: The 4 new `fig*.py` scripts (they work with real data)
2. **Delete**: All old `generate_*.py` scripts (12+ files causing confusion)
3. **Keep**: The `agririchter/` package (it's your research code)
4. **Add**: Steering rule to never use synthetic data (DONE ✅)

This gives you:
- ✅ 4 simple scripts to run
- ✅ Real data only
- ✅ No confusion from old scripts
- ✅ Nothing breaks

## The Real Issue

The complexity isn't from over-engineering - it's from:
1. **Accumulated scripts** from multiple attempts (12+ generate files)
2. **Synthetic data functions** in the package that shouldn't be used
3. **Lack of clear documentation** about what's real vs fake

## Next Steps

Want me to:
1. **Clean up** - Delete all old generate scripts, keep only fig1-4?
2. **Document** - Create a simple "HOW TO RUN" guide?
3. **Simplify** - Move fig scripts to a `scripts/` folder?

**Tell me which option you prefer and I'll execute it.**
