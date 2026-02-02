import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# Set up logging to avoid cluttering verification output
logging.basicConfig(level=logging.WARNING)

# Ensure we can import the agrichter engine
sys.path.append(str(Path(__file__).parent.parent))
from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.analysis.envelope_v2 import HPEnvelopeCalculatorV2
from scripts.fig4_risk_probability import calculate_shocks, load_real_events

def verify_and_explain_fig4():
    """
    Consolidated script to verify the mathematical logic of Figure 4.
    """
    print("\n" + "="*60)
    print("FIGURE 4 LOGIC VERIFICATION & EXPLANATION")
    print("="*60)

    print("\n--- STEP 1: CALIBRATING HISTORICAL VOLATILITY ---")
    # This pulls global 'Allgrain' production from 1960-2023, 
    # removes the linear trend, and calculates fractional shocks.
    stats = calculate_shocks()
    sigma = stats['sigma']  # Historical standard deviation of shocks
    print(f"Historical Volatility (sigma): {sigma:.2%}")
    
    # Define constants
    config = Config(crop_type='allgrain', root_dir='.')
    caloric_content = config.get_caloric_content()
    kcal_per_tmt = 1e9 * caloric_content
    current_prod_kcal = stats['current_trend_prod'] * kcal_per_tmt
    
    # THE CORE CALIBRATION:
    # We anchor our exponential tail to the historical record.
    # In a normal distribution, ~32% of events are outside 1 sigma.
    # We solve for 'lambda' (the scale) such that P(X > 1 sigma) = 0.32.
    # P(X > x) = exp(-x / lambda)
    # 0.32 = exp(-(sigma * current_prod) / lambda)
    # log(0.32) = -(sigma * current_prod) / lambda
    # lambda = (sigma * current_prod) / -log(0.32)
    main_scale = (sigma * current_prod_kcal) / (-np.log(0.32))
    print(f"Current trend production: {current_prod_kcal:.2e} kcal")
    print(f"Calibrated Risk Scale (lambda): {main_scale:.2e} kcal")

    print("\n--- STEP 2: LOADING THE SYSTEMIC FRAGILITY BOUNDARY (ENVELOPE) ---")
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    calculator = HPEnvelopeCalculatorV2(config)
    envelope = calculator.calculate_hp_envelope(prod_df, harv_df)
    
    # These are the coordinates of the 'worst case' red line
    H_km2 = envelope['upper_bound_harvest']
    P_up = envelope['upper_bound_production']
    M_scale = np.log10(H_km2)
    print(f"Envelope loaded: {len(M_scale)} points spanning M {M_scale[0]:.1f} to {M_scale[-1]:.1f}")

    print("\n--- STEP 3: MAPPING AREA TO PROBABILITY ---")
    mags_plot = np.linspace(3, 7.5, 500)
    # Get the production loss (kcal) for every magnitude on our x-axis
    losses_at_mags = np.interp(mags_plot, M_scale, P_up)
    
    # THE PROBABILITY FORMULA:
    # Annual Exceedance Prob = exp(-Loss / Scale)
    prob_median = np.exp(-losses_at_mags / main_scale)
    print(f"Probability mapping complete for 500 points.")
    
    print("\n--- STEP 4: BOOTSTRAPPING FOR UNCERTAINTY ---")
    n_boot = 1000
    shocks = stats['shocks']
    boot_res = []
    print(f"Performing {n_boot} iterations...")
    for _ in range(n_boot):
        sample = np.random.choice(shocks, size=len(shocks), replace=True)
        s_boot = np.std(sample)
        scale_boot = (s_boot * current_prod_kcal) / (-np.log(0.32))
        boot_res.append(np.exp(-losses_at_mags / scale_boot))
    
    prob_low = np.percentile(boot_res, 2.5, axis=0)
    prob_high = np.percentile(boot_res, 97.5, axis=0)
    print("95% Confidence Interval generated via bootstrap.")

    print("\n--- STEP 5: VERIFYING HISTORICAL MARKERS ---")
    events = load_real_events(config, grid_manager)
    # For each event, we compute:
    # 1. Its real production loss (L)
    # 2. Its annual probability: P = exp(-L / main_scale)
    event_probs = np.exp(-events['production_loss_kcal'] / main_scale)
    
    # Print diagnostic table for high-magnitude events
    print(f"{'Event Name':<25} | {'Magnitude':<6} | {'Loss (kcal)':<10} | {'Return Period':<10}")
    print("-" * 65)
    sorted_events = events.assign(prob=event_probs).sort_values('magnitude', ascending=False)
    for i, row in sorted_events.iterrows():
        if row['magnitude'] > 4.5:
            rp = 1.0 / row['prob']
            rp_text = f"{rp:,.0f} yr" if rp < 10000 else ">10,000 yr"
            print(f"{row['event_name'][:25]:<25} | {row['magnitude']:<9.2f} | {row['production_loss_kcal']:<10.2e} | {rp_text}")

    # Final Plot Generation (Verification Figure)
    plt.figure(figsize=(10, 8))
    plt.fill_between(mags_plot, prob_low, prob_high, color='gray', alpha=0.2, label='95% CI (Bootstrap)')
    plt.plot(mags_plot, prob_median, color='firebrick', lw=3, label='Theoretical Risk Curve (H-P Upper Bound)')
    plt.scatter(events['magnitude'], event_probs, color='black', s=70, alpha=0.7, label='Historical Realizations (Mapped)')
    
    plt.yscale('log')
    plt.xlim(3, 7.5)
    plt.ylim(1e-4, 1)
    plt.xlabel('AgRichter Magnitude ($M_D$)', fontsize=12, fontweight='bold')
    plt.ylabel('Annual Exceedance Probability', fontsize=12, fontweight='bold')
    plt.title('Figure 4 Logic Verification: Area Disruption vs. Systemic Risk', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', alpha=0.15)
    
    output_path = Path('results/fig4_verification_standalone.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nVerification plot saved to: {output_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    verify_and_explain_fig4()
