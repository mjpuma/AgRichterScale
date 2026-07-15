#!/usr/bin/env python3
"""
Figure S5: Fractional-Loss Sensitivity of Historical Event Magnitudes

The main-text analysis maps each historical event to its "modern-equivalent
footprint": 100% loss of current (SPAM 2020) production across the affected
countries/states. This is an upper-bound convention — real events destroy
only a fraction of the harvest within their footprint.

This figure shows how event positions on the H-P diagram shift under partial
loss intensities f = 100%, 50%, and 25%. Both the disrupted area and the
production loss scale linearly with f, so each event slides down-left along
a line of slope 1 in log-log space:

    M_D(f) = M_D(1) + log10(f)        L(f) = f * L(1)

Because the upper envelope is concave and the lower envelope convex (both
through the origin), scaled points remain inside the envelope, so the bounds
hold for any loss intensity.

Output: results/figureS5_fractional_loss_sensitivity.png / .svg
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('figS5')

FRACTIONS = [1.0, 0.5, 0.25]
FRACTION_STYLE = {
    1.0: dict(color='#1f77b4', size=90, alpha=0.95, label='100% loss (main text)'),
    0.5: dict(color='#7fb3d5', size=60, alpha=0.9, label='50% loss'),
    0.25: dict(color='#c8dcec', size=40, alpha=0.9, label='25% loss'),
}

# No per-event text labels: the triplet trajectories are the message.
# (Named events are identifiable from the exported events CSV if needed.)


def main():
    logger.info("Generating Figure S5: fractional-loss sensitivity...")

    # Envelope + thresholds (allgrain, consumption-based thresholds)
    config = Config(crop_type='allgrain', root_dir='.', use_dynamic_thresholds=True)
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    calculator = HPEnvelopeCalculatorV2(config)
    env = calculator.calculate_hp_envelope(prod_df, harv_df)

    H_up = env['upper_bound_harvest']
    P_up = env['upper_bound_production']
    H_low = env['lower_bound_harvest']
    P_low = env['lower_bound_production']

    # Events (computed by fig1 with the corrected mapping / per-crop kcal)
    events_path = Path('results/historical_events_computed.csv')
    events = pd.read_csv(events_path)
    events = events[(events['crop_type'] == 'allgrain')
                    & (events['harvest_area_km2'] > 0)].copy()
    logger.info(f"Loaded {len(events)} allgrain events from {events_path}")

    fig, ax = plt.subplots(figsize=(11, 9))

    # Envelope band
    M_up = np.log10(H_up)
    M_low = np.log10(H_low)
    ax.plot(M_up, P_up, color='black', lw=2.5, label='Upper Bound', zorder=3)
    ax.plot(M_low, P_low, color='blue', lw=2.5, label='Lower Bound', zorder=3)
    Mq = np.linspace(max(M_up.min(), M_low.min()), min(M_up.max(), M_low.max()), 500)
    ax.fill_between(
        Mq,
        np.interp(Mq, M_low, P_low),
        np.interp(Mq, M_up, P_up),
        color='gray', alpha=0.15, zorder=1,
    )

    # Threshold lines (kcal, corrected 1000-MT conversion, 2019-2021 window)
    threshold_colors = {'1 Month': '#FFD700', '3 Months': '#FF4500',
                        'Total Stocks': '#800080'}
    for name, value in config.thresholds.items():
        ax.axhline(value, color=threshold_colors.get(name, 'gray'),
                   ls='--', lw=2, alpha=0.85, zorder=2)
        ax.text(1.15, value * 1.15, name, color=threshold_colors.get(name, 'gray'),
                fontsize=11, fontweight='bold', va='bottom')

    # Event trajectories under fractional loss
    for _, row in events.iterrows():
        M1 = row['magnitude_MD']
        L1 = row['production_loss_kcal']
        traj_M = [M1 + np.log10(f) for f in FRACTIONS]
        traj_L = [L1 * f for f in FRACTIONS]
        ax.plot(traj_M, traj_L, color='gray', lw=0.8, alpha=0.5, zorder=4)
        for f, m, l in zip(FRACTIONS, traj_M, traj_L):
            s = FRACTION_STYLE[f]
            ax.scatter(m, l, s=s['size'], color=s['color'], alpha=s['alpha'],
                       edgecolors='black', linewidths=0.5, zorder=5)

    # Legend (envelope + fraction markers)
    fraction_handles = [
        mlines.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=FRACTION_STYLE[f]['color'],
                      markeredgecolor='black',
                      markersize=np.sqrt(FRACTION_STYLE[f]['size']),
                      label=FRACTION_STYLE[f]['label'])
        for f in FRACTIONS
    ]
    envelope_handles = [
        mlines.Line2D([0], [0], color='black', lw=2.5, label='Upper Bound'),
        mlines.Line2D([0], [0], color='blue', lw=2.5, label='Lower Bound'),
    ]
    ax.legend(handles=envelope_handles + fraction_handles,
              loc='lower right', fontsize=11, framealpha=0.95)

    ax.set_yscale('log')
    ax.set_xlim(1, 7)
    ax.set_ylim(1e8, 2e16)
    ax.set_xlabel(r'AgRichter Magnitude ($M_D = \log_{10}(A_H/\mathrm{km}^2)$)',
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Production Loss (kcal)', fontsize=14, fontweight='bold')
    ax.text(0.02, 0.98,
            'Each grey segment links one event at 100%, 50%, and 25% loss\n'
            '(down-left along unit slope in log--log space). Events unlabeled.',
            transform=ax.transAxes, fontsize=9, va='top', color='dimgray',
            style='italic')
    ax.set_title('Sensitivity of Event Positions to Loss Intensity\n'
                 '(All Grains; modern-equivalent footprint at three intensities)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, which='both', alpha=0.15)

    plt.tight_layout()
    out = Path('results/figureS5_fractional_loss_sensitivity.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out.with_suffix('.svg'), format='svg', bbox_inches='tight',
                facecolor='white')
    logger.info(f"Saved {out} and {out.with_suffix('.svg')}")
    plt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
