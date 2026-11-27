#!/usr/bin/env python3
"""
Figure S2: Comparative national vulnerability (normalized upper envelopes).

Shows, for each country, the AgRichter magnitude at which the upper H-P
envelope reaches a given fraction of national production.  Semi-log axes
(linear M_D vs linear % loss) make the horizontal ordering legible: stiff,
concentrated countries (e.g. Egypt) reach 100% at lower magnitudes than
geographically dispersed exporters (e.g. Australia).

The global curve is shown dashed as a reference only — it uses the world
grid and is not directly comparable on the same x-axis ranking as nations.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Core countries from Figure 3 + a spread of typologies
COUNTRY_CONFIGS = [
    {'key': 'Egypt', 'label': 'Egypt', 'spam_name': 'Egypt',
     'color': '#9467bd', 'linewidth': 2.5},
    {'key': 'France', 'label': 'France', 'spam_name': 'France',
     'color': '#8c564b', 'linewidth': 2.0},
    {'key': 'Germany', 'label': 'Germany', 'spam_name': 'Germany',
     'color': '#bcbd22', 'linewidth': 2.0},
    {'key': 'Brazil', 'label': 'Brazil', 'spam_name': 'Brazil',
     'color': '#2ca02c', 'linewidth': 2.0},
    {'key': 'Australia', 'label': 'Australia', 'spam_name': 'Australia',
     'color': '#e377c2', 'linewidth': 2.0},
    {'key': 'USA', 'label': 'USA', 'spam_name': 'United States of America',
     'color': '#1f77b4', 'linewidth': 2.5},
    {'key': 'India', 'label': 'India', 'spam_name': 'India',
     'color': '#ff7f0e', 'linewidth': 2.5},
    {'key': 'China', 'label': 'China', 'spam_name': 'China',
     'color': '#d62728', 'linewidth': 2.5},
]


def _curve(env):
    magnitude = np.log10(np.asarray(env['upper_bound_harvest'], dtype=float))
    total = float(env['upper_bound_production'][-1])
    pct = (np.asarray(env['upper_bound_production'], dtype=float) / total) * 100
    return magnitude, pct


def generate_comparative_figure():
    logger.info("Generating Figure S2: comparative national vulnerability...")

    config = Config(crop_type='allgrain', root_dir='.')
    grid_manager = GridDataManager(config)
    prod_df, harv_df = grid_manager.load_spam_data()
    calculator = HPEnvelopeCalculatorV2(config)

    envelopes = {}
    for country in COUNTRY_CONFIGS:
        key = country['key']
        mask = prod_df['ADM0_NAME'] == country['spam_name']
        if mask.sum() == 0:
            logger.warning(f"No data for {key}")
            continue
        try:
            envelopes[key] = calculator.calculate_hp_envelope(
                prod_df[mask], harv_df[mask]
            )
        except Exception as e:
            logger.error(f"Failed {key}: {e}")

    # Global reference (dashed; not ranked against nations)
    try:
        envelopes['Global'] = calculator.calculate_hp_envelope(prod_df, harv_df)
    except Exception as e:
        logger.warning(f"Global envelope failed: {e}")

    fig, ax = plt.subplots(figsize=(12, 8))

    endpoints = []
    for country in COUNTRY_CONFIGS:
        key = country['key']
        if key not in envelopes:
            continue
        M, pct = _curve(envelopes[key])
        ax.plot(M, pct, color=country['color'], linewidth=country['linewidth'],
                alpha=0.9, zorder=5)
        endpoints.append({
            'label': country['label'],
            'color': country['color'],
            'M_end': float(M[-1]),
            'bold': key in ('USA', 'China', 'India', 'Egypt'),
        })

    if 'Global' in envelopes:
        M, pct = _curve(envelopes['Global'])
        ax.plot(M, pct, color='black', linewidth=2.5, ls='--', alpha=0.55,
                zorder=3, label='Global (reference)')

    # Direct labels at the 100% endpoint (staggered to avoid overlap)
    endpoints.sort(key=lambda e: e['M_end'])
    label_rows = [101.5, 106.5]
    min_dx = 0.28
    last_x = {0: -np.inf, 1: -np.inf}
    for i, e in enumerate(endpoints):
        row = i % 2
        x_label = max(e['M_end'], last_x[row] + min_dx)
        last_x[row] = x_label
        ax.annotate(
            e['label'], xy=(e['M_end'], 100), xytext=(x_label, label_rows[row]),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold' if e['bold'] else 'normal',
            color=e['color'],
            arrowprops=dict(arrowstyle='-', color=e['color'], lw=0.8,
                            alpha=0.55, shrinkA=0, shrinkB=1),
        )

    ax.set_xlim(3.5, 6.5)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.axhline(100, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.set_xlabel(
        r'AgRichter Magnitude ($M_D = \log_{10}(A_H / \mathrm{km}^2)$)',
        fontsize=14, fontweight='bold',
    )
    ax.set_ylabel(
        'Worst-case production loss (% of national total)',
        fontsize=14, fontweight='bold',
    )
    ax.set_title(
        'National agricultural vulnerability fingerprints (all grains)\n'
        'Label marks the magnitude where the upper envelope reaches 100% '
        'of national production',
        fontsize=13, fontweight='bold',
    )
    ax.grid(True, ls='-', alpha=0.12)
    if 'Global' in envelopes:
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out = Path('results/figureS2_comparative_envelopes_semilogx.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out.with_suffix('.svg'), format='svg', bbox_inches='tight',
                facecolor='white')
    logger.info(f"Saved {out}")
    plt.close()


if __name__ == '__main__':
    generate_comparative_figure()
