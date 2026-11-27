#!/usr/bin/env python3
"""
Figure S4: Tail-Model Sensitivity for the Exceedance-Probability Analysis (Figure 4)

Figure 4 uses a deliberately conservative exponential tail calibrated so that
P(|shock| > sigma) = 0.32. This supplementary analysis quantifies how sensitive
the headline return periods are to that modeling choice by comparing four tail
models fitted to the same detrended USDA production shocks (1960-present):

  1. Conservative exponential (as used in Fig. 4): lambda = sigma / (-ln 0.32)
  2. Maximum-likelihood exponential fitted to observed annual losses
  3. Generalized Pareto (peaks-over-threshold, threshold = 0) MLE fit
  4. Gaussian (one-sided) tail, the "standard" assumption

Panel A: annual exceedance probability vs fractional production loss for each
model, with the empirical exceedance of the historical record overlaid.
Panel B: implied return periods for the two policy thresholds (3-month supply
buffer and total ending stocks), with 95% bootstrap confidence intervals.

Outputs: results/figureS4_tail_sensitivity.png / .svg
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

sys.path.append(str(Path(__file__).parent.parent))

from agrichter.data.usda_wide import load_usda_wide

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('figS4')

RNG_SEED = 42
N_BOOT = 1000


def compute_shocks():
    """Detrended fractional production shocks, matching Figure 4 (linear trend)."""
    prod_df = load_usda_wide("production")
    allgrain = prod_df[
        (prod_df['Commodity'] == 'Allgrain') & (prod_df['Country'] == 'World')
    ].sort_values('Year')
    years = allgrain['Year'].values
    vals = allgrain['Value'].values

    trend = np.poly1d(np.polyfit(years, vals, 1))(years)
    shocks = (vals - trend) / trend

    cons_df = load_usda_wide("consumption")
    stock_df = load_usda_wide("endingstocks")
    ref_years = [2019, 2020, 2021]

    def ref_val(df):
        sub = df[(df['Commodity'] == 'Allgrain') & (df['Country'] == 'World')
                 & (df['Year'].isin(ref_years))]
        return sub['Value'].mean()

    return {
        'shocks': shocks,
        'trend_prod': trend[-1],
        'avg_cons': ref_val(cons_df),
        'avg_stocks': ref_val(stock_df),
    }


def fit_models(shocks):
    """Fit all four tail models to a shock sample. Losses are positive fractions."""
    sigma = np.std(shocks)
    losses = -shocks[shocks < 0]
    n = len(shocks)
    p_loss = len(losses) / n  # probability a given year is a loss year

    models = {}

    # 1. Conservative exponential (Figure 4 calibration)
    lam_c = sigma / (-np.log(0.32))
    models['Conservative Exp. (Fig. 4)'] = lambda x, lam=lam_c: np.exp(-x / lam)

    # 2. MLE exponential on losses (exceedance = p_loss * survival)
    lam_mle = np.mean(losses)
    models['MLE Exponential'] = (
        lambda x, lam=lam_mle, p=p_loss: p * np.exp(-x / lam)
    )

    # 3. Generalized Pareto (POT with threshold at zero loss)
    #    fit shape (xi) and scale (beta); location fixed at 0
    xi, _, beta = sps.genpareto.fit(losses, floc=0)
    models['GPD (POT)'] = (
        lambda x, c=xi, b=beta, p=p_loss: p * sps.genpareto.sf(x, c, loc=0, scale=b)
    )

    # 4. Gaussian one-sided tail
    models['Gaussian'] = lambda x, s=sigma: sps.norm.sf(x / s)

    params = {
        'sigma': sigma, 'lam_c': lam_c, 'lam_mle': lam_mle,
        'gpd_shape': xi, 'gpd_scale': beta, 'p_loss': p_loss,
        'n_losses': len(losses),
    }
    return models, params, losses


def main():
    logger.info("Generating Figure S4: tail-model sensitivity...")
    data = compute_shocks()
    shocks = data['shocks']
    n_years = len(shocks)

    models, params, losses = fit_models(shocks)
    logger.info(f"Fitted parameters: {params}")

    # Goodness of fit (KS) for the two fitted loss models
    ks_exp = sps.kstest(losses, 'expon', args=(0, params['lam_mle']))
    ks_gpd = sps.kstest(losses, 'genpareto',
                        args=(params['gpd_shape'], 0, params['gpd_scale']))
    logger.info(f"KS exponential: D={ks_exp.statistic:.3f}, p={ks_exp.pvalue:.3f}")
    logger.info(f"KS GPD:         D={ks_gpd.statistic:.3f}, p={ks_gpd.pvalue:.3f}")

    # Policy thresholds as fractional loss of trend production (all in 1000 MT)
    thresholds = {
        '3-Month Supply': (data['avg_cons'] / 4.0) / data['trend_prod'],
        'Total Stocks': data['avg_stocks'] / data['trend_prod'],
    }
    logger.info(f"Thresholds (fraction of production): {thresholds}")

    # Bootstrap return periods per model
    rng = np.random.default_rng(RNG_SEED)
    model_names = list(models.keys())
    boot_rp = {name: {t: [] for t in thresholds} for name in model_names}
    for _ in range(N_BOOT):
        sample = rng.choice(shocks, size=n_years, replace=True)
        if np.sum(sample < 0) < 5:
            continue  # degenerate resample
        try:
            boot_models, _, _ = fit_models(sample)
        except Exception:
            continue
        for name in model_names:
            for tname, tval in thresholds.items():
                p = float(boot_models[name](tval))
                boot_rp[name][tname].append(1.0 / max(p, 1e-12))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    colors = {
        'Conservative Exp. (Fig. 4)': 'firebrick',
        'MLE Exponential': 'darkorange',
        'GPD (POT)': 'steelblue',
        'Gaussian': 'black',
    }
    styles = {
        'Conservative Exp. (Fig. 4)': '-',
        'MLE Exponential': '--',
        'GPD (POT)': '-.',
        'Gaussian': ':',
    }

    # Panel A: exceedance curves + empirical record
    ax = axes[0]
    x = np.linspace(1e-4, 0.35, 400)
    for name, fn in models.items():
        ax.semilogy(x * 100, fn(x), styles[name], color=colors[name], lw=2.5, label=name)

    sorted_losses = np.sort(losses)[::-1]
    emp_p = np.arange(1, len(sorted_losses) + 1) / n_years
    ax.semilogy(sorted_losses * 100, emp_p, 'ko', ms=6, mfc='white',
                label=f'Empirical ({n_years} yr record)')

    for tname, tval in thresholds.items():
        ax.axvline(tval * 100, color='gray', ls='--', lw=1.5, alpha=0.7)
        ax.text(tval * 100 + 0.2, 2e-4, tname, rotation=90, va='bottom',
                fontsize=10, color='gray')

    ax.set_xlim(0, 35)
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel('Annual Production Loss (% of trend)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Annual Exceedance Probability', fontsize=13, fontweight='bold')
    ax.set_title('A) Tail Models vs Historical Record', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, which='both', alpha=0.2)

    # Panel B: return periods at policy thresholds with bootstrap CIs
    ax = axes[1]
    width = 0.18
    xpos = np.arange(len(thresholds))
    for i, name in enumerate(model_names):
        med, lo, hi = [], [], []
        for tname, tval in thresholds.items():
            rps = np.array(boot_rp[name][tname])
            point = 1.0 / max(float(models[name](tval)), 1e-12)
            med.append(point)
            if len(rps) > 50:
                lo.append(max(point - np.percentile(rps, 2.5), 0))
                hi.append(max(np.percentile(rps, 97.5) - point, 0))
            else:
                lo.append(0)
                hi.append(0)
        offset = (i - (len(model_names) - 1) / 2) * width
        ax.bar(xpos + offset, med, width * 0.9, yerr=[lo, hi], capsize=3,
               color=colors[name], alpha=0.85, label=name,
               error_kw=dict(ecolor='gray', lw=1))

    ax.set_yscale('log')
    ax.set_xticks(xpos)
    ax.set_xticklabels(list(thresholds.keys()), fontsize=12)
    ax.set_ylabel('Return Period (years)', fontsize=13, fontweight='bold')
    ax.set_title('B) Implied Return Periods (95% bootstrap CI)',
                 fontsize=14, fontweight='bold')
    ax.axhline(100, color='gray', ls=':', lw=1, alpha=0.6)
    ax.axhline(1000, color='gray', ls=':', lw=1, alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', axis='y', alpha=0.2)

    # Fit summary box
    summary = (
        f"$\\sigma$ = {params['sigma']*100:.2f}%   "
        f"$\\lambda_{{cons}}$ = {params['lam_c']*100:.2f}%   "
        f"$\\lambda_{{MLE}}$ = {params['lam_mle']*100:.2f}%\n"
        f"GPD: $\\xi$ = {params['gpd_shape']:.2f}, "
        f"$\\beta$ = {params['gpd_scale']*100:.2f}%   "
        f"KS p-values: exp {ks_exp.pvalue:.2f}, GPD {ks_gpd.pvalue:.2f}"
    )
    fig.text(0.5, -0.02, summary, ha='center', fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))

    plt.tight_layout()
    out = Path('results/figureS4_tail_sensitivity.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out.with_suffix('.svg'), format='svg', bbox_inches='tight', facecolor='white')
    logger.info(f"Saved {out} and {out.with_suffix('.svg')}")
    plt.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
