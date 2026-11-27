#!/usr/bin/env python3
"""
Table S2: Verification & Validation (V&V) report.

Standard computational-science V&V, made visible as a supplementary table:

VERIFICATION (is the math solved correctly?)
  For every envelope used in the paper (4 global crops + 8 countries), check
  the five mathematical invariants at runtime:
    1. starts at origin            L(0) = 0
    2. converges at endpoint       L(A_total) = P_total
    3. upper dominates lower       upper(M) >= lower(M) for all M
    4. conservation                cumulative sums preserve totals
    5. monotonic harvest           bounds are non-decreasing
  and report the endpoint residual (relative error at convergence).

VALIDATION (do outputs match independent data?)
  - SPAM 2020 global production totals vs USDA PSD 2019-2021 mean production
    (fully independent data sources; rice differs by the paddy-vs-milled
    convention, factor ~0.70, which the table footnotes).
  - Historical events must lie inside the H-P envelope by construction;
    the table counts containment for all computed events per crop.

Outputs:
  results/tableS2_verification.tex   (LaTeX, two panels)
  results/tableS2_verification.csv   (machine-readable)
Exit code is non-zero if any verification check fails, so this script can
serve as an end-to-end integration test:  python3 scripts/tableS2_verification.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from agrichter.core.config import Config
from agrichter.data.grid_manager import GridDataManager
from agrichter.data.usda_wide import load_usda_wide
from agrichter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tableS2')

CROPS = ['allgrain', 'wheat', 'maize', 'rice']

# USDA PSD commodity rows corresponding to each SPAM crop selection
USDA_COMMODITY = {
    'allgrain': 'Allgrain',
    'wheat': 'Wheat',
    'maize': 'Corn',
    'rice': 'Rice, Milled',
}

# Countries featured in Figure 3 (SPAM ADM0_NAME)
COUNTRIES = [
    'United States of America', 'China', 'India', 'Brazil',
    'France', 'Egypt', 'Australia', 'Argentina',
]
COUNTRY_DISPLAY = {
    'United States of America': 'USA',
}

REF_YEARS = [2019, 2020, 2021]

INVARIANT_KEYS = [
    'starts_at_origin', 'converges_at_endpoint', 'upper_dominates_lower',
    'conservation_satisfied', 'monotonic_harvest',
]


def endpoint_residual(env, total_production):
    """Relative error between envelope endpoint and true total production."""
    p_up_end = float(np.asarray(env['upper_bound_production'])[-1])
    p_low_end = float(np.asarray(env['lower_bound_production'])[-1])
    return max(abs(p_up_end - total_production), abs(p_low_end - total_production)) / total_production


def check_event_containment(env, events):
    """Count events inside the envelope band (with 1% interpolation slack)."""
    M_up = np.log10(np.asarray(env['upper_bound_harvest'], dtype=float))
    M_low = np.log10(np.asarray(env['lower_bound_harvest'], dtype=float))
    P_up = np.asarray(env['upper_bound_production'], dtype=float)
    P_low = np.asarray(env['lower_bound_production'], dtype=float)

    n_inside = 0
    outside = []
    for _, row in events.iterrows():
        M = row['magnitude_MD']
        L = row['production_loss_kcal']
        up = np.interp(M, M_up, P_up)
        low = np.interp(M, M_low, P_low)
        if low * 0.99 <= L <= up * 1.01:
            n_inside += 1
        else:
            outside.append(row['event_name'])
    if outside:
        logger.warning(f"Events outside envelope: {outside}")
    return n_inside, len(events)


def fmt_check(ok):
    return r'\checkmark' if ok else r'$\times$'


def main():
    all_pass = True
    events_all = pd.read_csv('results/historical_events_computed.csv')
    usda_prod = load_usda_wide('production')

    # ---------- Panel A: global crops ----------
    rows_a = []
    for crop in CROPS:
        logger.info(f"=== Global verification: {crop} ===")
        config = Config(crop_type=crop, root_dir='.', use_dynamic_thresholds=False)
        gm = GridDataManager(config)
        prod_df, harv_df = gm.load_spam_data()

        calc = HPEnvelopeCalculatorV2(config)
        env = calc.calculate_hp_envelope(prod_df, harv_df)

        props = env.get('mathematical_properties', {})
        invariants_ok = all(props.get(k, False) for k in INVARIANT_KEYS)
        n_ok = sum(bool(props.get(k, False)) for k in INVARIANT_KEYS)

        # SPAM totals in mass units for the independent cross-check
        crop_indices = config.get_crop_indices()
        spam_prod_mt = gm.get_crop_production(prod_df, crop_indices, convert_to_kcal=False)
        spam_harv_ha = gm.get_crop_harvest_area(harv_df, crop_indices)

        total_production_kcal, _ = env['convergence_point'][1], env['convergence_point'][0]
        residual = endpoint_residual(env, total_production_kcal)

        usda_row = usda_prod[(usda_prod['Commodity'] == USDA_COMMODITY[crop])
                             & (usda_prod['Country'] == 'World')
                             & (usda_prod['Year'].isin(REF_YEARS))]
        usda_prod_mt = float(usda_row['Value'].mean()) * 1000.0  # TMT -> MT

        events = events_all[(events_all['crop_type'] == crop)
                            & (events_all['harvest_area_km2'] > 0)]
        n_inside, n_events = check_event_containment(env, events)

        crop_pass = invariants_ok and (residual < 1e-3) and (n_inside == n_events)
        all_pass &= crop_pass

        rows_a.append({
            'Crop': crop.capitalize() if crop != 'allgrain' else 'All grains',
            'SPAM_P_Mt': spam_prod_mt / 1e6,
            'USDA_P_Mt': usda_prod_mt / 1e6,
            'Ratio': spam_prod_mt / usda_prod_mt,
            'Harvest_1000km2': spam_harv_ha * 0.01 / 1e3,
            'Invariants': f'{n_ok}/5',
            'Endpoint_residual': residual,
            'Events_inside': f'{n_inside}/{n_events}',
            'PASS': crop_pass,
        })
        logger.info(f"{crop}: invariants {n_ok}/5, residual {residual:.2e}, "
                    f"events {n_inside}/{n_events}, SPAM/USDA {spam_prod_mt/usda_prod_mt:.3f}")

    # ---------- Panel B: countries (allgrain) ----------
    logger.info("=== Country verification (allgrain) ===")
    config = Config(crop_type='allgrain', root_dir='.', use_dynamic_thresholds=False)
    gm = GridDataManager(config)
    prod_df, harv_df = gm.load_spam_data()
    calc = HPEnvelopeCalculatorV2(config)
    crop_indices = config.get_crop_indices()
    global_prod_mt = gm.get_crop_production(prod_df, crop_indices, convert_to_kcal=False)

    rows_b = []
    for adm0 in COUNTRIES:
        mask = prod_df['ADM0_NAME'] == adm0
        if mask.sum() == 0:
            logger.error(f"No SPAM cells for {adm0}")
            all_pass = False
            continue
        c_prod, c_harv = prod_df[mask], harv_df[mask]
        env = calc.calculate_hp_envelope(c_prod, c_harv)
        props = env.get('mathematical_properties', {})
        n_ok = sum(bool(props.get(k, False)) for k in INVARIANT_KEYS)
        residual = endpoint_residual(env, env['convergence_point'][1])

        country_prod_mt = gm.get_crop_production(c_prod, crop_indices, convert_to_kcal=False)
        country_pass = (n_ok == 5) and (residual < 1e-3)
        all_pass &= country_pass

        rows_b.append({
            'Country': COUNTRY_DISPLAY.get(adm0, adm0),
            'Cells': int(mask.sum()),
            'Share_pct': 100.0 * country_prod_mt / global_prod_mt,
            'Invariants': f'{n_ok}/5',
            'Endpoint_residual': residual,
            'PASS': country_pass,
        })
        logger.info(f"{adm0}: invariants {n_ok}/5, residual {residual:.2e}")

    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)

    # ---------- CSV ----------
    out_csv = Path('results/tableS2_verification.csv')
    with open(out_csv, 'w') as f:
        f.write('# Panel A: global crops\n')
        df_a.to_csv(f, index=False)
        f.write('# Panel B: countries (allgrain)\n')
        df_b.to_csv(f, index=False)
    logger.info(f"Saved {out_csv}")

    # ---------- LaTeX ----------
    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'    \centering')
    tex.append(r'    \caption{\textbf{Verification and validation summary.}')
    tex.append(r'    \emph{Panel A (global crops):} runtime verification of the five mathematical')
    tex.append(r'    invariants of each H--P envelope (origin, endpoint convergence, dominance,')
    tex.append(r'    conservation, monotonicity), the endpoint residual')
    tex.append(r'    $\max(|L_{\mathrm{up/low}}(A_{\mathrm{tot}}) - P_{\mathrm{tot}}|)/P_{\mathrm{tot}}$,')
    tex.append(r'    and containment of all computed historical events within the envelope band.')
    tex.append(r'    SPAM 2020 production totals are validated against the independent USDA PSD')
    tex.append(r'    world production (2019--2021 mean). The rice ratio reflects the paddy (SPAM)')
    tex.append(r'    vs.\ milled (USDA) convention (milling rate $\approx$ 0.70); all-grain baskets')
    tex.append(r'    differ slightly in minor-grain composition.')
    tex.append(r'    \emph{Panel B (national envelopes, all grains):} the same invariant checks for')
    tex.append(r'    every country envelope shown in Figure~3 and Figure~S2.}')
    tex.append(r'    \label{tab:S2_verification}')
    tex.append(r'    \vspace{0.3cm}')
    tex.append(r'    \textbf{Panel A: Global crop envelopes}\\[0.15cm]')
    tex.append(r'    \begin{tabular}{lrrrrccc}')
    tex.append(r'        \toprule')
    tex.append(r'        \textbf{Crop} & \textbf{SPAM P} & \textbf{USDA P} & \textbf{Ratio} & \textbf{Harvest} & \textbf{Invariants} & \textbf{Endpoint} & \textbf{Events} \\')
    tex.append(r'         & (Mt) & (Mt) &  & ($10^3$\,km$^2$) & \textbf{passed} & \textbf{residual} & \textbf{inside} \\')
    tex.append(r'        \midrule')
    for _, r in df_a.iterrows():
        tex.append(f"        {r['Crop']} & {r['SPAM_P_Mt']:,.0f} & {r['USDA_P_Mt']:,.0f} & "
                   f"{r['Ratio']:.2f} & {r['Harvest_1000km2']:,.0f} & {r['Invariants']} & "
                   f"$<10^{{{max(int(np.ceil(np.log10(max(r['Endpoint_residual'], 1e-16)))), -16)}}}$ & "
                   f"{r['Events_inside']} \\\\")
    tex.append(r'        \bottomrule')
    tex.append(r'    \end{tabular}')
    tex.append(r'    \\[0.4cm]')
    tex.append(r'    \textbf{Panel B: National envelopes (all grains)}\\[0.15cm]')
    tex.append(r'    \begin{tabular}{lrrcc}')
    tex.append(r'        \toprule')
    tex.append(r'        \textbf{Country} & \textbf{Grid cells} & \textbf{Production share} & \textbf{Invariants passed} & \textbf{Endpoint residual} \\')
    tex.append(r'        \midrule')
    for _, r in df_b.iterrows():
        tex.append(f"        {r['Country']} & {r['Cells']:,} & {r['Share_pct']:.1f}\\% & "
                   f"{r['Invariants']} & "
                   f"$<10^{{{max(int(np.ceil(np.log10(max(r['Endpoint_residual'], 1e-16)))), -16)}}}$ \\\\")
    tex.append(r'        \bottomrule')
    tex.append(r'    \end{tabular}')
    tex.append(r'\end{table}')

    out_tex = Path('results/tableS2_verification.tex')
    out_tex.write_text('\n'.join(tex) + '\n')
    logger.info(f"Saved {out_tex}")

    print('\n================ V&V SUMMARY ================')
    print(df_a.to_string(index=False))
    print()
    print(df_b.to_string(index=False))
    print(f'\nOVERALL: {"PASS" if all_pass else "FAIL"}')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
