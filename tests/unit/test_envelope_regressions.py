"""Regression tests for envelope-builder numerical robustness.

Guards the float32-cumsum bug: SPAM columns are loaded as float32 for memory
efficiency, and a float32 cumulative sum over ~1e6 cells silently drops
increments smaller than ~0.5 once the running total exceeds ~1e7 (24-bit
mantissa), breaking conservation by ~0.04%. build_envelope must promote
inputs to float64.
"""

import numpy as np
import pytest

from agrichter.analysis.envelope_builder import build_envelope
from agrichter.analysis import EnvelopeData


class TestFloat32Conservation:
    def test_float32_inputs_conserve_totals(self):
        """Large float32 arrays must still satisfy exact conservation."""
        rng = np.random.default_rng(7)
        n = 500_000
        # Harvest areas spanning 4 orders of magnitude, totals ~1e7 (the
        # regime where float32 cumsum visibly loses small increments)
        H_ha = rng.lognormal(mean=3.0, sigma=1.5, size=n).astype(np.float32)
        Y = rng.uniform(0.5, 10.0, size=n).astype(np.float32)
        P_mt = (H_ha * Y).astype(np.float32)

        result = build_envelope(P_mt, H_ha)

        total_H_km2 = np.sum(H_ha.astype(np.float64)) * 0.01
        total_P_mt = np.sum(P_mt.astype(np.float64))

        for side in ('lower', 'upper'):
            cum_H_final = result[side]['cum_H_km2'][-1]
            cum_P_final = result[side]['cum_P_mt'][-1]
            assert cum_H_final == pytest.approx(total_H_km2, rel=1e-9), (
                f"{side} envelope H not conserved with float32 inputs"
            )
            assert cum_P_final == pytest.approx(total_P_mt, rel=1e-9), (
                f"{side} envelope P not conserved with float32 inputs"
            )

    def test_outputs_are_float64(self):
        H_ha = np.array([100, 200, 300], dtype=np.float32)
        P_mt = np.array([300, 400, 300], dtype=np.float32)
        result = build_envelope(P_mt, H_ha)
        assert result['lower']['cum_H_km2'].dtype == np.float64
        assert result['upper']['cum_P_mt'].dtype == np.float64


class TestEnvelopeInvariants:
    """Mathematical properties that must hold for any valid input."""

    @pytest.fixture
    def result(self):
        rng = np.random.default_rng(11)
        H_ha = rng.uniform(1, 1000, size=5000)
        Y = rng.uniform(0.1, 12.0, size=5000)
        return build_envelope(H_ha * Y, H_ha)

    def test_monotonic_nondecreasing(self, result):
        for side in ('lower', 'upper'):
            assert np.all(np.diff(result[side]['cum_H_km2']) >= 0)
            assert np.all(np.diff(result[side]['cum_P_mt']) >= 0)

    def test_upper_dominates_lower_at_common_harvest(self, result):
        """P_up(H) >= P_low(H) on a shared harvest grid."""
        H_common = np.linspace(
            max(result['lower']['cum_H_km2'][0], result['upper']['cum_H_km2'][0]),
            min(result['lower']['cum_H_km2'][-1], result['upper']['cum_H_km2'][-1]),
            200,
        )
        P_low = np.interp(H_common, result['lower']['cum_H_km2'],
                          result['lower']['cum_P_mt'])
        P_up = np.interp(H_common, result['upper']['cum_H_km2'],
                         result['upper']['cum_P_mt'])
        assert np.all(P_up >= P_low * (1 - 1e-12))

    def test_endpoints_converge(self, result):
        """Both envelopes reach the same total (H_total, P_total)."""
        assert result['lower']['cum_H_km2'][-1] == pytest.approx(
            result['upper']['cum_H_km2'][-1])
        assert result['lower']['cum_P_mt'][-1] == pytest.approx(
            result['upper']['cum_P_mt'][-1])

    def test_yield_sort_order(self, result):
        assert np.all(np.diff(result['lower']['Y_sorted']) >= 0)
        assert np.all(np.diff(result['upper']['Y_sorted']) <= 0)


class TestEnvelopeDataDataclass:
    def test_importable_from_analysis_package(self):
        assert EnvelopeData is not None

    def test_roundtrip(self):
        d = EnvelopeData(
            disruption_areas=np.array([1.0, 2.0]),
            lower_bound_harvest=np.array([1.0, 2.0]),
            lower_bound_production=np.array([10.0, 20.0]),
            upper_bound_harvest=np.array([1.0, 2.0]),
            upper_bound_production=np.array([30.0, 40.0]),
            convergence_point=(2.0, 40.0),
            convergence_validated=True,
            mathematical_properties={'monotonic': True},
            convergence_statistics={'err': 0.0},
            crop_type='wheat',
        )
        d2 = EnvelopeData.from_dict(d.to_dict())
        assert d2.crop_type == 'wheat'
        assert d2.validate_convergence()
