"""Unit tests for unit conversions used throughout the AgRichter pipeline.

These tests guard against the two unit-bug classes found during development:
1. Treating USDA PSD values (1000 MT) as plain MT (thresholds 1000x too low).
2. Applying a blended caloric factor to a multi-crop sum instead of
   converting each crop with its own factor.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from agrichter.core.config import Config
from agrichter.core.constants import (
    CALORIC_CONTENT,
    CROP_INDICES,
    GRAMS_PER_METRIC_TON,
    HECTARES_TO_KM2,
)
from agrichter.data.grid_manager import GridDataManager
from agrichter.analysis.event_calculator import EventCalculator


class TestConstants:
    def test_grams_per_metric_ton(self):
        assert GRAMS_PER_METRIC_TON == 1_000_000.0

    def test_hectares_to_km2(self):
        assert HECTARES_TO_KM2 == 0.01

    def test_caloric_content_plausible(self):
        """All grain caloric densities must be in the physical 3-4 kcal/g range."""
        for crop, kcal_per_g in CALORIC_CONTENT.items():
            assert 3.0 <= kcal_per_g <= 4.0, f"{crop}: {kcal_per_g} kcal/g implausible"

    def test_allgrain_blend_within_component_range(self):
        components = [CALORIC_CONTENT[c] for c in
                      ('Wheat', 'Rice', 'Corn', 'Barley', 'Millet', 'Sorghum', 'MixedGrain')]
        assert min(components) <= CALORIC_CONTENT['Allgrain'] <= max(components)


class TestConfigDefaults:
    def test_default_usda_year_range_matches_spam2020(self):
        config = Config(crop_type='allgrain', root_dir='.', use_dynamic_thresholds=False)
        assert config.usda_year_range == (2019, 2021)

    def test_unit_conversions_exposed(self):
        config = Config(crop_type='wheat', root_dir='.', use_dynamic_thresholds=False)
        conv = config.get_unit_conversions()
        assert conv['grams_per_metric_ton'] == 1_000_000.0
        assert conv['hectares_to_km2'] == 0.01


class TestDynamicThresholds:
    """Regression test for the 1000x USDA PSD unit bug.

    Global annual grain consumption is ~2.5e6 TMT (2.5 Gt). In kcal:
    2.5e6 TMT * 1e9 g/TMT * 3.45 kcal/g ~ 8.6e15 kcal/yr, so
    1 month ~ 7e14 kcal. Under the old bug these came out ~1000x lower.
    """

    @pytest.fixture(scope='class')
    def thresholds(self):
        config = Config(crop_type='allgrain', root_dir='.', use_dynamic_thresholds=True)
        return config.thresholds

    def test_threshold_names(self, thresholds):
        assert set(thresholds.keys()) == {'1 Month', '3 Months', 'Total Stocks'}

    def test_threshold_magnitudes(self, thresholds):
        assert 1e14 < thresholds['1 Month'] < 5e15
        assert 5e14 < thresholds['3 Months'] < 1e16
        assert 5e14 < thresholds['Total Stocks'] < 1e16

    def test_threshold_ordering(self, thresholds):
        assert thresholds['1 Month'] < thresholds['3 Months']
        # 3-month buffer is exactly 3x the 1-month buffer
        assert thresholds['3 Months'] == pytest.approx(3 * thresholds['1 Month'])


class TestMagnitudeFormula:
    """M_D = log10(area in km2), computed from hectares."""

    @pytest.fixture
    def calculator(self):
        config = Config(crop_type='allgrain', root_dir='.', use_dynamic_thresholds=False)
        return EventCalculator(config, MagicMock(), MagicMock())

    def test_100_ha_is_magnitude_zero(self, calculator):
        # 100 ha = 1 km2 -> M = 0
        assert calculator.calculate_magnitude(100.0) == pytest.approx(0.0)

    def test_one_million_ha_is_magnitude_four(self, calculator):
        # 1e6 ha = 1e4 km2 -> M = 4
        assert calculator.calculate_magnitude(1e6) == pytest.approx(4.0)

    def test_zero_area_is_nan(self, calculator):
        assert np.isnan(calculator.calculate_magnitude(0.0))

    def test_negative_area_is_nan(self, calculator):
        assert np.isnan(calculator.calculate_magnitude(-10.0))


class TestPerCropKcalConversion:
    """get_crop_production must convert each crop with its own caloric factor."""

    @pytest.fixture
    def grid_manager(self):
        config = Config(crop_type='allgrain', root_dir='.', use_dynamic_thresholds=False)
        return GridDataManager(config)

    def test_single_crop_wheat(self, grid_manager):
        cells = pd.DataFrame({'WHEA_A': [10.0, 20.0]})  # 30 MT wheat
        kcal = grid_manager.get_crop_production(cells, CROP_INDICES['wheat'],
                                                convert_to_kcal=True)
        expected = 30.0 * 1e6 * CALORIC_CONTENT['Wheat']
        assert kcal == pytest.approx(expected)

    def test_multi_crop_uses_per_crop_factors(self, grid_manager):
        cells = pd.DataFrame({'WHEA_A': [100.0], 'RICE_A': [50.0], 'MAIZ_A': [25.0]})
        kcal = grid_manager.get_crop_production(cells, CROP_INDICES['allgrain'],
                                                convert_to_kcal=True)
        expected = 1e6 * (100.0 * CALORIC_CONTENT['Wheat']
                          + 50.0 * CALORIC_CONTENT['Rice']
                          + 25.0 * CALORIC_CONTENT['Corn'])
        assert kcal == pytest.approx(expected)
        # And it must NOT equal the blended-factor result
        blended = 175.0 * 1e6 * CALORIC_CONTENT['Allgrain']
        assert kcal != pytest.approx(blended, rel=1e-6)

    def test_metric_ton_path_is_plain_sum(self, grid_manager):
        cells = pd.DataFrame({'WHEA_A': [100.0], 'RICE_A': [50.0]})
        mt = grid_manager.get_crop_production(cells, CROP_INDICES['allgrain'],
                                              convert_to_kcal=False)
        assert mt == pytest.approx(150.0)


class TestUsdaKcalFactor:
    """USDA PSD data is in 1000 MT: kcal = TMT * 1e9 * kcal/g."""

    def test_tmt_to_kcal_factor(self):
        config = Config(crop_type='allgrain', root_dir='.', use_dynamic_thresholds=False)
        kcal_per_tmt = 1e9 * config.get_caloric_content()
        # 1 TMT of allgrain = 1e9 g * 3.45 kcal/g = 3.45e9 kcal
        assert kcal_per_tmt == pytest.approx(3.45e9)
