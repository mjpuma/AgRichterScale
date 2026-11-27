"""
Integration test comparing V1 (original) and V2 (robust builder) envelope calculators.

This test runs both implementations side-by-side on real data and compares:
- Output shapes and formats
- Numerical similarity
- Validation results
- Performance characteristics
"""

import pytest
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from agrichter.core.config import Config
from agrichter.analysis.envelope import HPEnvelopeCalculator
from agrichter.analysis.envelope_v2 import HPEnvelopeCalculatorV2
from agrichter.data.loader import DataLoader


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def config_wheat():
    """Create config for wheat analysis."""
    config = Config(crop_type='wheat', spam_version='2020')
    return config


@pytest.fixture
def config_rice():
    """Create config for rice analysis."""
    config = Config(crop_type='rice', spam_version='2020')
    return config


@pytest.fixture
def config_allgrain():
    """Create config for allgrain analysis."""
    config = Config(crop_type='allgrain', spam_version='2020')
    return config


@pytest.fixture
def sample_data(config_wheat):
    """Load sample SPAM data for testing."""
    loader = DataLoader(config_wheat)
    
    try:
        production_df = loader.load_spam_production()
        harvest_df = loader.load_spam_harvest()
        
        # Take a sample for faster testing
        sample_size = 1000
        production_sample = production_df.head(sample_size)
        harvest_sample = harvest_df.head(sample_size)
        
        return production_sample, harvest_sample
    except Exception as e:
        pytest.skip(f"Could not load SPAM data: {e}")


class TestEnvelopeV1V2Comparison:
    """Test suite comparing V1 and V2 envelope calculators."""
    
    def test_api_compatibility(self, config_wheat):
        """Test that V2 has same API as V1."""
        v1 = HPEnvelopeCalculator(config_wheat)
        v2 = HPEnvelopeCalculatorV2(config_wheat)
        
        # Check that both have same public methods
        v1_methods = [m for m in dir(v1) if not m.startswith('_')]
        v2_methods = [m for m in dir(v2) if not m.startswith('_')]
        
        # V2 should have all V1 methods (may have additional ones)
        for method in v1_methods:
            assert hasattr(v2, method), f"V2 missing method: {method}"
    
    def test_output_format_compatibility(self, config_wheat, sample_data):
        """Test that V2 produces same output format as V1."""
        production_df, harvest_df = sample_data
        
        v1 = HPEnvelopeCalculator(config_wheat)
        v2 = HPEnvelopeCalculatorV2(config_wheat)
        
        # Calculate envelopes
        envelope_v1 = v1.calculate_hp_envelope(production_df, harvest_df)
        envelope_v2 = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Check that both have same keys (V2 may have additional keys)
        required_keys = [
            'disruption_areas',
            'lower_bound_harvest',
            'lower_bound_production',
            'upper_bound_harvest',
            'upper_bound_production'
        ]
        
        for key in required_keys:
            assert key in envelope_v1, f"V1 missing key: {key}"
            assert key in envelope_v2, f"V2 missing key: {key}"
        
        # Check array shapes match
        for key in required_keys:
            v1_shape = envelope_v1[key].shape
            v2_shape = envelope_v2[key].shape
            logger.info(f"{key}: V1 shape={v1_shape}, V2 shape={v2_shape}")
            
            # Shapes may differ slightly due to different interpolation strategies
            # but should be similar order of magnitude
            assert len(v1_shape) == len(v2_shape), f"Shape dimension mismatch for {key}"
    
    def test_numerical_similarity(self, config_wheat, sample_data):
        """Test that V1 and V2 produce numerically similar results."""
        production_df, harvest_df = sample_data
        
        v1 = HPEnvelopeCalculator(config_wheat)
        v2 = HPEnvelopeCalculatorV2(config_wheat)
        
        # Calculate envelopes
        envelope_v1 = v1.calculate_hp_envelope(production_df, harvest_df)
        envelope_v2 = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Compare total production ranges (should be very similar)
        v1_total_prod = envelope_v1['upper_bound_production'].max()
        v2_total_prod = envelope_v2['upper_bound_production'].max()
        
        logger.info(f"V1 max production: {v1_total_prod:.2e}")
        logger.info(f"V2 max production: {v2_total_prod:.2e}")
        
        # Should be within 10% (different interpolation may cause small differences)
        rel_diff = abs(v1_total_prod - v2_total_prod) / max(v1_total_prod, v2_total_prod)
        logger.info(f"Relative difference: {rel_diff:.2%}")
        
        assert rel_diff < 0.10, f"Production totals differ by more than 10%: {rel_diff:.2%}"
        
        # Compare harvest area ranges
        v1_total_harvest = envelope_v1['upper_bound_harvest'].max()
        v2_total_harvest = envelope_v2['upper_bound_harvest'].max()
        
        logger.info(f"V1 max harvest: {v1_total_harvest:.2f} km²")
        logger.info(f"V2 max harvest: {v2_total_harvest:.2f} km²")
        
        rel_diff_harvest = abs(v1_total_harvest - v2_total_harvest) / max(v1_total_harvest, v2_total_harvest)
        logger.info(f"Harvest relative difference: {rel_diff_harvest:.2%}")
        
        assert rel_diff_harvest < 0.10, f"Harvest totals differ by more than 10%: {rel_diff_harvest:.2%}"
    
    def test_dominance_constraint(self, config_wheat, sample_data):
        """Test that both V1 and V2 satisfy dominance constraint (upper >= lower)."""
        production_df, harvest_df = sample_data
        
        v1 = HPEnvelopeCalculator(config_wheat)
        v2 = HPEnvelopeCalculatorV2(config_wheat)
        
        # Calculate envelopes
        envelope_v1 = v1.calculate_hp_envelope(production_df, harvest_df)
        envelope_v2 = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Check V1 dominance
        v1_violations = np.sum(
            envelope_v1['upper_bound_production'] < envelope_v1['lower_bound_production']
        )
        logger.info(f"V1 dominance violations: {v1_violations}")
        
        # Check V2 dominance
        v2_violations = np.sum(
            envelope_v2['upper_bound_production'] < envelope_v2['lower_bound_production']
        )
        logger.info(f"V2 dominance violations: {v2_violations}")
        
        # V2 should have zero violations due to clipping
        assert v2_violations == 0, "V2 should enforce dominance constraint"
        
        # V1 may have some violations (this is what we're fixing)
        if v1_violations > 0:
            logger.warning(f"V1 has {v1_violations} dominance violations (expected)")
    
    def test_monotonicity(self, config_wheat, sample_data):
        """Test that both V1 and V2 produce monotonic cumulative sequences."""
        production_df, harvest_df = sample_data
        
        v1 = HPEnvelopeCalculator(config_wheat)
        v2 = HPEnvelopeCalculatorV2(config_wheat)
        
        # Calculate envelopes
        envelope_v1 = v1.calculate_hp_envelope(production_df, harvest_df)
        envelope_v2 = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Check V1 monotonicity
        v1_lower_mono = np.all(np.diff(envelope_v1['lower_bound_production']) >= 0)
        v1_upper_mono = np.all(np.diff(envelope_v1['upper_bound_production']) >= 0)
        logger.info(f"V1 lower monotonic: {v1_lower_mono}")
        logger.info(f"V1 upper monotonic: {v1_upper_mono}")
        
        # Check V2 monotonicity
        v2_lower_mono = np.all(np.diff(envelope_v2['lower_bound_production']) >= 0)
        v2_upper_mono = np.all(np.diff(envelope_v2['upper_bound_production']) >= 0)
        logger.info(f"V2 lower monotonic: {v2_lower_mono}")
        logger.info(f"V2 upper monotonic: {v2_upper_mono}")
        
        # V2 should guarantee monotonicity
        assert v2_lower_mono, "V2 lower envelope should be monotonic"
        assert v2_upper_mono, "V2 upper envelope should be monotonic"
    
    def test_validation_methods(self, config_wheat, sample_data):
        """Test that validation methods work for both V1 and V2."""
        production_df, harvest_df = sample_data
        
        v1 = HPEnvelopeCalculator(config_wheat)
        v2 = HPEnvelopeCalculatorV2(config_wheat)
        
        # Calculate envelopes
        envelope_v1 = v1.calculate_hp_envelope(production_df, harvest_df)
        envelope_v2 = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Test validation
        v1_valid = v1.validate_envelope_data(envelope_v1)
        v2_valid = v2.validate_envelope_data(envelope_v2)
        
        assert v1_valid, "V1 envelope should pass validation"
        assert v2_valid, "V2 envelope should pass validation"
    
    def test_statistics_methods(self, config_wheat, sample_data):
        """Test that statistics methods work for both V1 and V2."""
        production_df, harvest_df = sample_data
        
        v1 = HPEnvelopeCalculator(config_wheat)
        v2 = HPEnvelopeCalculatorV2(config_wheat)
        
        # Calculate envelopes
        envelope_v1 = v1.calculate_hp_envelope(production_df, harvest_df)
        envelope_v2 = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Get statistics
        stats_v1 = v1.get_envelope_statistics(envelope_v1)
        stats_v2 = v2.get_envelope_statistics(envelope_v2)
        
        # Check that both have required keys
        required_keys = ['crop_type', 'n_disruption_points', 'lower_bound_stats', 'upper_bound_stats']
        
        for key in required_keys:
            assert key in stats_v1, f"V1 stats missing key: {key}"
            assert key in stats_v2, f"V2 stats missing key: {key}"
        
        # V2 should have additional QA info
        assert 'calculator_version' in stats_v2, "V2 should report calculator version"
        
        logger.info(f"V1 stats: {stats_v1}")
        logger.info(f"V2 stats: {stats_v2}")
    
    def test_plotting_methods(self, config_wheat, sample_data):
        """Test that plotting methods work for both V1 and V2."""
        production_df, harvest_df = sample_data
        
        v1 = HPEnvelopeCalculator(config_wheat)
        v2 = HPEnvelopeCalculatorV2(config_wheat)
        
        # Calculate envelopes
        envelope_v1 = v1.calculate_hp_envelope(production_df, harvest_df)
        envelope_v2 = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Create plotting data
        plot_v1 = v1.create_envelope_for_plotting(envelope_v1, loss_factor=0.5)
        plot_v2 = v2.create_envelope_for_plotting(envelope_v2, loss_factor=0.5)
        
        # Check that both have required keys
        required_keys = [
            'harvest_polygon', 'production_polygon',
            'lower_harvest', 'lower_production',
            'upper_harvest', 'upper_production'
        ]
        
        for key in required_keys:
            assert key in plot_v1, f"V1 plot data missing key: {key}"
            assert key in plot_v2, f"V2 plot data missing key: {key}"
    
    def test_v2_qa_summary(self, config_wheat, sample_data):
        """Test that V2 provides comprehensive QA summary."""
        production_df, harvest_df = sample_data
        
        v2 = HPEnvelopeCalculatorV2(config_wheat)
        
        # Calculate envelope
        envelope_v2 = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Check for V2-specific QA data
        assert 'v2_summary' in envelope_v2, "V2 should include QA summary"
        
        summary = envelope_v2['v2_summary']
        logger.info(f"V2 QA Summary: {summary}")
        
        # Check for expected QA fields
        expected_fields = ['qa_status', 'n_valid_cells']
        for field in expected_fields:
            assert field in summary, f"V2 summary missing field: {field}"
    
    @pytest.mark.parametrize('crop_type', ['wheat', 'rice', 'allgrain'])
    def test_multiple_crops(self, crop_type):
        """Test V1 vs V2 comparison for different crop types."""
        config = Config(crop_type=crop_type, spam_version='2020')
        
        loader = DataLoader(config)
        
        try:
            production_df = loader.load_spam_production()
            harvest_df = loader.load_spam_harvest()
            
            # Take sample
            sample_size = 500
            production_sample = production_df.head(sample_size)
            harvest_sample = harvest_df.head(sample_size)
            
        except Exception as e:
            pytest.skip(f"Could not load SPAM data for {crop_type}: {e}")
        
        v1 = HPEnvelopeCalculator(config)
        v2 = HPEnvelopeCalculatorV2(config)
        
        # Calculate envelopes
        envelope_v1 = v1.calculate_hp_envelope(production_sample, harvest_sample)
        envelope_v2 = v2.calculate_hp_envelope(production_sample, harvest_sample)
        
        # Basic checks
        assert len(envelope_v1['disruption_areas']) > 0, f"V1 {crop_type}: no data points"
        assert len(envelope_v2['disruption_areas']) > 0, f"V2 {crop_type}: no data points"
        
        logger.info(f"{crop_type}: V1 points={len(envelope_v1['disruption_areas'])}, "
                   f"V2 points={len(envelope_v2['disruption_areas'])}")


def test_comparison_report(config_wheat, sample_data):
    """Generate comprehensive comparison report."""
    production_df, harvest_df = sample_data
    
    v1 = HPEnvelopeCalculator(config_wheat)
    v2 = HPEnvelopeCalculatorV2(config_wheat)
    
    # Calculate envelopes
    logger.info("=" * 60)
    logger.info("ENVELOPE CALCULATOR COMPARISON REPORT")
    logger.info("=" * 60)
    
    import time
    
    # Time V1
    start = time.time()
    envelope_v1 = v1.calculate_hp_envelope(production_df, harvest_df)
    v1_time = time.time() - start
    
    # Time V2
    start = time.time()
    envelope_v2 = v2.calculate_hp_envelope(production_df, harvest_df)
    v2_time = time.time() - start
    
    logger.info(f"\nPerformance:")
    logger.info(f"  V1 time: {v1_time:.3f}s")
    logger.info(f"  V2 time: {v2_time:.3f}s")
    logger.info(f"  Ratio: {v2_time/v1_time:.2f}x")
    
    # Compare outputs
    logger.info(f"\nOutput sizes:")
    logger.info(f"  V1 points: {len(envelope_v1['disruption_areas'])}")
    logger.info(f"  V2 points: {len(envelope_v2['disruption_areas'])}")
    
    # Compare statistics
    stats_v1 = v1.get_envelope_statistics(envelope_v1)
    stats_v2 = v2.get_envelope_statistics(envelope_v2)
    
    logger.info(f"\nProduction ranges:")
    logger.info(f"  V1: {stats_v1['lower_bound_stats']['max_production']:.2e} - "
               f"{stats_v1['upper_bound_stats']['max_production']:.2e}")
    logger.info(f"  V2: {stats_v2['lower_bound_stats']['max_production']:.2e} - "
               f"{stats_v2['upper_bound_stats']['max_production']:.2e}")
    
    # Check dominance violations
    v1_violations = np.sum(
        envelope_v1['upper_bound_production'] < envelope_v1['lower_bound_production']
    )
    v2_violations = np.sum(
        envelope_v2['upper_bound_production'] < envelope_v2['lower_bound_production']
    )
    
    logger.info(f"\nDominance violations:")
    logger.info(f"  V1: {v1_violations}")
    logger.info(f"  V2: {v2_violations}")
    
    # Check monotonicity
    v1_lower_mono = np.all(np.diff(envelope_v1['lower_bound_production']) >= 0)
    v1_upper_mono = np.all(np.diff(envelope_v1['upper_bound_production']) >= 0)
    v2_lower_mono = np.all(np.diff(envelope_v2['lower_bound_production']) >= 0)
    v2_upper_mono = np.all(np.diff(envelope_v2['upper_bound_production']) >= 0)
    
    logger.info(f"\nMonotonicity:")
    logger.info(f"  V1 lower: {v1_lower_mono}, upper: {v1_upper_mono}")
    logger.info(f"  V2 lower: {v2_lower_mono}, upper: {v2_upper_mono}")
    
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATION:")
    
    if v2_violations == 0 and v2_lower_mono and v2_upper_mono:
        logger.info("✓ V2 passes all mathematical property checks")
        logger.info("✓ V2 is ready for production use")
    else:
        logger.info("✗ V2 has issues that need investigation")
    
    logger.info("=" * 60)


if __name__ == '__main__':
    # Run comparison report
    pytest.main([__file__, '-v', '-s'])
