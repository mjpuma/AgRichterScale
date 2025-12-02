"""
Basic integration test for HPEnvelopeCalculatorV2 with synthetic data.

This test verifies V2 works correctly without requiring SPAM data files.
"""

import pytest
import numpy as np
import pandas as pd
import logging

from agririchter.core.config import Config
from agririchter.analysis.envelope_v2 import HPEnvelopeCalculatorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_spam_data(n_cells=100, n_crops=3):
    """Create synthetic SPAM-like data for testing."""
    # Create production data (metric tons)
    production_data = {}
    for i in range(n_crops):
        crop_name = f'CROP{i}_A'
        production_data[crop_name] = np.random.uniform(10, 1000, n_cells)
    
    production_df = pd.DataFrame(production_data)
    
    # Create harvest data (hectares)
    harvest_data = {}
    for i in range(n_crops):
        crop_name = f'CROP{i}_A'
        harvest_data[crop_name] = np.random.uniform(10, 500, n_cells)
    
    harvest_df = pd.DataFrame(harvest_data)
    
    return production_df, harvest_df


class TestHPEnvelopeCalculatorV2Basic:
    """Basic tests for V2 calculator with synthetic data."""
    
    def test_v2_initialization(self):
        """Test that V2 can be initialized."""
        config = Config(crop_type='wheat', spam_version='2020')
        v2 = HPEnvelopeCalculatorV2(config)
        
        assert v2 is not None
        assert v2.config.crop_type == 'wheat'
        assert hasattr(v2, 'calculate_hp_envelope')
    
    def test_v2_with_synthetic_data(self):
        """Test V2 calculation with synthetic data."""
        config = Config(crop_type='wheat', spam_version='2020')
        v2 = HPEnvelopeCalculatorV2(config)
        
        # Create synthetic data
        production_df, harvest_df = create_synthetic_spam_data(n_cells=100, n_crops=1)
        
        # Rename to wheat crop
        production_df.columns = ['WHEA_A']
        harvest_df.columns = ['WHEA_A']
        
        # Calculate envelope
        envelope = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Verify output structure
        assert 'disruption_areas' in envelope
        assert 'lower_bound_harvest' in envelope
        assert 'lower_bound_production' in envelope
        assert 'upper_bound_harvest' in envelope
        assert 'upper_bound_production' in envelope
        assert 'v2_summary' in envelope
        
        # Verify arrays are not empty
        assert len(envelope['disruption_areas']) > 0
        assert len(envelope['lower_bound_harvest']) > 0
        
        logger.info(f"V2 produced {len(envelope['disruption_areas'])} data points")
    
    def test_v2_dominance_constraint(self):
        """Test that V2 enforces dominance constraint."""
        config = Config(crop_type='wheat', spam_version='2020')
        v2 = HPEnvelopeCalculatorV2(config)
        
        # Create synthetic data
        production_df, harvest_df = create_synthetic_spam_data(n_cells=100, n_crops=1)
        production_df.columns = ['WHEA_A']
        harvest_df.columns = ['WHEA_A']
        
        # Calculate envelope
        envelope = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Check dominance: upper >= lower at all points
        violations = np.sum(
            envelope['upper_bound_production'] < envelope['lower_bound_production']
        )
        
        assert violations == 0, f"V2 should enforce dominance, found {violations} violations"
        logger.info("✓ V2 enforces dominance constraint (0 violations)")
    
    def test_v2_monotonicity(self):
        """Test that V2 produces monotonic sequences."""
        config = Config(crop_type='wheat', spam_version='2020')
        v2 = HPEnvelopeCalculatorV2(config)
        
        # Create synthetic data
        production_df, harvest_df = create_synthetic_spam_data(n_cells=100, n_crops=1)
        production_df.columns = ['WHEA_A']
        harvest_df.columns = ['WHEA_A']
        
        # Calculate envelope
        envelope = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Check monotonicity
        lower_mono = np.all(np.diff(envelope['lower_bound_production']) >= 0)
        upper_mono = np.all(np.diff(envelope['upper_bound_production']) >= 0)
        
        assert lower_mono, "Lower envelope should be monotonic"
        assert upper_mono, "Upper envelope should be monotonic"
        logger.info("✓ V2 produces monotonic sequences")
    
    def test_v2_validation_method(self):
        """Test V2 validation method."""
        config = Config(crop_type='wheat', spam_version='2020')
        v2 = HPEnvelopeCalculatorV2(config)
        
        # Create synthetic data
        production_df, harvest_df = create_synthetic_spam_data(n_cells=100, n_crops=1)
        production_df.columns = ['WHEA_A']
        harvest_df.columns = ['WHEA_A']
        
        # Calculate envelope
        envelope = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Validate
        is_valid = v2.validate_envelope_data(envelope)
        
        assert is_valid, "V2 envelope should pass validation"
        logger.info("✓ V2 envelope passes validation")
    
    def test_v2_statistics_method(self):
        """Test V2 statistics method."""
        config = Config(crop_type='wheat', spam_version='2020')
        v2 = HPEnvelopeCalculatorV2(config)
        
        # Create synthetic data
        production_df, harvest_df = create_synthetic_spam_data(n_cells=100, n_crops=1)
        production_df.columns = ['WHEA_A']
        harvest_df.columns = ['WHEA_A']
        
        # Calculate envelope
        envelope = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Get statistics
        stats = v2.get_envelope_statistics(envelope)
        
        # Verify statistics structure
        assert 'crop_type' in stats
        assert 'calculator_version' in stats
        assert 'n_disruption_points' in stats
        assert 'lower_bound_stats' in stats
        assert 'upper_bound_stats' in stats
        assert 'v2_qa_summary' in stats
        
        assert stats['calculator_version'] == 'V2 (robust builder)'
        logger.info(f"✓ V2 statistics: {stats['n_disruption_points']} points")
    
    def test_v2_plotting_method(self):
        """Test V2 plotting method."""
        config = Config(crop_type='wheat', spam_version='2020')
        v2 = HPEnvelopeCalculatorV2(config)
        
        # Create synthetic data
        production_df, harvest_df = create_synthetic_spam_data(n_cells=100, n_crops=1)
        production_df.columns = ['WHEA_A']
        harvest_df.columns = ['WHEA_A']
        
        # Calculate envelope
        envelope = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Create plotting data
        plot_data = v2.create_envelope_for_plotting(envelope, loss_factor=0.5)
        
        # Verify plotting data structure
        required_keys = [
            'harvest_polygon', 'production_polygon',
            'lower_harvest', 'lower_production',
            'upper_harvest', 'upper_production'
        ]
        
        for key in required_keys:
            assert key in plot_data, f"Missing key: {key}"
        
        logger.info("✓ V2 plotting data created successfully")
    
    def test_v2_qa_summary(self):
        """Test that V2 includes comprehensive QA summary."""
        config = Config(crop_type='wheat', spam_version='2020')
        v2 = HPEnvelopeCalculatorV2(config)
        
        # Create synthetic data
        production_df, harvest_df = create_synthetic_spam_data(n_cells=100, n_crops=1)
        production_df.columns = ['WHEA_A']
        harvest_df.columns = ['WHEA_A']
        
        # Calculate envelope
        envelope = v2.calculate_hp_envelope(production_df, harvest_df)
        
        # Check QA summary exists
        assert 'v2_summary' in envelope, "V2 should include QA summary"
        summary = envelope['v2_summary']
        
        # Verify summary is a dict with content
        assert isinstance(summary, dict), "Summary should be a dictionary"
        assert len(summary) > 0, "Summary should not be empty"
        
        # Log summary info
        logger.info(f"✓ V2 QA Summary keys: {list(summary.keys())}")
        
        # Check for key QA indicators
        if 'all_checks_passed' in summary:
            logger.info(f"✓ V2 All Checks Passed: {summary['all_checks_passed']}")
            assert summary['all_checks_passed'], "All QA checks should pass"
        
        # Verify summary has useful information
        assert 'dropped_counts' in summary or 'validation_result' in summary, \
            "Summary should include validation information"
        
        logger.info("✓ V2 includes comprehensive QA summary")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
