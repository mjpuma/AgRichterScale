"""
Unit tests for envelope builder export functionality (TASK 7).

Tests cover:
- 7.1: Save envelope curves (interpolated for plotting)
- 7.2: Save discrete cumulative sequences (for validation)
- 7.3: Save summary and QA report
- 7.4: Generate visualization with discrete points visible
"""

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path
import shutil

from agrichter.analysis.envelope_builder import build_envelope, export_envelope_results


@pytest.fixture
def synthetic_data():
    """Create synthetic test data."""
    np.random.seed(42)
    n_cells = 200
    
    Y_mt_per_ha = np.random.uniform(1.0, 10.0, n_cells)
    H_ha = np.random.uniform(100, 1000, n_cells)
    P_mt = Y_mt_per_ha * H_ha
    
    return {
        'P_mt': P_mt,
        'H_ha': H_ha,
        'Y_mt_per_ha': Y_mt_per_ha,
        'n_cells': n_cells
    }


@pytest.fixture
def envelope_result_with_interpolation(synthetic_data):
    """Build envelope with interpolation."""
    return build_envelope(
        P_mt=synthetic_data['P_mt'],
        H_ha=synthetic_data['H_ha'],
        Y_mt_per_ha=synthetic_data['Y_mt_per_ha'],
        interpolate=True,
        n_points=200
    )


@pytest.fixture
def envelope_result_without_interpolation(synthetic_data):
    """Build envelope without interpolation."""
    return build_envelope(
        P_mt=synthetic_data['P_mt'],
        H_ha=synthetic_data['H_ha'],
        interpolate=False
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "test_export"
    output_dir.mkdir()
    return output_dir


class TestTask71InterpolatedEnvelopeCurves:
    """Tests for TASK 7.1: Save envelope curves (interpolated for plotting)."""
    
    def test_lower_envelope_csv_created(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that lower envelope CSV is created with correct format."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        assert 'lower_csv' in output_paths
        assert output_paths['lower_csv'].exists()
    
    def test_lower_envelope_csv_format(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that lower envelope CSV has correct format and columns."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        # Read CSV
        df = pd.read_csv(output_paths['lower_csv'], comment='#')
        
        # Check columns
        assert 'Hq_km2' in df.columns
        assert 'lower_P_mt' in df.columns
        assert len(df.columns) == 2
        
        # Check number of points
        assert len(df) == 200
        
        # Check values are non-negative
        assert (df['Hq_km2'] >= 0).all()
        assert (df['lower_P_mt'] >= 0).all()
        
        # Check monotonicity
        assert (df['Hq_km2'].diff().dropna() > 0).all()
        assert (df['lower_P_mt'].diff().dropna() >= 0).all()
    
    def test_lower_envelope_csv_header(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that lower envelope CSV has correct header comments."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        with open(output_paths['lower_csv'], 'r') as f:
            lines = [f.readline() for _ in range(3)]
        
        assert lines[0].startswith("# Interpolated envelope")
        assert "Lower envelope" in lines[1]
        assert "Columns:" in lines[2]
    
    def test_upper_envelope_csv_created(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that upper envelope CSV is created with correct format."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        assert 'upper_csv' in output_paths
        assert output_paths['upper_csv'].exists()
    
    def test_upper_envelope_csv_format(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that upper envelope CSV has correct format and columns."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        # Read CSV
        df = pd.read_csv(output_paths['upper_csv'], comment='#')
        
        # Check columns
        assert 'Hq_km2' in df.columns
        assert 'upper_P_mt' in df.columns
        assert len(df.columns) == 2
        
        # Check number of points
        assert len(df) == 200
        
        # Check values are non-negative
        assert (df['Hq_km2'] >= 0).all()
        assert (df['upper_P_mt'] >= 0).all()
        
        # Check monotonicity
        assert (df['Hq_km2'].diff().dropna() > 0).all()
        assert (df['upper_P_mt'].diff().dropna() >= 0).all()
    
    def test_dominance_in_exported_curves(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that upper >= lower in exported curves."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        df_lower = pd.read_csv(output_paths['lower_csv'], comment='#')
        df_upper = pd.read_csv(output_paths['upper_csv'], comment='#')
        
        # Check dominance
        assert (df_upper['upper_P_mt'] >= df_lower['lower_P_mt']).all()


class TestTask72DiscreteSequences:
    """Tests for TASK 7.2: Save discrete cumulative sequences (for validation)."""
    
    def test_lower_discrete_csv_created(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that lower discrete CSV is created."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        assert 'lower_discrete_csv' in output_paths
        assert output_paths['lower_discrete_csv'].exists()
    
    def test_lower_discrete_csv_format(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that lower discrete CSV has correct format and columns."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        # Read CSV
        df = pd.read_csv(output_paths['lower_discrete_csv'], comment='#')
        
        # Check columns
        assert 'cum_H_km2' in df.columns
        assert 'cum_P_mt' in df.columns
        assert 'cell_index' in df.columns
        assert len(df.columns) == 3
        
        # Check number of points matches valid cells
        n_valid = envelope_result_with_interpolation['summary']['n_valid']
        assert len(df) == n_valid
        
        # Check monotonicity
        assert (df['cum_H_km2'].diff().dropna() > 0).all()
        assert (df['cum_P_mt'].diff().dropna() >= 0).all()
    
    def test_lower_discrete_csv_header(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that lower discrete CSV has correct header comments."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        with open(output_paths['lower_discrete_csv'], 'r') as f:
            lines = [f.readline() for _ in range(3)]
        
        assert lines[0].startswith("# Discrete cumulative sums")
        assert "no interpolation" in lines[0]
        assert "Lower envelope" in lines[1]
    
    def test_upper_discrete_csv_created(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that upper discrete CSV is created."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        assert 'upper_discrete_csv' in output_paths
        assert output_paths['upper_discrete_csv'].exists()
    
    def test_discrete_sequences_conservation(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that discrete sequences conserve totals."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        df_lower = pd.read_csv(output_paths['lower_discrete_csv'], comment='#')
        df_upper = pd.read_csv(output_paths['upper_discrete_csv'], comment='#')
        
        summary = envelope_result_with_interpolation['summary']
        
        # Check final cumulative values match totals
        assert np.isclose(df_lower['cum_H_km2'].iloc[-1], summary['totals']['H_km2'], rtol=1e-6)
        assert np.isclose(df_lower['cum_P_mt'].iloc[-1], summary['totals']['P_mt'], rtol=1e-6)
        assert np.isclose(df_upper['cum_H_km2'].iloc[-1], summary['totals']['H_km2'], rtol=1e-6)
        assert np.isclose(df_upper['cum_P_mt'].iloc[-1], summary['totals']['P_mt'], rtol=1e-6)


class TestTask73SummaryAndQA:
    """Tests for TASK 7.3: Save summary and QA report."""
    
    def test_summary_json_created(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that summary JSON is created."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        assert 'summary_json' in output_paths
        assert output_paths['summary_json'].exists()
    
    def test_summary_json_structure(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that summary JSON has correct structure."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        with open(output_paths['summary_json'], 'r') as f:
            data = json.load(f)
        
        # Check top-level structure
        assert 'metadata' in data
        assert 'summary' in data
        
        # Check metadata
        metadata = data['metadata']
        assert 'output_dir' in metadata
        assert 'prefix' in metadata
        assert 'interpolated' in metadata
        
        # Check summary sections
        summary = data['summary']
        required_sections = [
            'n_total', 'n_valid', 'n_dropped',
            'dropped_counts', 'yield_validation',
            'totals', 'yield_stats',
            'conservation_checks', 'discrete_qa_results',
            'interpolation_qa', 'warnings', 'all_checks_passed'
        ]
        
        for section in required_sections:
            assert section in summary, f"Missing section: {section}"
    
    def test_summary_json_conservation_checks(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that summary JSON includes conservation checks."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        with open(output_paths['summary_json'], 'r') as f:
            data = json.load(f)
        
        conservation_checks = data['summary']['conservation_checks']
        
        # Check that all conservation checks are present
        assert 'unit_conversion' in conservation_checks
        assert 'lower_H_conservation' in conservation_checks
        assert 'lower_P_conservation' in conservation_checks
        assert 'upper_H_conservation' in conservation_checks
        assert 'upper_P_conservation' in conservation_checks
        
        # Check that all passed
        for check_name, check_data in conservation_checks.items():
            assert check_data['passed'], f"Conservation check failed: {check_name}"
    
    def test_summary_json_qa_results(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that summary JSON includes QA validation results."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        with open(output_paths['summary_json'], 'r') as f:
            data = json.load(f)
        
        # Check discrete QA results
        discrete_qa = data['summary']['discrete_qa_results']
        assert 'dominance_holds' in discrete_qa
        assert discrete_qa['dominance_holds'] is True
        
        # Check interpolation QA results
        interp_qa = data['summary']['interpolation_qa']
        assert interp_qa is not None
        assert 'Hq_strictly_increasing' in interp_qa
        assert 'P_low_monotonic' in interp_qa
        assert 'P_up_monotonic' in interp_qa
        assert 'dominance_after_clipping' in interp_qa


class TestTask74Visualization:
    """Tests for TASK 7.4: Generate visualization with discrete points visible."""
    
    def test_plot_created(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that plot is created."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        assert 'plot_png' in output_paths
        assert output_paths['plot_png'].exists()
    
    def test_plot_file_size(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that plot file has reasonable size."""
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        file_size = output_paths['plot_png'].stat().st_size
        
        # Should be at least 10 KB (non-trivial plot)
        assert file_size > 10000, f"Plot file too small: {file_size} bytes"
        
        # Should be less than 1 MB (reasonable size)
        assert file_size < 1_000_000, f"Plot file too large: {file_size} bytes"


class TestExportWithoutInterpolation:
    """Tests for export when interpolation is not performed."""
    
    def test_no_interpolated_files_created(self, envelope_result_without_interpolation, temp_output_dir):
        """Test that interpolated files are not created when interpolation is disabled."""
        output_paths = export_envelope_results(
            envelope_result_without_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        # Should NOT have interpolated curves or plot
        assert 'lower_csv' not in output_paths
        assert 'upper_csv' not in output_paths
        assert 'plot_png' not in output_paths
    
    def test_discrete_files_still_created(self, envelope_result_without_interpolation, temp_output_dir):
        """Test that discrete files are still created when interpolation is disabled."""
        output_paths = export_envelope_results(
            envelope_result_without_interpolation,
            temp_output_dir,
            prefix="test"
        )
        
        # Should still have discrete sequences and summary
        assert 'lower_discrete_csv' in output_paths
        assert 'upper_discrete_csv' in output_paths
        assert 'summary_json' in output_paths
        
        # Verify files exist
        assert output_paths['lower_discrete_csv'].exists()
        assert output_paths['upper_discrete_csv'].exists()
        assert output_paths['summary_json'].exists()


class TestExportWithCustomPrefix:
    """Tests for export with custom prefix."""
    
    def test_custom_prefix_in_filenames(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that custom prefix is used in filenames."""
        custom_prefix = "wheat_2023"
        
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix=custom_prefix
        )
        
        # Check that all files have custom prefix
        for file_type, file_path in output_paths.items():
            assert file_path.name.startswith(custom_prefix), \
                f"File {file_path.name} does not start with prefix {custom_prefix}"
    
    def test_custom_prefix_in_metadata(self, envelope_result_with_interpolation, temp_output_dir):
        """Test that custom prefix is recorded in metadata."""
        custom_prefix = "rice_2024"
        
        output_paths = export_envelope_results(
            envelope_result_with_interpolation,
            temp_output_dir,
            prefix=custom_prefix
        )
        
        with open(output_paths['summary_json'], 'r') as f:
            data = json.load(f)
        
        assert data['metadata']['prefix'] == custom_prefix
