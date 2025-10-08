"""Unit tests for PerformanceMonitor."""

import pytest
import time
from pathlib import Path
from agririchter.core.performance import PerformanceMonitor


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor class."""
    
    def test_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()
        
        assert monitor.metrics == {}
        assert monitor.current_stage is None
        assert monitor.stage_start_time is None
        assert monitor.pipeline_start_time is None
    
    def test_start_pipeline(self):
        """Test starting pipeline monitoring."""
        monitor = PerformanceMonitor()
        monitor.start_pipeline()
        
        assert monitor.pipeline_start_time is not None
        assert isinstance(monitor.pipeline_start_time, float)
    
    def test_start_end_stage(self):
        """Test starting and ending a stage."""
        monitor = PerformanceMonitor()
        
        # Start a stage
        monitor.start_stage("test_stage")
        assert monitor.current_stage == "test_stage"
        assert monitor.stage_start_time is not None
        assert monitor.stage_start_memory is not None
        
        # Wait a bit
        time.sleep(0.1)
        
        # End the stage
        metrics = monitor.end_stage()
        
        assert monitor.current_stage is None
        assert metrics['stage_name'] == "test_stage"
        assert metrics['elapsed_time_seconds'] >= 0.1
        assert 'start_memory_mb' in metrics
        assert 'end_memory_mb' in metrics
        assert 'memory_delta_mb' in metrics
    
    def test_monitor_stage_context_manager(self):
        """Test using monitor_stage as context manager."""
        monitor = PerformanceMonitor()
        
        with monitor.monitor_stage("context_test"):
            time.sleep(0.1)
            assert monitor.current_stage == "context_test"
        
        # After context, stage should be ended
        assert monitor.current_stage is None
        assert "context_test" in monitor.metrics
        assert monitor.metrics["context_test"]['elapsed_time_seconds'] >= 0.1
    
    def test_log_memory_usage(self):
        """Test logging memory usage."""
        monitor = PerformanceMonitor()
        
        memory_mb = monitor.log_memory_usage("Test")
        
        assert isinstance(memory_mb, float)
        assert memory_mb > 0
    
    def test_get_total_pipeline_time(self):
        """Test getting total pipeline time."""
        monitor = PerformanceMonitor()
        
        # Before starting
        assert monitor.get_total_pipeline_time() == 0.0
        
        # After starting
        monitor.start_pipeline()
        time.sleep(0.1)
        
        total_time = monitor.get_total_pipeline_time()
        assert total_time >= 0.1
    
    def test_get_stage_metrics(self):
        """Test getting metrics for a specific stage."""
        monitor = PerformanceMonitor()
        
        # Non-existent stage
        assert monitor.get_stage_metrics("nonexistent") is None
        
        # Create a stage
        with monitor.monitor_stage("test_stage"):
            time.sleep(0.05)
        
        # Get metrics
        metrics = monitor.get_stage_metrics("test_stage")
        assert metrics is not None
        assert metrics['stage_name'] == "test_stage"
        assert metrics['elapsed_time_seconds'] >= 0.05
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        monitor = PerformanceMonitor()
        
        # Create multiple stages
        with monitor.monitor_stage("stage1"):
            time.sleep(0.05)
        
        with monitor.monitor_stage("stage2"):
            time.sleep(0.05)
        
        # Get all metrics
        all_metrics = monitor.get_all_metrics()
        
        assert len(all_metrics) == 2
        assert "stage1" in all_metrics
        assert "stage2" in all_metrics
    
    def test_generate_performance_report(self):
        """Test generating performance report."""
        monitor = PerformanceMonitor()
        monitor.start_pipeline()
        
        # Create some stages
        with monitor.monitor_stage("data_loading"):
            time.sleep(0.05)
        
        with monitor.monitor_stage("processing"):
            time.sleep(0.05)
        
        # Generate report
        report = monitor.generate_performance_report()
        
        assert isinstance(report, str)
        assert "PERFORMANCE REPORT" in report
        assert "Total Pipeline Time" in report
        assert "STAGE BREAKDOWN" in report
        assert "data_loading" in report
        assert "processing" in report
        assert "SUMMARY STATISTICS" in report
    
    def test_save_report(self, tmp_path):
        """Test saving performance report to file."""
        monitor = PerformanceMonitor()
        monitor.start_pipeline()
        
        with monitor.monitor_stage("test_stage"):
            time.sleep(0.05)
        
        # Save report
        report_path = tmp_path / "performance_report.txt"
        monitor.save_report(str(report_path))
        
        # Verify file exists and contains expected content
        assert report_path.exists()
        
        with open(report_path, 'r') as f:
            content = f.read()
        
        assert "PERFORMANCE REPORT" in content
        assert "test_stage" in content
    
    def test_reset(self):
        """Test resetting the monitor."""
        monitor = PerformanceMonitor()
        monitor.start_pipeline()
        
        with monitor.monitor_stage("test_stage"):
            time.sleep(0.05)
        
        # Verify data exists
        assert len(monitor.metrics) > 0
        assert monitor.pipeline_start_time is not None
        
        # Reset
        monitor.reset()
        
        # Verify everything is cleared
        assert len(monitor.metrics) == 0
        assert monitor.current_stage is None
        assert monitor.stage_start_time is None
        assert monitor.pipeline_start_time is None
    
    def test_nested_stage_warning(self):
        """Test that starting a new stage while one is active logs a warning."""
        monitor = PerformanceMonitor()
        
        monitor.start_stage("stage1")
        
        # Starting another stage should end the first one
        monitor.start_stage("stage2")
        
        # stage1 should be in metrics (it was auto-ended)
        assert "stage1" in monitor.metrics
        assert monitor.current_stage == "stage2"
    
    def test_performance_assessment(self):
        """Test performance assessment in report."""
        monitor = PerformanceMonitor()
        monitor.start_pipeline()
        
        # Simulate fast pipeline (< 10 minutes)
        with monitor.monitor_stage("fast_stage"):
            time.sleep(0.05)
        
        report = monitor.generate_performance_report()
        
        # Should indicate success
        assert "within target time" in report or "< 10 minutes" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
