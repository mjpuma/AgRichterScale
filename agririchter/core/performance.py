"""Performance monitoring utilities for AgriRichter pipeline."""

import logging
import time
import psutil
from typing import Dict, Any, Optional
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor and track performance metrics for pipeline stages.
    
    Tracks timing, memory usage, and provides performance reports.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.current_stage: Optional[str] = None
        self.stage_start_time: Optional[float] = None
        self.stage_start_memory: Optional[float] = None
        self.pipeline_start_time: Optional[float] = None
        
        # Get process for memory monitoring
        self.process = psutil.Process()
        
        logger.info("PerformanceMonitor initialized")
    
    def start_pipeline(self) -> None:
        """Mark the start of the complete pipeline."""
        self.pipeline_start_time = time.time()
        logger.info("Pipeline performance monitoring started")
    
    def start_stage(self, stage_name: str) -> None:
        """
        Start monitoring a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
        """
        if self.current_stage is not None:
            logger.warning(
                f"Starting new stage '{stage_name}' while stage '{self.current_stage}' "
                f"is still active. Ending previous stage."
            )
            self.end_stage()
        
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        
        # Get current memory usage
        memory_info = self.process.memory_info()
        self.stage_start_memory = memory_info.rss / 1024**2  # Convert to MB
        
        logger.info(f"Started stage: {stage_name}")
        logger.info(f"  Memory at start: {self.stage_start_memory:.2f} MB")
    
    def end_stage(self) -> Dict[str, Any]:
        """
        End monitoring current pipeline stage.
        
        Returns:
            Dictionary with stage metrics
        """
        if self.current_stage is None:
            logger.warning("No active stage to end")
            return {}
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.stage_start_time
        
        # Get current memory usage
        memory_info = self.process.memory_info()
        current_memory = memory_info.rss / 1024**2  # Convert to MB
        memory_delta = current_memory - self.stage_start_memory
        
        # Store metrics
        stage_metrics = {
            'stage_name': self.current_stage,
            'elapsed_time_seconds': elapsed_time,
            'start_memory_mb': self.stage_start_memory,
            'end_memory_mb': current_memory,
            'memory_delta_mb': memory_delta,
            'timestamp': time.time()
        }
        
        self.metrics[self.current_stage] = stage_metrics
        
        logger.info(f"Completed stage: {self.current_stage}")
        logger.info(f"  Elapsed time: {elapsed_time:.2f} seconds")
        logger.info(f"  Memory at end: {current_memory:.2f} MB")
        logger.info(f"  Memory delta: {memory_delta:+.2f} MB")
        
        # Reset current stage
        self.current_stage = None
        self.stage_start_time = None
        self.stage_start_memory = None
        
        return stage_metrics
    
    @contextmanager
    def monitor_stage(self, stage_name: str):
        """
        Context manager for monitoring a pipeline stage.
        
        Usage:
            with monitor.monitor_stage("data_loading"):
                # ... stage code ...
        
        Args:
            stage_name: Name of the pipeline stage
        """
        self.start_stage(stage_name)
        try:
            yield
        finally:
            self.end_stage()
    
    def log_memory_usage(self, label: str = "Current") -> float:
        """
        Log current memory usage.
        
        Args:
            label: Label for the memory measurement
        
        Returns:
            Current memory usage in MB
        """
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024**2
        
        logger.info(f"{label} memory usage: {memory_mb:.2f} MB")
        
        return memory_mb
    
    def get_total_pipeline_time(self) -> float:
        """
        Get total pipeline elapsed time.
        
        Returns:
            Total elapsed time in seconds
        """
        if self.pipeline_start_time is None:
            return 0.0
        
        return time.time() - self.pipeline_start_time
    
    def get_stage_metrics(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific stage.
        
        Args:
            stage_name: Name of the stage
        
        Returns:
            Dictionary with stage metrics or None if not found
        """
        return self.metrics.get(stage_name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all collected metrics.
        
        Returns:
            Dictionary mapping stage names to their metrics
        """
        return self.metrics.copy()
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Formatted performance report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Total pipeline time
        total_time = self.get_total_pipeline_time()
        report_lines.append(f"Total Pipeline Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        report_lines.append("")
        
        # Stage-by-stage breakdown
        if self.metrics:
            report_lines.append("-" * 80)
            report_lines.append("STAGE BREAKDOWN")
            report_lines.append("-" * 80)
            report_lines.append("")
            
            # Sort stages by timestamp
            sorted_stages = sorted(
                self.metrics.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            total_stage_time = 0.0
            
            for stage_name, metrics in sorted_stages:
                elapsed = metrics['elapsed_time_seconds']
                memory_delta = metrics['memory_delta_mb']
                end_memory = metrics['end_memory_mb']
                
                total_stage_time += elapsed
                
                report_lines.append(f"Stage: {stage_name}")
                report_lines.append(f"  Time:         {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
                report_lines.append(f"  Memory Delta: {memory_delta:+.2f} MB")
                report_lines.append(f"  Memory End:   {end_memory:.2f} MB")
                report_lines.append("")
            
            # Summary statistics
            report_lines.append("-" * 80)
            report_lines.append("SUMMARY STATISTICS")
            report_lines.append("-" * 80)
            report_lines.append(f"Number of stages: {len(self.metrics)}")
            report_lines.append(f"Total stage time: {total_stage_time:.2f} seconds")
            
            if total_time > 0:
                overhead = total_time - total_stage_time
                overhead_pct = (overhead / total_time) * 100
                report_lines.append(f"Overhead time:    {overhead:.2f} seconds ({overhead_pct:.1f}%)")
            
            # Find slowest stage
            slowest_stage = max(sorted_stages, key=lambda x: x[1]['elapsed_time_seconds'])
            report_lines.append(f"Slowest stage:    {slowest_stage[0]} ({slowest_stage[1]['elapsed_time_seconds']:.2f}s)")
            
            # Find stage with highest memory usage
            highest_memory_stage = max(sorted_stages, key=lambda x: x[1]['end_memory_mb'])
            report_lines.append(f"Highest memory:   {highest_memory_stage[0]} ({highest_memory_stage[1]['end_memory_mb']:.2f} MB)")
            
            report_lines.append("")
        
        # Performance assessment
        report_lines.append("-" * 80)
        report_lines.append("PERFORMANCE ASSESSMENT")
        report_lines.append("-" * 80)
        
        if total_time < 600:  # Less than 10 minutes
            report_lines.append("✓ Pipeline completed within target time (< 10 minutes)")
        else:
            report_lines.append("⚠ Pipeline exceeded target time of 10 minutes")
            report_lines.append(f"  Consider optimization for stages taking > 2 minutes")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_report(self, output_path: str) -> None:
        """
        Save performance report to file.
        
        Args:
            output_path: Path to save the report
        """
        report = self.generate_performance_report()
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Performance report saved to: {output_path}")
    
    def reset(self) -> None:
        """Reset all metrics and timers."""
        self.metrics.clear()
        self.current_stage = None
        self.stage_start_time = None
        self.stage_start_memory = None
        self.pipeline_start_time = None
        
        logger.info("Performance monitor reset")
