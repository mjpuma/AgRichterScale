#!/usr/bin/env python3
"""
Demo script showing how to use convergence validation in the AgriRichter pipeline.

This example demonstrates:
1. How to configure convergence validation settings
2. How to run the pipeline with convergence validation enabled
3. How to customize convergence validation behavior
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agririchter.core.config import Config
from agririchter.pipeline.events_pipeline import EventsPipeline


def demo_default_convergence_validation():
    """Demonstrate default convergence validation settings."""
    print("=" * 60)
    print("Demo 1: Default Convergence Validation")
    print("=" * 60)
    
    # Create config with default convergence validation (enabled by default)
    config = Config('wheat')
    
    print("Default convergence validation settings:")
    convergence_config = config.get_convergence_validation_config()
    for key, value in convergence_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nConvergence validation enabled: {config.is_convergence_validation_enabled()}")
    print(f"Convergence enforcement enabled: {config.should_enforce_convergence()}")
    print(f"Fallback on failure: {config.should_fallback_on_convergence_failure()}")
    print(f"Convergence tolerance: {config.get_convergence_tolerance()}")
    
    # Initialize pipeline
    pipeline = EventsPipeline(config, output_dir='demo_output', enable_performance_monitoring=False)
    print("\n✓ Pipeline initialized with default convergence validation")


def demo_custom_convergence_validation():
    """Demonstrate custom convergence validation settings."""
    print("\n" + "=" * 60)
    print("Demo 2: Custom Convergence Validation")
    print("=" * 60)
    
    # Create config with custom convergence validation settings
    custom_convergence = {
        'enabled': True,
        'enforce_convergence': True,
        'fallback_on_failure': False,  # Strict mode - fail if convergence can't be enforced
        'tolerance': 1e-8,  # Stricter tolerance
        'max_iterations': 20,  # More iterations for convergence
        'log_validation_details': True,  # Enable detailed logging
        'backward_compatible': False  # Disable backward compatibility mode
    }
    
    config = Config('wheat', convergence_validation=custom_convergence)
    
    print("Custom convergence validation settings:")
    convergence_config = config.get_convergence_validation_config()
    for key, value in convergence_config.items():
        print(f"  {key}: {value}")
    
    # Initialize pipeline
    pipeline = EventsPipeline(config, output_dir='demo_output', enable_performance_monitoring=False)
    print("\n✓ Pipeline initialized with custom convergence validation")


def demo_disabled_convergence_validation():
    """Demonstrate disabled convergence validation for backward compatibility."""
    print("\n" + "=" * 60)
    print("Demo 3: Disabled Convergence Validation (Backward Compatibility)")
    print("=" * 60)
    
    # Create config with convergence validation disabled
    disabled_convergence = {
        'enabled': False,
        'backward_compatible': True
    }
    
    config = Config('wheat', convergence_validation=disabled_convergence)
    
    print("Disabled convergence validation settings:")
    convergence_config = config.get_convergence_validation_config()
    for key, value in convergence_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nConvergence validation enabled: {config.is_convergence_validation_enabled()}")
    print(f"Backward compatible mode: {config.is_backward_compatible_mode()}")
    
    # Initialize pipeline
    pipeline = EventsPipeline(config, output_dir='demo_output', enable_performance_monitoring=False)
    print("\n✓ Pipeline initialized with convergence validation disabled")


def demo_convergence_validation_usage():
    """Demonstrate how convergence validation works in practice."""
    print("\n" + "=" * 60)
    print("Demo 4: Convergence Validation Usage")
    print("=" * 60)
    
    print("When running the full pipeline with convergence validation enabled:")
    print("1. Envelope calculation will be performed as usual")
    print("2. Convergence validation will check mathematical properties:")
    print("   - Bounds start at origin (0,0)")
    print("   - Bounds converge at maximum harvest area")
    print("   - Upper bound dominates lower bound")
    print("   - Conservation laws are satisfied")
    print("   - Harvest areas are monotonic")
    print("3. If validation fails and enforcement is enabled:")
    print("   - Convergence point will be explicitly added")
    print("   - Envelope will be re-validated")
    print("4. If enforcement fails and fallback is enabled:")
    print("   - Original envelope data will be used with warning")
    print("5. If fallback is disabled:")
    print("   - Pipeline will raise an exception")
    
    print("\nExample pipeline usage:")
    print("```python")
    print("config = Config('wheat', convergence_validation={'enabled': True})")
    print("pipeline = EventsPipeline(config, 'output')")
    print("results = pipeline.run_complete_pipeline()")
    print("```")


if __name__ == "__main__":
    print("AgriRichter Convergence Validation Demo")
    print("This demo shows how to configure and use convergence validation")
    print("in the AgriRichter pipeline for mathematically correct H-P envelopes.")
    
    try:
        demo_default_convergence_validation()
        demo_custom_convergence_validation()
        demo_disabled_convergence_validation()
        demo_convergence_validation_usage()
        
        print("\n" + "=" * 60)
        print("✓ All convergence validation demos completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("- Run the pipeline with your data to see convergence validation in action")
        print("- Check the logs for convergence validation messages")
        print("- Customize convergence settings based on your requirements")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        sys.exit(1)