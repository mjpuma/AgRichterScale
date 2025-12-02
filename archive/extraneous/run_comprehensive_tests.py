#!/usr/bin/env python3
"""
Comprehensive test runner for multi-tier envelope integration system.

This script runs all test suites and generates comprehensive validation reports
to confirm the system is ready for production deployment.
"""

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """Comprehensive test runner for multi-tier envelope system."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize test runner."""
        self.output_dir = output_dir or Path('./test_results')
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_results = {}
        self.overall_status = True
        
        logger.info(f"Test runner initialized - Output directory: {self.output_dir}")
    
    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        logger.info("Running unit tests...")
        
        try:
            # Run unit tests with pytest
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/unit/test_multi_tier_envelope_unit.py',
                '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=600)
            
            success = result.returncode == 0
            
            self.test_results['unit_tests'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': datetime.now().isoformat()
            }
            
            if success:
                logger.info("✅ Unit tests passed")
            else:
                logger.error("❌ Unit tests failed")
                logger.error(f"Error output: {result.stderr}")
            
            return success
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Unit tests timed out")
            self.test_results['unit_tests'] = {'success': False, 'error': 'Timeout'}
            return False
        except Exception as e:
            logger.error(f"❌ Unit tests failed with exception: {e}")
            self.test_results['unit_tests'] = {'success': False, 'error': str(e)}
            return False
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        logger.info("Running integration tests...")
        
        try:
            # Run integration tests with pytest (use our comprehensive system validation)
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/integration/test_comprehensive_system_validation.py',
                '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=1800)  # 30 minutes timeout
            
            success = result.returncode == 0
            
            self.test_results['integration_tests'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': datetime.now().isoformat()
            }
            
            if success:
                logger.info("✅ Integration tests passed")
            else:
                logger.error("❌ Integration tests failed")
                logger.error(f"Error output: {result.stderr}")
            
            return success
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Integration tests timed out")
            self.test_results['integration_tests'] = {'success': False, 'error': 'Timeout'}
            return False
        except Exception as e:
            logger.error(f"❌ Integration tests failed with exception: {e}")
            self.test_results['integration_tests'] = {'success': False, 'error': str(e)}
            return False
    
    def run_performance_tests(self) -> bool:
        """Run performance benchmark tests."""
        logger.info("Running performance tests...")
        
        try:
            # Run performance tests with pytest
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/performance/test_multi_tier_performance.py',
                '-v', '--tb=short', '-s'
            ], capture_output=True, text=True, timeout=2400)  # 40 minutes timeout
            
            success = result.returncode == 0
            
            self.test_results['performance_tests'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': datetime.now().isoformat()
            }
            
            if success:
                logger.info("✅ Performance tests passed")
            else:
                logger.error("❌ Performance tests failed")
                logger.error(f"Error output: {result.stderr}")
            
            return success
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Performance tests timed out")
            self.test_results['performance_tests'] = {'success': False, 'error': 'Timeout'}
            return False
        except Exception as e:
            logger.error(f"❌ Performance tests failed with exception: {e}")
            self.test_results['performance_tests'] = {'success': False, 'error': str(e)}
            return False
    
    def run_validation_tests(self) -> bool:
        """Run comprehensive validation tests."""
        logger.info("Running validation tests...")
        
        try:
            # Run validation tests directly (they have their own runner)
            result = subprocess.run([
                sys.executable, 'tests/validation/test_multi_tier_validation.py'
            ], capture_output=True, text=True, timeout=1200)  # 20 minutes timeout
            
            success = result.returncode == 0
            
            self.test_results['validation_tests'] = {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': datetime.now().isoformat()
            }
            
            # Copy validation report to output directory
            validation_report_path = Path('./multi_tier_validation_report.md')
            if validation_report_path.exists():
                import shutil
                shutil.copy(validation_report_path, self.output_dir / 'validation_report.md')
                
                # Also copy JSON report
                json_report_path = Path('./multi_tier_validation_report.json')
                if json_report_path.exists():
                    shutil.copy(json_report_path, self.output_dir / 'validation_report.json')
            
            if success:
                logger.info("✅ Validation tests passed")
            else:
                logger.error("❌ Validation tests failed")
                logger.error(f"Error output: {result.stderr}")
            
            return success
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Validation tests timed out")
            self.test_results['validation_tests'] = {'success': False, 'error': 'Timeout'}
            return False
        except Exception as e:
            logger.error(f"❌ Validation tests failed with exception: {e}")
            self.test_results['validation_tests'] = {'success': False, 'error': str(e)}
            return False
    
    def run_all_tests(self, skip_performance: bool = False) -> bool:
        """Run all test suites."""
        logger.info("Starting comprehensive test execution...")
        
        test_suites = [
            ('Unit Tests', self.run_unit_tests),
            ('Integration Tests', self.run_integration_tests),
            ('Validation Tests', self.run_validation_tests)
        ]
        
        if not skip_performance:
            test_suites.insert(-1, ('Performance Tests', self.run_performance_tests))
        
        all_passed = True
        
        for suite_name, test_function in test_suites:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {suite_name}")
            logger.info(f"{'='*60}")
            
            try:
                success = test_function()
                if not success:
                    all_passed = False
                    self.overall_status = False
            except Exception as e:
                logger.error(f"Failed to run {suite_name}: {e}")
                all_passed = False
                self.overall_status = False
        
        self.overall_status = all_passed
        return all_passed
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        logger.info("Generating summary report...")
        
        report_lines = [
            "# Multi-Tier Envelope Integration: Comprehensive Test Report",
            "",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Overall Status:** {'✅ PASS' if self.overall_status else '❌ FAIL'}",
            "",
            "## Executive Summary",
            ""
        ]
        
        if self.overall_status:
            report_lines.extend([
                "🎉 **ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION DEPLOYMENT**",
                "",
                "The multi-tier envelope integration system has successfully passed all",
                "comprehensive tests including unit tests, integration tests, performance",
                "benchmarks, and validation requirements.",
                ""
            ])
        else:
            failed_suites = [
                suite for suite, result in self.test_results.items() 
                if not result.get('success', False)
            ]
            
            report_lines.extend([
                "⚠️ **TESTS FAILED - SYSTEM NOT READY FOR PRODUCTION**",
                "",
                f"**Failed Test Suites:** {', '.join(failed_suites)}",
                "",
                "Please review the detailed test results below and address all",
                "failing tests before proceeding with production deployment.",
                ""
            ])
        
        # Test Suite Results
        report_lines.extend([
            "## Test Suite Results",
            "",
            "| Test Suite | Status | Details |",
            "|------------|--------|---------|"
        ])
        
        for suite_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            details = result.get('error', 'See detailed logs') if not result.get('success', False) else "All tests passed"
            
            report_lines.append(f"| {suite_name.replace('_', ' ').title()} | {status} | {details} |")
        
        report_lines.extend([
            "",
            "## Detailed Results",
            ""
        ])
        
        # Add detailed results for each suite
        for suite_name, result in self.test_results.items():
            report_lines.extend([
                f"### {suite_name.replace('_', ' ').title()}",
                ""
            ])
            
            if result.get('success', False):
                report_lines.append("✅ **Status:** PASSED")
            else:
                report_lines.append("❌ **Status:** FAILED")
                if 'error' in result:
                    report_lines.append(f"**Error:** {result['error']}")
            
            if 'execution_time' in result:
                report_lines.append(f"**Execution Time:** {result['execution_time']}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        
        if self.overall_status:
            report_lines.extend([
                "- ✅ **Deploy to Production:** All tests passed successfully",
                "- 📊 **Monitor Performance:** Set up production monitoring",
                "- 🔄 **Regular Testing:** Run tests with new data regularly",
                "- 📚 **Documentation:** Ensure user guides are up to date",
                ""
            ])
        else:
            report_lines.extend([
                "- ❌ **Do Not Deploy:** Fix all failing tests first",
                "- 🔍 **Review Logs:** Check detailed test output for specific issues",
                "- 🧪 **Re-run Tests:** After fixes, run comprehensive tests again",
                "- 🆘 **Get Support:** Contact development team if issues persist",
                ""
            ])
        
        # File locations
        report_lines.extend([
            "## Test Artifacts",
            "",
            "The following test artifacts have been generated:",
            "",
            f"- **Summary Report:** `{self.output_dir}/comprehensive_test_report.md`",
            f"- **Test Results:** `{self.output_dir}/test_results_summary.json`",
            f"- **Unit Test Results:** `{self.output_dir}/unit_test_results.json`",
            f"- **Integration Test Results:** `{self.output_dir}/integration_test_results.json`",
            f"- **Performance Test Results:** `{self.output_dir}/performance_test_results.json`",
            f"- **Validation Report:** `{self.output_dir}/validation_report.md`",
            f"- **Execution Log:** `test_execution.log`",
            ""
        ])
        
        # Write summary report
        summary_path = self.output_dir / 'comprehensive_test_report.md'
        with open(summary_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Write JSON summary
        json_summary = {
            'overall_status': self.overall_status,
            'execution_timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'summary': {
                'total_suites': len(self.test_results),
                'passed_suites': sum(1 for r in self.test_results.values() if r.get('success', False)),
                'failed_suites': sum(1 for r in self.test_results.values() if not r.get('success', False))
            }
        }
        
        json_path = self.output_dir / 'test_results_summary.json'
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2)
        
        logger.info(f"Summary report written to: {summary_path}")
        logger.info(f"JSON summary written to: {json_path}")
    
    def print_final_status(self):
        """Print final test status."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST EXECUTION COMPLETE")
        print("="*80)
        
        if self.overall_status:
            print("🎉 SUCCESS: All tests passed!")
            print("✅ System is ready for production deployment")
        else:
            print("❌ FAILURE: Some tests failed")
            print("⚠️  System is NOT ready for production deployment")
        
        print(f"\n📊 Test Results Summary:")
        for suite_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"   {suite_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n📁 Detailed reports available in: {self.output_dir}")
        print("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run comprehensive tests for multi-tier envelope system')
    parser.add_argument('--output-dir', type=Path, default=Path('./test_results'),
                       help='Output directory for test results')
    parser.add_argument('--skip-performance', action='store_true',
                       help='Skip performance tests (for faster execution)')
    parser.add_argument('--unit-only', action='store_true',
                       help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true',
                       help='Run only integration tests')
    parser.add_argument('--validation-only', action='store_true',
                       help='Run only validation tests')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ComprehensiveTestRunner(args.output_dir)
    
    try:
        if args.unit_only:
            success = runner.run_unit_tests()
        elif args.integration_only:
            success = runner.run_integration_tests()
        elif args.validation_only:
            success = runner.run_validation_tests()
        else:
            success = runner.run_all_tests(skip_performance=args.skip_performance)
        
        # Generate reports
        runner.generate_summary_report()
        runner.print_final_status()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.error("Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test execution failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()