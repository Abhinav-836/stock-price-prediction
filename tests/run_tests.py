"""
Test Runner - Enhanced with Coverage, Reports, and Multiple Test Types
"""
import unittest
import sys
import os
import time
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing coverage
try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False
    print("⚠️  coverage not installed. Run: pip install coverage")

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


class TestRunner:
    """
    Enhanced test runner with coverage, reporting, and multiple test modes
    """
    
    def __init__(self, verbose: bool = True, with_coverage: bool = False):
        self.verbose = verbose
        self.with_coverage = with_coverage
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.coverage_data = None
        
    def run_all_tests(self, test_pattern: str = 'test_*.py') -> Tuple[int, Dict]:
        """
        Discover and run all tests
        
        Args:
            test_pattern: Pattern for discovering test files
            
        Returns:
            Tuple of (exit_code, test_results)
        """
        print("="*70)
        print(" "*20 + "🧪 RUNNING TEST SUITE")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        
        # Start coverage
        if self.with_coverage and COVERAGE_AVAILABLE:
            cov = coverage.Coverage(
                source=['src'],
                omit=['*/tests/*', '*/__pycache__/*', '*/venv/*']
            )
            cov.start()
        
        # Discover all tests
        loader = unittest.TestLoader()
        start_dir = os.path.dirname(os.path.abspath(__file__))
        suite = loader.discover(start_dir, pattern=test_pattern)
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=2 if self.verbose else 1,
            stream=sys.stdout
        )
        result = runner.run(suite)
        
        # Stop coverage
        if self.with_coverage and COVERAGE_AVAILABLE:
            cov.stop()
            self.coverage_data = cov
        
        self.end_time = time.time()
        
        # Store results
        self.results = {
            'total_tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'successful': result.testsRun - len(result.failures) - len(result.errors),
            'time_taken': self.end_time - self.start_time,
            'failures_details': [str(f) for f in result.failures],
            'errors_details': [str(e) for e in result.errors]
        }
        
        # Print summary
        self._print_summary()
        
        # Save report
        self._save_report()
        
        # Generate coverage report
        if self.with_coverage and COVERAGE_AVAILABLE:
            self._print_coverage_report()
        
        return 0 if result.wasSuccessful() else 1
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print(" "*25 + "📊 TEST SUMMARY")
        print("="*70)
        
        results = self.results
        print(f"\n📈 Total Tests: {results['total_tests']}")
        print(f"✅ Successful: {results['successful']}")
        print(f"❌ Failures: {results['failures']}")
        print(f"⚠️  Errors: {results['errors']}")
        print(f"⏭️  Skipped: {results['skipped']}")
        print(f"⏱️  Time: {results['time_taken']:.3f} seconds")
        
        if results['failures'] > 0:
            print(f"\n❌ Failure Details:")
            for i, failure in enumerate(results['failures_details'], 1):
                print(f"  {i}. {failure[:100]}...")
        
        if results['errors'] > 0:
            print(f"\n⚠️  Error Details:")
            for i, error in enumerate(results['errors_details'], 1):
                print(f"  {i}. {error[:100]}...")
        
        # Overall status
        if results['failures'] == 0 and results['errors'] == 0:
            print(f"\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n❌ SOME TESTS FAILED!")
        
        print("="*70)
    
    def _save_report(self):
        """Save test report to file"""
        report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f'test_report_{timestamp}.json')
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📝 Test report saved to: {report_path}")
        
        # Also save as HTML summary
        html_path = os.path.join(report_dir, f'test_report_{timestamp}.html')
        self._generate_html_report(html_path)
    
    def _generate_html_report(self, filepath: str):
        """Generate HTML test report"""
        results = self.results
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report - Stock Prediction</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #1f77b4; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                .summary {{ background: #f0f2f6; padding: 15px; border-radius: 5px; }}
                .details {{ margin-top: 20px; }}
                .failure {{ background: #ffebee; padding: 10px; margin: 5px 0; border-left: 4px solid red; }}
            </style>
        </head>
        <body>
            <h1>🧪 Test Report - Stock Price Prediction</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: {results['total_tests']}</p>
                <p class="pass">✅ Successful: {results['successful']}</p>
                <p class="fail">❌ Failures: {results['failures']}</p>
                <p>⚠️ Errors: {results['errors']}</p>
                <p>⏭️ Skipped: {results['skipped']}</p>
                <p>⏱️ Time: {results['time_taken']:.3f} seconds</p>
            </div>
        """
        
        if results['failures'] > 0 or results['errors'] > 0:
            html += '<div class="details"><h2>Failures & Errors</h2>'
            for failure in results['failures_details']:
                html += f'<div class="failure">{failure}</div>'
            for error in results['errors_details']:
                html += f'<div class="failure">{error}</div>'
            html += '</div>'
        
        if results['successful'] == results['total_tests']:
            html += '<h2 class="pass">🎉 All tests passed successfully!</h2>'
        
        html += """
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html)
        
        print(f"📊 HTML report saved to: {filepath}")
    
    def _print_coverage_report(self):
        """Print coverage report"""
        if not COVERAGE_AVAILABLE or self.coverage_data is None:
            return
        
        print("\n" + "="*70)
        print(" "*25 + "📊 COVERAGE REPORT")
        print("="*70)
        
        # Generate coverage report
        try:
            from io import StringIO
            import sys
            
            # Capture coverage report output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            self.coverage_data.report(show_missing=True)
            coverage_output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            print(coverage_output)
            
            # Save coverage report
            report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports')
            os.makedirs(report_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            coverage_path = os.path.join(report_dir, f'coverage_{timestamp}.txt')
            with open(coverage_path, 'w') as f:
                f.write(coverage_output)
            
            # Generate HTML coverage report
            html_dir = os.path.join(report_dir, f'coverage_html_{timestamp}')
            self.coverage_data.html_report(directory=html_dir)
            print(f"\n🌐 HTML coverage report saved to: {html_dir}/index.html")
            
            # Get coverage percentage
            total_statements = 0
            covered_statements = 0
            for filename, analysis in self.coverage_data.analysis2().items():
                statements = len(analysis[1])
                executed = len(analysis[2])
                total_statements += statements
                covered_statements += executed
            
            coverage_pct = (covered_statements / total_statements * 100) if total_statements > 0 else 0
            print(f"\n📊 Overall Coverage: {coverage_pct:.1f}%")
            
        except Exception as e:
            print(f"⚠️  Error generating coverage report: {e}")
    
    def run_specific_test(self, test_class: str, test_method: str = None):
        """
        Run a specific test class or method
        
        Args:
            test_class: Name of the test class
            test_method: Name of the test method (optional)
        """
        print(f"\n🧪 Running specific test: {test_class}")
        if test_method:
            print(f"   Method: {test_method}")
        
        # Import the test module
        try:
            module = __import__(f'tests.{test_class.lower()}', fromlist=[test_class])
            test_class_obj = getattr(module, test_class)
            
            if test_method:
                suite = unittest.TestLoader().loadTestsFromName(
                    f'{test_class}.{test_method}', module
                )
            else:
                suite = unittest.TestLoader().loadTestsFromTestCase(test_class_obj)
            
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            return 0 if result.wasSuccessful() else 1
            
        except ImportError as e:
            print(f"❌ Could not import test class: {e}")
            return 1
        except AttributeError as e:
            print(f"❌ Could not find test method: {e}")
            return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Test Runner for Stock Price Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python tests/run_tests.py
  
  # Run with coverage
  python tests/run_tests.py --coverage
  
  # Run specific test class
  python tests/run_tests.py --class TestModelTraining
  
  # Run specific test method
  python tests/run_tests.py --class TestModelTraining --method test_linear_regression
  
  # Run with minimal output
  python tests/run_tests.py --quiet
        """
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    parser.add_argument(
        '--class',
        dest='test_class',
        type=str,
        help='Run specific test class'
    )
    
    parser.add_argument(
        '--method',
        dest='test_method',
        type=str,
        help='Run specific test method (requires --class)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Run with minimal output'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='test_*.py',
        help='Pattern for test discovery (default: test_*.py)'
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(
        verbose=not args.quiet,
        with_coverage=args.coverage
    )
    
    # Run tests
    if args.test_class:
        exit_code = runner.run_specific_test(args.test_class, args.test_method)
    else:
        exit_code = runner.run_all_tests(test_pattern=args.pattern)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()