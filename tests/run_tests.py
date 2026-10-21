"""
Test Runner - FIXED

Fixes vs previous version:
  * Discovery used the tests folder as the top-level dir, so `from tests.x import ...`
    inside test modules failed. It now discovers with the project root as top level.
  * --class imported `tests.<classname lowercased>` (e.g. tests.testmodeltraining), a module
    that never exists. It now searches the discovered suite for the class/method.
  * Coverage percentage used `coverage.analysis2().items()` - not a real API -> crashed.
    `Coverage.report()` returns the total percentage directly.
  * Coverage was started AFTER src was already imported via discovery side effects in some
    setups; it is now started before discovery.
  * Reports go to <project>/reports/ (already git-ignored) and HTML-escape failure text.
"""
import argparse
import html
import json
import os
import sys
import time
import unittest
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)
REPORT_DIR = os.path.join(ROOT_DIR, 'reports')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _iter_tests(item)
        else:
            yield item


class TestRunner:
    def __init__(self, verbose: bool = True, with_coverage: bool = False):
        self.verbose = verbose
        self.with_coverage = with_coverage and COVERAGE_AVAILABLE
        if with_coverage and not COVERAGE_AVAILABLE:
            print("⚠️  coverage not installed. Run: pip install coverage")
        self.results = {}
        self.cov = None

    def _discover(self, pattern='test_*.py'):
        return unittest.TestLoader().discover(TESTS_DIR, pattern=pattern, top_level_dir=ROOT_DIR)

    def _run(self, suite):
        print("=" * 70 + "\n" + " " * 20 + "🧪 RUNNING TEST SUITE\n" + "=" * 70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        start = time.time()
        result = unittest.TextTestRunner(verbosity=2 if self.verbose else 1, stream=sys.stdout).run(suite)
        self.results = {
            'total_tests': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'successful': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
            'time_taken': time.time() - start,
            'failures_details': [f"{t.id()}\n{tb}" for t, tb in result.failures],
            'errors_details': [f"{t.id()}\n{tb}" for t, tb in result.errors],
        }
        return result

    def run_all_tests(self, test_pattern: str = 'test_*.py') -> int:
        if self.with_coverage:
            self.cov = coverage.Coverage(source=[os.path.join(ROOT_DIR, 'src'), os.path.join(ROOT_DIR, 'utils')])
            self.cov.start()

        result = self._run(self._discover(test_pattern))

        if self.cov:
            self.cov.stop()
            self.cov.save()

        self._print_summary()
        self._save_report()
        if self.cov:
            self._print_coverage_report()
        return 0 if result.wasSuccessful() else 1

    def run_specific_test(self, test_class: str, test_method: str = None) -> int:
        selected = unittest.TestSuite()
        for test in _iter_tests(self._discover()):
            if isinstance(test, unittest.loader._FailedTest):
                selected.addTest(test)  # surface import errors instead of hiding them
                continue
            if type(test).__name__ == test_class and (test_method is None or test._testMethodName == test_method):
                selected.addTest(test)

        if selected.countTestCases() == 0:
            print(f"❌ No tests found for {test_class}{'.' + test_method if test_method else ''}")
            return 1
        result = self._run(selected)
        self._print_summary()
        return 0 if result.wasSuccessful() else 1

    def _print_summary(self):
        r = self.results
        print("\n" + "=" * 70 + "\n" + " " * 25 + "📊 TEST SUMMARY\n" + "=" * 70)
        print(f"📈 Total: {r['total_tests']}  ✅ Passed: {r['successful']}  ❌ Failures: {r['failures']}  "
              f"⚠️  Errors: {r['errors']}  ⏭️  Skipped: {r['skipped']}  ⏱️  {r['time_taken']:.2f}s")
        for label, key in (("❌ Failures", 'failures_details'), ("⚠️  Errors", 'errors_details')):
            if r[key]:
                print(f"\n{label}:")
                for i, detail in enumerate(r[key], 1):
                    print(f"  {i}. {detail.strip().splitlines()[0]}")
        print("\n🎉 ALL TESTS PASSED!" if r['failures'] == r['errors'] == 0 else "\n❌ SOME TESTS FAILED!")
        print("=" * 70)

    def _save_report(self):
        os.makedirs(REPORT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(REPORT_DIR, f'test_report_{stamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        r = self.results
        details = "".join(f'<pre class="failure">{html.escape(d)}</pre>'
                          for d in r['failures_details'] + r['errors_details'])
        page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Test Report</title>
<style>body{{font-family:Arial,sans-serif;margin:20px}}.summary{{background:#f0f2f6;padding:15px;border-radius:5px}}
.failure{{background:#ffebee;padding:10px;border-left:4px solid red;white-space:pre-wrap}}</style></head><body>
<h1>🧪 Test Report - Stock Price Prediction</h1><p>Generated: {datetime.now():%Y-%m-%d %H:%M:%S}</p>
<div class="summary"><p>Total: {r['total_tests']}</p><p>✅ Passed: {r['successful']}</p>
<p>❌ Failures: {r['failures']}</p><p>⚠️ Errors: {r['errors']}</p><p>⏭️ Skipped: {r['skipped']}</p>
<p>⏱️ {r['time_taken']:.2f}s</p></div>{details}</body></html>"""
        with open(os.path.join(REPORT_DIR, f'test_report_{stamp}.html'), 'w', encoding='utf-8') as f:
            f.write(page)
        print(f"📝 Reports saved to: {REPORT_DIR}")

    def _print_coverage_report(self):
        print("\n" + "=" * 70 + "\n" + " " * 25 + "📊 COVERAGE REPORT\n" + "=" * 70)
        try:
            total = self.cov.report(show_missing=False)
            html_dir = os.path.join(REPORT_DIR, f'coverage_html_{datetime.now():%Y%m%d_%H%M%S}')
            self.cov.html_report(directory=html_dir)
            print(f"\n📊 Overall Coverage: {total:.1f}%\n🌐 HTML report: {html_dir}/index.html")
        except Exception as e:
            print(f"⚠️  Error generating coverage report: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test Runner for Stock Price Prediction')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--class', dest='test_class', type=str, help='Run one test class, e.g. TestForecasting')
    parser.add_argument('--method', dest='test_method', type=str, help='Run one method (needs --class)')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--pattern', type=str, default='test_*.py')
    args = parser.parse_args()

    if args.test_method and not args.test_class:
        parser.error("--method requires --class")

    runner = TestRunner(verbose=not args.quiet, with_coverage=args.coverage)
    if args.test_class:
        sys.exit(runner.run_specific_test(args.test_class, args.test_method))
    sys.exit(runner.run_all_tests(test_pattern=args.pattern))


if __name__ == '__main__':
    main()