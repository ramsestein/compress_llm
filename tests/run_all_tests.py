#!/usr/bin/env python3
"""Main test runner for the comprehensive test suite.

Runs compression system, LoRA, main scripts, and utilities tests.
"""

import importlib.util
import sys
import time
import traceback
import unittest
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_test_files():
    """Discovers all test files in the tests/ directory."""
    test_dir = Path(__file__).parent
    test_files = []

    # Define test categories
    test_categories = {
        "compression": [
            "test_compression_system_comprehensive.py",
            "test_compression_methods_specific.py",
            "test_compression_engine.py",
            "test_compression_verification.py",
        ],
        "lora": [
            "test_lora_system_comprehensive.py",
            "test_peft_methods_specific.py",
            "test_lora_trainer.py",
            "test_lora_model.py",
            "test_peft_methods_config.py",
            "test_peft_universal_trainer.py",
            "test_dataset_manager.py",
            "test_dataset_manager_comprehensive.py",
            "test_training_execution.py",
        ],
        "scripts": [
            "test_main_scripts_comprehensive.py",
            "test_merge_lora.py",
            "test_ollama_server.py",
        ],
        "utilities": ["test_utilities_comprehensive.py"],
    }

    # Collect all test files
    for category, files in test_categories.items():
        for file_name in files:
            file_path = test_dir / file_name
            if file_path.exists():
                test_files.append((category, file_path))

    return test_files


def run_tests_by_category():
    """Run tests organized by category."""
    print("Discovering test files...")

    test_files = discover_test_files()

    if not test_files:
        print("No test files found")
        return False

    print(f"Found {len(test_files)} test files:")

    # Organize by category
    categories = {}
    for category, file_path in test_files:
        if category not in categories:
            categories[category] = []
        categories[category].append(file_path)
        print(f"   - {category}: {file_path.name}")

    print("\nRunning tests by category...")

    # Run tests by category
    results = {}
    total_tests = 0
    total_failures = 0
    total_errors = 0

    for category, files in categories.items():
        print(f"\n{'=' * 60}")
        print(f"Running tests for {category.upper()}")
        print(f"{'=' * 60}")

        category_results = run_category_tests(category, files)
        results[category] = category_results

        total_tests += category_results["total_tests"]
        total_failures += category_results["failures"]
        total_errors += category_results["errors"]

        # Show category summary
        print(f"\nSummary {category}:")
        print(
            f"   Tests passed: {category_results['total_tests'] - category_results['failures'] - category_results['errors']}"
        )
        print(f"   Failures: {category_results['failures']}")
        print(f"    Errors: {category_results['errors']}")
        print(f"   Time: {category_results['execution_time']:.2f}s")

    # Summary general
    print(f"\n{'=' * 60}")
    print("GENERAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"   Categories executed: {len(categories)}")
    print(f"    Total tests: {total_tests}")
    print(f"   Tests passed: {total_tests - total_failures - total_errors}")
    print(f"   Total failures: {total_failures}")
    print(f"    Total errors: {total_errors}")
    print(f"   Total time: {sum(r['execution_time'] for r in results.values()):.2f}s")

    success_rate = (
        ((total_tests - total_failures - total_errors) / total_tests * 100)
        if total_tests > 0
        else 0
    )
    print(f"   Success rate: {success_rate:.1f}%")

    return total_failures == 0 and total_errors == 0


def run_category_tests(category, test_files):
    """Run tests for a specific category."""
    start_time = time.time()

    # Use unittest
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    total_tests = 0
    errors = 0

    for test_file in test_files:
        try:
            # Import the test module
            module_name = test_file.stem
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Load tests from module
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)

            # Count tests
            test_count = tests.countTestCases()
            total_tests += test_count

        except Exception as e:
            print(f" Error loading {test_file.name}: {e}")
            errors += 1
            continue

    # Run tests
    runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
    result = runner.run(suite)

    execution_time = time.time() - start_time

    return {
        "total_tests": total_tests,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "execution_time": execution_time,
        "result": result,
    }


def run_specific_test(test_file_path):
    """Run a specific test file."""
    print(f" Running specific test: {test_file_path.name}")

    try:
        # Import the test module
        module_name = test_file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, test_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Run test
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return result.wasSuccessful()

    except Exception as e:
        print(f"Error running {test_file_path.name}: {e}")
        traceback.print_exc()
        return False


def run_quick_tests():
    """Run quick tests for basic verification."""
    print("Running quick tests...")

    quick_tests = ["test_compression_engine.py", "test_lora_model.py", "test_dataset_manager.py"]

    test_dir = Path(__file__).parent
    success_count = 0

    for test_file in quick_tests:
        test_path = test_dir / test_file
        if test_path.exists():
            if run_specific_test(test_path):
                success_count += 1
                print(f"{test_file} - PASSED")
            else:
                print(f"{test_file} - FAILED")
        else:
            print(f" {test_file} - NOT FOUND")

    print(f"\nQuick tests: {success_count}/{len(quick_tests)} passed")
    return success_count == len(quick_tests)


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("Running comprehensive tests...")

    comprehensive_tests = [
        "test_compression_system_comprehensive.py",
        "test_lora_system_comprehensive.py",
        "test_main_scripts_comprehensive.py",
        "test_utilities_comprehensive.py",
    ]

    test_dir = Path(__file__).parent
    success_count = 0

    for test_file in comprehensive_tests:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"\n{'=' * 50}")
            print(f"Running: {test_file}")
            print(f"{'=' * 50}")

            if run_specific_test(test_path):
                success_count += 1
                print(f"{test_file} - PASSED")
            else:
                print(f"{test_file} - FAILED")
        else:
            print(f" {test_file} - NOT FOUND")

    print(f"\nComprehensive tests: {success_count}/{len(comprehensive_tests)} passed")
    return success_count == len(comprehensive_tests)


def main():
    """Serve as the main entry point."""
    print("Starting comprehensive test execution...")
    print("=" * 60)

    # Check command-line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "quick":
            print("Mode: Quick tests")
            success = run_quick_tests()
        elif mode == "comprehensive":
            print("Mode: Comprehensive tests")
            success = run_comprehensive_tests()
        elif mode == "category":
            print("Mode: Tests by category")
            success = run_tests_by_category()
        elif mode == "specific" and len(sys.argv) > 2:
            test_file = sys.argv[2]
            test_path = Path(__file__).parent / test_file
            if test_path.exists():
                print(f"Mode: Specific test - {test_file}")
                success = run_specific_test(test_path)
            else:
                print(f"Test file not found: {test_file}")
                success = False
        else:
            print("Mode not recognized. Use: quick, comprehensive, category, or specific <file>")
            success = False
    else:
        # Default mode: tests by category
        print("Default mode: Tests by category")
        success = run_tests_by_category()

    print("\n" + "=" * 60)
    if success:
        print("All tests passed successfully!")
        exit_code = 0
    else:
        print("Some tests failed. Review the errors above.")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
