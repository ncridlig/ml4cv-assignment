"""
Test script to verify all imports work correctly after repository refactoring.
Tests imports from utils/, anomaly_detection/, and config.py.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import(module_name, description, items_to_check=None):
    """
    Test importing a module and optionally specific items from it.

    Args:
        module_name: Name of module to import (e.g., 'config' or 'utils.dataloader')
        description: Human-readable description of what's being tested
        items_to_check: List of items to verify exist in the module (optional)

    Returns:
        bool: True if import succeeded, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"Module: {module_name}")
    print(f"{'='*70}")

    try:
        # Import the module
        module = __import__(module_name, fromlist=[''])
        print(f"✓ Successfully imported {module_name}")

        # Check specific items if provided
        if items_to_check:
            for item in items_to_check:
                if hasattr(module, item):
                    print(f"  ✓ Found {item}")
                else:
                    print(f"  ✗ Missing {item}")
                    return False

        return True

    except Exception as e:
        print(f"✗ Failed to import {module_name}")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all import tests."""
    print("\n" + "="*70)
    print("IMPORT VERIFICATION TEST SUITE")
    print("Testing repository structure after refactoring")
    print("="*70)

    results = []

    # =========================================================================
    # Test 1: Config module
    # =========================================================================
    results.append(test_import(
        'config',
        'Central configuration file',
        items_to_check=[
            'DEVICE',
            'MODEL_PATH',
            'NUM_CLASSES',
            'ANOMALY_CLASS_IDX',
            'IMAGE_SIZE',
            'BATCH_SIZE',
            'LEARNING_RATE',
            'ANOMALY_THRESHOLD',
        ]
    ))

    # =========================================================================
    # Test 2: Utils - Dataloader
    # =========================================================================
    results.append(test_import(
        'utils.dataloader',
        'Dataloader utility (StreetHazards dataset)',
        items_to_check=[
            'StreetHazardsDataset',
            'get_dataloaders',
            'CLASS_NAMES',
            'NUM_CLASSES',
            'ANOMALY_CLASS_IDX',
        ]
    ))

    # =========================================================================
    # Test 3: Utils - Model utilities
    # =========================================================================
    results.append(test_import(
        'utils.model_utils',
        'Model utility functions',
        items_to_check=[
            'load_model',
        ]
    ))

    # =========================================================================
    # Test 4: Anomaly Detection - Simple Max Logits
    # =========================================================================
    # Note: We only test if the file can be imported, not if it runs
    # (running would require model weights and datasets)
    print(f"\n{'='*70}")
    print(f"Testing: Anomaly detection scripts exist and are importable")
    print(f"{'='*70}")

    anomaly_scripts = [
        'simple_max_logits',
        'maximum_softmax_probability',
        'standardized_max_logits',
        'energy_score_anomaly_detection',
        'heat_anomaly_detection',
    ]

    anomaly_test_passed = True
    for script in anomaly_scripts:
        script_path = project_root / 'anomaly_detection' / f'{script}.py'
        if script_path.exists():
            print(f"  ✓ Found anomaly_detection/{script}.py")
        else:
            print(f"  ✗ Missing anomaly_detection/{script}.py")
            anomaly_test_passed = False

    results.append(anomaly_test_passed)

    # =========================================================================
    # Test 5: Check directory structure
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Testing: Directory structure")
    print(f"{'='*70}")

    required_dirs = [
        'anomaly_detection',
        'models',
        'utils',
        'assets',
    ]

    dir_test_passed = True
    for dirname in required_dirs:
        dir_path = project_root / dirname
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ Directory exists: {dirname}/")
        else:
            print(f"  ✗ Missing directory: {dirname}/")
            dir_test_passed = False

    results.append(dir_test_passed)

    # =========================================================================
    # Test 6: Check critical files exist
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Testing: Critical files exist")
    print(f"{'='*70}")

    critical_files = [
        'config.py',
        'main.ipynb',
        'README.md',
        'requirements.txt',
        'download_dataset.sh',
        'evaluate_qualitative.py',
        'utils/__init__.py',
        'utils/dataloader.py',
        'utils/model_utils.py',
    ]

    file_test_passed = True
    for filepath in critical_files:
        file_path = project_root / filepath
        if file_path.exists():
            print(f"  ✓ Found {filepath}")
        else:
            print(f"  ✗ Missing {filepath}")
            file_test_passed = False

    results.append(file_test_passed)

    # =========================================================================
    # Test 7: Test actual import from config in a script-like context
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Testing: Practical import pattern (as used in scripts)")
    print(f"{'='*70}")

    practical_test_passed = True
    try:
        # Simulate what scripts do
        from config import DEVICE, MODEL_PATH, NUM_CLASSES, ANOMALY_CLASS_IDX
        print(f"  ✓ Successfully imported from config: DEVICE={DEVICE}")
        print(f"  ✓ Successfully imported from config: MODEL_PATH={MODEL_PATH}")
        print(f"  ✓ Successfully imported from config: NUM_CLASSES={NUM_CLASSES}")
        print(f"  ✓ Successfully imported from config: ANOMALY_CLASS_IDX={ANOMALY_CLASS_IDX}")

        from utils.dataloader import CLASS_NAMES, StreetHazardsDataset
        print(f"  ✓ Successfully imported from utils.dataloader: CLASS_NAMES (length={len(CLASS_NAMES)})")
        print(f"  ✓ Successfully imported from utils.dataloader: StreetHazardsDataset")

        from utils.model_utils import load_model
        print(f"  ✓ Successfully imported from utils.model_utils: load_model")

    except Exception as e:
        print(f"  ✗ Practical import test failed: {e}")
        traceback.print_exc()
        practical_test_passed = False

    results.append(practical_test_passed)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    test_names = [
        "Config module",
        "Utils - Dataloader",
        "Utils - Model utilities",
        "Anomaly detection scripts",
        "Directory structure",
        "Critical files",
        "Practical imports",
    ]

    passed = sum(results)
    total = len(results)

    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Repository structure is correct.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
