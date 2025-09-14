#!/usr/bin/env python
"""Check if word experiment prerequisites are installed correctly."""

import os
import sys


def check_word_experiment_setup():
    """Check if all requirements for word experiment are met."""

    print("="*70)
    print("Checking Word Experiment Setup")
    print("="*70)

    all_good = True

    # Check MorphoLEX
    print("\n1. Checking MorphoLEX dataset...")
    morpholex_path = "data/MorphoLEX_en.xlsx"
    if os.path.exists(morpholex_path):
        size_mb = os.path.getsize(morpholex_path) / (1024 * 1024)
        print(f"   ✓ MorphoLEX found at {morpholex_path} ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ MorphoLEX not found at {morpholex_path}")
        print("     Download from: http://www.lexique.org/?page_id=250")
        all_good = False

    # Check GoogleNews Word2Vec
    print("\n2. Checking GoogleNews Word2Vec model...")
    googlenews_paths = [
        "data/GoogleNews-vectors-negative300.bin.gz",
        "data/GoogleNews-vectors-negative300.bin",
        os.path.expanduser("~/Downloads/GoogleNews-vectors-negative300.bin.gz"),
        os.path.expanduser("~/Downloads/GoogleNews-vectors-negative300.bin"),
    ]

    model_found = False
    for path in googlenews_paths:
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / (1024**3)
            print(f"   ✓ GoogleNews model found at {path} ({size_gb:.2f} GB)")
            model_found = True
            break

    if not model_found:
        print("   ✗ GoogleNews model not found")
        print("     Download GoogleNews-vectors-negative300.bin.gz from:")
        print("     https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/")
        print("     Save to: data/GoogleNews-vectors-negative300.bin.gz")
        all_good = False

    # Check Python packages
    print("\n3. Checking Python packages...")
    required_packages = {
        'gensim': 'Word2Vec loading',
        'openpyxl': 'MorphoLEX Excel reading',
        'sklearn': 'CCA analysis',
        'numpy': 'Numerical operations',
        'pandas': 'Data processing',
        'matplotlib': 'Plotting'
    }

    for package, purpose in required_packages.items():
        try:
            __import__(package)
            print(f"   ✓ {package} installed ({purpose})")
        except ImportError:
            print(f"   ✗ {package} not installed (needed for {purpose})")
            all_good = False

    # Summary
    print("\n" + "="*70)
    if all_good:
        print("✓ All requirements met! You can run the word experiment:")
        print("\n  python run_experiments.py --experiment word --plot --verbose")
    else:
        print("✗ Some requirements are missing. Please install them first.")
        print("\nFor detailed instructions, see the README.md file.")
    print("="*70)

    return all_good


if __name__ == "__main__":
    success = check_word_experiment_setup()
    sys.exit(0 if success else 1)