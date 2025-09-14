#!/usr/bin/env python
"""Check if sentence experiment prerequisites are installed correctly."""

import os
import sys


def check_sentence_experiment_setup():
    """Check if all requirements for sentence experiment are met."""

    print("="*70)
    print("Checking Sentence Experiment Setup")
    print("="*70)

    all_good = True

    # Check data files
    print("\n1. Checking sentence data files...")
    data_dir = "data/sentence"
    user_texts_path = os.path.join(data_dir, "user_texts.txt")
    dialogue_csv_path = os.path.join(data_dir, "dialogue_data.csv")

    if os.path.exists(user_texts_path):
        with open(user_texts_path, 'r') as f:
            n_sentences = len([line for line in f if line.strip()])
        print(f"   ✓ user_texts.txt found ({n_sentences} sentences)")
    else:
        print(f"   ✗ user_texts.txt not found at {user_texts_path}")
        all_good = False

    if os.path.exists(dialogue_csv_path):
        size_mb = os.path.getsize(dialogue_csv_path) / (1024 * 1024)
        print(f"   ✓ dialogue_data.csv found ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ dialogue_data.csv not found at {dialogue_csv_path}")
        all_good = False

    # Check Python packages
    print("\n2. Checking Python packages...")
    required_packages = {
        'sentence_transformers': 'SBERT models',
        'transformers': 'Transformer models',
        'torch': 'PyTorch backend',
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

    # Check SBERT model availability
    print("\n3. Checking SBERT model...")
    try:
        from sentence_transformers import SentenceTransformer
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        print(f"   Loading {model_name}...")
        model = SentenceTransformer(model_name)
        print(f"   ✓ SBERT model loaded successfully")
        print(f"     - Embedding dimension: {model.get_sentence_embedding_dimension()}")
        print(f"     - Max sequence length: {model.max_seq_length}")
    except Exception as e:
        print(f"   ✗ Failed to load SBERT model: {e}")
        all_good = False

    # Summary
    print("\n" + "="*70)
    if all_good:
        print("✓ All requirements met! You can run the sentence experiment:")
        print("\n  python run_experiments.py --experiment sentence --plot --verbose")
        print("\nThis will:")
        print("  - Load 2,458 sentences from data/sentence/user_texts.txt")
        print("  - Load attributes from data/sentence/dialogue_data.csv")
        print("  - Extract layer 6 embeddings using all-MiniLM-L6-v2")
        print("  - Perform CCA with 15 components")
        print("  - Group by unique slot combinations for decomposition")
    else:
        print("✗ Some requirements are missing. Please install them first.")
        print("\nFor the data files, ensure you have:")
        print("  - data/sentence/user_texts.txt")
        print("  - data/sentence/dialogue_data.csv")
    print("="*70)

    return all_good


if __name__ == "__main__":
    success = check_sentence_experiment_setup()
    sys.exit(0 if success else 1)