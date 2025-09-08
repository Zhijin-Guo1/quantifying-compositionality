"""
Test script for attribute generators.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from attributes import (
    SentenceConceptAttributes,
    WordMorphologyAttributes,
    WordMorphologyAdditive,
    KGDemographicAttributes,
    KGDemographicAttributesForCCA,
    KGDemographicAttributesForAdditive
)


def test_sentence_attributes():
    """Test sentence concept attribute generation."""
    print("\n" + "="*60)
    print("Testing Sentence Concept Attributes")
    print("="*60)
    
    generator = SentenceConceptAttributes(
        data_path=Path("data/output"),
        cache_dir=Path("cache/attributes")
    )
    
    try:
        results = generator.generate()
        
        print(f"✓ Generated attributes for {results['metadata']['n_entities']} sentences")
        print(f"✓ Number of concepts: {results['metadata']['n_attributes']}")
        print(f"✓ Mean concepts per sentence: {results['metadata']['mean_attributes_per_entity']:.2f}")
        print(f"✓ Matrix sparsity: {results['metadata']['sparsity']:.2%}")
        
        # Show some statistics
        stats = generator.get_statistics()
        print("\nTop 5 most frequent concepts:")
        print(stats.head())
        
        # Show concept combinations
        combos = generator.get_concept_combinations()
        print(f"\n✓ Found {len(combos)} unique concept combinations")
        print("Top 5 combinations:")
        print(combos.head())
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_word_morphology_attributes():
    """Test word morphology attribute generation."""
    print("\n" + "="*60)
    print("Testing Word Morphology Attributes")
    print("="*60)
    
    generator = WordMorphologyAttributes(
        data_path=Path("data/MorphoLEX_en.xlsx"),
        cache_dir=Path("cache/attributes"),
        min_suffix_frequency=10
    )
    
    try:
        results = generator.generate()
        
        print(f"✓ Generated attributes for {results['metadata']['n_entities']} words")
        print(f"✓ Number of suffixes: {results['metadata']['n_attributes']}")
        print(f"✓ Mean suffixes per word: {results['metadata']['mean_attributes_per_entity']:.2f}")
        print(f"✓ Matrix sparsity: {results['metadata']['sparsity']:.2%}")
        
        # Show suffix statistics
        suffix_stats = generator.get_suffix_statistics()
        print("\nTop 5 most frequent suffixes:")
        print(suffix_stats.head())
        
        # Test word decomposition
        test_words = ['weightier', 'weightiest', 'weightily']
        print("\nWord decompositions:")
        for word in test_words:
            if word in generator.entity_ids:
                suffixes = generator.get_word_decomposition(word)
                print(f"  {word}: {suffixes}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_word_morphology_additive():
    """Test additive word morphology attributes."""
    print("\n" + "="*60)
    print("Testing Word Morphology Additive Attributes")
    print("="*60)
    
    generator = WordMorphologyAdditive(
        data_path=Path("data/MorphoLEX_en.xlsx"),
        cache_dir=Path("cache/attributes"),
        n_suffixes_required=3,
        min_suffix_frequency=10
    )
    
    try:
        results = generator.generate()
        
        print(f"✓ Generated attributes for {results['metadata']['n_entities']} words")
        print(f"✓ Number of attributes (roots + suffixes): {results['metadata']['n_attributes']}")
        print(f"✓ Mean attributes per word: {results['metadata']['mean_attributes_per_entity']:.2f}")
        
        # Count roots and suffixes
        n_roots = sum(1 for name in results['attribute_names'] if name.startswith('root:'))
        n_suffixes = sum(1 for name in results['attribute_names'] if name.startswith('suffix:'))
        print(f"✓ {n_roots} roots + {n_suffixes} suffixes")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_kg_demographic_attributes():
    """Test KG demographic attribute generation."""
    print("\n" + "="*60)
    print("Testing KG Demographic Attributes")
    print("="*60)
    
    generator = KGDemographicAttributes(
        data_path=Path("data/ml-1m"),
        cache_dir=Path("cache/attributes"),
        include_occupation=True
    )
    
    try:
        results = generator.generate()
        
        print(f"✓ Generated attributes for {results['metadata']['n_entities']} users")
        print(f"✓ Number of attributes: {results['metadata']['n_attributes']}")
        print(f"✓ Mean attributes per user: {results['metadata']['mean_attributes_per_entity']:.2f}")
        
        # Show demographic statistics
        demo_stats = generator.get_demographic_statistics()
        print("\nDemographic distribution:")
        for category in demo_stats['category'].unique():
            cat_stats = demo_stats[demo_stats['category'] == category]
            print(f"\n{category}:")
            for _, row in cat_stats.head(5).iterrows():
                print(f"  {row['value']}: {row['count']} ({row['percentage']:.1f}%)")
        
        # Test user lookup
        test_user = generator.entity_ids[0] if generator.entity_ids else None
        if test_user:
            user_info = generator.get_user_demographics(test_user)
            print(f"\nExample user ({test_user}):")
            print(f"  Gender: {user_info['gender']}")
            print(f"  Age: {user_info['age']} ({user_info['age_group']})")
            print(f"  Occupation: {user_info['occupation']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_kg_demographic_cca():
    """Test KG demographic attributes for CCA."""
    print("\n" + "="*60)
    print("Testing KG Demographic Attributes for CCA")
    print("="*60)
    
    generator = KGDemographicAttributesForCCA(
        data_path=Path("data/ml-1m"),
        cache_dir=Path("cache/attributes")
    )
    
    try:
        results = generator.generate()
        
        print(f"✓ Generated CCA attributes for {results['metadata']['n_entities']} users")
        print(f"✓ Number of attributes (gender + age): {results['metadata']['n_attributes']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_kg_demographic_additive():
    """Test KG demographic attributes for additive experiments."""
    print("\n" + "="*60)
    print("Testing KG Demographic Attributes for Additive")
    print("="*60)
    
    generator = KGDemographicAttributesForAdditive(
        data_path=Path("data/ml-1m"),
        cache_dir=Path("cache/attributes")
    )
    
    try:
        results = generator.generate()
        grouped = generator.get_grouped_attributes()
        
        print(f"✓ Generated additive attributes for {results['metadata']['n_entities']} users")
        print(f"✓ Grouped into {len(grouped['combination_ids'])} demographic combinations")
        print("\nExample combinations:")
        for combo in grouped['combination_ids'][:5]:
            print(f"  {combo}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING ATTRIBUTE GENERATORS")
    print("="*60)
    
    tests = [
        ("Sentence Concepts", test_sentence_attributes),
        ("Word Morphology", test_word_morphology_attributes),
        ("Word Morphology Additive", test_word_morphology_additive),
        ("KG Demographics", test_kg_demographic_attributes),
        ("KG Demographics CCA", test_kg_demographic_cca),
        ("KG Demographics Additive", test_kg_demographic_additive),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with unexpected error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)