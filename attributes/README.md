# Attribute Generation Framework

This module provides a unified framework for extracting and encoding attributes from different data modalities for compositionality analysis.

## Overview

The framework implements attribute extraction for three domains:
- **Sentences**: Semantic concepts from Schema-Guided Dialogue
- **Words**: Morphological suffixes from MorphoLex
- **Knowledge Graphs**: User demographics from MovieLens

## Architecture

```
attributes/
├── base.py                  # Abstract base class
├── sentence_attributes.py   # Sentence concept extraction
├── word_attributes.py       # Word morphology extraction
└── kg_attributes.py         # KG demographic extraction
```

## Base Class

All attribute generators inherit from `BaseAttributeGenerator` which provides:
- Unified interface for attribute generation
- Caching support for processed attributes
- Validation and metadata computation
- Statistics and analysis methods

## Usage

### Sentence Concepts

```python
from attributes import SentenceConceptAttributes

generator = SentenceConceptAttributes(
    data_path=Path("data/output"),
    cache_dir=Path("cache/attributes"),
    min_concepts=3,
    max_concepts=4
)

results = generator.generate()
# results['matrix']         # Binary matrix (n_sentences x n_concepts)
# results['entity_ids']     # List of sentence IDs
# results['attribute_names'] # List of concept names
```

### Word Morphology

```python
from attributes import WordMorphologyAttributes

generator = WordMorphologyAttributes(
    data_path=Path("data/MorphoLEX_en.xlsx"),
    min_suffix_frequency=10
)

results = generator.generate()
# Get word decomposition
suffixes = generator.get_word_decomposition("weightily")
# ['weight', 'ly']
```

### KG Demographics

```python
from attributes import KGDemographicAttributes

generator = KGDemographicAttributes(
    data_path=Path("data/ml-1m"),
    include_occupation=True
)

results = generator.generate()
# Get user demographics
user_info = generator.get_user_demographics("user_1")
```

## Specialized Variants

### For CCA Analysis
- `KGDemographicAttributesForCCA`: Simplified attributes (gender + age only)

### For Additive Compositionality
- `WordMorphologyAdditive`: Root + suffix decomposition
- `KGDemographicAttributesForAdditive`: Grouped by demographic combinations

## Output Format

All generators return a dictionary with:
```python
{
    'matrix': np.ndarray,        # Binary attribute matrix
    'entity_ids': List[str],     # Entity identifiers
    'attribute_names': List[str], # Attribute names
    'metadata': {
        'n_entities': int,
        'n_attributes': int,
        'sparsity': float,
        'mean_attributes_per_entity': float,
        'attribute_frequency': np.ndarray
    }
}
```

## Caching

Processed attributes can be cached to disk:
```python
generator = SentenceConceptAttributes(
    data_path=Path("data/output"),
    cache_dir=Path("cache/attributes")  # Enable caching
)

# First call processes and caches
results = generator.generate()

# Subsequent calls load from cache
results = generator.generate()

# Force regeneration
results = generator.generate(force_regenerate=True)
```

## Testing

Run the test suite:
```bash
python test_attributes.py
```

This will test all attribute generators and verify they produce valid output.