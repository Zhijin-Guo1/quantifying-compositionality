# Quantifying Compositionality - Architecture

## Overview

This repository implements the methodology from "Quantifying Compositionality of Classic and State-of-the-Art Embeddings" paper. The codebase provides a unified framework for measuring compositional structure in embeddings across three domains: sentences, words, and knowledge graphs.

## Core Methodology

The framework implements a two-step diagnostic approach:

1. **Linearity Assessment (CCA)**: Measures linear alignment between binary attributes and continuous embeddings
2. **Additive Generalization (Leave-One-Out)**: Tests if embeddings can be reconstructed from attribute combinations

## Project Structure

```
quantifying-compositionality/
├── attributes/              # Attribute generation framework
│   ├── base.py             # Abstract base class
│   ├── sentence_attributes.py  # Sentence concepts
│   ├── word_attributes.py      # Word morphology
│   └── kg_attributes.py        # User demographics
│
├── embeddings/             # Embedding generation (TODO)
│   ├── sentence_embeddings.py  # SBERT, GPT, Llama
│   ├── word_embeddings.py      # Word2Vec
│   └── kg_embeddings.py        # TransE, DistMult
│
├── compositionality/       # Analysis framework (TODO)
│   ├── cca.py             # Canonical Correlation Analysis
│   ├── linear_decomposition.py # Linear system solver
│   └── evaluation.py      # Metrics and visualization
│
├── data/                   # Raw data
│   ├── ml-1m/             # MovieLens dataset
│   ├── MorphoLEX_en.xlsx  # Word morphology
│   └── output/            # Schema-Guided Dialogue
│
├── KG_embedding/          # Pretrained KG models
│   ├── 300_epochs_DistMult_gpu34.pt
│   └── 300_epochs_TransE_gpu34.pt
│
├── notebooks/             # Original experiments
│   ├── sentence_concept.ipynb
│   ├── compositionality_KG.ipynb
│   └── morphology_linear_decomposition.ipynb
│
└── figures/               # Paper figures
```

## Implementation Status

### ✅ Completed

1. **Attribute Generation Framework**
   - Base abstract class with caching and validation
   - Sentence concept extraction from SGD dataset
   - Word morphology extraction from MorphoLex
   - User demographic extraction from MovieLens
   - Specialized variants for CCA and additive experiments

### 🚧 TODO

1. **Embedding Generation Framework**
   - Unified interface for all embedding types
   - SBERT/GPT/Llama sentence embeddings
   - Word2Vec word embeddings
   - TransE/DistMult KG embeddings

2. **Compositionality Analysis**
   - CCA implementation with permutation testing
   - Linear decomposition with leave-one-out
   - Evaluation metrics (L2, cosine, retrieval)
   - Statistical significance testing
   - Visualization utilities

3. **Experiment Pipeline**
   - End-to-end experiment scripts
   - Cross-layer analysis for transformers
   - Training stage analysis
   - Results aggregation and reporting

## Usage

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Generating Attributes

```python
from attributes import SentenceConceptAttributes

# Generate sentence concept attributes
generator = SentenceConceptAttributes(
    data_path=Path("data/output"),
    cache_dir=Path("cache/attributes")
)
results = generator.generate()

# Access the binary attribute matrix
attribute_matrix = results['matrix']  # (n_sentences, n_concepts)
entity_ids = results['entity_ids']
concept_names = results['attribute_names']
```

### Testing

```bash
# Test attribute generators
python test_attributes.py
```

## Key Design Principles

1. **Unified Interface**: All generators follow the same API pattern
2. **Caching**: Processed attributes are cached for efficiency
3. **Validation**: Automatic validation of generated matrices
4. **Metadata**: Rich metadata about attributes and statistics
5. **Modularity**: Clean separation between data loading, processing, and analysis

## Data Requirements

### Sentences (Schema-Guided Dialogue)
- Location: `data/output/dialogue_data.csv`
- Format: CSV with sentences and concept annotations
- Size: ~2,500 sentences with 47 concepts

### Words (MorphoLex)
- Location: `data/MorphoLEX_en.xlsx`
- Format: Excel with morphological features
- Size: ~68,000 words with suffix annotations

### Knowledge Graph (MovieLens)
- Location: `data/ml-1m/`
- Files: `users.dat`, `movies.dat`, `ratings.dat`
- Size: 6,040 users, 3,900 movies, 1M ratings

## Paper Reference

"Quantifying Compositionality of Classic and State-of-the-Art Embeddings"
- Two-step diagnostic: CCA for linearity, Linear Decomposition for generalization
- Three domains: sentences (concepts), words (morphology), KG (demographics)
- Key finding: Embeddings show increasing compositionality during training