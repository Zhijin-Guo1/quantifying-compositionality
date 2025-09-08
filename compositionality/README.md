# Compositionality Analysis Module

Unified framework for quantifying compositionality between embeddings and structured attributes using the two-step method from "Quantifying Compositionality of Classic and State-of-the-Art Embeddings".

## Overview

This module implements:
1. **CCA (Canonical Correlation Analysis)** - Measures linear alignment
2. **Linear Decomposition** - Tests additive compositionality
3. **Evaluation Metrics** - Cosine similarity, L2 loss, Hits@k
4. **Permutation Testing** - Statistical significance assessment

## Structure

```
compositionality/
├── cca.py                 # CCA analysis with permutation testing
├── linear_decomposition.py # Pseudo-inverse based reconstruction
├── metrics.py             # Evaluation metrics (cosine, L2, Hits@k)
├── analyzer.py            # Unified analyzer combining all methods
└── __init__.py           # Module exports
```

## Key Components

### 1. CCA Analysis (`cca.py`)
- Computes canonical correlations between embeddings and attributes
- Compares real correlations with shuffled baselines
- Provides statistical significance testing

### 2. Linear Decomposition (`linear_decomposition.py`)
- Learns attribute embeddings via pseudo-inverse
- Reconstructs entity embeddings from attributes
- Leave-one-out evaluation for generalization testing

### 3. Metrics (`metrics.py`)
- **Cosine Similarity**: Measures angular similarity
- **L2 Loss**: Reconstruction error
- **Hits@k**: Retrieval accuracy at different k values
- **Permutation Testing**: Random baseline comparisons

### 4. Unified Analyzer (`analyzer.py`)
- Combines all methods in a single pipeline
- Automatic data preprocessing (grouping by attributes)
- Comprehensive visualization of results

## Usage

### Basic Example

```python
from compositionality import CompositionalityAnalyzer

# Initialize analyzer
analyzer = CompositionalityAnalyzer(
    cca_components=10,
    decomposition_method='pseudo_inverse'
)

# Analyze compositionality
results = analyzer.analyze_compositionality(
    embeddings=embeddings,  # (n_samples, embedding_dim)
    attributes=attributes,  # (n_samples, n_attributes)
    methods=['cca', 'decomposition', 'metrics'],
    n_permutations=100,
    n_trials=100
)

# Visualize results
analyzer.plot_results(results, save_path='results.png')
```

### Interpreting Results

#### Overall Score
- **>0.7**: Strong compositionality - embeddings capture attributes well
- **0.4-0.7**: Moderate compositionality - partial alignment
- **<0.4**: Weak compositionality - embeddings don't reflect attributes

#### Individual Metrics

1. **CCA Score**: Linear correlation strength (0-1)
2. **Decomposition Score**: Reconstruction quality (0-1)
3. **Cosine Similarity**: Angular similarity between original and reconstructed
4. **Hits@k**: Percentage of correct retrievals in top-k

#### Statistical Significance
- P-values < 0.05 indicate significant compositionality
- Compares against random permutation baselines

## Advanced Features

### Layer-wise Analysis
Analyze compositionality across transformer layers:

```python
for layer in range(n_layers):
    embeddings = extractor.extract(sentences, layer=layer)
    results = analyzer.analyze_compositionality(embeddings, attributes)
    print(f"Layer {layer}: {results['overall_compositionality_score']:.4f}")
```

### Custom Metrics
Add your own evaluation metrics:

```python
from compositionality.metrics import CompositionalityMetrics

metrics = CompositionalityMetrics()
custom_results = metrics.compute_all_metrics(
    attributes, 
    embeddings,
    n_permutations=100
)
```

### Grouped Analysis
Automatically groups samples with identical attributes:

```python
# Handles duplicate attribute combinations
results = analyzer.analyze_compositionality(
    embeddings, 
    attributes,
    group_by_attributes=True  # Automatic grouping
)
```

## Visualization

The analyzer provides comprehensive plots:
- CCA correlation curves (real vs permuted)
- L2 loss distributions
- Cosine similarity histograms
- Retrieval accuracy bars
- P-value significance tests
- Overall compositionality scores

## Mathematical Background

### CCA
Finds linear transformations that maximize correlation:
```
max corr(Xa, Yb) subject to ||a||=||b||=1
```

### Linear Decomposition
Learns attribute embeddings via pseudo-inverse:
```
A = B^+ E
where B is attributes, E is embeddings, B^+ is pseudo-inverse
```

### Leave-One-Out
Tests generalization by predicting held-out samples:
```
1. Train on n-1 samples
2. Predict held-out sample
3. Measure similarity/accuracy
```

## Performance Considerations

- **Grouping**: Reduces computation for duplicate attributes
- **Batch Processing**: Efficient matrix operations
- **Permutation Count**: Balance accuracy vs speed (50-100 typical)
- **Trial Count**: More trials improve leave-one-out estimates

## Requirements

- `numpy`: Array operations
- `scikit-learn`: CCA, metrics
- `scipy`: Statistical functions
- `matplotlib`: Visualization

## Citation

Based on the methodology from:
```
Guo et al. (2024). "Quantifying Compositionality of Classic and 
State-of-the-Art Embeddings"
```