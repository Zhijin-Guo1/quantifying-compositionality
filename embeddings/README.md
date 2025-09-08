# Embedding Extraction Module

Unified framework for extracting embeddings across three domains: sentences, words, and knowledge graphs.

## Structure

```
embeddings/
├── base.py              # Abstract base class for all extractors
├── sentence_bert.py     # SBERT extractor with layer-wise support
├── word2vec.py         # Word2Vec extractor for word embeddings
├── kg_embeddings.py    # Loader for pre-trained KG embeddings
└── __init__.py         # Module exports
```

## Features

### 1. Sentence Embeddings (SBERT)
- **Models**: Any sentence-transformers model
- **Layer-wise extraction**: Extract from any layer (0 to n)
- **Batch processing**: Efficient GPU utilization
- **Normalization**: L2 normalization support

```python
from embeddings import SentenceBERTExtractor

# Initialize
extractor = SentenceBERTExtractor(model_name='all-MiniLM-L6-v2')

# Extract from final layer
embeddings = extractor.extract(sentences)

# Extract from specific layer
layer4_embeddings = extractor.extract(sentences, layer=4)

# Extract from all layers
all_layers = extractor.extract_all_layers(sentences)
```

### 2. Word Embeddings (Word2Vec)
- **Pretrained models**: Load from gensim-data
- **Custom training**: Train on your corpus
- **OOV handling**: Zero vectors or random initialization
- **Sentence aggregation**: Mean, sum, or max pooling

```python
from embeddings import Word2VecExtractor

# Use pretrained
extractor = Word2VecExtractor(pretrained_model='glove-wiki-gigaword-100')

# Extract word embeddings
word_embeddings = extractor.extract(words)

# Aggregate to sentences
sentence_embeddings = extractor.extract_sentence_embeddings(sentences)
```

### 3. KG Embeddings
- **Pre-trained models**: TransE and DistMult
- **Flexible loading**: From .pt checkpoint files
- **Entity mapping**: Support for entity ID mappings
- **Relation embeddings**: Access relation vectors

```python
from embeddings import KGEmbeddingLoader

# Load TransE
loader = KGEmbeddingLoader(model_type='TransE')

# Extract entity embeddings
embeddings = loader.extract(entity_ids)

# Get all entities
all_embeddings = loader.extract_all_entities()
```

## Usage Example

Run the complete pipeline:

```bash
python example_embedding_pipeline.py
```

This demonstrates:
- SBERT layer-wise extraction
- Word2Vec embedding extraction
- KG embedding loading
- Comparative statistics

## Output Format

All extractors produce numpy arrays:
- **Shape**: `(n_samples, embedding_dim)`
- **Type**: `float32` or `float64`
- **Normalization**: Optional L2 normalization

## Integration with Compositionality Analysis

These embeddings serve as distributional representations to compare against structured attribute vectors:

```python
# 1. Extract embeddings (distributional)
embeddings = extractor.extract(data)

# 2. Extract attributes (structured)
attributes = attribute_extractor.extract(data)

# 3. Analyze compositionality
results = analyzer.analyze_compositionality(embeddings, attributes)
```

## Requirements

- `torch`: PyTorch for model loading
- `sentence-transformers`: SBERT models
- `transformers`: Hugging Face transformers
- `gensim`: Word2Vec models
- `numpy`: Array operations

## Notes

- GPU acceleration supported for SBERT
- Caching available for pretrained models
- Memory-efficient batch processing
- Compatible with compositionality analysis framework