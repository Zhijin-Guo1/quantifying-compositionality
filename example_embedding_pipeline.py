"""
Unified embedding extraction pipeline example.

This script demonstrates how to extract embeddings from different domains:
1. Sentence embeddings using SBERT (with layer-wise extraction)
2. Word embeddings using Word2Vec
3. Knowledge Graph embeddings from pre-trained models
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import embedding extractors
from embeddings import (
    SentenceBERTExtractor,
    Word2VecExtractor,
    KGEmbeddingLoader
)


def extract_sentence_embeddings_demo():
    """Demonstrate sentence embedding extraction with SBERT."""
    print("\n" + "="*60)
    print("SENTENCE EMBEDDING EXTRACTION (SBERT)")
    print("="*60)
    
    # Sample sentences
    sentences = [
        "I'd like to book a table for dinner tomorrow.",
        "Can you show me flights from New York to London?",
        "What movies are playing tonight?",
        "I need a hotel room for next weekend.",
        "Book me an appointment with the dentist."
    ]
    
    # Initialize SBERT extractor
    sbert_extractor = SentenceBERTExtractor(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    # Extract embeddings from final layer
    print("\n1. Extracting final layer embeddings...")
    final_embeddings = sbert_extractor.extract(
        sentences,
        layer=None,  # Use final layer
        normalize=True
    )
    print(f"   Shape: {final_embeddings.shape}")
    print(f"   Sample embedding (first 5 dims): {final_embeddings[0][:5]}")
    
    # Extract embeddings from specific layer
    print("\n2. Extracting layer 4 embeddings...")
    layer4_embeddings = sbert_extractor.extract(
        sentences,
        layer=4,
        normalize=True
    )
    print(f"   Shape: {layer4_embeddings.shape}")
    
    # Extract embeddings from all layers
    print("\n3. Extracting embeddings from all layers...")
    all_layer_embeddings = sbert_extractor.extract_all_layers(
        sentences,
        normalize=True
    )
    print(f"   Number of layers: {len(all_layer_embeddings)}")
    for layer_idx, embeddings in all_layer_embeddings.items():
        print(f"   Layer {layer_idx}: shape {embeddings.shape}")
    
    # Save embeddings
    save_path = 'output/sentence_embeddings.npz'
    sbert_extractor.save_embeddings(final_embeddings, sentences, save_path)
    print(f"\n4. Saved embeddings to {save_path}")
    
    return final_embeddings, sentences


def extract_word_embeddings_demo():
    """Demonstrate word embedding extraction with Word2Vec."""
    print("\n" + "="*60)
    print("WORD EMBEDDING EXTRACTION (Word2Vec)")
    print("="*60)
    
    # Sample words
    words = [
        "book", "booking", "booked", "books",
        "fly", "flying", "flight", "flights",
        "hotel", "restaurant", "movie", "appointment"
    ]
    
    # Initialize Word2Vec extractor with pretrained model
    print("\n1. Loading pretrained Word2Vec model...")
    print("   (This may take a while for first-time download)")
    
    try:
        # Try to use a smaller pretrained model for demo
        word2vec_extractor = Word2VecExtractor(
            pretrained_model='glove-wiki-gigaword-100',  # Smaller model
            use_pretrained=True
        )
    except:
        print("   Failed to load pretrained model. Training custom model instead...")
        # Train a simple model if pretrained fails
        sample_sentences = [
            ["book", "table", "dinner", "restaurant"],
            ["book", "flight", "travel", "airplane"],
            ["movie", "theater", "watch", "film"],
            ["hotel", "room", "stay", "night"],
            ["appointment", "doctor", "schedule", "meeting"]
        ]
        
        word2vec_extractor = Word2VecExtractor(use_pretrained=False)
        word2vec_extractor.train_model(
            sample_sentences,
            embedding_dim=100,
            min_count=1,
            epochs=10
        )
    
    # Extract word embeddings
    print("\n2. Extracting word embeddings...")
    word_embeddings = word2vec_extractor.extract(
        words,
        use_zero_for_oov=True,
        normalize=False
    )
    print(f"   Shape: {word_embeddings.shape}")
    print(f"   Sample embedding (first 5 dims): {word_embeddings[0][:5]}")
    
    # Extract sentence embeddings by aggregating words
    print("\n3. Extracting sentence embeddings via word aggregation...")
    sample_sentences = [
        "book a table",
        "flight to london",
        "hotel room available"
    ]
    sentence_embeddings = word2vec_extractor.extract_sentence_embeddings(
        sample_sentences,
        aggregation='mean',
        normalize=True
    )
    print(f"   Shape: {sentence_embeddings.shape}")
    
    # Find similar words
    print("\n4. Finding similar words to 'book':")
    try:
        similar_words = word2vec_extractor.get_most_similar('book', topn=5)
        for word, score in similar_words:
            print(f"   {word}: {score:.4f}")
    except:
        print("   Word not in vocabulary or model doesn't support similarity")
    
    # Save embeddings
    save_path = 'output/word_embeddings.npz'
    word2vec_extractor.save_embeddings(word_embeddings, words, save_path)
    print(f"\n5. Saved embeddings to {save_path}")
    
    return word_embeddings, words


def extract_kg_embeddings_demo():
    """Demonstrate KG embedding loading from pre-trained models."""
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH EMBEDDING EXTRACTION")
    print("="*60)
    
    # Load TransE embeddings
    print("\n1. Loading TransE embeddings...")
    transe_loader = KGEmbeddingLoader(
        model_type='TransE',
        kg_embedding_dir='KG_embedding'
    )
    
    # Get checkpoint info
    info = transe_loader.get_checkpoint_info()
    print(f"   Model type: {info['model_type']}")
    print(f"   Number of entities: {info['num_entities']}")
    print(f"   Embedding dimension: {info['embedding_dim']}")
    
    # Extract embeddings for specific entity IDs
    print("\n2. Extracting entity embeddings...")
    # Use integer IDs as example (in real use, these would correspond to actual entities)
    entity_ids = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100]
    entity_embeddings = transe_loader.extract(
        entity_ids,
        normalize=False
    )
    print(f"   Shape: {entity_embeddings.shape}")
    print(f"   Sample embedding (first 5 dims): {entity_embeddings[0][:5]}")
    
    # Load DistMult embeddings
    print("\n3. Loading DistMult embeddings...")
    distmult_loader = KGEmbeddingLoader(
        model_type='DistMult',
        kg_embedding_dir='KG_embedding'
    )
    
    # Extract all entity embeddings
    print("\n4. Extracting all entity embeddings from DistMult...")
    all_entities = distmult_loader.extract_all_entities(normalize=True)
    print(f"   Shape: {all_entities.shape}")
    print(f"   Min value: {all_entities.min():.4f}")
    print(f"   Max value: {all_entities.max():.4f}")
    print(f"   Mean value: {all_entities.mean():.4f}")
    
    # Save as numpy for faster loading
    save_path = 'output/kg_embeddings_transe.npz'
    transe_loader.save_as_numpy(save_path)
    print(f"\n5. Saved TransE embeddings to {save_path}")
    
    return entity_embeddings, entity_ids


def compare_embeddings(embeddings_dict: Dict[str, np.ndarray]):
    """Compare statistics across different embedding types."""
    print("\n" + "="*60)
    print("EMBEDDING COMPARISON")
    print("="*60)
    
    for name, embeddings in embeddings_dict.items():
        print(f"\n{name}:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Dimension: {embeddings.shape[1]}")
        print(f"  L2 norm (mean): {np.linalg.norm(embeddings, axis=1).mean():.4f}")
        print(f"  Sparsity: {(embeddings == 0).sum() / embeddings.size:.4f}")
        print(f"  Value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")


def main():
    """Run the complete embedding extraction pipeline."""
    print("\n" + "="*60)
    print("UNIFIED EMBEDDING EXTRACTION PIPELINE")
    print("="*60)
    
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    # Extract embeddings from different domains
    embeddings_dict = {}
    
    # 1. Sentence embeddings
    try:
        sentence_embeddings, sentences = extract_sentence_embeddings_demo()
        embeddings_dict['Sentence (SBERT)'] = sentence_embeddings
    except Exception as e:
        logger.error(f"Failed to extract sentence embeddings: {e}")
    
    # 2. Word embeddings
    try:
        word_embeddings, words = extract_word_embeddings_demo()
        embeddings_dict['Word (Word2Vec)'] = word_embeddings
    except Exception as e:
        logger.error(f"Failed to extract word embeddings: {e}")
    
    # 3. KG embeddings
    try:
        kg_embeddings, entity_ids = extract_kg_embeddings_demo()
        embeddings_dict['KG (TransE)'] = kg_embeddings
    except Exception as e:
        logger.error(f"Failed to extract KG embeddings: {e}")
    
    # Compare embeddings
    if embeddings_dict:
        compare_embeddings(embeddings_dict)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Load attribute matrices using the attributes module")
    print("2. Perform compositionality analysis (CCA, linear decomposition)")
    print("3. Evaluate with leave-one-out and permutation tests")
    print("4. Generate visualization and statistical reports")


if __name__ == "__main__":
    main()