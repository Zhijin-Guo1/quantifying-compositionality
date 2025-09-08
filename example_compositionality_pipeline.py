"""
Complete pipeline for quantifying compositionality.

This script demonstrates the full workflow:
1. Extract attributes from data (structured representation)
2. Extract embeddings from data (distributional representation)
3. Analyze compositionality between the two representations
"""

import numpy as np
import logging
import os
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import attribute extractors
from attributes import (
    SentenceAttributeExtractor,
    WordAttributeExtractor,
    KGAttributeExtractor
)

# Import embedding extractors
from embeddings import (
    SentenceBERTExtractor,
    Word2VecExtractor,
    KGEmbeddingLoader
)

# Import compositionality analyzer
from compositionality import CompositionalityAnalyzer


def analyze_sentence_compositionality():
    """Analyze compositionality for sentences."""
    print("\n" + "="*70)
    print("SENTENCE COMPOSITIONALITY ANALYSIS")
    print("="*70)
    
    # Sample sentences (in practice, load from your dataset)
    sentences = [
        "I'd like to book a table for two at an Italian restaurant in downtown.",
        "Can you find me a flight from New York to London next Monday?",
        "What movies are playing tonight at the theater near me?",
        "I need a hotel room with a view for this weekend.",
        "Book me an appointment with Dr. Smith for tomorrow afternoon.",
        "Show me vegetarian restaurants that are open now.",
        "Find flights departing before 10 AM to San Francisco.",
        "Are there any action movies playing this evening?",
        "I want to reserve a suite at the Hilton for three nights.",
        "Schedule a dental cleaning for next Tuesday morning."
    ]
    
    # Duplicate some sentences to test grouping
    sentences = sentences * 3  # 30 sentences total
    
    print(f"\nProcessing {len(sentences)} sentences...")
    
    # Step 1: Extract attributes (structured representation)
    print("\n1. Extracting attributes...")
    attr_extractor = SentenceAttributeExtractor(
        concept_vocabulary=['location', 'time', 'service', 'genre', 'quantity']
    )
    attributes, concept_names = attr_extractor.extract(sentences)
    print(f"   Attributes shape: {attributes.shape}")
    print(f"   Concepts: {concept_names[:5]}...")  # Show first 5
    
    # Step 2: Extract embeddings (distributional representation)
    print("\n2. Extracting SBERT embeddings...")
    embed_extractor = SentenceBERTExtractor(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    # Extract from final layer
    embeddings = embed_extractor.extract(sentences, normalize=True)
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Step 3: Analyze compositionality
    print("\n3. Analyzing compositionality...")
    analyzer = CompositionalityAnalyzer(
        cca_components=10,
        decomposition_method='pseudo_inverse'
    )
    
    results = analyzer.analyze_compositionality(
        embeddings=embeddings,
        attributes=attributes,
        methods=['cca', 'decomposition', 'metrics'],
        n_permutations=50,  # Reduced for demo
        n_trials=50,  # Reduced for demo
        group_by_attributes=True,
        verbose=True
    )
    
    # Plot results
    print("\n4. Generating visualization...")
    fig = analyzer.plot_results(results, save_path='output/sentence_compositionality.png')
    
    return results


def analyze_word_compositionality():
    """Analyze compositionality for words."""
    print("\n" + "="*70)
    print("WORD COMPOSITIONALITY ANALYSIS")
    print("="*70)
    
    # Sample words with morphological patterns
    words = [
        # Base forms
        "book", "play", "run", "write", "read",
        # -ing forms
        "booking", "playing", "running", "writing", "reading",
        # -ed forms
        "booked", "played", "ran", "wrote", "read",
        # -er forms
        "booker", "player", "runner", "writer", "reader",
        # -s forms
        "books", "plays", "runs", "writes", "reads"
    ]
    
    print(f"\nProcessing {len(words)} words...")
    
    # Step 1: Extract attributes (morphological features)
    print("\n1. Extracting morphological attributes...")
    attr_extractor = WordAttributeExtractor(attribute_type='morphological')
    attributes, feature_names = attr_extractor.extract(words)
    print(f"   Attributes shape: {attributes.shape}")
    print(f"   Features: {feature_names[:5]}...")
    
    # Step 2: Extract embeddings
    print("\n2. Extracting Word2Vec embeddings...")
    try:
        # Try to use pretrained model
        embed_extractor = Word2VecExtractor(
            pretrained_model='glove-wiki-gigaword-100',
            use_pretrained=True
        )
        embeddings = embed_extractor.extract(words, normalize=False)
    except:
        print("   Failed to load pretrained model. Using random embeddings for demo...")
        # Generate random embeddings for demo
        np.random.seed(42)
        embeddings = np.random.randn(len(words), 100)
    
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Step 3: Analyze compositionality
    print("\n3. Analyzing compositionality...")
    analyzer = CompositionalityAnalyzer(
        cca_components=5,
        decomposition_method='pseudo_inverse'
    )
    
    results = analyzer.analyze_compositionality(
        embeddings=embeddings,
        attributes=attributes,
        methods=['cca', 'decomposition'],
        n_permutations=30,  # Reduced for demo
        n_trials=30,  # Reduced for demo
        group_by_attributes=False,  # Words are unique
        verbose=True
    )
    
    return results


def analyze_kg_compositionality():
    """Analyze compositionality for knowledge graph entities."""
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH COMPOSITIONALITY ANALYSIS")
    print("="*70)
    
    # Check if KG embeddings exist
    kg_path = 'KG_embedding/300_epochs_TransE_gpu34.pt'
    if not os.path.exists(kg_path):
        print(f"KG embeddings not found at {kg_path}")
        print("Skipping KG analysis...")
        return None
    
    # Generate synthetic user data for demo
    n_users = 100
    print(f"\nProcessing {n_users} users...")
    
    # Step 1: Extract attributes (demographics)
    print("\n1. Extracting demographic attributes...")
    attr_extractor = KGAttributeExtractor(attribute_type='demographic')
    
    # Generate synthetic entities with demographic info
    entities = []
    for i in range(n_users):
        entity = {
            'id': i,
            'age': np.random.choice(['young', 'middle', 'senior']),
            'gender': np.random.choice(['M', 'F']),
            'occupation': np.random.choice(['student', 'professional', 'retired'])
        }
        entities.append(entity)
    
    attributes, feature_names = attr_extractor.extract(entities)
    print(f"   Attributes shape: {attributes.shape}")
    print(f"   Features: {feature_names}")
    
    # Step 2: Load KG embeddings
    print("\n2. Loading TransE embeddings...")
    embed_loader = KGEmbeddingLoader(model_type='TransE')
    
    # Extract embeddings for user IDs
    entity_ids = list(range(n_users))
    embeddings = embed_loader.extract(entity_ids, normalize=True)
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Step 3: Analyze compositionality
    print("\n3. Analyzing compositionality...")
    analyzer = CompositionalityAnalyzer(
        cca_components=5,
        decomposition_method='pseudo_inverse'
    )
    
    results = analyzer.analyze_compositionality(
        embeddings=embeddings,
        attributes=attributes,
        methods=['cca', 'decomposition'],
        n_permutations=30,  # Reduced for demo
        n_trials=30,  # Reduced for demo
        group_by_attributes=True,
        verbose=True
    )
    
    return results


def analyze_layerwise_compositionality():
    """Analyze how compositionality changes across SBERT layers."""
    print("\n" + "="*70)
    print("LAYER-WISE COMPOSITIONALITY ANALYSIS")
    print("="*70)
    
    # Sample sentences
    sentences = [
        "Book a table for dinner",
        "Find flights to Paris",
        "Show movie times",
        "Reserve hotel room",
        "Schedule appointment"
    ] * 4  # 20 sentences
    
    # Extract attributes
    print("\n1. Extracting attributes...")
    attr_extractor = SentenceAttributeExtractor()
    attributes, _ = attr_extractor.extract(sentences)
    
    # Initialize SBERT
    print("\n2. Initializing SBERT...")
    embed_extractor = SentenceBERTExtractor()
    n_layers = embed_extractor.get_num_layers()
    print(f"   Model has {n_layers} layers")
    
    # Analyze each layer
    print("\n3. Analyzing compositionality across layers...")
    analyzer = CompositionalityAnalyzer(cca_components=5)
    
    layer_scores = []
    for layer in range(n_layers):
        print(f"\n   Layer {layer}/{n_layers-1}:")
        
        # Extract embeddings from this layer
        embeddings = embed_extractor.extract(sentences, layer=layer, normalize=True)
        
        # Quick analysis (reduced parameters for speed)
        results = analyzer.analyze_compositionality(
            embeddings=embeddings,
            attributes=attributes,
            methods=['cca', 'decomposition'],
            n_permutations=10,
            n_trials=10,
            verbose=False
        )
        
        score = results['overall_compositionality_score']
        layer_scores.append(score)
        print(f"     Compositionality score: {score:.4f}")
    
    # Find best layer
    best_layer = np.argmax(layer_scores)
    print(f"\n4. Results:")
    print(f"   Best layer: {best_layer} (score: {layer_scores[best_layer]:.4f})")
    print(f"   Layer scores: {[f'{s:.3f}' for s in layer_scores]}")
    
    return layer_scores


def main():
    """Run complete compositionality analysis pipeline."""
    print("\n" + "="*70)
    print("QUANTIFYING COMPOSITIONALITY - COMPLETE PIPELINE")
    print("="*70)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Run analyses for different domains
    results = {}
    
    # 1. Sentence compositionality
    try:
        print("\n[1/4] Analyzing sentence compositionality...")
        sentence_results = analyze_sentence_compositionality()
        results['sentences'] = sentence_results
    except Exception as e:
        logger.error(f"Sentence analysis failed: {e}")
    
    # 2. Word compositionality
    try:
        print("\n[2/4] Analyzing word compositionality...")
        word_results = analyze_word_compositionality()
        results['words'] = word_results
    except Exception as e:
        logger.error(f"Word analysis failed: {e}")
    
    # 3. KG compositionality
    try:
        print("\n[3/4] Analyzing KG compositionality...")
        kg_results = analyze_kg_compositionality()
        if kg_results:
            results['kg'] = kg_results
    except Exception as e:
        logger.error(f"KG analysis failed: {e}")
    
    # 4. Layer-wise analysis
    try:
        print("\n[4/4] Analyzing layer-wise compositionality...")
        layer_scores = analyze_layerwise_compositionality()
        results['layers'] = layer_scores
    except Exception as e:
        logger.error(f"Layer-wise analysis failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    
    for domain, domain_results in results.items():
        if domain == 'layers':
            print(f"\n{domain.upper()}:")
            print(f"  Best layer: {np.argmax(domain_results)}")
            print(f"  Max score: {np.max(domain_results):.4f}")
        elif isinstance(domain_results, dict) and 'overall_compositionality_score' in domain_results:
            print(f"\n{domain.upper()}:")
            print(f"  Overall Score: {domain_results['overall_compositionality_score']:.4f}")
            
            if 'cca_score' in domain_results:
                print(f"  CCA Score: {domain_results['cca_score']:.4f}")
            if 'decomposition_score' in domain_results:
                print(f"  Decomposition Score: {domain_results['decomposition_score']:.4f}")
            
            # Statistical significance
            if 'cca' in domain_results and 'overall_p_value' in domain_results['cca']:
                p_val = domain_results['cca']['overall_p_value']
                print(f"  Statistical Significance: p={p_val:.4f} {'✓' if p_val < 0.05 else '✗'}")
    
    print("\n" + "="*70)
    print("All results saved to output/")
    print("="*70)
    
    # Save results
    np.savez_compressed(
        'output/compositionality_results.npz',
        **{k: v for k, v in results.items() if v is not None}
    )
    
    return results


if __name__ == "__main__":
    results = main()