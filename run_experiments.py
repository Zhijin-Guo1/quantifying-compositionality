#!/usr/bin/env python
"""
Flexible experiment runner for quantifying compositionality.

Usage:
    python run_experiments.py --experiment sentence
    python run_experiments.py --experiment word
    python run_experiments.py --experiment kg
    python run_experiments.py --experiment all
    python run_experiments.py --experiment layer-wise
"""

import argparse
import numpy as np
import logging
import os
import sys
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_sentence_experiment(args) -> Optional[Dict]:
    """Run sentence compositionality experiment."""
    try:
        from embeddings import SentenceBERTExtractor
        from compositionality import CompositionalityAnalyzer
        from data_loaders import DialogueLoader
        
        print("\n" + "="*70)
        print("SENTENCE COMPOSITIONALITY EXPERIMENT")
        print("="*70)
        
        # Load Schema-Guided Dialogue data
        print(f"\nLoading Schema-Guided Dialogue data from {args.dialogue_dir}...")
        dialogue_loader = DialogueLoader(
            data_dir=args.dialogue_dir,
            min_slots=args.min_slots
        )
        
        try:
            dialogue_data = dialogue_loader.prepare_dialogue_data()
            sentences = dialogue_data['sentences']
            attributes = dialogue_data['attributes']
            feature_names = dialogue_data['feature_names']
            
            print(f"Loaded {len(sentences)} sentences")
            print(f"Unique slot combinations: {dialogue_data['n_unique_patterns']}")
            print(f"   Attributes shape: {attributes.shape}")
            print(f"   Number of slots: {len(feature_names)}")
            
        except Exception as e:
            print(f"Failed to load dialogue data: {e}")
            print("Please ensure Schema-Guided Dialogue data is in the specified directory")
            print("Download from: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue")
            
            # Fall back to demo sentences if data not available
            if args.use_demo_fallback:
                print("\nFalling back to demo sentences...")
                sentences = [
                    "I'd like to book a table for two at an Italian restaurant in downtown.",
                    "Can you find me a flight from New York to London next Monday?",
                    "What movies are playing tonight at the theater near me?",
                ] * 10  # Replicate for testing
                
                from attributes import SentenceAttributeExtractor
                attr_extractor = SentenceAttributeExtractor()
                attributes, feature_names = attr_extractor.extract(sentences)
            else:
                return None
        
        # Extract embeddings
        print("\n2. Extracting SBERT embeddings...")
        embed_extractor = SentenceBERTExtractor(
            model_name=args.sbert_model or 'sentence-transformers/all-MiniLM-L6-v2'
        )
        
        if args.layer is not None:
            print(f"   Extracting from layer {args.layer}...")
            embeddings = embed_extractor.extract(sentences, layer=args.layer, normalize=True)
        else:
            embeddings = embed_extractor.extract(sentences, normalize=True)
        
        print(f"   Embeddings shape: {embeddings.shape}")
        
        # Analyze compositionality
        print("\n3. Analyzing compositionality...")
        analyzer = CompositionalityAnalyzer(
            cca_components=args.cca_components,
            decomposition_method=args.decomposition_method
        )
        
        results = analyzer.analyze_compositionality(
            embeddings=embeddings,
            attributes=attributes,
            methods=args.methods.split(','),
            n_permutations=args.n_permutations,
            n_trials=args.n_trials,
            group_by_attributes=args.group_by_attributes,
            verbose=args.verbose
        )
        
        # Save results
        if args.save_results:
            save_path = os.path.join(args.output_dir, 'sentence_results.npz')
            np.savez_compressed(save_path, **results)
            print(f"\nResults saved to {save_path}")
        
        # Plot if requested
        if args.plot:
            plot_path = os.path.join(args.output_dir, 'sentence_compositionality.png')
            analyzer.plot_results(results, save_path=plot_path)
            print(f"Plot saved to {plot_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Sentence experiment failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return None


def run_word_experiment(args) -> Optional[Dict]:
    """Run word compositionality experiment."""
    try:
        from attributes import WordAttributeExtractor
        from embeddings import Word2VecExtractor
        from compositionality import CompositionalityAnalyzer
        from data_loaders import MorphoLEXLoader
        
        print("\n" + "="*70)
        print("WORD COMPOSITIONALITY EXPERIMENT")
        print("="*70)
        
        # Try to load MorphoLEX data first
        use_morpholex = False
        if args.morpholex_path and os.path.exists(args.morpholex_path):
            print(f"\nLoading MorphoLEX data from {args.morpholex_path}...")
            ml_loader = MorphoLEXLoader(excel_path=args.morpholex_path)
            use_morpholex = True
        elif os.path.exists('Downloads/MorphoLEX_en.xlsx'):
            print("\nFound MorphoLEX data at default location...")
            ml_loader = MorphoLEXLoader()
            use_morpholex = True
        
        if use_morpholex:
            # Load word list if provided
            word_list = None
            if args.data_path and os.path.exists(args.data_path):
                with open(args.data_path, 'r') as f:
                    word_list = [line.strip() for line in f if line.strip()]
                print(f"Filtering to {len(word_list)} words from {args.data_path}")
            
            word_data = ml_loader.prepare_word_data(word_list=word_list)
            words = word_data['words']
            attributes = word_data['attributes']
            feature_names = word_data['feature_names']
        else:
            # Fallback to demo data or word list
            if args.data_path and os.path.exists(args.data_path):
                print(f"\nLoading words from {args.data_path}...")
                with open(args.data_path, 'r') as f:
                    words = [line.strip() for line in f if line.strip()]
            else:
                print("\nUsing demo words with morphological patterns...")
                print("NOTE: For real experiments, download MorphoLEX_en.xlsx from:")
                print("      http://www.lexique.org/?page_id=250")
                words = [
                    # Base forms
                    "book", "play", "run", "write", "read", "walk", "talk", "work",
                    # -ing forms
                    "booking", "playing", "running", "writing", "reading", "walking", "talking", "working",
                    # -ed forms
                    "booked", "played", "ran", "written", "read", "walked", "talked", "worked",
                    # -er forms
                    "booker", "player", "runner", "writer", "reader", "walker", "talker", "worker",
                    # -s forms
                    "books", "plays", "runs", "writes", "reads", "walks", "talks", "works",
                    # Prefixes
                    "unbook", "replay", "rerun", "rewrite", "reread", "unwalk", "retalk", "rework",
                    # Compounds
                    "bookshelf", "playground", "runway", "writeup", "readout", "walkway", "talkshow", "workday"
                ]
            
            # Extract attributes using WordAttributeExtractor
            print("\n1. Extracting morphological attributes...")
            attr_extractor = WordAttributeExtractor(
                attribute_type=args.word_attribute_type or 'morphological'
            )
            attributes, feature_names = attr_extractor.extract(words)
        
        print(f"Processing {len(words)} words...")
        print(f"   Attributes shape: {attributes.shape}")
        print(f"   Number of features: {len(feature_names)}")
        
        # Extract embeddings
        print("\n2. Extracting Word2Vec embeddings...")
        
        if args.word2vec_model:
            # Load custom model
            embed_extractor = Word2VecExtractor(model_path=args.word2vec_model)
            embeddings = embed_extractor.extract(words, normalize=False)
        else:
            try:
                # Try pretrained model
                embed_extractor = Word2VecExtractor(
                    pretrained_model=args.pretrained_word2vec or 'glove-wiki-gigaword-100',
                    use_pretrained=True
                )
                embeddings = embed_extractor.extract(words, normalize=False)
            except:
                print("   Failed to load pretrained model. Using random embeddings for demo...")
                np.random.seed(42)
                embeddings = np.random.randn(len(words), 100)
        
        print(f"   Embeddings shape: {embeddings.shape}")
        
        # Analyze compositionality
        print("\n3. Analyzing compositionality...")
        analyzer = CompositionalityAnalyzer(
            cca_components=min(args.cca_components, 10),  # Fewer components for words
            decomposition_method=args.decomposition_method
        )
        
        results = analyzer.analyze_compositionality(
            embeddings=embeddings,
            attributes=attributes,
            methods=args.methods.split(','),
            n_permutations=args.n_permutations,
            n_trials=args.n_trials,
            group_by_attributes=False,  # Words are usually unique
            verbose=args.verbose
        )
        
        # Save results
        if args.save_results:
            save_path = os.path.join(args.output_dir, 'word_results.npz')
            np.savez_compressed(save_path, **results)
            print(f"\nResults saved to {save_path}")
        
        # Plot if requested
        if args.plot:
            plot_path = os.path.join(args.output_dir, 'word_compositionality.png')
            analyzer.plot_results(results, save_path=plot_path)
            print(f"Plot saved to {plot_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Word experiment failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return None


def run_kg_experiment(args) -> Optional[Dict]:
    """Run knowledge graph compositionality experiment."""
    try:
        from embeddings import KGEmbeddingLoader
        from compositionality import CompositionalityAnalyzer
        from data_loaders import MovieLensLoader
        import pandas as pd
        
        print("\n" + "="*70)
        print("KNOWLEDGE GRAPH COMPOSITIONALITY EXPERIMENT")
        print("="*70)
        
        # Check if KG embeddings exist
        kg_path = os.path.join(args.kg_embedding_dir, f'300_epochs_{args.kg_model}_gpu34.pt')
        if not os.path.exists(kg_path):
            print(f"\nERROR: KG embeddings not found at {kg_path}")
            print("Please ensure you have pre-trained KG embeddings in the KG_embedding/ directory")
            return None
        
        # Load MovieLens data
        print(f"\nLoading MovieLens data from {args.movielens_dir}...")
        ml_loader = MovieLensLoader(data_dir=args.movielens_dir)
        
        try:
            kg_data = ml_loader.prepare_kg_data()
            users_df = kg_data['users_df']
            user_ids = kg_data['user_ids']
            attributes = kg_data['attributes']
            feature_names = kg_data['feature_names']
            
            print(f"Loaded {len(users_df)} users from MovieLens")
            print(f"Demographics: {len(feature_names)} features")
        except Exception as e:
            print(f"Failed to load MovieLens data: {e}")
            print("Please ensure MovieLens 1M data is in the specified directory")
            print("Download from: https://grouplens.org/datasets/movielens/1m/")
            return None
        
        print(f"   Attributes shape: {attributes.shape}")
        print(f"   Features (first 10): {feature_names[:10]}...")
        
        # Load KG embeddings
        print(f"\n2. Loading {args.kg_model} embeddings...")
        kg_loader = KGEmbeddingLoader(
            model_type=args.kg_model,
            kg_embedding_dir=args.kg_embedding_dir
        )
        
        # Extract embeddings for user IDs (0-indexed)
        embeddings = kg_loader.extract(user_ids, normalize=args.normalize_kg)
        print(f"   Embeddings shape: {embeddings.shape}")
        
        # Print embedding statistics
        print(f"   Embedding statistics:")
        print(f"     Min: {embeddings.min():.4f}")
        print(f"     Max: {embeddings.max():.4f}")
        print(f"     Mean: {embeddings.mean():.4f}")
        print(f"     Std: {embeddings.std():.4f}")
        
        # Analyze compositionality
        print("\n3. Analyzing compositionality...")
        analyzer = CompositionalityAnalyzer(
            cca_components=min(args.cca_components, attributes.shape[1]),
            decomposition_method=args.decomposition_method
        )
        
        results = analyzer.analyze_compositionality(
            embeddings=embeddings,
            attributes=attributes,
            methods=args.methods.split(','),
            n_permutations=args.n_permutations,
            n_trials=args.n_trials,
            group_by_attributes=args.group_by_attributes,
            verbose=args.verbose
        )
        
        # Save results
        if args.save_results:
            save_path = os.path.join(args.output_dir, f'kg_{args.kg_model}_results.npz')
            np.savez_compressed(save_path, **results)
            print(f"\nResults saved to {save_path}")
        
        # Plot if requested
        if args.plot:
            plot_path = os.path.join(args.output_dir, f'kg_{args.kg_model}_compositionality.png')
            analyzer.plot_results(results, save_path=plot_path)
            print(f"Plot saved to {plot_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"KG experiment failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return None


def run_layerwise_experiment(args) -> Optional[Dict]:
    """Run layer-wise compositionality analysis for SBERT."""
    try:
        from attributes import SentenceAttributeExtractor
        from embeddings import SentenceBERTExtractor
        from compositionality import CompositionalityAnalyzer
        
        print("\n" + "="*70)
        print("LAYER-WISE COMPOSITIONALITY ANALYSIS")
        print("="*70)
        
        # Use demo sentences
        sentences = [
            "Book a table for dinner",
            "Find flights to Paris",
            "Show movie times",
            "Reserve hotel room",
            "Schedule appointment",
            "Order pizza delivery",
            "Call taxi service",
            "Buy concert tickets"
        ] * 3  # Replicate for better statistics
        
        print(f"Processing {len(sentences)} sentences...")
        
        # Extract attributes once
        print("\n1. Extracting attributes...")
        attr_extractor = SentenceAttributeExtractor()
        attributes, _ = attr_extractor.extract(sentences)
        
        # Initialize SBERT
        print("\n2. Initializing SBERT...")
        embed_extractor = SentenceBERTExtractor(
            model_name=args.sbert_model or 'sentence-transformers/all-MiniLM-L6-v2'
        )
        n_layers = embed_extractor.get_num_layers()
        print(f"   Model has {n_layers} layers")
        
        # Analyze each layer
        print("\n3. Analyzing compositionality across layers...")
        analyzer = CompositionalityAnalyzer(
            cca_components=min(args.cca_components, 5),
            decomposition_method=args.decomposition_method
        )
        
        layer_results = {}
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
                n_permutations=args.n_permutations // 5,  # Fewer permutations
                n_trials=args.n_trials // 5,  # Fewer trials
                group_by_attributes=True,
                verbose=False
            )
            
            score = results['overall_compositionality_score']
            layer_results[f'layer_{layer}'] = results
            layer_scores.append(score)
            print(f"     Compositionality score: {score:.4f}")
        
        # Find best layer
        best_layer = np.argmax(layer_scores)
        print(f"\n4. Results Summary:")
        print(f"   Best layer: {best_layer} (score: {layer_scores[best_layer]:.4f})")
        print(f"   Layer scores: {[f'{s:.3f}' for s in layer_scores]}")
        
        # Plot layer scores
        if args.plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(range(n_layers), layer_scores, 'b-o', linewidth=2, markersize=8)
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Moderate')
            plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Strong')
            plt.xlabel('Layer', fontsize=12)
            plt.ylabel('Compositionality Score', fontsize=12)
            plt.title('Compositionality Across SBERT Layers', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plot_path = os.path.join(args.output_dir, 'layerwise_compositionality.png')
            plt.savefig(plot_path, dpi=150)
            plt.show()
            print(f"\nPlot saved to {plot_path}")
        
        # Save results
        if args.save_results:
            save_path = os.path.join(args.output_dir, 'layerwise_results.npz')
            np.savez_compressed(save_path, 
                               layer_scores=layer_scores,
                               best_layer=best_layer,
                               **layer_results)
            print(f"Results saved to {save_path}")
        
        return {'layer_scores': layer_scores, 'best_layer': best_layer, 'layer_results': layer_results}
        
    except Exception as e:
        logger.error(f"Layer-wise experiment failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run compositionality experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sentence experiment with Schema-Guided Dialogue data
  python run_experiments.py --experiment sentence --dialogue-dir train
  
  # Run word experiment
  python run_experiments.py --experiment word
  
  # Run KG experiment with MovieLens data
  python run_experiments.py --experiment kg --movielens-dir ml-1m --kg-model TransE
  
  # Run layer-wise analysis
  python run_experiments.py --experiment layer-wise --plot
  
  # Run all experiments
  python run_experiments.py --experiment all --save-results --plot
        """
    )
    
    # Experiment selection
    parser.add_argument('--experiment', '-e', 
                       choices=['sentence', 'word', 'kg', 'layer-wise', 'all'],
                       default='all',
                       help='Which experiment to run')
    
    # Data options
    parser.add_argument('--output-dir', '-o',
                       default='output',
                       help='Output directory for results')
    
    # Sentence-specific options
    parser.add_argument('--dialogue-dir',
                       default='train',
                       help='Directory containing Schema-Guided Dialogue data')
    parser.add_argument('--min-slots', type=int,
                       default=3,
                       help='Minimum slots per sentence')
    parser.add_argument('--sbert-model',
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='SBERT model name')
    parser.add_argument('--layer', type=int,
                       help='Specific layer to extract (for sentence)')
    parser.add_argument('--use-demo-fallback', action='store_true',
                       help='Use demo sentences if real data not available')
    
    # Word-specific options
    parser.add_argument('--morpholex-path',
                       help='Path to MorphoLEX_en.xlsx file')
    parser.add_argument('--data-path',
                       help='Path to text file with words (one per line)')
    parser.add_argument('--word2vec-model',
                       help='Path to custom Word2Vec model')
    parser.add_argument('--pretrained-word2vec',
                       default='glove-wiki-gigaword-100',
                       help='Pretrained Word2Vec model name')
    parser.add_argument('--word-attribute-type',
                       default='morphological',
                       choices=['morphological', 'semantic'],
                       help='Type of word attributes')
    
    # KG-specific options
    parser.add_argument('--movielens-dir',
                       default='ml-1m',
                       help='Directory containing MovieLens 1M data')
    parser.add_argument('--kg-model',
                       default='TransE',
                       choices=['TransE', 'DistMult'],
                       help='KG embedding model type')
    parser.add_argument('--kg-embedding-dir',
                       default='KG_embedding',
                       help='Directory containing KG embeddings')
    parser.add_argument('--normalize-kg', action='store_true',
                       help='Normalize KG embeddings')
    
    # Analysis options
    parser.add_argument('--cca-components', type=int,
                       default=10,
                       help='Number of CCA components')
    parser.add_argument('--decomposition-method',
                       default='pseudo_inverse',
                       choices=['pseudo_inverse', 'lstsq'],
                       help='Linear decomposition method')
    parser.add_argument('--methods',
                       default='cca,decomposition,metrics',
                       help='Comma-separated analysis methods')
    parser.add_argument('--n-permutations', type=int,
                       default=50,
                       help='Number of permutations for significance testing')
    parser.add_argument('--n-trials', type=int,
                       default=50,
                       help='Number of trials for leave-one-out')
    parser.add_argument('--group-by-attributes', action='store_true',
                       default=True,
                       help='Group samples with identical attributes')
    
    # Output options
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and save plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with full error traces')
    parser.add_argument('--random-seed', type=int,
                       default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected experiments
    results = {}
    
    if args.experiment in ['sentence', 'all']:
        print("\n" + "="*70)
        print("RUNNING SENTENCE EXPERIMENT")
        print("="*70)
        sentence_results = run_sentence_experiment(args)
        if sentence_results:
            results['sentence'] = sentence_results
    
    if args.experiment in ['word', 'all']:
        print("\n" + "="*70)
        print("RUNNING WORD EXPERIMENT")
        print("="*70)
        word_results = run_word_experiment(args)
        if word_results:
            results['word'] = word_results
    
    if args.experiment in ['kg', 'all']:
        print("\n" + "="*70)
        print("RUNNING KG EXPERIMENT")
        print("="*70)
        kg_results = run_kg_experiment(args)
        if kg_results:
            results['kg'] = kg_results
    
    if args.experiment in ['layer-wise', 'all']:
        print("\n" + "="*70)
        print("RUNNING LAYER-WISE ANALYSIS")
        print("="*70)
        layerwise_results = run_layerwise_experiment(args)
        if layerwise_results:
            results['layer-wise'] = layerwise_results
    
    # Print summary
    if results:
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        
        for exp_name, exp_results in results.items():
            if exp_name == 'layer-wise':
                print(f"\n{exp_name.upper()}:")
                print(f"  Best layer: {exp_results['best_layer']}")
                print(f"  Max score: {np.max(exp_results['layer_scores']):.4f}")
            elif isinstance(exp_results, dict) and 'overall_compositionality_score' in exp_results:
                print(f"\n{exp_name.upper()}:")
                print(f"  Overall Score: {exp_results['overall_compositionality_score']:.4f}")
                
                if 'cca' in exp_results:
                    print(f"  CCA Score: {exp_results.get('cca_score', 'N/A'):.4f}")
                if 'decomposition' in exp_results:
                    print(f"  Decomposition Score: {exp_results.get('decomposition_score', 'N/A'):.4f}")
                
                # Statistical significance
                if 'cca' in exp_results and 'overall_p_value' in exp_results['cca']:
                    p_val = exp_results['cca']['overall_p_value']
                    print(f"  Statistical Significance: p={p_val:.4f} {'✓' if p_val < 0.05 else '✗'}")
    else:
        print("\nNo experiments completed successfully.")
        return 1
    
    print(f"\nAll results saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())