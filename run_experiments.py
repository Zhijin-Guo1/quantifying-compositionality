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
        from data_loaders.sentence_loader import SentenceDataLoader
        
        print("\n" + "="*70)
        print("SENTENCE COMPOSITIONALITY EXPERIMENT")
        print("="*70)
        
        # Load pre-prepared sentence data (matching notebook approach)
        print(f"\n1. Loading sentence data from {args.sentence_data_dir}...")

        # Check if using pre-prepared data (notebook format)
        sentence_loader = SentenceDataLoader(data_dir=args.sentence_data_dir)

        try:
            # Load data matching notebook format
            sentence_data = sentence_loader.load_data()
            sentences = sentence_data['sentences']
            attributes = sentence_data['attributes']
            feature_names = sentence_data['feature_names']

            print(f"   Loaded {len(sentences)} sentences")
            print(f"   Unique slot combinations: {sentence_data['n_unique_combinations']}")
            print(f"   Attributes shape: {attributes.shape}")
            print(f"   Number of features: {len(feature_names)}")
            
        except FileNotFoundError as e:
            print(f"\nERROR: {e}")
            print("\nThe sentence experiment requires pre-prepared data files:")
            print("  - data/sentence/user_texts.txt")
            print("  - data/sentence/dialogue_data.csv")
            print("\nThese files contain pre-extracted sentences and attributes.")
            print("Please ensure these files exist before running the experiment.")
            return None
        except Exception as e:
            print(f"Failed to load sentence data: {e}")
            return None
        
        # Extract embeddings (matching notebook: all-MiniLM-L6-v2, layer 6, normalized)
        print("\n2. Extracting SBERT embeddings...")
        model_name = args.sbert_model or 'sentence-transformers/all-MiniLM-L6-v2'
        embed_extractor = SentenceBERTExtractor(model_name=model_name)

        # Default to layer 6 if not specified (matching notebook)
        layer = args.layer if args.layer is not None else 6

        print(f"   Model: {model_name}")
        print(f"   Extracting from layer {layer}...")
        embeddings = embed_extractor.extract(sentences, layer=layer, normalize=True)

        print(f"   Embeddings shape: {embeddings.shape}")
        
        # Analyze compositionality
        print("\n3. Analyzing compositionality...")

        # Set CCA components (notebook uses 15 for sentences)
        cca_components = args.cca_components if args.cca_components else 15

        analyzer = CompositionalityAnalyzer(
            cca_components=cca_components,
            decomposition_method=args.decomposition_method,
            random_seed=2  # Match notebook's random seed
        )

        # Note: The notebook uses:
        # - ALL data for CCA (no grouping)
        # - GROUPED data for linear decomposition
        # This is handled by group_by_attributes=True
        results = analyzer.analyze_compositionality(
            embeddings=embeddings,
            attributes=attributes,
            methods=args.methods.split(','),
            n_permutations=args.n_permutations,
            n_trials=args.n_trials,
            group_by_attributes=True,  # Group for decomposition (like notebook)
            verbose=args.verbose
        )
        
        # Always save results
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, 'sentence_results.npz')
        np.savez_compressed(save_path, **results)
        print(f"\nResults saved to {save_path}")
        
        # Plot if requested
        if args.plot:
            # Generate 4 individual plots matching notebook style
            analyzer.plot_results_individual(results, data_type='sentence', output_dir=args.output_dir)
            print(f"Plots saved to {args.output_dir}/")
        
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
        elif os.path.exists('data/MorphoLEX_en.xlsx'):
            print("\nFound MorphoLEX data at default location...")
            ml_loader = MorphoLEXLoader(excel_path='data/MorphoLEX_en.xlsx')
            use_morpholex = True
        
        if not use_morpholex:
            print("\nERROR: MorphoLEX_en.xlsx file not found!")
            print("Expected location: data/MorphoLEX_en.xlsx")
            print("Please ensure the file exists before running word experiments.")
            return None
            
        if use_morpholex:
            # Load Word2Vec model (REQUIRED for accurate results)
            print("\n1. Loading Word2Vec model...")
            embed_extractor = None
            word2vec_model = None

            # Check for GoogleNews model in standard locations
            googlenews_paths = [
                "data/GoogleNews-vectors-negative300.bin.gz",
                "data/GoogleNews-vectors-negative300.bin",
                os.path.expanduser("~/Downloads/GoogleNews-vectors-negative300.bin.gz"),
                os.path.expanduser("~/Downloads/GoogleNews-vectors-negative300.bin"),
                "./GoogleNews-vectors-negative300.bin.gz",
                "./GoogleNews-vectors-negative300.bin",
            ]

            # Add custom path if specified
            if args.word2vec_model:
                googlenews_paths.insert(0, args.word2vec_model)

            # Find the model
            model_path = None
            for path in googlenews_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path:
                try:
                    print(f"   Found GoogleNews model at: {model_path}")
                    embed_extractor = Word2VecExtractor(model_path=model_path)
                    word2vec_model = embed_extractor.keyed_vectors
                    print(f"   ✓ Loaded GoogleNews Word2Vec model ({len(word2vec_model):,} words)")
                except Exception as e:
                    print(f"   ✗ Failed to load model: {e}")
                    word2vec_model = None
            else:
                print("\n" + "="*70)
                print("ERROR: GoogleNews Word2Vec Model Not Found")
                print("="*70)
                print("The word experiment requires the GoogleNews Word2Vec model")
                print("to reproduce the results from the notebook.")
                print("\nPlease download the model:")
                print("\n1. Download GoogleNews-vectors-negative300.bin.gz from:")
                print("   https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/")
                print("   (File size: ~1.5 GB)")
                print("\n2. Save it to one of these locations:")
                print("   - data/GoogleNews-vectors-negative300.bin.gz")
                print("   - ~/Downloads/GoogleNews-vectors-negative300.bin.gz")
                print("\n3. Run the experiment again")
                print("\nAlternatively, specify the path:")
                print("   python run_experiments.py --experiment word --word2vec-model /path/to/model.bin.gz")
                print("="*70)
                return None
            
            # Prepare CCA data WITHOUT Word2Vec filtering (like notebook)
            print("\n2. Preparing CCA data (filtered suffixes)...")
            # Pass None for word2vec_model to skip filtering during preparation
            cca_data = ml_loader.prepare_cca_data(word2vec_model=None)
            if cca_data is None:
                print("Failed to prepare CCA data")
                return None
                
            words = cca_data['words']
            attributes = cca_data['attributes']
            feature_names = cca_data['feature_names']
            combined_df = cca_data['combined_df']
            
            print(f"   Initial data: {len(words)} words, {len(feature_names)} suffix features")

            # Now filter to words that exist in Word2Vec (matching notebook approach)
            if word2vec_model:
                print("\n   Filtering to words in Word2Vec vocabulary...")
                words_to_keep = []
                indices_to_keep = []
                for i, word in enumerate(words):
                    if word in word2vec_model:
                        words_to_keep.append(word)
                        indices_to_keep.append(i)

                # Update data to only include words with embeddings
                words = words_to_keep
                attributes = attributes[indices_to_keep]
                combined_df = combined_df.iloc[indices_to_keep].reset_index(drop=True)

                print(f"   After filtering: {len(words)} words with embeddings")
            else:
                print("   No Word2Vec model - using all words")

            print(f"   Final CCA data: {len(words)} words, {len(feature_names)} suffix features")
        
        print(f"Processing {len(words)} words...")
        print(f"   Attributes shape: {attributes.shape}")
        print(f"   Number of features: {len(feature_names)}")
        
        # Extract embeddings using the same Word2Vec model
        print("\n3. Extracting Word2Vec embeddings...")

        if embed_extractor and word2vec_model:
            # Extract embeddings in exact order of words list
            # Following notebook approach: iterate through words and get vectors
            embeddings_list = []
            missing_words = []

            for word in words:
                if word in word2vec_model:
                    embeddings_list.append(word2vec_model[word])
                else:
                    # This should not happen if prepare_cca_data worked correctly
                    missing_words.append(word)
                    logger.warning(f"Word '{word}' not found in Word2Vec model despite filtering")

            if missing_words:
                raise ValueError(f"Found {len(missing_words)} words without embeddings after filtering: {missing_words[:5]}...")

            embeddings = np.array(embeddings_list)
            print(f"   Successfully extracted embeddings for all {len(words)} words")
        else:
            print("   No embedding model available. Using random embeddings for demo...")
            np.random.seed(42)
            embeddings = np.random.randn(len(words), 100)
        
        print(f"   Embeddings shape: {embeddings.shape}")
        
        # Prepare separate data for Linear Decomposition if using MorphoLEX
        decomp_embeddings = None
        decomp_attributes = None
        
        if use_morpholex:
            print("\n4. Preparing filtered data for Linear Decomposition...")
            # Pass the combined_df from CCA preparation to maintain consistency
            decomp_data = ml_loader.prepare_decomposition_data(combined_df=combined_df)
            
            if decomp_data:
                decomp_words = decomp_data['words']
                decomp_attributes = decomp_data['attributes']
                
                # Extract embeddings for decomposition words using the same Word2Vec model
                if word2vec_model:
                    # Extract embeddings in exact order
                    decomp_embeddings_list = []
                    for word in decomp_words:
                        if word in word2vec_model:
                            decomp_embeddings_list.append(word2vec_model[word])
                        else:
                            # This should not happen for decomposition words
                            logger.warning(f"Decomposition word '{word}' not found in Word2Vec")
                            # Use zero vector as fallback
                            decomp_embeddings_list.append(np.zeros(embeddings.shape[1]))

                    decomp_embeddings = np.array(decomp_embeddings_list)
                else:
                    # Use random for demo
                    np.random.seed(43)
                    decomp_embeddings = np.random.randn(len(decomp_words), embeddings.shape[1])
                
                print(f"   Decomposition data: {len(decomp_words)} words, {decomp_attributes.shape[1]} features")
                print(f"   - {decomp_data['n_suffix_features']} suffix features")
                print(f"   - {decomp_data['n_root_features']} root features")
        
        # Analyze compositionality
        print("\n5. Analyzing compositionality...")
        # Use 20 components for words to match notebook
        word_cca_components = 20 if use_morpholex else min(args.cca_components, 10)
        analyzer = CompositionalityAnalyzer(
            cca_components=word_cca_components,
            decomposition_method=args.decomposition_method,
            random_seed=2  # Match notebook's random seed
        )
        
        results = analyzer.analyze_compositionality(
            embeddings=embeddings,
            attributes=attributes,
            methods=args.methods.split(','),
            n_permutations=args.n_permutations,
            n_trials=args.n_trials,
            group_by_attributes=False,  # Words are usually unique
            verbose=args.verbose,
            decomp_embeddings=decomp_embeddings,  # Separate data for decomposition
            decomp_attributes=decomp_attributes   # Separate data for decomposition
        )
        
        # Always save results
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, 'word_results.npz')
        np.savez_compressed(save_path, **results)
        print(f"\nResults saved to {save_path}")
        
        # Plot if requested
        if args.plot:
            # Generate 4 individual plots matching notebook style
            analyzer.plot_results_individual(results, data_type='word', output_dir=args.output_dir)
            print(f"Plots saved to {args.output_dir}/")
        
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
            # Prepare KG data - default is to exclude occupation (only age and gender)
            include_occupation = args.include_occupation  # False by default
            kg_data = ml_loader.prepare_kg_data(include_occupation=include_occupation)
            users_df = kg_data['users_df']
            user_ids = kg_data['user_ids']
            attributes = kg_data['attributes']
            feature_names = kg_data['feature_names']
            
            print(f"Loaded {len(users_df)} users from MovieLens")
            if include_occupation:
                print(f"Demographics: {len(feature_names)} features (gender, age, occupation)")
            else:
                print(f"Demographics: {len(feature_names)} features (gender, age only)")
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
        
        # Always save results
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save as NPZ
        save_path = os.path.join(args.output_dir, f'kg_{args.kg_model}_results.npz')
        np.savez_compressed(save_path, **results)
        print(f"\nResults saved to {save_path}")
        
        # Also save as JSON for easier reading
        json_path = os.path.join(args.output_dir, f'kg_{args.kg_model}_results.json')
        # Convert numpy arrays to lists for JSON serialization
        import json
        
        def convert_to_json_serializable(obj, skip_keys=None):
            """Convert numpy types to Python native types for JSON serialization."""
            if skip_keys is None:
                skip_keys = {'cca_model', 'decomposer', 'analyzer'}  # Skip non-serializable objects
            
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                  np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if key not in skip_keys:  # Skip non-serializable keys
                        try:
                            result[key] = convert_to_json_serializable(value, skip_keys)
                        except (TypeError, ValueError):
                            # Skip values that can't be serialized
                            pass
                return result
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item, skip_keys) for item in obj]
            else:
                # For any other type, try to convert to basic Python type
                try:
                    if hasattr(obj, '__dict__'):
                        # Skip complex objects
                        return str(type(obj).__name__)
                    else:
                        return obj
                except:
                    return str(obj)
        
        json_results = convert_to_json_serializable(results)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Results also saved as JSON to {json_path}")
        
        # Plot if requested
        if args.plot:
            # Generate 4 individual plots matching notebook style
            analyzer.plot_results_individual(results, data_type='kg', output_dir=args.output_dir)
            print(f"Plots saved to {args.output_dir}/")
        
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
    parser.add_argument('--sentence-data-dir',
                       default='data/sentence',
                       help='Directory containing pre-prepared sentence data (user_texts.txt and dialogue_data.csv)')
    parser.add_argument('--dialogue-dir',
                       default='train',
                       help='(Deprecated) Directory containing Schema-Guided Dialogue data')
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
                       default='data/MorphoLEX_en.xlsx',
                       help='Path to MorphoLEX_en.xlsx file')
    parser.add_argument('--data-path',
                       help='Path to text file with words (one per line)')
    parser.add_argument('--word2vec-model',
                       help='Path to GoogleNews Word2Vec model (e.g., path/to/GoogleNews-vectors-negative300.bin.gz)')
    parser.add_argument('--pretrained-word2vec',
                       default='word2vec-google-news-300',
                       help='Pretrained Word2Vec model name')
    parser.add_argument('--word-attribute-type',
                       default='morphological',
                       choices=['morphological', 'semantic'],
                       help='Type of word attributes')
    
    # KG-specific options
    parser.add_argument('--movielens-dir',
                       default='data/ml-1m',
                       help='Directory containing MovieLens 1M data')
    parser.add_argument('--kg-embedding', '--kg-model',
                       dest='kg_model',
                       default='TransE',
                       choices=['TransE', 'DistMult'],
                       help='KG embedding model type')
    parser.add_argument('--kg-embedding-dir',
                       default='KG_embedding',
                       help='Directory containing KG embeddings')
    parser.add_argument('--normalize-kg', action='store_true',
                       help='Normalize KG embeddings')
    parser.add_argument('--include-occupation', action='store_true',
                       help='Include occupation in KG attributes (default: only age and gender)')
    
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
                       default=100,
                       help='Number of permutations for significance testing')
    parser.add_argument('--n-trials', type=int,
                       default=100,
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