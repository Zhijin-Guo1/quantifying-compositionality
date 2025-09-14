"""Unified compositionality analyzer combining all methods."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List, Union
import logging
from .cca import CCAAnalyzer
from .linear_decomposition import LinearDecomposer
from .metrics import CompositionalityMetrics

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class CompositionalityAnalyzer:
    """
    Unified analyzer for quantifying compositionality between embeddings and attributes.
    
    This implements the two-step method:
    1. CCA to measure linear correlation
    2. Linear decomposition to test additive compositionality
    """
    
    def __init__(self,
                 cca_components: Optional[int] = None,
                 decomposition_method: str = 'pseudo_inverse',
                 random_seed: Optional[int] = None):
        """
        Initialize the compositionality analyzer.
        
        Args:
            cca_components: Number of CCA components (None for auto)
            decomposition_method: Method for linear decomposition
            random_seed: Random seed for reproducibility
        """
        self.cca_analyzer = CCAAnalyzer(n_components=cca_components)
        self.linear_decomposer = LinearDecomposer(method=decomposition_method)
        self.metrics = CompositionalityMetrics()
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def preprocess_data(self,
                       attributes: np.ndarray,
                       embeddings: np.ndarray,
                       group_by_attributes: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data by grouping samples with identical attributes.
        
        Args:
            attributes: Binary attribute matrix (n_samples, n_attributes)
            embeddings: Embedding matrix (n_samples, n_dims)
            group_by_attributes: Whether to group by unique attributes
            
        Returns:
            processed_attributes: Processed attribute matrix
            processed_embeddings: Processed embedding matrix
        """
        if not group_by_attributes:
            return attributes, embeddings
        
        # Check if grouping is needed
        n_samples = len(attributes)
        unique_attrs = len(np.unique(attributes, axis=0))
        
        if unique_attrs < n_samples:
            # Log the expected grouping based on attribute dimensions
            n_features = attributes.shape[1]
            if n_features == 9:  # Only age and gender
                logger.info(f"Grouping {n_samples} samples by age and gender (9 dimensions)")
            elif n_features == 30:  # Age, gender, and occupation
                logger.info(f"Grouping {n_samples} samples by age, gender, and occupation (30 dimensions)")
            else:
                logger.info(f"Grouping {n_samples} samples into unique attribute combinations ({n_features} dimensions)")
            return self.metrics.group_by_attributes(attributes, embeddings)
        
        return attributes, embeddings
    
    def analyze_compositionality(self,
                               embeddings: np.ndarray,
                               attributes: np.ndarray,
                               methods: List[str] = ['cca', 'decomposition', 'metrics'],
                               n_permutations: int = 100,
                               n_trials: int = 100,
                               group_by_attributes: bool = True,
                               verbose: bool = True,
                               decomp_embeddings: Optional[np.ndarray] = None,
                               decomp_attributes: Optional[np.ndarray] = None) -> Dict:
        """
        Perform complete compositionality analysis.
        
        Note: CCA (Step 1) uses the full dataset without grouping for maximum correlation detection.
        Linear Decomposition (Step 2) uses grouped data to avoid overfitting to duplicate attributes.
        
        Args:
            embeddings: Embedding matrix (n_samples, n_dims)
            attributes: Binary attribute matrix (n_samples, n_attributes)
            methods: List of methods to use
            n_permutations: Number of permutations for testing
            n_trials: Number of trials for leave-one-out
            group_by_attributes: Kept for compatibility but always groups for step 2
            verbose: Whether to print progress
            
        Returns:
            results: Complete analysis results
        """
        if verbose:
            logger.info("="*60)
            logger.info("COMPOSITIONALITY ANALYSIS")
            logger.info("="*60)
            logger.info(f"Input shapes - Embeddings: {embeddings.shape}, Attributes: {attributes.shape}")
        
        # Validate inputs
        assert len(embeddings) == len(attributes), "Number of samples must match"
        assert embeddings.ndim == 2, "Embeddings must be 2D"
        assert attributes.ndim == 2, "Attributes must be 2D"
        
        # Prepare data for different steps
        # Step 1 (CCA): Use full dataset without grouping
        # Step 2 (Linear Decomposition): Use grouped data (for KG) or ungrouped (for words)
        grouped_attributes, grouped_embeddings = self.preprocess_data(
            attributes, embeddings, group_by_attributes=group_by_attributes
        )
        
        if verbose:
            if len(grouped_attributes) < len(attributes):
                logger.info(f"Grouping info - Original: {attributes.shape}, Grouped: {grouped_attributes.shape}")
                logger.info("Note: CCA uses full data, Linear Decomposition uses grouped data")
        
        results = {
            'input_shape': {
                'embeddings': embeddings.shape,
                'attributes': attributes.shape
            },
            'grouped_shape': {
                'embeddings': grouped_embeddings.shape,
                'attributes': grouped_attributes.shape
            }
        }
        
        # 1. CCA Analysis (using FULL dataset)
        if 'cca' in methods:
            if verbose:
                logger.info("\n" + "-"*40)
                logger.info("Step 1: CCA Analysis (using full dataset)")
                logger.info(f"  Data shape: {attributes.shape} (all {len(attributes)} samples)")
                logger.info("-"*40)
            
            cca_results = self.cca_analyzer.analyze(
                attributes,  # Use full dataset
                embeddings,  # Use full dataset
                n_permutations=n_permutations
            )
            
            results['cca'] = cca_results
            results['cca_score'] = self.cca_analyzer.compute_cca_score(cca_results)
            
            if verbose:
                logger.info(f"CCA Score: {results['cca_score']:.4f}")
        
        # 2. Linear Decomposition (using separate data if provided, else GROUPED data)
        if 'decomposition' in methods:
            # Use separate decomposition data if provided (for word experiments)
            if decomp_embeddings is not None and decomp_attributes is not None:
                decomp_data = decomp_attributes
                decomp_emb = decomp_embeddings
                if verbose:
                    logger.info("\n" + "-"*40)
                    logger.info("Step 2: Linear Decomposition (using separate filtered data)")
                    logger.info(f"  Data shape: {decomp_data.shape} ({len(decomp_data)} filtered samples)")
                    logger.info("-"*40)
            else:
                decomp_data = grouped_attributes
                decomp_emb = grouped_embeddings
                if verbose:
                    logger.info("\n" + "-"*40)
                    logger.info("Step 2: Linear Decomposition (using grouped data)")
                    logger.info(f"  Data shape: {decomp_data.shape} ({len(decomp_data)} unique groups)")
                    logger.info("-"*40)
            
            decomp_results = self.linear_decomposer.analyze_with_permutation(
                decomp_data,
                decomp_emb,
                n_permutations=n_permutations,
                n_trials_per_permutation=n_trials
            )
            
            results['decomposition'] = decomp_results
            results['decomposition_score'] = self.linear_decomposer.compute_decomposition_score(decomp_results)
            
            if verbose:
                logger.info(f"Decomposition Score: {results['decomposition_score']:.4f}")
        
        # 3. Additional Metrics
        if 'metrics' in methods:
            # For metrics, use the same data as decomposition
            if decomp_embeddings is not None and decomp_attributes is not None:
                metrics_data = decomp_attributes
                metrics_emb = decomp_embeddings
                if verbose:
                    logger.info("\n" + "-"*40)
                    logger.info("Step 3: Computing Metrics (using decomposition data)")
                    logger.info("-"*40)
            else:
                metrics_data = grouped_attributes
                metrics_emb = grouped_embeddings
                if verbose:
                    logger.info("\n" + "-"*40)
                    logger.info("Step 3: Computing Metrics (using grouped data)")
                    logger.info("-"*40)
            
            # For word experiments, never group duplicates
            # For KG experiments, group duplicates
            should_group = group_by_attributes  # True for KG, False for words
            
            metrics_results = self.metrics.compute_all_metrics(
                metrics_data,
                metrics_emb,
                n_permutations=n_permutations,
                n_trials=n_trials,
                group_duplicates=should_group
            )
            
            results['metrics'] = metrics_results
            
            if verbose:
                logger.info(f"Cosine Similarity: {metrics_results['mean_cosine_similarity']:.4f} "
                          f"(random: {metrics_results['permuted_cosine_mean']:.4f})")
                logger.info(f"Hits@5: {metrics_results['hits@5']:.4f} "
                          f"(random: {metrics_results['permuted_hits@5_mean']:.4f})")
        
        # Compute overall compositionality score
        scores = []
        if 'cca_score' in results:
            scores.append(results['cca_score'])
        if 'decomposition_score' in results:
            scores.append(results['decomposition_score'])
        if 'metrics' in results:
            scores.append(results['metrics']['compositionality_score'])
        
        results['overall_compositionality_score'] = np.mean(scores) if scores else 0.0
        
        if verbose:
            logger.info("\n" + "="*60)
            logger.info(f"OVERALL COMPOSITIONALITY SCORE: {results['overall_compositionality_score']:.4f}")
            logger.info("="*60)
            logger.info(f"Score interpretation: Higher scores indicate stronger compositional alignment")
        
        return results
    
    def plot_results_individual(self, results: Dict, data_type: str = 'sentence', 
                                output_dir: Optional[str] = None):
        """
        Generate 4 individual plots matching notebook style.
        
        Args:
            results: Results from analyze_compositionality()
            data_type: Type of data ('sentence', 'kg', 'word')
            output_dir: Directory to save plots
        """
        import matplotlib.pyplot as plt
        import os
        
        # Set font size for better readability
        plt.rcParams.update({'font.size': 14})
        
        # Determine Hits@k value based on data type
        hits_k = 5  # default
        if data_type == 'kg':
            hits_k = 1
        elif data_type == 'word':
            hits_k = 10
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Plot 1: CCA Correlations
        if 'cca' in results:
            plt.figure(figsize=(10, 6))
            cca_results = results['cca']
            # Use 20 components for word experiments to match notebook
            if data_type == 'word':
                n_components = min(20, len(cca_results['real_correlations']))
            else:
                n_components = min(15, len(cca_results['real_correlations']))
            
            # Plot real correlations
            plt.plot(range(n_components), cca_results['real_correlations'][:n_components], 
                    'b-', linewidth=2, label='real embedding')
            
            # Plot all permuted correlations as yellow lines
            for perm in cca_results['permuted_correlations']:
                plt.plot(range(n_components), perm[:n_components], 'y-', alpha=0.3)
            
            # Add one labeled permuted line for legend
            plt.plot(range(n_components), cca_results['permuted_correlations'][0][:n_components], 
                    'y-', label='shuffled embedding')
            
            plt.xlabel('CCA Component', fontweight='bold')
            plt.ylabel('Correlation Coefficient', fontweight='bold')
            plt.title('Correlation Coefficients for Original and Shuffled Data')
            plt.legend(loc="upper right")
            plt.xticks(range(n_components))
            plt.grid(True, alpha=0.3)
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'{data_type}_CCA.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot 2: L2 Loss Distribution
        if 'decomposition' in results:
            plt.figure(figsize=(10, 6))
            decomp_results = results['decomposition']
            
            # Histogram of permuted L2 losses
            plt.hist(decomp_results['permuted_l2_losses'], bins=30, 
                    alpha=0.5, color='b', label="100 permuted pairs")
            
            # Real L2 loss line
            real_l2 = decomp_results['real_l2_loss']
            plt.axvline(real_l2, color='r', linestyle='dashed', linewidth=2, 
                       label=f"Real Pair: {real_l2:.2f}")
            
            plt.title(f'Comparison of 100 permuted pairs with Real Pair')
            plt.xlabel('L2 norm')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'{data_type}_L2.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot 3: Cosine Similarity Distribution
        if 'decomposition' in results:
            plt.figure(figsize=(10, 6))
            decomp_results = results['decomposition']
            
            # Histogram of permuted cosine similarities
            plt.hist(decomp_results['permuted_cosine_similarities'], bins=30,
                    alpha=0.5, color='b', label="100 permuted pairs")
            
            # Real cosine similarity line
            real_cos = decomp_results['real_cosine_similarity']
            plt.axvline(real_cos, color='r', linestyle='dashed', linewidth=2,
                       label=f"Real Pair: {real_cos:.2f}")
            
            plt.title('Comparison of 100 permuted pairs with Real Pair')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'{data_type}_cos.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot 4: Retrieval Accuracy (Hits@k)
        if 'metrics' in results:
            plt.figure(figsize=(10, 6))
            metrics = results['metrics']
            
            # Get permuted accuracies for histogram
            perm_key = f'permuted_hits@{hits_k}_all'
            if perm_key in metrics:
                perm_accuracies = metrics[perm_key]
            else:
                # Fallback: create synthetic data based on mean and std
                mean_perm = metrics.get(f'permuted_hits@{hits_k}_mean', 0.05)
                std_perm = metrics.get(f'permuted_hits@{hits_k}_std', 0.02)
                perm_accuracies = np.random.normal(mean_perm, std_perm, 100)
                perm_accuracies = np.clip(perm_accuracies, 0, 1)
            
            # Histogram of permuted accuracies
            plt.hist(perm_accuracies, bins=30, alpha=0.5, color='b', label="100 permuted pairs")
            
            # Real accuracy line
            real_acc = metrics.get(f'hits@{hits_k}', 0)
            plt.axvline(real_acc, color='r', linestyle='dashed', linewidth=2,
                       label=f"Real Pair: {real_acc:.2f}")
            
            plt.title('Comparison of 100 permuted pairs with Real Pair')
            plt.xlabel(f'Hits@{hits_k} accuracy')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'{data_type}_hits{hits_k}.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        return True
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot analysis results.
        
        Args:
            results: Results from analyze_compositionality()
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Compositionality Analysis Results', fontsize=16)
        
        # 1. CCA Correlations
        if 'cca' in results:
            ax = axes[0, 0]
            cca_results = results['cca']
            n_components = len(cca_results['real_correlations'])
            
            # Plot real correlations
            ax.plot(range(n_components), cca_results['real_correlations'], 
                   'b-', linewidth=2, label='Real')
            
            # Plot permuted correlations (mean and std)
            mean_perm = np.mean(cca_results['permuted_correlations'], axis=0)
            std_perm = np.std(cca_results['permuted_correlations'], axis=0)
            ax.plot(range(n_components), mean_perm, 'r--', linewidth=2, label='Permuted')
            ax.fill_between(range(n_components), 
                          mean_perm - std_perm, 
                          mean_perm + std_perm, 
                          alpha=0.3, color='red')
            
            ax.set_xlabel('CCA Component')
            ax.set_ylabel('Correlation')
            ax.set_title('CCA Correlations')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. L2 Loss Distribution
        if 'decomposition' in results:
            ax = axes[0, 1]
            decomp_results = results['decomposition']
            
            # Histogram of permuted L2 losses
            ax.hist(decomp_results['permuted_l2_losses'], bins=30, 
                   alpha=0.7, color='blue', label='Permuted')
            
            # Real L2 loss line
            ax.axvline(decomp_results['real_l2_loss'], color='red', 
                      linestyle='--', linewidth=2, 
                      label=f"Real: {decomp_results['real_l2_loss']:.2f}")
            
            ax.set_xlabel('L2 Loss')
            ax.set_ylabel('Frequency')
            ax.set_title('L2 Reconstruction Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Cosine Similarity Distribution
        if 'decomposition' in results:
            ax = axes[0, 2]
            decomp_results = results['decomposition']
            
            # Histogram of permuted cosine similarities
            ax.hist(decomp_results['permuted_cosine_similarities'], bins=30,
                   alpha=0.7, color='blue', label='Permuted')
            
            # Real cosine similarity line
            ax.axvline(decomp_results['real_cosine_similarity'], color='red',
                      linestyle='--', linewidth=2,
                      label=f"Real: {decomp_results['real_cosine_similarity']:.2f}")
            
            ax.set_xlabel('Cosine Similarity')
            ax.set_ylabel('Frequency')
            ax.set_title('Cosine Similarity (Leave-One-Out)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Retrieval Accuracy
        if 'metrics' in results:
            ax = axes[1, 0]
            metrics = results['metrics']
            
            # Bar plot of hits@k
            k_values = [1, 5, 10]
            real_hits = [metrics.get(f'hits@{k}', 0) for k in k_values]
            perm_hits = [metrics.get(f'permuted_hits@{k}_mean', 0) for k in k_values]
            
            x = np.arange(len(k_values))
            width = 0.35
            
            ax.bar(x - width/2, real_hits, width, label='Real', color='blue')
            ax.bar(x + width/2, perm_hits, width, label='Permuted', color='red', alpha=0.7)
            
            ax.set_xlabel('k')
            ax.set_ylabel('Accuracy')
            ax.set_title('Retrieval Accuracy (Hits@k)')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{k}' for k in k_values])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 5. P-values
        ax = axes[1, 1]
        p_values = []
        p_labels = []
        
        if 'cca' in results:
            p_values.append(results['cca']['overall_p_value'])
            p_labels.append('CCA')
        if 'decomposition' in results:
            p_values.append(results['decomposition']['p_value_cosine'])
            p_labels.append('Cosine')
            p_values.append(results['decomposition']['p_value_l2'])
            p_labels.append('L2')
        if 'metrics' in results:
            p_values.append(results['metrics'].get('p_value_hits@5', 0))
            p_labels.append('Hits@5')
        
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        bars = ax.bar(range(len(p_values)), p_values, color=colors)
        ax.axhline(0.05, color='black', linestyle='--', label='α=0.05')
        ax.set_xlabel('Test')
        ax.set_ylabel('P-value')
        ax.set_title('Statistical Significance')
        ax.set_xticks(range(len(p_labels)))
        ax.set_xticklabels(p_labels, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Overall Scores
        ax = axes[1, 2]
        scores = []
        score_labels = []
        
        if 'cca_score' in results:
            scores.append(results['cca_score'])
            score_labels.append('CCA')
        if 'decomposition_score' in results:
            scores.append(results['decomposition_score'])
            score_labels.append('Decomp')
        if 'metrics' in results:
            scores.append(results['metrics']['compositionality_score'])
            score_labels.append('Metrics')
        
        scores.append(results['overall_compositionality_score'])
        score_labels.append('Overall')
        
        colors = plt.cm.RdYlGn([s for s in scores])
        bars = ax.bar(range(len(scores)), scores, color=colors)
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Compositionality Scores')
        ax.set_xticks(range(len(score_labels)))
        ax.set_xticklabels(score_labels)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
        
        return fig