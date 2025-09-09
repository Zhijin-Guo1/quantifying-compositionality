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
            logger.info(f"Grouping {n_samples} samples into {unique_attrs} unique attribute combinations")
            return self.metrics.group_by_attributes(attributes, embeddings)
        
        return attributes, embeddings
    
    def analyze_compositionality(self,
                               embeddings: np.ndarray,
                               attributes: np.ndarray,
                               methods: List[str] = ['cca', 'decomposition', 'metrics'],
                               n_permutations: int = 100,
                               n_trials: int = 100,
                               group_by_attributes: bool = True,
                               verbose: bool = True) -> Dict:
        """
        Perform complete compositionality analysis.
        
        Args:
            embeddings: Embedding matrix (n_samples, n_dims)
            attributes: Binary attribute matrix (n_samples, n_attributes)
            methods: List of methods to use
            n_permutations: Number of permutations for testing
            n_trials: Number of trials for leave-one-out
            group_by_attributes: Whether to group by unique attributes
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
        
        # Preprocess data
        proc_attributes, proc_embeddings = self.preprocess_data(
            attributes, embeddings, group_by_attributes
        )
        
        if verbose and group_by_attributes:
            logger.info(f"After grouping - Embeddings: {proc_embeddings.shape}, Attributes: {proc_attributes.shape}")
        
        results = {
            'input_shape': {
                'embeddings': embeddings.shape,
                'attributes': attributes.shape
            },
            'processed_shape': {
                'embeddings': proc_embeddings.shape,
                'attributes': proc_attributes.shape
            }
        }
        
        # 1. CCA Analysis
        if 'cca' in methods:
            if verbose:
                logger.info("\n" + "-"*40)
                logger.info("Step 1: CCA Analysis")
                logger.info("-"*40)
            
            cca_results = self.cca_analyzer.analyze(
                proc_attributes,
                proc_embeddings,
                n_permutations=n_permutations
            )
            
            results['cca'] = cca_results
            results['cca_score'] = self.cca_analyzer.compute_cca_score(cca_results)
            
            if verbose:
                logger.info(f"CCA Score: {results['cca_score']:.4f}")
        
        # 2. Linear Decomposition
        if 'decomposition' in methods:
            if verbose:
                logger.info("\n" + "-"*40)
                logger.info("Step 2: Linear Decomposition")
                logger.info("-"*40)
            
            decomp_results = self.linear_decomposer.analyze_with_permutation(
                proc_attributes,
                proc_embeddings,
                n_permutations=n_permutations,
                n_trials_per_permutation=n_trials
            )
            
            results['decomposition'] = decomp_results
            results['decomposition_score'] = self.linear_decomposer.compute_decomposition_score(decomp_results)
            
            if verbose:
                logger.info(f"Decomposition Score: {results['decomposition_score']:.4f}")
        
        # 3. Additional Metrics
        if 'metrics' in methods:
            if verbose:
                logger.info("\n" + "-"*40)
                logger.info("Step 3: Computing Metrics")
                logger.info("-"*40)
            
            metrics_results = self.metrics.compute_all_metrics(
                proc_attributes,
                proc_embeddings,
                n_permutations=n_permutations,
                n_trials=n_trials
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