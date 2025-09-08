"""Canonical Correlation Analysis for compositionality measurement."""

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.utils import shuffle
from scipy import stats
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class CCAAnalyzer:
    """Canonical Correlation Analysis for measuring linear alignment."""
    
    def __init__(self, n_components: Optional[int] = None, regularization: float = 0.0):
        """
        Initialize CCA analyzer.
        
        Args:
            n_components: Number of CCA components (None for auto)
            regularization: Regularization parameter for CCA
        """
        self.n_components = n_components
        self.regularization = regularization
        
    def fit_cca(self, 
                X: np.ndarray, 
                Y: np.ndarray,
                n_components: Optional[int] = None) -> Tuple[CCA, np.ndarray, np.ndarray]:
        """
        Fit CCA model and transform data.
        
        Args:
            X: First set of variables (e.g., attributes)
            Y: Second set of variables (e.g., embeddings)
            n_components: Number of components to use
            
        Returns:
            cca_model: Fitted CCA model
            X_transformed: Transformed X
            Y_transformed: Transformed Y
        """
        if n_components is None:
            n_components = self.n_components or min(X.shape[1], Y.shape[1], X.shape[0] - 1)
        else:
            n_components = min(n_components, X.shape[1], Y.shape[1], X.shape[0] - 1)
        
        # Fit CCA
        cca_model = CCA(n_components=n_components)
        cca_model.fit(X, Y)
        
        # Transform data
        X_transformed, Y_transformed = cca_model.transform(X, Y)
        
        return cca_model, X_transformed, Y_transformed
    
    def compute_correlations(self,
                           X_transformed: np.ndarray,
                           Y_transformed: np.ndarray) -> np.ndarray:
        """
        Compute canonical correlations between transformed variables.
        
        Args:
            X_transformed: Transformed X variables
            Y_transformed: Transformed Y variables
            
        Returns:
            correlations: Array of canonical correlations
        """
        n_components = X_transformed.shape[1]
        correlations = np.zeros(n_components)
        
        for i in range(n_components):
            correlations[i] = np.corrcoef(X_transformed[:, i], Y_transformed[:, i])[0, 1]
        
        return correlations
    
    def analyze(self,
                attributes: np.ndarray,
                embeddings: np.ndarray,
                n_components: Optional[int] = None,
                n_permutations: int = 100) -> Dict:
        """
        Perform complete CCA analysis with permutation testing.
        
        Args:
            attributes: Attribute matrix (n_samples, n_attributes)
            embeddings: Embedding matrix (n_samples, n_dims)
            n_components: Number of CCA components
            n_permutations: Number of permutations for testing
            
        Returns:
            results: Dictionary containing analysis results
        """
        # Determine number of components
        if n_components is None:
            n_components = min(15, attributes.shape[1], embeddings.shape[1], attributes.shape[0] - 1)
        
        # Fit CCA on real data
        cca_model, X_c, Y_c = self.fit_cca(attributes, embeddings, n_components)
        real_correlations = self.compute_correlations(X_c, Y_c)
        
        # Permutation testing
        permuted_correlations = []
        for _ in range(n_permutations):
            # Shuffle embeddings
            embeddings_shuffled = shuffle(embeddings, random_state=None)
            
            # Fit CCA on shuffled data
            _, X_c_shuf, Y_c_shuf = self.fit_cca(attributes, embeddings_shuffled, n_components)
            perm_corr = self.compute_correlations(X_c_shuf, Y_c_shuf)
            permuted_correlations.append(perm_corr)
        
        permuted_correlations = np.array(permuted_correlations)
        
        # Compute statistics
        mean_real = np.mean(real_correlations)
        mean_permuted = np.mean(permuted_correlations)
        std_permuted = np.std(permuted_correlations)
        
        # Statistical significance (p-value)
        p_values = []
        for i in range(n_components):
            # Count how many permuted correlations exceed real
            p_val = np.mean(permuted_correlations[:, i] >= real_correlations[i])
            p_values.append(p_val)
        
        # Overall significance
        overall_p_value = np.mean(np.mean(permuted_correlations, axis=1) >= mean_real)
        
        results = {
            'n_components': n_components,
            'real_correlations': real_correlations,
            'permuted_correlations': permuted_correlations,
            'mean_real_correlation': mean_real,
            'mean_permuted_correlation': mean_permuted,
            'std_permuted_correlation': std_permuted,
            'p_values': np.array(p_values),
            'overall_p_value': overall_p_value,
            'significant': overall_p_value < 0.05,
            'cca_model': cca_model
        }
        
        logger.info(f"CCA Analysis Complete:")
        logger.info(f"  Mean real correlation: {mean_real:.4f}")
        logger.info(f"  Mean permuted correlation: {mean_permuted:.4f}")
        logger.info(f"  P-value: {overall_p_value:.4f}")
        logger.info(f"  Significant: {results['significant']}")
        
        return results
    
    def compute_cca_score(self, results: Dict) -> float:
        """
        Compute a single CCA-based compositionality score.
        
        Args:
            results: Results from analyze()
            
        Returns:
            score: Compositionality score (0 to 1)
        """
        mean_real = results['mean_real_correlation']
        mean_permuted = results['mean_permuted_correlation']
        
        # Normalize the difference
        if mean_permuted > 0:
            score = (mean_real - mean_permuted) / (1 - mean_permuted)
        else:
            score = mean_real
        
        # Clip to [0, 1]
        score = np.clip(score, 0, 1)
        
        return score