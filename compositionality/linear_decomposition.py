"""Linear decomposition methods for compositionality analysis."""

import numpy as np
from sklearn.utils import shuffle
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class LinearDecomposer:
    """Linear decomposition for testing additive compositionality."""
    
    def __init__(self, method: str = 'pseudo_inverse'):
        """
        Initialize linear decomposer.
        
        Args:
            method: Decomposition method ('pseudo_inverse' or 'lstsq')
        """
        self.method = method
    
    def compute_attribute_embeddings(self,
                                    attributes: np.ndarray,
                                    embeddings: np.ndarray) -> np.ndarray:
        """
        Compute attribute embeddings using linear decomposition.
        
        Args:
            attributes: Binary attribute matrix (n_samples, n_attributes)
            embeddings: Embedding matrix (n_samples, n_dims)
            
        Returns:
            attribute_embeddings: Learned attribute embeddings (n_attributes, n_dims)
        """
        if self.method == 'pseudo_inverse':
            # Use Moore-Penrose pseudo-inverse
            attribute_embeddings = np.linalg.pinv(attributes).dot(embeddings)
        elif self.method == 'lstsq':
            # Use least squares
            attribute_embeddings, _, _, _ = np.linalg.lstsq(attributes, embeddings, rcond=None)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return attribute_embeddings
    
    def reconstruct_embeddings(self,
                             attributes: np.ndarray,
                             attribute_embeddings: np.ndarray) -> np.ndarray:
        """
        Reconstruct embeddings from attributes and attribute embeddings.
        
        Args:
            attributes: Binary attribute matrix (n_samples, n_attributes)
            attribute_embeddings: Attribute embeddings (n_attributes, n_dims)
            
        Returns:
            reconstructed: Reconstructed embeddings (n_samples, n_dims)
        """
        return np.dot(attributes, attribute_embeddings)
    
    def compute_reconstruction_loss(self,
                                   original: np.ndarray,
                                   reconstructed: np.ndarray,
                                   metric: str = 'l2') -> float:
        """
        Compute reconstruction loss.
        
        Args:
            original: Original embeddings
            reconstructed: Reconstructed embeddings
            metric: Loss metric ('l2' or 'cosine')
            
        Returns:
            loss: Reconstruction loss
        """
        if metric == 'l2':
            return np.linalg.norm(original - reconstructed)
        elif metric == 'cosine':
            # Compute mean cosine similarity
            similarities = []
            for i in range(len(original)):
                orig_norm = np.linalg.norm(original[i])
                recon_norm = np.linalg.norm(reconstructed[i])
                if orig_norm > 0 and recon_norm > 0:
                    sim = np.dot(original[i], reconstructed[i]) / (orig_norm * recon_norm)
                    similarities.append(sim)
            return 1 - np.mean(similarities)  # Convert to loss
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def leave_one_out_evaluation(self,
                                attributes: np.ndarray,
                                embeddings: np.ndarray,
                                n_trials: int = 100) -> Dict:
        """
        Perform leave-one-out evaluation.
        
        Args:
            attributes: Binary attribute matrix
            embeddings: Embedding matrix
            n_trials: Number of leave-one-out trials
            
        Returns:
            results: Evaluation results
        """
        n_samples = attributes.shape[0]
        cosine_similarities = []
        l2_losses = []
        
        for _ in range(n_trials):
            # Select random sample to leave out
            idx = np.random.randint(n_samples)
            
            # Split data
            train_attributes = np.delete(attributes, idx, axis=0)
            train_embeddings = np.delete(embeddings, idx, axis=0)
            test_attributes = attributes[idx:idx+1]
            test_embedding = embeddings[idx]
            
            # Learn attribute embeddings from training data
            attribute_embeddings = self.compute_attribute_embeddings(
                train_attributes, train_embeddings
            )
            
            # Reconstruct test embedding
            reconstructed = self.reconstruct_embeddings(
                test_attributes, attribute_embeddings
            )[0]
            
            # Compute metrics
            # Cosine similarity
            norm_orig = np.linalg.norm(test_embedding)
            norm_recon = np.linalg.norm(reconstructed)
            if norm_orig > 0 and norm_recon > 0:
                cosine_sim = np.dot(test_embedding, reconstructed) / (norm_orig * norm_recon)
                cosine_similarities.append(np.clip(cosine_sim, -1, 1))
            
            # L2 loss
            l2_loss = np.linalg.norm(test_embedding - reconstructed)
            l2_losses.append(l2_loss)
        
        return {
            'cosine_similarities': np.array(cosine_similarities),
            'mean_cosine_similarity': np.mean(cosine_similarities),
            'std_cosine_similarity': np.std(cosine_similarities),
            'l2_losses': np.array(l2_losses),
            'mean_l2_loss': np.mean(l2_losses),
            'std_l2_loss': np.std(l2_losses)
        }
    
    def analyze_with_permutation(self,
                                attributes: np.ndarray,
                                embeddings: np.ndarray,
                                n_permutations: int = 100,
                                n_trials_per_permutation: int = 100) -> Dict:
        """
        Analyze linear decomposition with permutation testing.
        
        Args:
            attributes: Binary attribute matrix
            embeddings: Embedding matrix
            n_permutations: Number of permutations
            n_trials_per_permutation: Number of trials per permutation
            
        Returns:
            results: Complete analysis results
        """
        # Real data evaluation
        logger.info("Evaluating on real data...")
        
        # Full reconstruction (no leave-one-out)
        attribute_embeddings = self.compute_attribute_embeddings(attributes, embeddings)
        reconstructed = self.reconstruct_embeddings(attributes, attribute_embeddings)
        real_l2_loss = self.compute_reconstruction_loss(embeddings, reconstructed, 'l2')
        
        # Leave-one-out evaluation
        loo_results = self.leave_one_out_evaluation(
            attributes, embeddings, n_trials_per_permutation
        )
        real_cosine_sim = loo_results['mean_cosine_similarity']
        
        # Permutation testing
        logger.info(f"Running {n_permutations} permutation tests...")
        permuted_l2_losses = []
        permuted_cosine_sims = []
        
        for i in range(n_permutations):
            # Shuffle embeddings
            shuffled_embeddings = shuffle(embeddings, random_state=None)
            
            # Full reconstruction
            attr_emb_shuf = self.compute_attribute_embeddings(attributes, shuffled_embeddings)
            recon_shuf = self.reconstruct_embeddings(attributes, attr_emb_shuf)
            perm_l2 = self.compute_reconstruction_loss(shuffled_embeddings, recon_shuf, 'l2')
            permuted_l2_losses.append(perm_l2)
            
            # Leave-one-out (subset for efficiency)
            loo_shuf = self.leave_one_out_evaluation(
                attributes, shuffled_embeddings, min(20, n_trials_per_permutation)
            )
            permuted_cosine_sims.append(loo_shuf['mean_cosine_similarity'])
        
        permuted_l2_losses = np.array(permuted_l2_losses)
        permuted_cosine_sims = np.array(permuted_cosine_sims)
        
        # Compute p-values
        p_value_l2 = np.mean(permuted_l2_losses <= real_l2_loss)
        p_value_cosine = np.mean(permuted_cosine_sims >= real_cosine_sim)
        
        results = {
            # Real data results
            'attribute_embeddings': attribute_embeddings,
            'real_l2_loss': real_l2_loss,
            'real_cosine_similarity': real_cosine_sim,
            'loo_results': loo_results,
            
            # Permutation results
            'permuted_l2_losses': permuted_l2_losses,
            'mean_permuted_l2': np.mean(permuted_l2_losses),
            'std_permuted_l2': np.std(permuted_l2_losses),
            
            'permuted_cosine_similarities': permuted_cosine_sims,
            'mean_permuted_cosine': np.mean(permuted_cosine_sims),
            'std_permuted_cosine': np.std(permuted_cosine_sims),
            
            # Statistical significance
            'p_value_l2': p_value_l2,
            'p_value_cosine': p_value_cosine,
            'significant_l2': p_value_l2 < 0.05,
            'significant_cosine': p_value_cosine < 0.05
        }
        
        logger.info(f"Linear Decomposition Analysis Complete:")
        logger.info(f"  Real L2 loss: {real_l2_loss:.4f}")
        logger.info(f"  Mean permuted L2: {results['mean_permuted_l2']:.4f}")
        logger.info(f"  Real cosine similarity: {real_cosine_sim:.4f}")
        logger.info(f"  Mean permuted cosine: {results['mean_permuted_cosine']:.4f}")
        logger.info(f"  P-value (L2): {p_value_l2:.4f}")
        logger.info(f"  P-value (cosine): {p_value_cosine:.4f}")
        
        return results
    
    def compute_decomposition_score(self, results: Dict) -> float:
        """
        Compute a single decomposition-based compositionality score.
        
        Args:
            results: Results from analyze_with_permutation()
            
        Returns:
            score: Compositionality score (0 to 1)
        """
        # Combine cosine similarity improvement over random
        real_cosine = results['real_cosine_similarity']
        perm_cosine = results['mean_permuted_cosine']
        
        if perm_cosine < 1:
            score = (real_cosine - perm_cosine) / (1 - perm_cosine)
        else:
            score = real_cosine
        
        # Clip to [0, 1]
        score = np.clip(score, 0, 1)
        
        return score