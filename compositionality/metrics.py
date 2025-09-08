"""Evaluation metrics for compositionality analysis."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CompositionalityMetrics:
    """Metrics for evaluating compositional alignment."""
    
    @staticmethod
    def cosine_similarity_matrix(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cosine similarity matrix.
        
        Args:
            X: First matrix (n_samples, n_features)
            Y: Second matrix (optional, defaults to X)
            
        Returns:
            similarity_matrix: Cosine similarity matrix
        """
        return cosine_similarity(X, Y)
    
    @staticmethod
    def hits_at_k(predictions: np.ndarray,
                  targets: np.ndarray,
                  k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Compute Hits@k metrics.
        
        Args:
            predictions: Predicted embeddings (n_samples, n_dims)
            targets: Target embeddings (n_samples, n_dims)
            k_values: List of k values to compute
            
        Returns:
            hits: Dictionary with hits@k for each k
        """
        n_samples = len(predictions)
        
        # Compute similarity matrix
        similarities = cosine_similarity(predictions, targets)
        
        hits = {}
        for k in k_values:
            k = min(k, n_samples)
            correct = 0
            
            for i in range(n_samples):
                # Get top k indices for prediction i
                top_k = np.argsort(-similarities[i])[:k]
                
                # Check if correct target is in top k
                if i in top_k:
                    correct += 1
            
            hits[f'hits@{k}'] = correct / n_samples
        
        return hits
    
    @staticmethod
    def retrieval_accuracy_leave_one_out(attributes: np.ndarray,
                                        embeddings: np.ndarray,
                                        attribute_embeddings: np.ndarray,
                                        k_values: List[int] = [1, 5, 10],
                                        n_trials: int = 100) -> Dict:
        """
        Compute retrieval accuracy using leave-one-out evaluation.
        
        Args:
            attributes: Binary attribute matrix
            embeddings: Embedding matrix
            attribute_embeddings: Learned attribute embeddings
            k_values: List of k values for hits@k
            n_trials: Number of trials
            
        Returns:
            results: Retrieval accuracy results
        """
        n_samples = len(embeddings)
        hits_counts = {k: 0 for k in k_values}
        
        for _ in range(n_trials):
            # Select random sample to leave out
            idx = np.random.randint(n_samples)
            
            # Split data
            train_attributes = np.delete(attributes, idx, axis=0)
            train_embeddings = np.delete(embeddings, idx, axis=0)
            test_attributes = attributes[idx:idx+1]
            
            # Learn attribute embeddings from training data
            attr_emb = np.linalg.pinv(train_attributes).dot(train_embeddings)
            
            # Reconstruct test embedding
            predicted = np.dot(test_attributes, attr_emb)[0]
            
            # Compute similarities to all embeddings
            similarities = cosine_similarity([predicted], embeddings)[0]
            
            # Check hits@k
            for k in k_values:
                k_actual = min(k, n_samples)
                top_k = np.argsort(-similarities)[:k_actual]
                if idx in top_k:
                    hits_counts[k] += 1
        
        # Compute accuracies
        results = {}
        for k in k_values:
            results[f'hits@{k}'] = hits_counts[k] / n_trials
        
        return results
    
    @staticmethod
    def group_by_attributes(attributes: np.ndarray,
                           embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Group samples by unique attribute combinations.
        
        Args:
            attributes: Binary attribute matrix
            embeddings: Embedding matrix
            
        Returns:
            unique_attributes: Unique attribute combinations
            mean_embeddings: Mean embedding for each group
        """
        # Convert to tuples for grouping
        attr_tuples = [tuple(row) for row in attributes]
        
        # Find unique combinations
        unique_tuples = list(set(attr_tuples))
        unique_attributes = np.array([list(t) for t in unique_tuples])
        
        # Compute mean embeddings for each group
        mean_embeddings = []
        for unique_attr in unique_tuples:
            # Find all samples with this attribute combination
            indices = [i for i, attr in enumerate(attr_tuples) if attr == unique_attr]
            
            # Compute mean embedding
            group_embeddings = embeddings[indices]
            mean_embedding = np.mean(group_embeddings, axis=0)
            mean_embeddings.append(mean_embedding)
        
        mean_embeddings = np.array(mean_embeddings)
        
        logger.info(f"Grouped {len(attributes)} samples into {len(unique_attributes)} unique combinations")
        
        return unique_attributes, mean_embeddings
    
    @staticmethod
    def compute_all_metrics(attributes: np.ndarray,
                           embeddings: np.ndarray,
                           n_permutations: int = 100,
                           n_trials: int = 100) -> Dict:
        """
        Compute all compositionality metrics with permutation testing.
        
        Args:
            attributes: Binary attribute matrix
            embeddings: Embedding matrix
            n_permutations: Number of permutations
            n_trials: Number of trials for leave-one-out
            
        Returns:
            metrics: Dictionary of all metrics
        """
        metrics = {}
        
        # Group by unique attributes if there are duplicates
        n_samples = len(attributes)
        unique_attrs = len(np.unique(attributes, axis=0))
        
        if unique_attrs < n_samples:
            logger.info("Grouping by unique attribute combinations...")
            attributes, embeddings = CompositionalityMetrics.group_by_attributes(
                attributes, embeddings
            )
        
        # Learn attribute embeddings
        attribute_embeddings = np.linalg.pinv(attributes).dot(embeddings)
        reconstructed = np.dot(attributes, attribute_embeddings)
        
        # 1. L2 reconstruction loss
        real_l2 = np.linalg.norm(embeddings - reconstructed)
        metrics['l2_loss'] = real_l2
        
        # 2. Mean cosine similarity
        cosine_sims = []
        for i in range(len(embeddings)):
            norm_orig = np.linalg.norm(embeddings[i])
            norm_recon = np.linalg.norm(reconstructed[i])
            if norm_orig > 0 and norm_recon > 0:
                sim = np.dot(embeddings[i], reconstructed[i]) / (norm_orig * norm_recon)
                cosine_sims.append(sim)
        metrics['mean_cosine_similarity'] = np.mean(cosine_sims)
        
        # 3. Retrieval accuracy (Hits@k)
        k_values = [1, 5, 10]
        retrieval_results = CompositionalityMetrics.retrieval_accuracy_leave_one_out(
            attributes, embeddings, attribute_embeddings, k_values, n_trials
        )
        metrics.update(retrieval_results)
        
        # 4. Permutation testing
        logger.info(f"Running {n_permutations} permutation tests...")
        perm_l2_losses = []
        perm_cosine_sims = []
        perm_hits_at_5 = []
        
        for _ in range(n_permutations):
            # Shuffle embeddings
            perm_embeddings = shuffle(embeddings, random_state=None)
            
            # Learn on shuffled data
            perm_attr_emb = np.linalg.pinv(attributes).dot(perm_embeddings)
            perm_reconstructed = np.dot(attributes, perm_attr_emb)
            
            # L2 loss
            perm_l2 = np.linalg.norm(perm_embeddings - perm_reconstructed)
            perm_l2_losses.append(perm_l2)
            
            # Cosine similarity
            perm_cos_sims = []
            for i in range(len(perm_embeddings)):
                norm_orig = np.linalg.norm(perm_embeddings[i])
                norm_recon = np.linalg.norm(perm_reconstructed[i])
                if norm_orig > 0 and norm_recon > 0:
                    sim = np.dot(perm_embeddings[i], perm_reconstructed[i]) / (norm_orig * norm_recon)
                    perm_cos_sims.append(sim)
            perm_cosine_sims.append(np.mean(perm_cos_sims))
            
            # Hits@5 (subset for efficiency)
            perm_retrieval = CompositionalityMetrics.retrieval_accuracy_leave_one_out(
                attributes, perm_embeddings, perm_attr_emb, [5], min(20, n_trials)
            )
            perm_hits_at_5.append(perm_retrieval['hits@5'])
        
        # Add permutation statistics
        metrics['permuted_l2_mean'] = np.mean(perm_l2_losses)
        metrics['permuted_l2_std'] = np.std(perm_l2_losses)
        metrics['permuted_cosine_mean'] = np.mean(perm_cosine_sims)
        metrics['permuted_cosine_std'] = np.std(perm_cosine_sims)
        metrics['permuted_hits@5_mean'] = np.mean(perm_hits_at_5)
        metrics['permuted_hits@5_std'] = np.std(perm_hits_at_5)
        
        # Compute p-values
        metrics['p_value_l2'] = np.mean(np.array(perm_l2_losses) <= real_l2)
        metrics['p_value_cosine'] = np.mean(np.array(perm_cosine_sims) >= metrics['mean_cosine_similarity'])
        metrics['p_value_hits@5'] = np.mean(np.array(perm_hits_at_5) >= metrics['hits@5'])
        
        # Overall compositionality score (0 to 1)
        # Combine normalized improvements over random baseline
        cosine_improvement = (metrics['mean_cosine_similarity'] - metrics['permuted_cosine_mean']) / (1 - metrics['permuted_cosine_mean'] + 1e-10)
        hits_improvement = (metrics['hits@5'] - metrics['permuted_hits@5_mean']) / (1 - metrics['permuted_hits@5_mean'] + 1e-10)
        
        metrics['compositionality_score'] = np.clip((cosine_improvement + hits_improvement) / 2, 0, 1)
        
        return metrics