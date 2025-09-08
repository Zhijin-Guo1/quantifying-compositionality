"""Base class for all embedding extractors."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class EmbeddingExtractor(ABC):
    """Abstract base class for extracting embeddings from raw data."""
    
    def __init__(self, embedding_dim: Optional[int] = None):
        """
        Initialize the embedding extractor.
        
        Args:
            embedding_dim: Dimension of the output embeddings
        """
        self.embedding_dim = embedding_dim
        self.model = None
        
    @abstractmethod
    def extract(self, data: Union[List[str], List[Any]], **kwargs) -> np.ndarray:
        """
        Extract embeddings from raw data.
        
        Args:
            data: List of raw data items (sentences, words, entities)
            **kwargs: Additional arguments specific to the extractor
            
        Returns:
            embeddings: numpy array of shape (n_samples, embedding_dim)
        """
        pass
    
    def normalize(self, embeddings: np.ndarray, norm_type: str = 'l2') -> np.ndarray:
        """
        Normalize embeddings.
        
        Args:
            embeddings: numpy array of embeddings
            norm_type: Type of normalization ('l2', 'l1', or None)
            
        Returns:
            normalized_embeddings: Normalized embeddings
        """
        if norm_type == 'l2':
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embeddings / norms
        elif norm_type == 'l1':
            norms = np.abs(embeddings).sum(axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embeddings / norms
        else:
            return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       data: Optional[List[Any]] = None,
                       save_path: str = 'embeddings.npz') -> None:
        """
        Save embeddings to file.
        
        Args:
            embeddings: numpy array of embeddings
            data: Original data items (optional)
            save_path: Path to save the embeddings
        """
        save_dict = {'embeddings': embeddings}
        if data is not None:
            save_dict['data'] = data
        
        np.savez_compressed(save_path, **save_dict)
        logger.info(f"Embeddings saved to {save_path}")
    
    def load_embeddings(self, load_path: str) -> Tuple[np.ndarray, Optional[List[Any]]]:
        """
        Load embeddings from file.
        
        Args:
            load_path: Path to load embeddings from
            
        Returns:
            embeddings: numpy array of embeddings
            data: Original data items (if available)
        """
        loaded = np.load(load_path, allow_pickle=True)
        embeddings = loaded['embeddings']
        data = loaded.get('data', None)
        
        logger.info(f"Loaded embeddings from {load_path}")
        return embeddings, data
    
    def get_embedding_dim(self) -> Optional[int]:
        """Get the dimension of the embeddings."""
        return self.embedding_dim
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim})"