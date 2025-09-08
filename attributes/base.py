"""
Base class for attribute generation across different modalities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAttributeGenerator(ABC):
    """
    Abstract base class for attribute generation.
    
    This class defines the interface for extracting and encoding attributes
    from different data sources (sentences, words, knowledge graphs).
    """
    
    def __init__(self, data_path: Path, cache_dir: Optional[Path] = None):
        """
        Initialize attribute generator.
        
        Args:
            data_path: Path to the data source
            cache_dir: Optional directory for caching processed attributes
        """
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.attribute_matrix = None
        self.entity_ids = None
        self.attribute_names = None
        self.metadata = {}
        
    @abstractmethod
    def load_data(self) -> Any:
        """Load raw data from source."""
        pass
    
    @abstractmethod
    def extract_attributes(self, data: Any) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract attributes from raw data.
        
        Args:
            data: Raw data loaded from source
            
        Returns:
            Tuple of:
                - attribute_matrix: Binary matrix (n_entities x n_attributes)
                - entity_ids: List of entity identifiers
                - attribute_names: List of attribute names
        """
        pass
    
    def generate(self, force_regenerate: bool = False) -> Dict[str, Any]:
        """
        Generate attribute matrix and metadata.
        
        Args:
            force_regenerate: If True, regenerate even if cached
            
        Returns:
            Dictionary containing:
                - 'matrix': Binary attribute matrix (numpy array)
                - 'entity_ids': List of entity identifiers
                - 'attribute_names': List of attribute names
                - 'metadata': Additional metadata dict
        """
        # Check cache if available
        if not force_regenerate and self.cache_dir and self._load_from_cache():
            logger.info(f"Loaded attributes from cache: {self.cache_dir}")
            return self._get_results()
        
        # Generate attributes
        logger.info(f"Generating attributes from: {self.data_path}")
        data = self.load_data()
        
        self.attribute_matrix, self.entity_ids, self.attribute_names = \
            self.extract_attributes(data)
        
        # Validate output
        self._validate_attributes()
        
        # Compute metadata
        self.metadata = self._compute_metadata()
        
        # Save to cache if available
        if self.cache_dir:
            self._save_to_cache()
            logger.info(f"Saved attributes to cache: {self.cache_dir}")
        
        return self._get_results()
    
    def _validate_attributes(self):
        """Validate generated attributes."""
        assert self.attribute_matrix is not None, "Attribute matrix is None"
        assert self.entity_ids is not None, "Entity IDs are None"
        assert self.attribute_names is not None, "Attribute names are None"
        
        n_entities, n_attributes = self.attribute_matrix.shape
        assert len(self.entity_ids) == n_entities, \
            f"Entity count mismatch: {len(self.entity_ids)} != {n_entities}"
        assert len(self.attribute_names) == n_attributes, \
            f"Attribute count mismatch: {len(self.attribute_names)} != {n_attributes}"
        
        # Check binary values
        unique_vals = np.unique(self.attribute_matrix)
        assert set(unique_vals).issubset({0, 1}), \
            f"Non-binary values found: {unique_vals}"
        
        logger.info(f"Validated attributes: {n_entities} entities x {n_attributes} attributes")
    
    def _compute_metadata(self) -> Dict[str, Any]:
        """Compute metadata about the attributes."""
        n_entities, n_attributes = self.attribute_matrix.shape
        
        metadata = {
            'n_entities': n_entities,
            'n_attributes': n_attributes,
            'sparsity': 1.0 - (self.attribute_matrix.sum() / (n_entities * n_attributes)),
            'mean_attributes_per_entity': self.attribute_matrix.sum(axis=1).mean(),
            'attribute_frequency': self.attribute_matrix.sum(axis=0) / n_entities,
        }
        
        return metadata
    
    def _get_results(self) -> Dict[str, Any]:
        """Get results dictionary."""
        return {
            'matrix': self.attribute_matrix,
            'entity_ids': self.entity_ids,
            'attribute_names': self.attribute_names,
            'metadata': self.metadata
        }
    
    def _load_from_cache(self) -> bool:
        """Load attributes from cache if available."""
        if not self.cache_dir:
            return False
        
        cache_file = self.cache_dir / f"{self.__class__.__name__}.npz"
        if not cache_file.exists():
            return False
        
        try:
            data = np.load(cache_file, allow_pickle=True)
            self.attribute_matrix = data['matrix']
            self.entity_ids = data['entity_ids'].tolist()
            self.attribute_names = data['attribute_names'].tolist()
            self.metadata = data['metadata'].item()
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False
    
    def _save_to_cache(self):
        """Save attributes to cache."""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{self.__class__.__name__}.npz"
        
        np.savez(
            cache_file,
            matrix=self.attribute_matrix,
            entity_ids=self.entity_ids,
            attribute_names=self.attribute_names,
            metadata=self.metadata
        )
    
    def get_statistics(self) -> pd.DataFrame:
        """Get statistics about attributes."""
        if self.attribute_matrix is None:
            raise ValueError("Attributes not generated yet. Call generate() first.")
        
        stats = []
        for i, attr_name in enumerate(self.attribute_names):
            count = self.attribute_matrix[:, i].sum()
            stats.append({
                'attribute': attr_name,
                'count': count,
                'frequency': count / len(self.entity_ids),
            })
        
        return pd.DataFrame(stats).sort_values('count', ascending=False)