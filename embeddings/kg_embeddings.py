"""Knowledge Graph embedding loader for pre-trained embeddings."""

import torch
import numpy as np
from typing import List, Optional, Union, Dict, Tuple
import logging
import os
from .base import EmbeddingExtractor

logger = logging.getLogger(__name__)


class KGEmbeddingLoader(EmbeddingExtractor):
    """Load pre-trained Knowledge Graph embeddings from .pt files."""
    
    def __init__(self,
                 model_type: str = 'TransE',
                 embedding_path: Optional[str] = None,
                 kg_embedding_dir: str = 'KG_embedding'):
        """
        Initialize the KG embedding loader.
        
        Args:
            model_type: Type of KG model ('TransE' or 'DistMult')
            embedding_path: Direct path to .pt file (overrides model_type)
            kg_embedding_dir: Directory containing KG embeddings
        """
        super().__init__()
        
        self.model_type = model_type
        self.kg_embedding_dir = kg_embedding_dir
        self.embeddings_dict = None
        self.entity_embeddings = None
        self.relation_embeddings = None
        self.entity_to_id = {}
        self.id_to_entity = {}
        
        # Determine embedding file path
        if embedding_path:
            self.embedding_path = embedding_path
        else:
            # Use default paths based on model type
            if model_type == 'TransE':
                self.embedding_path = os.path.join(kg_embedding_dir, '300_epochs_TransE_gpu34.pt')
            elif model_type == 'DistMult':
                self.embedding_path = os.path.join(kg_embedding_dir, '300_epochs_DistMult_gpu34.pt')
            else:
                raise ValueError(f"Unknown model type: {model_type}. Use 'TransE' or 'DistMult'")
        
        # Load embeddings
        self.load_embeddings()
    
    def load_embeddings(self):
        """Load embeddings from .pt file."""
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_path}")
        
        logger.info(f"Loading KG embeddings from {self.embedding_path}")
        
        # Load the .pt file
        device = torch.device('cpu')  # Load to CPU by default
        checkpoint = torch.load(self.embedding_path, map_location=device)
        
        # Extract embeddings based on checkpoint structure
        if isinstance(checkpoint, dict):
            # Standard checkpoint format
            if 'entity_embeddings' in checkpoint:
                self.entity_embeddings = checkpoint['entity_embeddings']
            elif 'ent_embeddings' in checkpoint:
                self.entity_embeddings = checkpoint['ent_embeddings']
            elif 'model_state_dict' in checkpoint:
                # Extract from model state dict
                state_dict = checkpoint['model_state_dict']
                for key in state_dict.keys():
                    if 'entity' in key.lower() or 'ent' in key.lower():
                        self.entity_embeddings = state_dict[key]
                        break
            
            # Extract relation embeddings if available
            if 'relation_embeddings' in checkpoint:
                self.relation_embeddings = checkpoint['relation_embeddings']
            elif 'rel_embeddings' in checkpoint:
                self.relation_embeddings = checkpoint['rel_embeddings']
            
            # Extract entity mappings if available
            if 'entity_to_id' in checkpoint:
                self.entity_to_id = checkpoint['entity_to_id']
                self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
            elif 'ent2id' in checkpoint:
                self.entity_to_id = checkpoint['ent2id']
                self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
            
            # Store full checkpoint for reference
            self.embeddings_dict = checkpoint
        else:
            # Assume it's just the embedding tensor
            self.entity_embeddings = checkpoint
        
        # Convert to numpy if tensor
        if isinstance(self.entity_embeddings, torch.Tensor):
            self.entity_embeddings = self.entity_embeddings.detach().cpu().numpy()
        if isinstance(self.relation_embeddings, torch.Tensor):
            self.relation_embeddings = self.relation_embeddings.detach().cpu().numpy()
        
        # Set embedding dimension
        if self.entity_embeddings is not None:
            self.embedding_dim = self.entity_embeddings.shape[1]
            logger.info(f"Loaded entity embeddings: shape {self.entity_embeddings.shape}")
        
        if self.relation_embeddings is not None:
            logger.info(f"Loaded relation embeddings: shape {self.relation_embeddings.shape}")
        
        logger.info(f"KG embeddings loaded successfully")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def extract(self,
                entity_ids: Union[List[int], List[str], np.ndarray],
                normalize: bool = False) -> np.ndarray:
        """
        Extract embeddings for given entity IDs.
        
        Args:
            entity_ids: List of entity IDs (integers or strings)
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            embeddings: numpy array of shape (n_entities, embedding_dim)
        """
        if self.entity_embeddings is None:
            raise ValueError("No entity embeddings loaded")
        
        embeddings = []
        
        for entity_id in entity_ids:
            if isinstance(entity_id, str):
                # Convert string ID to integer if mapping exists
                if entity_id in self.entity_to_id:
                    idx = self.entity_to_id[entity_id]
                else:
                    logger.warning(f"Entity '{entity_id}' not found in mapping")
                    # Use zero vector for unknown entities
                    embeddings.append(np.zeros(self.embedding_dim))
                    continue
            else:
                idx = int(entity_id)
            
            if idx < 0 or idx >= len(self.entity_embeddings):
                logger.warning(f"Entity index {idx} out of range")
                embeddings.append(np.zeros(self.embedding_dim))
            else:
                embeddings.append(self.entity_embeddings[idx])
        
        embeddings = np.array(embeddings)
        
        if normalize:
            embeddings = self.normalize(embeddings, norm_type='l2')
        
        logger.info(f"Extracted embeddings: shape {embeddings.shape}")
        return embeddings
    
    def extract_all_entities(self, normalize: bool = False) -> np.ndarray:
        """
        Extract embeddings for all entities.
        
        Args:
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            embeddings: numpy array of shape (n_entities, embedding_dim)
        """
        if self.entity_embeddings is None:
            raise ValueError("No entity embeddings loaded")
        
        embeddings = self.entity_embeddings.copy()
        
        if normalize:
            embeddings = self.normalize(embeddings, norm_type='l2')
        
        return embeddings
    
    def extract_relations(self,
                         relation_ids: Optional[Union[List[int], List[str]]] = None,
                         normalize: bool = False) -> np.ndarray:
        """
        Extract relation embeddings.
        
        Args:
            relation_ids: List of relation IDs (None for all relations)
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            embeddings: numpy array of shape (n_relations, embedding_dim)
        """
        if self.relation_embeddings is None:
            raise ValueError("No relation embeddings available")
        
        if relation_ids is None:
            embeddings = self.relation_embeddings.copy()
        else:
            embeddings = []
            for rel_id in relation_ids:
                if isinstance(rel_id, int) and 0 <= rel_id < len(self.relation_embeddings):
                    embeddings.append(self.relation_embeddings[rel_id])
                else:
                    logger.warning(f"Relation {rel_id} not found")
                    embeddings.append(np.zeros(self.embedding_dim))
            embeddings = np.array(embeddings)
        
        if normalize:
            embeddings = self.normalize(embeddings, norm_type='l2')
        
        return embeddings
    
    def get_entity_embedding(self, entity_id: Union[int, str]) -> np.ndarray:
        """Get embedding for a single entity."""
        return self.extract([entity_id])[0]
    
    def get_checkpoint_info(self) -> Dict:
        """Get information about the loaded checkpoint."""
        if self.embeddings_dict is None:
            return {}
        
        info = {
            'model_type': self.model_type,
            'embedding_path': self.embedding_path,
            'num_entities': len(self.entity_embeddings) if self.entity_embeddings is not None else 0,
            'num_relations': len(self.relation_embeddings) if self.relation_embeddings is not None else 0,
            'embedding_dim': self.embedding_dim,
            'has_entity_mapping': len(self.entity_to_id) > 0,
            'checkpoint_keys': list(self.embeddings_dict.keys()) if isinstance(self.embeddings_dict, dict) else []
        }
        
        return info
    
    def save_as_numpy(self, save_path: str):
        """Save embeddings as numpy arrays."""
        save_dict = {}
        
        if self.entity_embeddings is not None:
            save_dict['entity_embeddings'] = self.entity_embeddings
        
        if self.relation_embeddings is not None:
            save_dict['relation_embeddings'] = self.relation_embeddings
        
        if self.entity_to_id:
            save_dict['entity_to_id'] = self.entity_to_id
        
        np.savez_compressed(save_path, **save_dict)
        logger.info(f"Saved KG embeddings to {save_path}")
    
    def __repr__(self) -> str:
        num_entities = len(self.entity_embeddings) if self.entity_embeddings is not None else 0
        return f"KGEmbeddingLoader(model='{self.model_type}', num_entities={num_entities}, embedding_dim={self.embedding_dim})"