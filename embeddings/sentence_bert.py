"""SBERT embedding extractor with layer-wise extraction support."""

import torch
import numpy as np
from typing import List, Optional, Union, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import logging
from .base import EmbeddingExtractor

logger = logging.getLogger(__name__)


class SentenceBERTExtractor(EmbeddingExtractor):
    """Extract sentence embeddings using Sentence-BERT models with layer-wise support."""
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 device: Optional[str] = None):
        """
        Initialize the Sentence-BERT extractor.
        
        Args:
            model_name: Name or path of the sentence-transformers model
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        # Access transformer and pooling components
        self.transformer_model = self.model[0].auto_model
        self.pooling_layer = self.model[1]
        
        # Enable hidden states output
        self.transformer_model.config.output_hidden_states = True
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get embedding dimension
        super().__init__(embedding_dim=self.model.get_sentence_embedding_dimension())
        
        logger.info(f"Loaded model: {model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def extract(self, 
                sentences: List[str],
                layer: Optional[int] = None,
                batch_size: int = 32,
                show_progress: bool = True,
                normalize: bool = True) -> np.ndarray:
        """
        Extract sentence embeddings.
        
        Args:
            sentences: List of sentences to encode
            layer: Specific layer to extract (None for final layer)
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            embeddings: numpy array of shape (n_sentences, embedding_dim)
        """
        if layer is None:
            # Use the standard sentence-transformers encoding
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return embeddings
        else:
            # Extract from specific layer
            return self.extract_layer_embeddings(
                sentences, 
                layer=layer,
                batch_size=batch_size,
                normalize=normalize
            )
    
    def extract_layer_embeddings(self,
                                sentences: List[str],
                                layer: int,
                                batch_size: int = 32,
                                normalize: bool = True) -> np.ndarray:
        """
        Extract embeddings from a specific layer.
        
        Args:
            sentences: List of sentences to encode
            layer: Layer index (0 to num_layers)
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            embeddings: numpy array of shape (n_sentences, embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Get hidden states
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
            
            # Extract the specified layer's hidden states
            hidden_states = outputs.hidden_states[layer]
            
            # Apply pooling
            pooling_input = {
                'token_embeddings': hidden_states,
                'attention_mask': inputs['attention_mask']
            }
            
            with torch.no_grad():
                pooling_output = self.pooling_layer(pooling_input)
            
            batch_embeddings = pooling_output['sentence_embedding']
            
            # Normalize if requested
            if normalize:
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Extracted embeddings from layer {layer}: shape {embeddings.shape}")
        return embeddings
    
    def extract_all_layers(self,
                          sentences: List[str],
                          batch_size: int = 32,
                          normalize: bool = True,
                          save_path: Optional[str] = None) -> Dict[int, np.ndarray]:
        """
        Extract embeddings from all layers.
        
        Args:
            sentences: List of sentences to encode
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize embeddings
            save_path: Optional path to save layer embeddings
            
        Returns:
            layer_embeddings: Dictionary mapping layer index to embeddings
        """
        # Get number of layers
        with torch.no_grad():
            dummy_input = self.tokenizer(
                ["test"],
                return_tensors='pt'
            )
            dummy_input = {key: value.to(self.device) for key, value in dummy_input.items()}
            dummy_output = self.transformer_model(**dummy_input)
            num_layers = len(dummy_output.hidden_states)
        
        layer_embeddings = {}
        
        for layer_idx in range(num_layers):
            logger.info(f"Processing layer {layer_idx}/{num_layers-1}")
            embeddings = self.extract_layer_embeddings(
                sentences,
                layer=layer_idx,
                batch_size=batch_size,
                normalize=normalize
            )
            layer_embeddings[layer_idx] = embeddings
            
            if save_path:
                layer_save_path = save_path.replace('.npz', f'_layer_{layer_idx}.npz')
                self.save_embeddings(embeddings, sentences, layer_save_path)
        
        logger.info(f"Extracted embeddings from all {num_layers} layers")
        return layer_embeddings
    
    def get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        with torch.no_grad():
            dummy_input = self.tokenizer(["test"], return_tensors='pt')
            dummy_input = {key: value.to(self.device) for key, value in dummy_input.items()}
            dummy_output = self.transformer_model(**dummy_input)
            return len(dummy_output.hidden_states)
    
    def __repr__(self) -> str:
        return f"SentenceBERTExtractor(model='{self.model_name}', device='{self.device}', embedding_dim={self.embedding_dim})"