"""Word2Vec embedding extractor."""

import numpy as np
from typing import List, Optional, Union, Dict
import logging
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from .base import EmbeddingExtractor

logger = logging.getLogger(__name__)


class Word2VecExtractor(EmbeddingExtractor):
    """Extract word embeddings using Word2Vec models."""
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 pretrained_model: str = 'word2vec-google-news-300',
                 embedding_dim: int = 300,
                 use_pretrained: bool = True):
        """
        Initialize the Word2Vec extractor.
        
        Args:
            model_path: Path to a trained Word2Vec model (if using custom model)
            pretrained_model: Name of pretrained model from gensim-data
            embedding_dim: Dimension of word embeddings
            use_pretrained: Whether to use pretrained model or train new one
        """
        super().__init__(embedding_dim=embedding_dim)
        
        self.model_path = model_path
        self.pretrained_model = pretrained_model
        self.use_pretrained = use_pretrained
        self.model = None
        self.keyed_vectors = None
        
        if model_path:
            # Load custom model
            self.load_model(model_path)
        elif use_pretrained:
            # Load pretrained model
            self.load_pretrained_model(pretrained_model)
    
    def load_model(self, model_path: str):
        """Load a Word2Vec model from file."""
        try:
            # Try loading as full model
            self.model = Word2Vec.load(model_path)
            self.keyed_vectors = self.model.wv
            self.embedding_dim = self.model.wv.vector_size
            logger.info(f"Loaded Word2Vec model from {model_path}")
        except:
            # Try loading as KeyedVectors
            self.keyed_vectors = KeyedVectors.load(model_path)
            self.embedding_dim = self.keyed_vectors.vector_size
            logger.info(f"Loaded Word2Vec KeyedVectors from {model_path}")
    
    def load_pretrained_model(self, model_name: str):
        """Load a pretrained Word2Vec model from gensim-data."""
        logger.info(f"Loading pretrained model: {model_name}")
        logger.info("This may take a while for first-time download...")
        
        try:
            self.keyed_vectors = api.load(model_name)
            self.embedding_dim = self.keyed_vectors.vector_size
            logger.info(f"Loaded pretrained model: {model_name}")
            logger.info(f"Vocabulary size: {len(self.keyed_vectors)}")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            logger.info("Available models:")
            for model_info in api.info()['models'].keys():
                if 'word2vec' in model_info.lower() or 'glove' in model_info.lower():
                    logger.info(f"  - {model_info}")
            raise
    
    def train_model(self, 
                   sentences: List[List[str]],
                   embedding_dim: int = 100,
                   window: int = 5,
                   min_count: int = 5,
                   workers: int = 4,
                   epochs: int = 10,
                   sg: int = 1):
        """
        Train a new Word2Vec model.
        
        Args:
            sentences: List of tokenized sentences (list of word lists)
            embedding_dim: Dimension of word embeddings
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            epochs: Number of training epochs
            sg: Training algorithm (1 for skip-gram, 0 for CBOW)
        """
        logger.info("Training new Word2Vec model...")
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=embedding_dim,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=sg
        )
        
        self.keyed_vectors = self.model.wv
        self.embedding_dim = embedding_dim
        
        logger.info(f"Trained Word2Vec model")
        logger.info(f"Vocabulary size: {len(self.keyed_vectors)}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def extract(self,
                words: List[str],
                use_zero_for_oov: bool = True,
                normalize: bool = False) -> np.ndarray:
        """
        Extract word embeddings.
        
        Args:
            words: List of words to get embeddings for
            use_zero_for_oov: Use zero vector for out-of-vocabulary words
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            embeddings: numpy array of shape (n_words, embedding_dim)
        """
        if self.keyed_vectors is None:
            raise ValueError("No Word2Vec model loaded. Load a model first.")
        
        embeddings = []
        oov_count = 0
        
        for word in words:
            if word in self.keyed_vectors:
                embedding = self.keyed_vectors[word]
            else:
                oov_count += 1
                if use_zero_for_oov:
                    embedding = np.zeros(self.embedding_dim)
                else:
                    # Use random vector for OOV
                    embedding = np.random.normal(0, 0.1, self.embedding_dim)
            
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        if oov_count > 0:
            logger.warning(f"Found {oov_count}/{len(words)} out-of-vocabulary words")
        
        if normalize:
            embeddings = self.normalize(embeddings, norm_type='l2')
        
        logger.info(f"Extracted embeddings: shape {embeddings.shape}")
        return embeddings
    
    def extract_sentence_embeddings(self,
                                   sentences: List[str],
                                   aggregation: str = 'mean',
                                   normalize: bool = False) -> np.ndarray:
        """
        Extract sentence embeddings by aggregating word embeddings.
        
        Args:
            sentences: List of sentences
            aggregation: How to aggregate word embeddings ('mean', 'sum', 'max')
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            embeddings: numpy array of shape (n_sentences, embedding_dim)
        """
        if self.keyed_vectors is None:
            raise ValueError("No Word2Vec model loaded. Load a model first.")
        
        sentence_embeddings = []
        
        for sentence in sentences:
            # Tokenize sentence (simple split, could use better tokenization)
            words = sentence.lower().split()
            
            # Get word embeddings
            word_embeddings = []
            for word in words:
                if word in self.keyed_vectors:
                    word_embeddings.append(self.keyed_vectors[word])
            
            if len(word_embeddings) == 0:
                # No known words in sentence
                embedding = np.zeros(self.embedding_dim)
            else:
                word_embeddings = np.array(word_embeddings)
                
                if aggregation == 'mean':
                    embedding = np.mean(word_embeddings, axis=0)
                elif aggregation == 'sum':
                    embedding = np.sum(word_embeddings, axis=0)
                elif aggregation == 'max':
                    embedding = np.max(word_embeddings, axis=0)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            sentence_embeddings.append(embedding)
        
        sentence_embeddings = np.array(sentence_embeddings)
        
        if normalize:
            sentence_embeddings = self.normalize(sentence_embeddings, norm_type='l2')
        
        logger.info(f"Extracted sentence embeddings: shape {sentence_embeddings.shape}")
        return sentence_embeddings
    
    def get_most_similar(self, word: str, topn: int = 10) -> List[tuple]:
        """Get most similar words to a given word."""
        if self.keyed_vectors is None:
            raise ValueError("No Word2Vec model loaded.")
        
        if word not in self.keyed_vectors:
            logger.warning(f"Word '{word}' not in vocabulary")
            return []
        
        return self.keyed_vectors.most_similar(word, topn=topn)
    
    def save_model(self, save_path: str):
        """Save the Word2Vec model."""
        if self.model:
            self.model.save(save_path)
            logger.info(f"Saved Word2Vec model to {save_path}")
        elif self.keyed_vectors:
            self.keyed_vectors.save(save_path)
            logger.info(f"Saved Word2Vec KeyedVectors to {save_path}")
        else:
            logger.warning("No model to save")
    
    def __repr__(self) -> str:
        model_info = "pretrained" if self.use_pretrained else "custom"
        return f"Word2VecExtractor(type='{model_info}', embedding_dim={self.embedding_dim})"