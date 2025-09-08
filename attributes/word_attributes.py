"""
Word morphology attribute extraction from MorphoLex dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Any, Optional, Set
import logging

from .base import BaseAttributeGenerator

logger = logging.getLogger(__name__)


class WordMorphologyAttributes(BaseAttributeGenerator):
    """
    Extract morphological attributes (suffixes) from words using MorphoLex dataset.
    
    Creates a binary matrix where each row is a word and each column is a suffix.
    """
    
    def __init__(self,
                 data_path: Path = Path("data/MorphoLEX_en.xlsx"),
                 cache_dir: Optional[Path] = None,
                 min_suffix_frequency: int = 10,
                 word2vec_vocab_file: Optional[Path] = None):
        """
        Initialize word morphology attribute generator.
        
        Args:
            data_path: Path to MorphoLex Excel file
            cache_dir: Optional cache directory
            min_suffix_frequency: Minimum frequency for suffix to be included
            word2vec_vocab_file: Optional Word2Vec vocabulary for filtering
        """
        super().__init__(data_path, cache_dir)
        self.min_suffix_frequency = min_suffix_frequency
        self.word2vec_vocab_file = word2vec_vocab_file
        self.word2vec_vocab = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load MorphoLex morphological data.
        
        Returns:
            DataFrame with words and their morphological features
        """
        logger.info(f"Loading MorphoLex data from {self.data_path}")
        
        # Load Excel file
        df = pd.read_excel(self.data_path)
        
        # Basic info
        logger.info(f"Loaded {len(df)} words from MorphoLex")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Load Word2Vec vocabulary if specified
        if self.word2vec_vocab_file and self.word2vec_vocab_file.exists():
            logger.info(f"Loading Word2Vec vocabulary from {self.word2vec_vocab_file}")
            with open(self.word2vec_vocab_file, 'r') as f:
                self.word2vec_vocab = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.word2vec_vocab)} words in Word2Vec vocabulary")
        
        return df
    
    def extract_attributes(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract morphological attributes from word data.
        
        Args:
            data: MorphoLex DataFrame
            
        Returns:
            Tuple of (attribute_matrix, entity_ids, attribute_names)
        """
        # Identify suffix columns (binary columns indicating suffix presence)
        # MorphoLex has columns like 'able', 'age', 'al', 'ance', etc. that are binary
        suffix_columns = []
        for col in data.columns:
            if col not in ['Item', 'Nletters', 'Nphon', 'NSyll', 'NMorph', 
                          'MorphSP', 'MorphPR', 'FamSize', 'DerivEntropy',
                          'InflEntropy', 'AffixFreq', 'PercentRootFreq']:
                # Check if it's a binary column
                unique_vals = data[col].dropna().unique()
                if set(unique_vals).issubset({0, 1}):
                    suffix_columns.append(col)
        
        logger.info(f"Found {len(suffix_columns)} suffix columns")
        
        # Filter words if Word2Vec vocabulary is provided
        if self.word2vec_vocab:
            data = data[data['Item'].isin(self.word2vec_vocab)].copy()
            logger.info(f"Filtered to {len(data)} words in Word2Vec vocabulary")
        
        # Filter suffixes by frequency
        suffix_frequencies = {}
        for suffix in suffix_columns:
            freq = data[suffix].sum()
            suffix_frequencies[suffix] = freq
        
        # Keep only suffixes with minimum frequency
        selected_suffixes = [
            suffix for suffix, freq in suffix_frequencies.items()
            if freq >= self.min_suffix_frequency
        ]
        
        logger.info(f"Selected {len(selected_suffixes)} suffixes with frequency >= {self.min_suffix_frequency}")
        
        # Filter words that have at least one selected suffix
        data_filtered = data.copy()
        data_filtered['has_suffix'] = data_filtered[selected_suffixes].sum(axis=1) > 0
        data_filtered = data_filtered[data_filtered['has_suffix']].copy()
        
        logger.info(f"Filtered to {len(data_filtered)} words with at least one suffix")
        
        # Create binary attribute matrix
        attribute_matrix = data_filtered[selected_suffixes].values.astype(np.int8)
        
        # Entity IDs are the words themselves
        entity_ids = data_filtered['Item'].tolist()
        
        # Attribute names are the suffix names
        attribute_names = selected_suffixes
        
        # Store additional metadata
        self.words = entity_ids
        self.suffix_frequencies = {s: suffix_frequencies[s] for s in selected_suffixes}
        
        logger.info(f"Created attribute matrix: {len(entity_ids)} words x {len(attribute_names)} suffixes")
        logger.info(f"Mean suffixes per word: {attribute_matrix.sum(axis=1).mean():.2f}")
        logger.info(f"Matrix sparsity: {1.0 - attribute_matrix.sum() / attribute_matrix.size:.2%}")
        
        return attribute_matrix, entity_ids, attribute_names
    
    def get_words_with_suffix(self, suffix: str) -> List[str]:
        """Get all words containing a specific suffix."""
        if self.attribute_matrix is None or suffix not in self.attribute_names:
            return []
        
        suffix_idx = self.attribute_names.index(suffix)
        word_indices = np.where(self.attribute_matrix[:, suffix_idx] == 1)[0]
        
        return [self.entity_ids[i] for i in word_indices]
    
    def get_suffix_statistics(self) -> pd.DataFrame:
        """
        Get statistics about suffix usage.
        
        Returns:
            DataFrame with suffix frequencies and example words
        """
        if self.attribute_matrix is None:
            raise ValueError("Attributes not generated yet. Call generate() first.")
        
        stats = []
        for i, suffix in enumerate(self.attribute_names):
            words_with_suffix = self.get_words_with_suffix(suffix)
            stats.append({
                'suffix': suffix,
                'count': len(words_with_suffix),
                'frequency': len(words_with_suffix) / len(self.entity_ids),
                'example_words': ', '.join(words_with_suffix[:5])
            })
        
        return pd.DataFrame(stats).sort_values('count', ascending=False)
    
    def get_word_decomposition(self, word: str) -> Optional[List[str]]:
        """
        Get morphological decomposition of a word.
        
        Args:
            word: Word to decompose
            
        Returns:
            List of suffixes present in the word, or None if word not found
        """
        if word not in self.entity_ids:
            return None
        
        word_idx = self.entity_ids.index(word)
        suffixes = [
            self.attribute_names[i]
            for i in range(len(self.attribute_names))
            if self.attribute_matrix[word_idx, i] == 1
        ]
        
        return suffixes


class WordMorphologyAdditive(WordMorphologyAttributes):
    """
    Extended word morphology attributes for additive compositionality experiments.
    
    This version creates attributes for both root words and suffixes,
    suitable for testing additive decomposition (root + suffixes = word).
    """
    
    def __init__(self,
                 data_path: Path = Path("data/MorphoLEX_en.xlsx"),
                 cache_dir: Optional[Path] = None,
                 n_suffixes_required: int = 3,
                 min_suffix_frequency: int = 10):
        """
        Initialize additive morphology attribute generator.
        
        Args:
            data_path: Path to MorphoLex Excel file
            cache_dir: Optional cache directory
            n_suffixes_required: Exact number of suffixes required per word
            min_suffix_frequency: Minimum frequency for suffix inclusion
        """
        super().__init__(data_path, cache_dir, min_suffix_frequency)
        self.n_suffixes_required = n_suffixes_required
        
    def extract_attributes(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract root + suffix attributes for additive experiments.
        
        Returns words with exactly n_suffixes_required suffixes,
        and creates attributes for both root words and their suffixes.
        """
        # First get suffix columns as in parent class
        suffix_columns = []
        for col in data.columns:
            if col not in ['Item', 'Nletters', 'Nphon', 'NSyll', 'NMorph',
                          'MorphSP', 'MorphPR', 'FamSize', 'DerivEntropy',
                          'InflEntropy', 'AffixFreq', 'PercentRootFreq']:
                unique_vals = data[col].dropna().unique()
                if set(unique_vals).issubset({0, 1}):
                    suffix_columns.append(col)
        
        # Filter suffixes by frequency
        suffix_frequencies = {col: data[col].sum() for col in suffix_columns}
        selected_suffixes = [
            s for s, freq in suffix_frequencies.items()
            if freq >= self.min_suffix_frequency
        ]
        
        # Filter words with exactly n_suffixes_required suffixes
        data_filtered = data.copy()
        data_filtered['n_suffixes'] = data_filtered[selected_suffixes].sum(axis=1)
        data_filtered = data_filtered[
            data_filtered['n_suffixes'] == self.n_suffixes_required
        ].copy()
        
        logger.info(f"Found {len(data_filtered)} words with exactly {self.n_suffixes_required} suffixes")
        
        # Extract root words (approximation: remove common suffixes)
        # This is simplified - real root extraction would be more complex
        def extract_root(word: str, suffixes: List[str]) -> str:
            """Simple heuristic to extract root."""
            root = word.lower()
            # Try to identify and remove suffix patterns
            for suffix in ['ness', 'less', 'ly', 'er', 'est', 'ing', 'ed', 
                          'able', 'ible', 'ful', 'ous', 'ive', 'ize', 'ise']:
                if root.endswith(suffix) and len(root) > len(suffix) + 2:
                    root = root[:-len(suffix)]
                    break
            return root
        
        # Get unique roots
        roots = set()
        word_to_root = {}
        for word in data_filtered['Item']:
            active_suffixes = [
                s for s in selected_suffixes
                if data_filtered[data_filtered['Item'] == word][s].iloc[0] == 1
            ]
            root = extract_root(word, active_suffixes)
            roots.add(root)
            word_to_root[word] = root
        
        # Create combined attribute matrix (roots + suffixes)
        roots_list = sorted(list(roots))
        root_to_idx = {r: i for i, r in enumerate(roots_list)}
        
        n_words = len(data_filtered)
        n_roots = len(roots_list)
        n_suffixes = len(selected_suffixes)
        
        # Combined attribute matrix: [root features | suffix features]
        attribute_matrix = np.zeros((n_words, n_roots + n_suffixes), dtype=np.int8)
        
        entity_ids = []
        for i, (_, row) in enumerate(data_filtered.iterrows()):
            word = row['Item']
            entity_ids.append(word)
            
            # Set root feature
            root = word_to_root[word]
            if root in root_to_idx:
                attribute_matrix[i, root_to_idx[root]] = 1
            
            # Set suffix features
            for j, suffix in enumerate(selected_suffixes):
                if row[suffix] == 1:
                    attribute_matrix[i, n_roots + j] = 1
        
        # Attribute names
        attribute_names = [f"root:{r}" for r in roots_list] + [f"suffix:{s}" for s in selected_suffixes]
        
        logger.info(f"Created additive attribute matrix: {n_words} words x ({n_roots} roots + {n_suffixes} suffixes)")
        
        return attribute_matrix, entity_ids, attribute_names