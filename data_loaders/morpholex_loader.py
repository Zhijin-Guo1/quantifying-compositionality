"""MorphoLEX data loader for word morphology experiments."""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class MorphoLEXLoader:
    """Load MorphoLEX dataset for word morphology analysis."""
    
    def __init__(self, excel_path='Downloads/MorphoLEX_en.xlsx'):
        """
        Initialize MorphoLEX loader.
        
        Args:
            excel_path: Path to MorphoLEX Excel file
        """
        self.excel_path = excel_path
        
    def load_morpholex_data(self) -> pd.DataFrame:
        """
        Load and combine MorphoLEX data from Excel sheets.
        
        Returns:
            DataFrame with word morphology information
        """
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"MorphoLEX file not found at {self.excel_path}")
        
        logger.info(f"Loading MorphoLEX data from {self.excel_path}")
        
        # Load relevant sheets (sheets 3, 4, 5 contain morphological data)
        df_sheet3 = pd.read_excel(self.excel_path, sheet_name=2)
        df_sheet4 = pd.read_excel(self.excel_path, sheet_name=3)
        df_sheet5 = pd.read_excel(self.excel_path, sheet_name=4)
        
        # Combine the dataframes
        # Sheet 3: Basic morphological info
        # Sheet 4: Additional attributes
        # Sheet 5: More detailed morphology
        
        # Merge dataframes on common word column
        combined_df = df_sheet3
        
        logger.info(f"Loaded {len(combined_df)} words from MorphoLEX")
        
        return combined_df
    
    def extract_morphological_features(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray, List[str]]:
        """
        Extract morphological features from MorphoLEX dataframe.
        
        Args:
            df: MorphoLEX dataframe
            
        Returns:
            words: List of words
            features: Binary feature matrix
            feature_names: List of feature names
        """
        # Get words
        words = df['Word'].tolist() if 'Word' in df.columns else df.iloc[:, 0].tolist()
        
        # Define morphological features to extract
        feature_columns = []
        
        # Check which columns are available
        available_cols = df.columns.tolist()
        
        # Common morphological features in MorphoLEX
        potential_features = [
            'PRS', 'PST', 'PLUR',  # Tense and number
            'PREFIX', 'SUFFIX',  # Affixes
            'Nmorph',  # Number of morphemes
            'MorphSP',  # Morphological family size
            'MorphPR',  # Morphological family frequency
        ]
        
        # Add available features
        for feat in potential_features:
            if feat in available_cols:
                feature_columns.append(feat)
        
        # Convert to binary features
        features = []
        for col in feature_columns:
            if df[col].dtype == bool or df[col].dtype == np.bool_:
                features.append(df[col].values.astype(int))
            else:
                # Binarize continuous features
                features.append((df[col] > 0).astype(int))
        
        if features:
            feature_matrix = np.column_stack(features)
        else:
            # Fallback: create basic features from word structure
            logger.warning("No morphological columns found, creating basic features")
            feature_matrix = self._create_basic_features(words)
            feature_columns = ['has_ing', 'has_ed', 'has_s', 'has_er', 'has_un', 'has_re']
        
        logger.info(f"Extracted {feature_matrix.shape[1]} morphological features")
        
        return words, feature_matrix, feature_columns
    
    def _create_basic_features(self, words: List[str]) -> np.ndarray:
        """
        Create basic morphological features from word endings.
        
        Args:
            words: List of words
            
        Returns:
            Binary feature matrix
        """
        features = []
        
        for word in words:
            word_lower = word.lower()
            feat = [
                int(word_lower.endswith('ing')),
                int(word_lower.endswith('ed')),
                int(word_lower.endswith('s')),
                int(word_lower.endswith('er')),
                int(word_lower.startswith('un')),
                int(word_lower.startswith('re'))
            ]
            features.append(feat)
        
        return np.array(features)
    
    def prepare_word_data(self, word_list: List[str] = None) -> Dict:
        """
        Prepare word data for compositionality analysis.
        
        Args:
            word_list: Optional list of specific words to use
            
        Returns:
            dict with:
                - words: List of words
                - attributes: Binary attribute matrix
                - feature_names: List of feature names
        """
        try:
            # Load MorphoLEX data
            df = self.load_morpholex_data()
            
            # Filter to specific words if provided
            if word_list:
                df_filtered = df[df['Word'].isin(word_list)] if 'Word' in df.columns else df
                logger.info(f"Filtered to {len(df_filtered)} words from provided list")
            else:
                df_filtered = df
            
            # Extract features
            words, attributes, feature_names = self.extract_morphological_features(df_filtered)
            
            return {
                'words': words,
                'attributes': attributes,
                'feature_names': feature_names,
                'n_words': len(words)
            }
            
        except FileNotFoundError:
            # If MorphoLEX not available, create demo data
            logger.warning("MorphoLEX data not found, creating demo morphological features")
            
            if not word_list:
                # Default demo words with clear morphological patterns
                word_list = [
                    # Base forms
                    "book", "play", "run", "write", "read",
                    # -ing forms
                    "booking", "playing", "running", "writing", "reading",
                    # -ed forms
                    "booked", "played", "written", 
                    # -er forms
                    "booker", "player", "runner", "writer", "reader",
                    # -s forms
                    "books", "plays", "runs", "writes", "reads",
                    # Prefixes
                    "unbook", "replay", "rerun", "rewrite", "reread"
                ]
            
            # Create basic morphological features
            attributes = self._create_basic_features(word_list)
            feature_names = ['has_ing', 'has_ed', 'has_s', 'has_er', 'has_un', 'has_re']
            
            return {
                'words': word_list,
                'attributes': attributes,
                'feature_names': feature_names,
                'n_words': len(word_list)
            }