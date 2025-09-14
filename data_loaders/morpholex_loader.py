"""MorphoLEX data loader for word morphology experiments."""

import pandas as pd
import numpy as np
import os
import logging
import re
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


class MorphoLEXLoader:
    """Load MorphoLEX dataset for word morphology analysis."""
    
    def __init__(self, excel_path='data/MorphoLEX_en.xlsx'):
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
        
        # Return all three sheets for processing
        return df_sheet3, df_sheet4, df_sheet5
    
    def extract_root_and_suffixes(self, entry: str) -> Tuple[str, List[str]]:
        """
        Extract root and suffixes from MorphoLexSegm entry.
        
        Args:
            entry: MorphoLexSegm string
            
        Returns:
            root: Root word
            suffixes: List of suffixes
        """
        # Extract all substrings that are enclosed in parentheses
        roots = re.findall(r'\((.*?)\)', entry)
        
        # The root is the first such substring
        root = roots[0] if roots else ''
        
        # Split the entry into segments by '>'
        segments = entry.split('>')
        
        # The suffixes are all the remaining segments, after removing non-alphabetic characters
        suffixes = [re.sub('[^a-zA-Z]', '', segment) for segment in segments[1:]]
        
        return root, suffixes
    
    def prepare_cca_data(self, word2vec_model=None) -> Dict:
        """
        Prepare data for CCA analysis following notebook preprocessing:
        1. Combine suffixes from all 3 sheets
        2. Remove suffixes occurring less than 10 times
        3. Remove words with any low-occurrence suffix
        4. Filter to words with Word2Vec embeddings (if model provided)
        
        Args:
            word2vec_model: Optional Word2Vec model for filtering
            
        Returns:
            dict with filtered words and suffix features
        """
        # Load all sheets
        df_sheet3, df_sheet4, df_sheet5 = self.load_morpholex_data()
        
        # Process each sheet to extract suffixes
        all_dfs = []
        
        for df, sheet_name in [(df_sheet3, 'sheet3'), (df_sheet4, 'sheet4'), (df_sheet5, 'sheet5')]:
            if 'MorphoLexSegm' not in df.columns:
                logger.warning(f"MorphoLexSegm not found in {sheet_name}, skipping")
                continue
                
            # Extract suffixes for each word
            suffixes_list = []
            words = []
            
            for _, row in df.iterrows():
                if pd.notna(row.get('MorphoLexSegm')) and pd.notna(row.get('Word')):
                    _, suffixes = self.extract_root_and_suffixes(row['MorphoLexSegm'])
                    suffixes_list.append(suffixes)
                    words.append(row['Word'])
            
            # Convert to binary matrix using MultiLabelBinarizer
            if suffixes_list:
                mlb = MultiLabelBinarizer()
                suffix_array = mlb.fit_transform(suffixes_list)
                suffix_df = pd.DataFrame(suffix_array, columns=mlb.classes_)
                
                # Remove empty string column if exists
                if '' in suffix_df.columns:
                    suffix_df = suffix_df.drop(columns=[''])
                
                # Add Word column
                suffix_df.insert(0, 'Word', words)
                all_dfs.append(suffix_df)
        
        if not all_dfs:
            logger.error("No valid suffix data found in MorphoLEX sheets")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.fillna(0)
        
        # Convert to int
        for col in combined_df.columns[1:]:
            combined_df[col] = combined_df[col].astype(int)
        
        logger.info(f"Combined data: {len(combined_df)} words, {len(combined_df.columns)-1} suffixes")
        
        # Remove low-occurrence suffixes (< 10 occurrences)
        suffix_counts = combined_df.iloc[:, 1:].sum()
        low_occurrence_suffixes = suffix_counts[suffix_counts < 10].index
        
        # Remove rows where any low-occurrence suffix occurs
        if len(low_occurrence_suffixes) > 0:
            mask = combined_df[low_occurrence_suffixes].sum(axis=1) > 0
            combined_df = combined_df[~mask]
            combined_df = combined_df.drop(columns=low_occurrence_suffixes)
        
        logger.info(f"After filtering: {len(combined_df)} words, {len(combined_df.columns)-1} suffixes")
        
        # Filter to words with embeddings if model provided
        # This follows the exact approach from the notebook to ensure proper pairing
        if word2vec_model is not None:
            # Build list of words to remove (those NOT in Word2Vec)
            words_to_remove = []
            for word in combined_df['Word']:
                if word not in word2vec_model:
                    words_to_remove.append(word)

            # Remove words not in Word2Vec vocabulary
            if words_to_remove:
                combined_df = combined_df[~combined_df['Word'].isin(words_to_remove)]

            # Reset index to ensure clean ordering
            combined_df = combined_df.reset_index(drop=True)
            logger.info(f"After Word2Vec filtering: {len(combined_df)} words")
        
        # Extract final data - maintain exact order from combined_df
        words = combined_df['Word'].tolist()
        attributes = combined_df.iloc[:, 1:].values.astype(int)
        feature_names = combined_df.columns[1:].tolist()

        # Double-check that we have the right number of words
        logger.info(f"Final data: {len(words)} words with {len(feature_names)} features")

        return {
            'words': words,
            'attributes': attributes,
            'feature_names': feature_names,
            'n_words': len(words),
            'combined_df': combined_df  # Keep for decomposition filtering
        }
    
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
    
    def prepare_decomposition_data(self, combined_df: pd.DataFrame = None) -> Dict:
        """
        Prepare filtered data specifically for Linear Decomposition following notebook logic:
        1. Start from CCA-filtered data
        2. Filter words with exactly 3 suffixes
        3. Remove suffixes that occur less than 10 times (in this subset)
        4. Add root features
        5. Remove roots that occur less than 3 times
        
        Args:
            combined_df: CCA-filtered dataframe (if None, prepares it first)
            
        Returns:
            dict with filtered words and expanded attributes
        """
        if combined_df is None:
            # First prepare CCA data to get the filtered combined_df
            cca_data = self.prepare_cca_data()
            if cca_data is None:
                return None
            combined_df = cca_data['combined_df']
        
        # Get suffix columns (all columns except 'Word')
        suffix_cols = [col for col in combined_df.columns[1:]]
        
        # Step 1: Filter words with exactly 3 suffixes
        suffix_df = combined_df.copy()
        suffix_sums = suffix_df[suffix_cols].sum(axis=1)
        three_suffix_mask = suffix_sums == 3
        filtered_df = suffix_df[three_suffix_mask].copy()
        
        logger.info(f"Step 1: {len(filtered_df)} words with exactly 3 suffixes")
        
        # Step 2: Remove suffixes that occur less than 10 times IN THIS SUBSET
        suffix_counts = filtered_df[suffix_cols].sum()
        low_occurrence_suffixes = suffix_counts[suffix_counts < 10].index.tolist()
        
        if low_occurrence_suffixes:
            # Drop low-occurrence suffix columns
            filtered_df = filtered_df.drop(columns=low_occurrence_suffixes)
        
        # Keep only non-empty columns
        filtered_df = filtered_df.loc[:, (filtered_df != 0).any(axis=0)]
        # Reset index to ensure clean ordering
        filtered_df = filtered_df.reset_index(drop=True)

        logger.info(f"Step 2: {len(filtered_df)} words after removing rare suffix columns")
        
        # Step 3: Extract roots from MorphoLexSegm
        # Load original sheets to get MorphoLexSegm data
        df_sheet3, df_sheet4, df_sheet5 = self.load_morpholex_data()
        
        # Create word->root mapping from all sheets
        word_root_dict = {}
        for df in [df_sheet3, df_sheet4, df_sheet5]:
            if 'MorphoLexSegm' in df.columns and 'Word' in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row.get('MorphoLexSegm')) and pd.notna(row.get('Word')):
                        root, _ = self.extract_root_and_suffixes(row['MorphoLexSegm'])
                        word_root_dict[row['Word']] = root
        
        # Map words to their roots
        roots = []
        for word in filtered_df['Word']:
            root = word_root_dict.get(word, word[:min(4, len(word))])  # Fallback to prefix if not found
            roots.append(root)
        
        filtered_df['Root'] = roots
        
        # Create one-hot encoding for roots
        root_dummies = pd.get_dummies(filtered_df['Root'])
        filtered_df = pd.concat([filtered_df, root_dummies], axis=1)
        
        # Step 4: Remove roots that occur less than 3 times
        root_cols = root_dummies.columns.tolist()
        root_counts = filtered_df[root_cols].sum()
        low_occurrence_roots = root_counts[root_counts < 3].index.tolist()
        
        if low_occurrence_roots:
            # Drop low-occurrence root columns
            filtered_df = filtered_df.drop(columns=low_occurrence_roots)
        
        # Keep only non-empty columns
        filtered_df = filtered_df.loc[:, (filtered_df != 0).any(axis=0)]
        # Reset index to ensure clean ordering
        filtered_df = filtered_df.reset_index(drop=True)

        logger.info(f"Step 4: {len(filtered_df)} words after removing rare root columns")
        
        # Prepare final attributes matrix
        # All columns except 'Word' and 'Root' are features
        feature_cols = [col for col in filtered_df.columns if col not in ['Word', 'Root']]
        
        # Identify which are suffix vs root features
        final_suffix_cols = [col for col in feature_cols if col in suffix_cols]
        final_root_cols = [col for col in feature_cols if col not in suffix_cols]
        
        attributes = filtered_df[feature_cols].values.astype(int)
        words = filtered_df['Word'].tolist()
        
        logger.info(f"Final: {len(words)} words with {len(final_suffix_cols)} suffixes and {len(final_root_cols)} roots")
        
        return {
            'words': words,
            'attributes': attributes,
            'feature_names': feature_cols,
            'n_words': len(words),
            'n_suffix_features': len(final_suffix_cols),
            'n_root_features': len(final_root_cols)
        }