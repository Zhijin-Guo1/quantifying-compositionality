"""Sentence data loader for pre-prepared dialogue data (matching notebook)."""

import pandas as pd
import numpy as np
import os
import ast
import logging

logger = logging.getLogger(__name__)


class SentenceDataLoader:
    """Load pre-prepared sentence data matching the notebook format."""

    def __init__(self, data_dir='data/sentence'):
        """
        Initialize sentence data loader.

        Args:
            data_dir: Directory containing user_texts.txt and dialogue_data.csv
        """
        self.data_dir = data_dir
        self.user_texts_path = os.path.join(data_dir, 'user_texts.txt')
        self.dialogue_csv_path = os.path.join(data_dir, 'dialogue_data.csv')

    def fix_format(self, matrix_str):
        """
        Fix format of binary matrix string (add commas between numbers).
        Matches notebook preprocessing.
        """
        return '[' + ','.join(matrix_str.strip('[]').split()) + ']'

    def load_data(self):
        """
        Load sentence data matching notebook approach.

        Returns:
            dict with sentences, attributes, and metadata
        """
        # Check if files exist
        if not os.path.exists(self.user_texts_path):
            raise FileNotFoundError(f"user_texts.txt not found at {self.user_texts_path}")
        if not os.path.exists(self.dialogue_csv_path):
            raise FileNotFoundError(f"dialogue_data.csv not found at {self.dialogue_csv_path}")

        # Load sentences from text file
        logger.info(f"Loading sentences from {self.user_texts_path}")
        with open(self.user_texts_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(sentences)} sentences")

        # Load dialogue data CSV
        logger.info(f"Loading dialogue data from {self.dialogue_csv_path}")
        df = pd.read_csv(self.dialogue_csv_path)

        # Extract binary slot matrix (attributes)
        # This matches the notebook's approach exactly
        attributes = np.array([
            ast.literal_eval(self.fix_format(matrix))
            for matrix in df['binary_slot_matrix']
        ])

        logger.info(f"Loaded attributes with shape: {attributes.shape}")

        # Verify sentence count matches
        if len(sentences) != len(attributes):
            logger.warning(f"Sentence count ({len(sentences)}) != attribute count ({len(attributes)})")
            # Use minimum length
            min_len = min(len(sentences), len(attributes))
            sentences = sentences[:min_len]
            attributes = attributes[:min_len]
            logger.info(f"Truncated to {min_len} samples")

        # Extract slots information
        slots_present = df['slots_present'].apply(ast.literal_eval).tolist()

        # Get unique slot combinations (for grouping)
        df['slots_combination'] = df['slots_present'].apply(
            lambda x: tuple(sorted(ast.literal_eval(x)))
        )
        unique_combinations = df['slots_combination'].nunique()

        logger.info(f"Number of unique slot combinations: {unique_combinations}")

        # Create feature names (slot names)
        # Infer from the first row's slots_present
        all_slots = set()
        for slots in slots_present:
            all_slots.update(slots)
        feature_names = sorted(list(all_slots))

        # If we can't determine feature names, use generic ones
        if not feature_names:
            feature_names = [f"slot_{i}" for i in range(attributes.shape[1])]

        logger.info(f"Feature names (slots): {len(feature_names)} slots")

        return {
            'sentences': sentences,
            'attributes': attributes,
            'feature_names': feature_names,
            'slots_present': slots_present,
            'dataframe': df,
            'n_unique_combinations': unique_combinations
        }

    def prepare_grouped_data(self, sentences, attributes, embeddings=None):
        """
        Group data by unique attribute combinations (for linear decomposition).
        Matches notebook's grouping approach.

        Args:
            sentences: List of sentences
            attributes: Binary attribute matrix
            embeddings: Optional embedding matrix to group

        Returns:
            dict with grouped data
        """
        import pandas as pd

        # Convert attributes to tuples for grouping
        attribute_tuples = [tuple(row) for row in attributes]

        # Create DataFrame for grouping
        df = pd.DataFrame({
            'group_id': attribute_tuples,
            'sentence': sentences
        })

        if embeddings is not None:
            df['embedding'] = list(embeddings)

        # Group by unique attribute combinations
        if embeddings is not None:
            # Compute mean embedding for each group
            grouped = df.groupby('group_id').agg({
                'sentence': 'first',  # Take first sentence as representative
                'embedding': lambda x: np.mean(np.vstack(x), axis=0)
            }).reset_index()

            # Extract results
            group_attributes = np.array(grouped['group_id'].tolist())
            group_embeddings = np.vstack(grouped['embedding'].values)
            group_sentences = grouped['sentence'].tolist()

            logger.info(f"Grouped to {len(group_attributes)} unique combinations")
            logger.info(f"Group attributes shape: {group_attributes.shape}")
            logger.info(f"Group embeddings shape: {group_embeddings.shape}")

            return {
                'group_attributes': group_attributes,
                'group_embeddings': group_embeddings,
                'group_sentences': group_sentences,
                'n_groups': len(group_attributes)
            }
        else:
            # Just group attributes
            grouped = df.groupby('group_id').agg({
                'sentence': 'first'
            }).reset_index()

            group_attributes = np.array(grouped['group_id'].tolist())
            group_sentences = grouped['sentence'].tolist()

            logger.info(f"Grouped to {len(group_attributes)} unique combinations")

            return {
                'group_attributes': group_attributes,
                'group_sentences': group_sentences,
                'n_groups': len(group_attributes)
            }