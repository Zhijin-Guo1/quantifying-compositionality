"""
Sentence concept attribute extraction for Schema-Guided Dialogue dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Any, Optional
import json
import logging

from .base import BaseAttributeGenerator

logger = logging.getLogger(__name__)


class SentenceConceptAttributes(BaseAttributeGenerator):
    """
    Extract concept attributes from sentences in Schema-Guided Dialogue dataset.
    
    Each sentence is annotated with semantic concepts (e.g., location, genre, time).
    We create a binary matrix where each row is a sentence and each column is a concept.
    """
    
    def __init__(self, 
                 data_path: Path = Path("data/output"),
                 cache_dir: Optional[Path] = None,
                 min_concepts: int = 3,
                 max_concepts: int = 4):
        """
        Initialize sentence concept attribute generator.
        
        Args:
            data_path: Path to dialogue data directory
            cache_dir: Optional cache directory
            min_concepts: Minimum concepts per sentence to include
            max_concepts: Maximum concepts per sentence to include
        """
        super().__init__(data_path, cache_dir)
        self.min_concepts = min_concepts
        self.max_concepts = max_concepts
        
    def load_data(self) -> pd.DataFrame:
        """
        Load dialogue data with concept annotations.
        
        Returns:
            DataFrame with sentences and their concept labels
        """
        # Try to load pre-processed CSV if available
        csv_path = self.data_path / "dialogue_data.csv"
        if csv_path.exists():
            logger.info(f"Loading processed data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Parse concept lists if stored as strings
            if 'concepts' in df.columns and isinstance(df['concepts'].iloc[0], str):
                df['concepts'] = df['concepts'].apply(eval)
            
            return df
        
        # Otherwise, process from raw Schema-Guided Dialogue files
        logger.info("Processing raw Schema-Guided Dialogue data...")
        sentences = []
        concepts = []
        
        # Look for test split annotation files
        for json_file in self.data_path.glob("test/*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Extract sentences and concepts from dialogues
            for dialogue in data.get('dialogues', []):
                for turn in dialogue.get('turns', []):
                    utterance = turn.get('utterance', '')
                    frames = turn.get('frames', [])
                    
                    # Extract concepts from frames
                    turn_concepts = set()
                    for frame in frames:
                        # Add slot names as concepts
                        for slot in frame.get('slots', []):
                            slot_name = slot.get('slot', '')
                            if slot_name:
                                turn_concepts.add(slot_name)
                        
                        # Add intents as concepts
                        intent = frame.get('intent', '')
                        if intent:
                            turn_concepts.add(intent)
                    
                    if turn_concepts:
                        sentences.append(utterance)
                        concepts.append(list(turn_concepts))
        
        if not sentences:
            raise ValueError(f"No dialogue data found in {self.data_path}")
        
        df = pd.DataFrame({
            'sentence': sentences,
            'concepts': concepts
        })
        
        # Save for future use
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved processed data to {csv_path}")
        
        return df
    
    def extract_attributes(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract concept attributes from sentence data.
        
        Args:
            data: DataFrame with sentences and concepts
            
        Returns:
            Tuple of (attribute_matrix, entity_ids, attribute_names)
        """
        # Filter sentences by concept count
        data = data.copy()
        data['n_concepts'] = data['concepts'].apply(len)
        data = data[
            (data['n_concepts'] >= self.min_concepts) & 
            (data['n_concepts'] <= self.max_concepts)
        ].reset_index(drop=True)
        
        logger.info(f"Filtered to {len(data)} sentences with {self.min_concepts}-{self.max_concepts} concepts")
        
        # Get all unique concepts
        all_concepts = set()
        for concept_list in data['concepts']:
            all_concepts.update(concept_list)
        
        concept_names = sorted(list(all_concepts))
        concept_to_idx = {c: i for i, c in enumerate(concept_names)}
        
        # Create binary matrix
        n_sentences = len(data)
        n_concepts = len(concept_names)
        attribute_matrix = np.zeros((n_sentences, n_concepts), dtype=np.int8)
        
        for i, concept_list in enumerate(data['concepts']):
            for concept in concept_list:
                if concept in concept_to_idx:
                    attribute_matrix[i, concept_to_idx[concept]] = 1
        
        # Use sentences as entity IDs (truncated for readability)
        entity_ids = [f"sent_{i:04d}" for i in range(n_sentences)]
        
        # Store full sentences for reference
        self.sentences = data['sentence'].tolist()
        self.concept_lists = data['concepts'].tolist()
        
        logger.info(f"Created attribute matrix: {n_sentences} sentences x {n_concepts} concepts")
        logger.info(f"Mean concepts per sentence: {attribute_matrix.sum(axis=1).mean():.2f}")
        
        return attribute_matrix, entity_ids, concept_names
    
    def get_sentence_by_id(self, entity_id: str) -> Optional[str]:
        """Get sentence text by entity ID."""
        if not hasattr(self, 'sentences'):
            return None
        
        if entity_id.startswith('sent_'):
            idx = int(entity_id.split('_')[1])
            if 0 <= idx < len(self.sentences):
                return self.sentences[idx]
        return None
    
    def get_concepts_by_id(self, entity_id: str) -> Optional[List[str]]:
        """Get concept list by entity ID."""
        if not hasattr(self, 'concept_lists'):
            return None
        
        if entity_id.startswith('sent_'):
            idx = int(entity_id.split('_')[1])
            if 0 <= idx < len(self.concept_lists):
                return self.concept_lists[idx]
        return None
    
    def get_concept_combinations(self) -> pd.DataFrame:
        """
        Get statistics about concept combinations.
        
        Returns:
            DataFrame with concept combination frequencies
        """
        if self.attribute_matrix is None:
            raise ValueError("Attributes not generated yet. Call generate() first.")
        
        # Convert each row to a tuple of active concepts
        combinations = []
        for i in range(len(self.entity_ids)):
            active_concepts = [
                self.attribute_names[j] 
                for j in range(len(self.attribute_names))
                if self.attribute_matrix[i, j] == 1
            ]
            combinations.append(tuple(sorted(active_concepts)))
        
        # Count combinations
        from collections import Counter
        combo_counts = Counter(combinations)
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'combination': ' + '.join(combo),
                'count': count,
                'frequency': count / len(combinations)
            }
            for combo, count in combo_counts.items()
        ])
        
        return df.sort_values('count', ascending=False)