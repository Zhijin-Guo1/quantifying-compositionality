"""Schema-Guided Dialogue data loader for sentence experiments."""

import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import logging

logger = logging.getLogger(__name__)


class DialogueLoader:
    """Load Schema-Guided Dialogue dataset for sentence compositionality analysis."""
    
    def __init__(self, data_dir='train', min_slots=3):
        """
        Initialize dialogue loader.
        
        Args:
            data_dir: Directory containing dialogue JSON files
            min_slots: Minimum number of slots required per sentence
        """
        self.data_dir = data_dir
        self.min_slots = min_slots
        self.schema_file = os.path.join(data_dir, 'schema.json')
        
    def load_schema(self):
        """Load schema to get all possible slot names."""
        if not os.path.exists(self.schema_file):
            raise FileNotFoundError(f"Schema file not found at {self.schema_file}")
        
        with open(self.schema_file, 'r') as f:
            schema_data = json.load(f)
        
        # Extract all unique slot names
        all_slots = sorted({
            slot['name'] for service in schema_data 
            for slot in service['slots']
        })
        
        logger.info(f"Loaded schema with {len(all_slots)} unique slots")
        return all_slots
    
    def load_dialogues(self):
        """
        Load and process dialogue files.
        
        Returns:
            dialogues_data: List of processed dialogue turns
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dialogue directory not found at {self.data_dir}")
        
        # Get all dialogue files
        files = [
            f for f in os.listdir(self.data_dir)
            if f.startswith('dialogues_') and f.endswith('.json')
        ]
        files.sort()
        
        if not files:
            raise FileNotFoundError(f"No dialogue files found in {self.data_dir}")
        
        logger.info(f"Found {len(files)} dialogue files")
        
        all_dialogues = []
        total_turns = 0
        filtered_turns = 0
        
        for filename in files:
            file_path = os.path.join(self.data_dir, filename)
            
            with open(file_path, 'r') as f:
                dialogues = json.load(f)
            
            # Process each dialogue
            for dialogue in dialogues:
                dialogue_id = dialogue['dialogue_id']
                
                for turn in dialogue['turns']:
                    # Only process user turns
                    if turn['speaker'] == 'USER':
                        total_turns += 1
                        user_text = turn['utterance']
                        
                        # Collect slots used in this turn
                        slots_present = [
                            slot['slot'] 
                            for frame in turn.get("frames", [])
                            for slot in frame.get("slots", [])
                        ]
                        
                        # Filter by minimum slots
                        if len(slots_present) >= self.min_slots:
                            filtered_turns += 1
                            all_dialogues.append({
                                'dialogue_id': dialogue_id,
                                'turn_id': turn.get('turn_id', None),
                                'user_text': user_text,
                                'slots_present': slots_present
                            })
        
        logger.info(f"Processed {total_turns} total turns")
        logger.info(f"Kept {filtered_turns} turns with >= {self.min_slots} slots")
        
        return all_dialogues
    
    def create_slot_attributes(self, dialogues_data, all_slots=None):
        """
        Create binary slot attribute matrix.
        
        Args:
            dialogues_data: List of dialogue turns
            all_slots: List of all possible slots (if None, derive from data)
            
        Returns:
            attributes: Binary attribute matrix (n_sentences, n_slots)
            feature_names: List of slot names
            sentences: List of sentence texts
        """
        if all_slots is None:
            # Get slots that actually appear in the data
            slot_counts = {}
            for item in dialogues_data:
                for slot in item['slots_present']:
                    slot_counts[slot] = slot_counts.get(slot, 0) + 1
            
            # Filter out slots that never appear
            all_slots = sorted([slot for slot, count in slot_counts.items() if count > 0])
            logger.info(f"Found {len(all_slots)} active slots in filtered data")
        
        # Initialize MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=all_slots)
        mlb.fit([all_slots])
        
        # Create binary matrix
        slots_lists = [item['slots_present'] for item in dialogues_data]
        attributes = mlb.transform(slots_lists)
        
        # Extract sentences
        sentences = [item['user_text'] for item in dialogues_data]
        
        logger.info(f"Created attribute matrix: shape {attributes.shape}")
        
        return attributes, all_slots, sentences
    
    def prepare_dialogue_data(self):
        """
        Prepare complete dialogue data for compositionality analysis.
        
        Returns:
            dict with:
                - sentences: List of sentence texts
                - attributes: Binary slot attribute matrix
                - feature_names: List of slot names
                - dialogue_df: DataFrame with all dialogue info
        """
        # Load schema
        all_slots = self.load_schema()
        
        # Load dialogues
        dialogues_data = self.load_dialogues()
        
        # Create attributes
        attributes, feature_names, sentences = self.create_slot_attributes(
            dialogues_data, all_slots
        )
        
        # Create DataFrame for reference
        dialogue_df = pd.DataFrame(dialogues_data)
        dialogue_df['slots_combination'] = dialogue_df['slots_present'].apply(
            lambda x: tuple(sorted(x))
        )
        
        # Count unique combinations
        unique_combinations = dialogue_df['slots_combination'].nunique()
        logger.info(f"Found {unique_combinations} unique slot combinations")
        
        return {
            'sentences': sentences,
            'attributes': attributes,
            'feature_names': feature_names,
            'dialogue_df': dialogue_df,
            'n_sentences': len(sentences),
            'n_unique_patterns': unique_combinations
        }
    
    def save_processed_data(self, output_dir='output'):
        """Save processed dialogue data to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        data = self.prepare_dialogue_data()
        
        # Save sentences
        sentences_file = os.path.join(output_dir, 'user_texts.txt')
        with open(sentences_file, 'w') as f:
            for sentence in data['sentences']:
                f.write(sentence + '\n')
        logger.info(f"Saved {len(data['sentences'])} sentences to {sentences_file}")
        
        # Save DataFrame
        csv_file = os.path.join(output_dir, 'dialogue_data.csv')
        data['dialogue_df'].to_csv(csv_file, index=False)
        logger.info(f"Saved dialogue data to {csv_file}")
        
        # Save attributes
        npz_file = os.path.join(output_dir, 'dialogue_attributes.npz')
        np.savez_compressed(
            npz_file,
            attributes=data['attributes'],
            feature_names=data['feature_names']
        )
        logger.info(f"Saved attributes to {npz_file}")
        
        return data