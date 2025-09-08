"""
Knowledge Graph (MovieLens) demographic attribute extraction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Any, Optional, Dict
import logging

from .base import BaseAttributeGenerator

logger = logging.getLogger(__name__)


class KGDemographicAttributes(BaseAttributeGenerator):
    """
    Extract demographic attributes from MovieLens users.
    
    Creates a binary matrix encoding gender, age group, and occupation.
    """
    
    # Age group boundaries
    AGE_GROUPS = {
        'Under18': (0, 18),
        '18-24': (18, 25),
        '25-34': (25, 35),
        '35-44': (35, 45),
        '45-49': (45, 50),
        '50-55': (50, 56),
        '56+': (56, 200)
    }
    
    # Occupation mapping from MovieLens
    OCCUPATIONS = {
        0: "other",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer"
    }
    
    def __init__(self,
                 data_path: Path = Path("data/ml-1m"),
                 cache_dir: Optional[Path] = None,
                 include_occupation: bool = True):
        """
        Initialize KG demographic attribute generator.
        
        Args:
            data_path: Path to MovieLens dataset directory
            cache_dir: Optional cache directory
            include_occupation: Whether to include occupation attributes
        """
        super().__init__(data_path, cache_dir)
        self.include_occupation = include_occupation
        
    def load_data(self) -> pd.DataFrame:
        """
        Load MovieLens user data.
        
        Returns:
            DataFrame with user demographics
        """
        users_file = self.data_path / "users.dat"
        
        if not users_file.exists():
            raise FileNotFoundError(f"Users file not found: {users_file}")
        
        logger.info(f"Loading MovieLens users from {users_file}")
        
        # Load users data
        # Format: UserID::Gender::Age::Occupation::Zip-code
        users_df = pd.read_csv(
            users_file,
            sep='::',
            header=None,
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            engine='python'
        )
        
        logger.info(f"Loaded {len(users_df)} users")
        logger.info(f"Gender distribution: {users_df['gender'].value_counts().to_dict()}")
        logger.info(f"Age range: {users_df['age'].min()}-{users_df['age'].max()}")
        logger.info(f"Unique occupations: {users_df['occupation'].nunique()}")
        
        return users_df
    
    def extract_attributes(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Extract demographic attributes from user data.
        
        Args:
            data: MovieLens users DataFrame
            
        Returns:
            Tuple of (attribute_matrix, entity_ids, attribute_names)
        """
        n_users = len(data)
        attributes = []
        attribute_names = []
        
        # Gender attributes (2 binary features)
        gender_features = np.zeros((n_users, 2), dtype=np.int8)
        gender_features[data['gender'] == 'M', 0] = 1  # Male
        gender_features[data['gender'] == 'F', 1] = 1  # Female
        attributes.append(gender_features)
        attribute_names.extend(['gender:M', 'gender:F'])
        
        # Age group attributes (7 binary features)
        age_features = np.zeros((n_users, len(self.AGE_GROUPS)), dtype=np.int8)
        for i, (age_group, (min_age, max_age)) in enumerate(self.AGE_GROUPS.items()):
            mask = (data['age'] >= min_age) & (data['age'] < max_age)
            age_features[mask, i] = 1
        attributes.append(age_features)
        attribute_names.extend([f'age:{group}' for group in self.AGE_GROUPS.keys()])
        
        # Occupation attributes (21 binary features) - optional
        if self.include_occupation:
            occupation_features = np.zeros((n_users, len(self.OCCUPATIONS)), dtype=np.int8)
            for occ_id, occ_name in self.OCCUPATIONS.items():
                mask = data['occupation'] == occ_id
                occupation_features[mask, occ_id] = 1
            attributes.append(occupation_features)
            attribute_names.extend([f'occ:{name}' for name in self.OCCUPATIONS.values()])
        
        # Combine all attributes
        attribute_matrix = np.hstack(attributes)
        
        # Entity IDs are user IDs
        entity_ids = [f"user_{uid}" for uid in data['user_id']]
        
        # Store original data for reference
        self.users_df = data.copy()
        
        logger.info(f"Created attribute matrix: {n_users} users x {len(attribute_names)} attributes")
        logger.info(f"Attributes: {2} gender + {len(self.AGE_GROUPS)} age groups" + 
                   (f" + {len(self.OCCUPATIONS)} occupations" if self.include_occupation else ""))
        
        return attribute_matrix, entity_ids, attribute_names
    
    def get_user_demographics(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get demographic information for a specific user.
        
        Args:
            entity_id: User entity ID (e.g., 'user_1')
            
        Returns:
            Dictionary with demographic information
        """
        if not hasattr(self, 'users_df'):
            return None
        
        if entity_id.startswith('user_'):
            user_id = int(entity_id.split('_')[1])
            user_data = self.users_df[self.users_df['user_id'] == user_id]
            
            if len(user_data) == 1:
                row = user_data.iloc[0]
                
                # Find age group
                age_group = None
                for group, (min_age, max_age) in self.AGE_GROUPS.items():
                    if min_age <= row['age'] < max_age:
                        age_group = group
                        break
                
                return {
                    'user_id': user_id,
                    'gender': row['gender'],
                    'age': row['age'],
                    'age_group': age_group,
                    'occupation': self.OCCUPATIONS.get(row['occupation'], 'unknown')
                }
        
        return None
    
    def get_demographic_statistics(self) -> pd.DataFrame:
        """
        Get statistics about demographic distributions.
        
        Returns:
            DataFrame with demographic statistics
        """
        if not hasattr(self, 'users_df'):
            raise ValueError("Data not loaded yet. Call generate() first.")
        
        stats = []
        
        # Gender statistics
        for gender in ['M', 'F']:
            count = (self.users_df['gender'] == gender).sum()
            stats.append({
                'category': 'Gender',
                'value': 'Male' if gender == 'M' else 'Female',
                'count': count,
                'percentage': count / len(self.users_df) * 100
            })
        
        # Age group statistics
        for age_group, (min_age, max_age) in self.AGE_GROUPS.items():
            mask = (self.users_df['age'] >= min_age) & (self.users_df['age'] < max_age)
            count = mask.sum()
            stats.append({
                'category': 'Age Group',
                'value': age_group,
                'count': count,
                'percentage': count / len(self.users_df) * 100
            })
        
        # Occupation statistics
        if self.include_occupation:
            for occ_id, occ_name in self.OCCUPATIONS.items():
                count = (self.users_df['occupation'] == occ_id).sum()
                if count > 0:  # Only include occupations that exist
                    stats.append({
                        'category': 'Occupation',
                        'value': occ_name,
                        'count': count,
                        'percentage': count / len(self.users_df) * 100
                    })
        
        return pd.DataFrame(stats)


class KGDemographicAttributesForCCA(KGDemographicAttributes):
    """
    Simplified demographic attributes for CCA experiments.
    
    Uses only gender and age for cleaner correlation analysis.
    """
    
    def __init__(self,
                 data_path: Path = Path("data/ml-1m"),
                 cache_dir: Optional[Path] = None):
        """Initialize CCA-specific demographic generator."""
        super().__init__(data_path, cache_dir, include_occupation=False)


class KGDemographicAttributesForAdditive(KGDemographicAttributes):
    """
    Demographic attributes for additive compositionality experiments.
    
    Groups users by gender-age combinations for testing generalization.
    """
    
    def __init__(self,
                 data_path: Path = Path("data/ml-1m"),
                 cache_dir: Optional[Path] = None):
        """Initialize additive-specific demographic generator."""
        super().__init__(data_path, cache_dir, include_occupation=False)
        
    def get_grouped_attributes(self) -> Dict[str, Any]:
        """
        Group users by demographic combinations and compute mean embeddings.
        
        Returns:
            Dictionary with grouped attribute matrix and metadata
        """
        if self.attribute_matrix is None:
            raise ValueError("Attributes not generated yet. Call generate() first.")
        
        # Create combination labels
        combinations = []
        for i in range(len(self.entity_ids)):
            # Get active attributes for this user
            active_attrs = [
                self.attribute_names[j]
                for j in range(len(self.attribute_names))
                if self.attribute_matrix[i, j] == 1
            ]
            # Filter to gender and age only
            gender = [a for a in active_attrs if a.startswith('gender:')][0]
            age = [a for a in active_attrs if a.startswith('age:')][0]
            combinations.append(f"{gender}_{age}")
        
        # Get unique combinations
        unique_combos = sorted(set(combinations))
        combo_to_idx = {c: i for i, c in enumerate(unique_combos)}
        
        # Create grouped attribute matrix
        n_combos = len(unique_combos)
        n_attrs = len(self.attribute_names)
        grouped_matrix = np.zeros((n_combos, n_attrs), dtype=np.int8)
        
        for combo in unique_combos:
            # Parse combination
            gender_age = combo.split('_')
            gender_attr = gender_age[0]
            age_attr = gender_age[1]
            
            # Set attributes
            combo_idx = combo_to_idx[combo]
            if gender_attr in self.attribute_names:
                grouped_matrix[combo_idx, self.attribute_names.index(gender_attr)] = 1
            if age_attr in self.attribute_names:
                grouped_matrix[combo_idx, self.attribute_names.index(age_attr)] = 1
        
        # Create mapping from users to combinations
        user_to_combo = {
            self.entity_ids[i]: combinations[i]
            for i in range(len(self.entity_ids))
        }
        
        logger.info(f"Created {n_combos} demographic combinations from {len(self.entity_ids)} users")
        
        return {
            'matrix': grouped_matrix,
            'combination_ids': unique_combos,
            'attribute_names': self.attribute_names,
            'user_to_combination': user_to_combo,
            'combination_to_index': combo_to_idx
        }