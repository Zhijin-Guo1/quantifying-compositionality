"""MovieLens data loader for KG experiments."""

import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class MovieLensLoader:
    """Load MovieLens 1M dataset for KG compositionality analysis."""
    
    def __init__(self, data_dir='ml-1m'):
        """
        Initialize MovieLens loader.
        
        Args:
            data_dir: Directory containing MovieLens data files
        """
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, 'users.dat')
        self.readme_file = os.path.join(data_dir, 'README')
        
    def load_users(self):
        """
        Load MovieLens user data with demographics.
        
        Returns:
            users_df: DataFrame with user demographics
            user_ids: List of user IDs (0-indexed for embeddings)
        """
        if not os.path.exists(self.users_file):
            raise FileNotFoundError(f"MovieLens users file not found at {self.users_file}")
        
        # Load occupation mapping from README
        occupation_dict = self._load_occupation_mapping()
        
        # Load users data
        logger.info(f"Loading MovieLens users from {self.users_file}")
        users_df = pd.read_csv(
            self.users_file,
            delimiter='::',
            engine='python',
            header=None,
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code']
        )
        
        # Map occupation IDs to names
        users_df['occupation'] = users_df['occupation'].replace(occupation_dict)
        
        # Convert user_id to 0-indexed for embedding alignment
        users_df['embedding_id'] = users_df['user_id'] - 1  # MovieLens IDs start at 1
        
        logger.info(f"Loaded {len(users_df)} users")
        logger.info(f"Age groups: {users_df['age'].unique()}")
        logger.info(f"Occupations: {len(users_df['occupation'].unique())} unique")
        
        return users_df, users_df['embedding_id'].tolist()
    
    def _load_occupation_mapping(self):
        """Load occupation ID to name mapping from README."""
        if not os.path.exists(self.readme_file):
            # Default occupation mapping if README not found
            return {
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
        
        try:
            readme_text = np.array(open(self.readme_file).read().splitlines())
            start_index = np.flatnonzero(
                np.core.defchararray.find(readme_text, 'Occupation is chosen') != -1
            )[0]
            end_index = np.flatnonzero(
                np.core.defchararray.find(readme_text, 'MOVIES FILE DESCRIPTION') != -1
            )[0]
            
            occupation_list = [
                x.split('"')[1] for x in readme_text[start_index:end_index][2:-1].tolist()
            ]
            occupation_dict = dict(zip(range(len(occupation_list)), occupation_list))
            
            logger.info(f"Loaded {len(occupation_dict)} occupation mappings from README")
            return occupation_dict
            
        except Exception as e:
            logger.warning(f"Failed to parse README, using default occupations: {e}")
            return self._load_occupation_mapping.__defaults__[0]
    
    def create_demographic_attributes(self, users_df):
        """
        Create binary demographic attribute matrix.
        
        Args:
            users_df: DataFrame with user demographics
            
        Returns:
            attributes: Binary attribute matrix (n_users, n_attributes)
            feature_names: List of attribute names
        """
        # Create one-hot encoding for demographics
        logger.info("Creating demographic attribute matrix...")
        
        # Encode gender, age, and occupation
        encode_attribute = pd.get_dummies(
            users_df[['gender', 'age', 'occupation']].astype(str)
        )
        
        # Convert to numpy array
        attributes = encode_attribute.to_numpy()
        feature_names = encode_attribute.columns.tolist()
        
        logger.info(f"Created attribute matrix: shape {attributes.shape}")
        logger.info(f"Features: {len(feature_names)} total")
        
        return attributes, feature_names
    
    def prepare_kg_data(self):
        """
        Prepare complete KG data for compositionality analysis.
        
        Returns:
            dict with:
                - users_df: DataFrame with user info
                - user_ids: List of embedding IDs
                - attributes: Binary attribute matrix
                - feature_names: List of feature names
        """
        # Load users
        users_df, user_ids = self.load_users()
        
        # Create attributes
        attributes, feature_names = self.create_demographic_attributes(users_df)
        
        return {
            'users_df': users_df,
            'user_ids': user_ids,
            'attributes': attributes,
            'feature_names': feature_names,
            'n_users': len(users_df)
        }