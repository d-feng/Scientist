"""
Data loader module for E-Gen framework.
"""

import os
import pickle
from typing import Dict, Optional
import pandas as pd

class ExperimentalDataLoader:
    """Data loader for experimental data."""
    
    def __init__(self, data_path: str, table_dict_selection: str = 'all_bio', data_sampling: int = -1):
        """
        Initialize the experimental data loader.
        
        Args:
            data_path (str): Path to data directory
            table_dict_selection (str): Type of table selection ('all_bio' or 'default')
            data_sampling (int): Number of datasets to sample (-1 for all)
        """
        self.data_path = data_path
        self.table_dict_selection = table_dict_selection
        self.data_sampling = data_sampling
        self.table_dict = {}
        self.data_desc = ""
        
        # Load the data
        self._load_data()
        
    def _load_data(self):
        """Load data from bio_database directory."""
        bio_db_path = os.path.join(self.data_path, 'bio_database')
        if not os.path.exists(bio_db_path):
            raise FileNotFoundError(f"Bio database not found at {bio_db_path}")
            
        # First try to load the standard files
        table_dict_file = os.path.join(bio_db_path, 'table_dict.pkl')
        desc_file = os.path.join(bio_db_path, 'data_desc.pkl')
        
        if os.path.exists(table_dict_file) and os.path.exists(desc_file):
            # Load standard format
            with open(table_dict_file, 'rb') as f:
                self.table_dict = pickle.load(f)
            with open(desc_file, 'rb') as f:
                self.data_desc = pickle.load(f)
        else:
            # Try to load eqtl data
            eqtl_file = os.path.join(bio_db_path, 'eqtl_ukbb.pkl')
            if os.path.exists(eqtl_file):
                with open(eqtl_file, 'rb') as f:
                    eqtl_data = pickle.load(f)
                    if isinstance(eqtl_data, pd.DataFrame):
                        self.table_dict = {'eqtl_ukbb': eqtl_data}
                        self.data_desc = "UKBB eQTL dataset containing genetic associations"
                    
        # Apply data sampling if specified
        if self.data_sampling > 0:
            # Sample from available tables
            table_keys = list(self.table_dict.keys())
            if len(table_keys) > self.data_sampling:
                sampled_keys = table_keys[:self.data_sampling]
                self.table_dict = {k: self.table_dict[k] for k in sampled_keys}

class DataManager:
    """Manager class for data registration and loading."""
    
    def __init__(self):
        """Initialize the data manager."""
        self.data_loader = None
        self.data_path = None
        
    def register_data(self, data_path: str, data_sampling: int = -1, 
                     loader_type: str = 'bio', metadata: Optional[Dict] = None):
        """
        Register data for hypothesis testing.
        
        Args:
            data_path (str): Path to data directory
            data_sampling (int): Number of datasets to sample (-1 for all)
            loader_type (str): Type of data loader to use ('bio' or 'bio_selected')
            metadata (Optional[Dict]): Additional metadata for loading
        """
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            
        self.data_path = data_path
        bio_db_path = os.path.join(data_path, 'bio_database')
        
        if not os.path.exists(bio_db_path):
            raise FileNotFoundError(
                f"Bio database not found at {bio_db_path}. "
                "Please ensure the bio_database folder containing table_dict.pkl "
                "and data_desc.pkl exists in the specified data path."
            )
        
        if loader_type == 'bio':
            self.data_loader = ExperimentalDataLoader(
                data_path=data_path,
                table_dict_selection='all_bio',
                data_sampling=data_sampling
            )
        elif loader_type == 'bio_selected':
            self.data_loader = ExperimentalDataLoader(
                data_path=data_path,
                table_dict_selection='default',
                data_sampling=data_sampling
            )
        else:
            raise ValueError(f"Unknown loader_type: {loader_type}. "
                           f"Supported types are: 'bio', 'bio_selected'")
            
    @property
    def data(self):
        """Get the loaded data."""
        if self.data_loader is None:
            raise ValueError("No data has been registered. Call register_data() first.")
        return self.data_loader 