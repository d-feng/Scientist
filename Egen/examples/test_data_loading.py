#Example script to test E-Gen data loading functionality.
#This script tests:
#1. Data loading from bio_database
#2. Table dictionary access
#3. Falsification test setup
#
#Usage:
#    From root directory (r"C:\Users\difen\Egen"):
#    python examples/test_data_loading.py
#
import os
import sys
import pandas as pd

# Add the parent directory to Python path to access e_gen package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e_gen.data_loader import DataManager
from e_gen.agents.falsification import SequentialFalsificationTest

def test_data_loading():
    """Test loading eQTL data and display the first few rows."""
    print("Testing eQTL data loading...")
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Get the path to the bio_database directory
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    try:
        # Register and load the data
        data_manager.register_data(
            data_path=data_path,
            loader_type='bio'
        )
        
        # Access the loaded data
        data_loader = data_manager.data
        if data_loader and hasattr(data_loader, 'table_dict'):
            # Print information about available tables
            print("\nAvailable tables:")
            for table_name, table_data in data_loader.table_dict.items():
                print(f"\nTable: {table_name}")
                print(f"Shape: {table_data.shape}")
                print("\nFirst 5 rows:")
                print(table_data.head())
                
                # Add cell type analysis
                if 'cell_type_name' in table_data.columns:
                    print("\nCell Type Analysis:")
                    cell_type_counts = table_data['cell_type_name'].value_counts()
                    print("\nUnique cell types and their counts:")
                    for cell_type, count in cell_type_counts.items():
                        print(f"{cell_type}: {count:,} entries")
                    print(f"\nTotal number of unique cell types: {len(cell_type_counts)}")
                
                print("\nColumn descriptions:")
                for column in table_data.columns:
                    print(f"{column}: {table_data[column].dtype}")
                
                # Print basic statistics for numeric columns
                numeric_cols = table_data.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    print("\nBasic statistics for numeric columns:")
                    print(table_data[numeric_cols].describe())
        else:
            print("\nNo data tables found in the loader.")
            
    except FileNotFoundError as e:
        print(f"\nError: Bio database not found at {data_path}/bio_database")
        print("Please ensure you have the correct data directory structure:")
        print("data/")
        print("└── bio_database/")
        print("    └── eqtl_ukbb.pkl")
    except Exception as e:
        print(f"\nError loading data: {str(e)}")

if __name__ == '__main__':
    test_data_loading() 