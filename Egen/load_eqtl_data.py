"""
Script to load and inspect real eQTL data.

This script demonstrates:
1. Loading the real eQTL data
2. Displaying basic statistics and information
3. Analyzing cell type distributions
"""

import os
import sys
import pandas as pd

# Add the parent directory to Python path to access e_gen package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from e_gen.data_loader import DataManager

def load_and_inspect_data():
    """Load and inspect the real eQTL data."""
    print("Loading eQTL data...")
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Get the path to the data directory
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    try:
        # Register and load the data
        data_manager.register_data(
            data_path=data_path,
            loader_type='bio'  # Use the bio loader for eQTL data
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
                
                # Display column information
                print("\nColumn descriptions:")
                for column in table_data.columns:
                    print(f"{column}: {table_data[column].dtype}")
                    n_unique = len(table_data[column].unique())
                    print(f"  - Unique values: {n_unique}")
                
                # Analyze cell types if available
                if 'cell_type_name' in table_data.columns:
                    print("\nCell Type Analysis:")
                    cell_type_counts = table_data['cell_type_name'].value_counts()
                    print("\nTop 10 cell types by frequency:")
                    print(cell_type_counts.head(10))
                    print(f"\nTotal number of unique cell types: {len(cell_type_counts)}")
                
                # Basic statistics for numeric columns
                numeric_cols = table_data.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    print("\nBasic statistics for numeric columns:")
                    print(table_data[numeric_cols].describe())
                    
                    # Additional analysis for qtl_scores if available
                    if 'qtl_score' in numeric_cols:
                        print("\nQTL Score Distribution:")
                        print("Top 10 highest QTL scores:")
                        top_scores = table_data.nlargest(10, 'qtl_score')[['gene_name', 'cell_type_name', 'qtl_score']]
                        print(top_scores)
                
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

if __name__ == "__main__":
    load_and_inspect_data() 