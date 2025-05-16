"""
Test script for CodeGeneratorAgent to analyze eQTL data.

This script demonstrates:
1. Loading the eQTL data
2. Using CodeGeneratorAgent to generate analysis code
3. Running and evaluating the generated code

To run this script:
1. Create a .env file in the root directory (C:\\Users\\difen\\Egen\\.env)
2. Add your OpenAI API key: OPENAI_API_KEY=your-api-key-here
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError(
        "OpenAI API key not found! Please create a .env file in the root directory "
        "(C:\\Users\\difen\\Egen\\.env) with: OPENAI_API_KEY=your-api-key-here"
    )

# Add parent directory to path to import e_gen
sys.path.append(str(Path(__file__).parent.parent))

from e_gen import DataManager
from e_gen.agents.falsification import CodeGeneratorAgent

def test_code_generator():
    """Test the CodeGeneratorAgent with eQTL data analysis tasks."""
    
    # Initialize data manager and load data
    data_manager = DataManager()
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    data_manager.register_data(
        data_path=data_path,
        data_sampling=-1,
        loader_type='bio'
    )
    
    # Get the data loader
    data_loader = data_manager.data
    
    print("\n=== eQTL Dataset Information ===")
    eqtl_data = data_loader.table_dict['eqtl_ukbb']
    print(f"Shape: {eqtl_data.shape}")
    print("\nColumns:")
    for col in eqtl_data.columns:
        print(f"- {col}")
    print("\nSample of data:")
    print(eqtl_data.head())
    
    print("\n=== Setting up CodeGeneratorAgent ===")
    # Initialize code generator with API key
    agent = CodeGeneratorAgent(
        data=data_loader,
        llm="gpt-4o",
        time_limit=5,  # 5 minutes timeout
        domain="biology",
        verbose=True,
        api_key=api_key  # Pass the API key here
    )
    
    # Test different analysis tasks
    analysis_tasks = [
        "Calculate summary statistics (mean, std, min, max) for qtl_scores grouped by cell_type_name",
        "Find the top 10 genes with the highest average qtl_scores",
        "Compute the distribution of qtl_scores and test if they follow a normal distribution",
        "Calculate the percentage of strong associations (qtl_score > 5.0) for each gene"
    ]
    
    print("\n=== Running Analysis Tasks ===")
    for i, task in enumerate(analysis_tasks, 1):
        print(f"\nTask {i}: {task}")
        print("-" * 50)
        
        # Generate and run code
        result = agent.generate(
            messages=[("user", f"Generate code to {task}. Use the eqtl_data DataFrame.")],
            iterations=0
        )
        
        if result.get("error") == "yes":
            print(f"Error: {result.get('status')}")
        else:
            print("\nGenerated code output:")
            print(result.get("captured_output", "No output captured"))
            
        print("-" * 50)
    
    return True

if __name__ == "__main__":
    success = test_code_generator()
    print(f"\nTest {'succeeded' if success else 'failed'}")
    sys.exit(0 if success else 1) 