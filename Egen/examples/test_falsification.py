"""
Test script for falsification testing using the eQTL dataset.

This script demonstrates:
1. Loading the eQTL data
2. Running statistical tests
3. Evaluating the hypothesis

To run this script:
1. Set your OpenAI API key in the environment:
   - Windows (PowerShell): $env:OPENAI_API_KEY="your-api-key-here"
   - Windows (CMD): set OPENAI_API_KEY=your-api-key-here
   - Linux/Mac: export OPENAI_API_KEY=your-api-key-here
2. Or create a .env file in the root directory with:
   OPENAI_API_KEY=your-api-key-here
"""

import os
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get API key from environment
api_key = os.getenv('OPENAI_API_KEY', 'EMPTY')
if api_key == 'EMPTY':
    print("Warning: OPENAI_API_KEY not found in environment variables or .env file")
    print("Please set your API key using:")
    print("  - Windows (PowerShell): $env:OPENAI_API_KEY=\"your-api-key-here\"")
    print("  - Windows (CMD): set OPENAI_API_KEY=your-api-key-here")
    print("  - Linux/Mac: export OPENAI_API_KEY=your-api-key-here")
    print("Or create a .env file with: OPENAI_API_KEY=your-api-key-here")

# Add parent directory to path to import e_gen
sys.path.append(str(Path(__file__).parent.parent))

from e_gen import DataManager
from e_gen.agents.falsification import SequentialFalsificationTest

def run_falsification_test():
    """Run statistical tests on the eQTL data."""
    
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
    
    print("\n=== Setting up Falsification Test ===")
    # Initialize falsification test with GPT-4o
    sft = SequentialFalsificationTest(llm="gpt-4o", api_key=api_key)
    
    # Configure the test
    sft.configure(
        data=data_loader,
        alpha=0.05,  # significance level
        beta=0.05,   # type II error rate
        aggregate_test='E-value',
        max_num_of_tests=5,
        domain="biology",
        llm_approx=True  # Use LLM approximation mode
    )
    
    print("\n=== Running Statistical Tests ===")
    
    # Test 1: Proportion of strong associations
    print("\nTest 1: Proportion of variants with strong effects")
    strong_threshold = 5.0  # qtl_score threshold for strong effect
    n_strong = (eqtl_data['qtl_score'] > strong_threshold).sum()
    prop_strong = n_strong / len(eqtl_data)
    
    # Binomial test for proportion > 10%
    null_prop = 0.10
    p_value = 1 - stats.binom.cdf(n_strong - 1, len(eqtl_data), null_prop)
    
    print(f"Number of variants with strong effects (score > {strong_threshold}): {n_strong}")
    print(f"Proportion: {prop_strong:.4f}")
    print(f"Null hypothesis: proportion <= {null_prop}")
    print(f"P-value: {p_value}")
    
    # Test 2: Distribution of QTL scores
    print("\nTest 2: Distribution of QTL scores")
    mean_score = eqtl_data['qtl_score'].mean()
    std_score = eqtl_data['qtl_score'].std()
    
    # One-sample t-test against baseline
    baseline_score = 0
    t_stat, t_p_value = stats.ttest_1samp(eqtl_data['qtl_score'], baseline_score)
    
    print(f"Mean QTL score: {mean_score:.4f}")
    print(f"Standard deviation: {std_score:.4f}")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {t_p_value}")
    
    # Combine p-values using Fisher's method
    combined_stat = -2 * np.sum(np.log([p_value, t_p_value]))
    combined_p = 1 - stats.chi2.cdf(combined_stat, df=4)
    
    print("\n=== Combined Results ===")
    print(f"Fisher's combined p-value: {combined_p}")
    
    # Make conclusion
    alpha = 0.05
    reject = combined_p < alpha
    
    print("\n=== Conclusion ===")
    print(f"At significance level {alpha}:")
    if reject:
        print("REJECT the null hypothesis")
        print("There is significant evidence of strong genetic associations")
    else:
        print("FAIL TO REJECT the null hypothesis")
        print("Insufficient evidence of strong genetic associations")
    
    return True

if __name__ == "__main__":
    success = run_falsification_test()
    print(f"\nTest {'succeeded' if success else 'failed'}")
    sys.exit(0 if success else 1) 