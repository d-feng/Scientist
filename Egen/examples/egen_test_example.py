"""
Example script demonstrating E-Gen framework functionality with real eQTL data.

This script demonstrates:
1. Loading real eQTL data
2. Running falsification test proposals
3. Generating and executing test code
4. Running sequential falsification tests
"""

import os
import sys
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add the parent directory to Python path to access e_gen package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from e_gen import DataManager
from e_gen.agents.falsification import (
    SequentialFalsificationTest,
    FalsificationTestProposalAgent,
    CodeGeneratorAgent
)

def load_api_key():
    """Load API key from environment variables."""
    # Try to load from .env file in current directory
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables "
            "or create a .env file in the project root with OPENAI_API_KEY=your-key"
        )
    return api_key

def load_real_data():
    """Load real eQTL data for testing."""
    print("Loading real eQTL data...")
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Get the path to the data directory
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # Register and load the data
    data_manager.register_data(
        data_path=data_path,
        data_sampling=-1,  # Use all data
        loader_type='bio'  # Use the bio loader for eQTL data
    )
    
    return data_manager.data.table_dict

def get_data_context(data_dict):
    """Generate context string describing available dataframes."""
    context = []
    for name, df in data_dict.items():
        context.append(f"DataFrame '{name}':")
        context.append("Columns:")
        for col in df.columns:
            context.append(f"- {col} ({df[col].dtype})")
        context.append("\nSample values:")
        context.append(str(df.head()))
        context.append("\n")
    return "\n".join(context)

def run_falsification_test_proposal(api_key):
    """Example of using FalsificationTestProposalAgent."""
    print("\nRunning FalsificationTestProposalAgent example...")
    
    real_data = load_real_data()
    try:
        agent = FalsificationTestProposalAgent(
            data=real_data,
            domain="biology",
            llm="gpt-4o",
            api_key=api_key
        )
        
        hypothesis = "IL2 impact immune cell than non-immune cells"
        log = {
            'designer': [],
            'executor': [],
            'relevance_checker': [],
            'summarizer': [],
            'sequential_testing': []
        }
        
        # Generate test proposal
        test_spec = agent.go(hypothesis, log=log)
        print("\nGenerated test specification:")
        print(test_spec)
        
        # Track the test
        agent.add_to_existing_tests(test_spec)
        print(f"\nNumber of tracked tests: {len(agent.existing_tests)}")
        
        return test_spec
    except Exception as e:
        print(f"\nError in test proposal generation: {str(e)}")
        return None

def run_code_generator(test_spec, api_key):
    """Example of using CodeGeneratorAgent."""
    if not test_spec:
        print("\nSkipping code generation due to failed test proposal")
        return None
        
    print("\nRunning CodeGeneratorAgent example...")
    
    real_data = load_real_data()
    try:
        # Create context string describing available dataframes
        context = get_data_context(real_data)
        
        agent = CodeGeneratorAgent(
            data=real_data,
            time_limit=5,
            max_retry=3,
            domain="biology",
            llm="gpt-4o",
            api_key=api_key
        )
        
        # Create messages list with the test specification
        messages = [("user", f"Generate a statistical test for this test specification:\n{test_spec}")]
        
        # Pass messages directly as a List
        result = agent.generate(messages)
        
        print("\nCode generation result:")
        print(f"Status: {result['status']}")
        if result.get('error') == 'no':
            print("Output:", result.get('captured_output', ''))
            print("P-value:", result.get('p_val', 'Not available'))
        
        return result
    except Exception as e:
        print(f"\nError in code generation: {str(e)}")
        return None

def run_sequential_falsification(api_key):
    """Example of using SequentialFalsificationTest."""
    print("\nRunning SequentialFalsificationTest example...")
    
    real_data = load_real_data()
    config = {
        'alpha': 0.1,
        'beta': 0.1,
        'aggregate_test': 'Fisher',
        'llm_approx': False,
        'max_num_of_tests': 3,
        'time_limit': 5,
        'max_retry': 3,
        'domain': "biology",
        'max_failed_tests': 3
    }
    
    try:
        # Initialize and configure test
        sequential_test = SequentialFalsificationTest(llm="gpt-4o", api_key=api_key)
        sequential_test.configure(data=real_data, **config)
        
        # Run test
        hypothesis = "IL2 impact immune cell than non-immune cells"
        results, conclusion, log = sequential_test.run(hypothesis)
        
        print("\nTest Results:")
        print(conclusion)
        
        return results, conclusion, log
    except Exception as e:
        print(f"\nError in sequential falsification: {str(e)}")
        return None, None, None

def main():
    """Run the E-Gen framework examples with real data."""
    print("Starting E-Gen framework examples with real eQTL data...")
    print("This example demonstrates how to use the E-Gen framework for hypothesis testing.")
    print("We'll use real eQTL data to test a hypothesis about gene expression.")
    
    try:
        # Load API key
        api_key = load_api_key()
        print("\nAPI key loaded successfully")
        
        # Run examples in sequence
        test_spec = run_falsification_test_proposal(api_key)
        code_result = run_code_generator(test_spec, api_key)
        results, conclusion, log = run_sequential_falsification(api_key)
        
        if all(x is not None for x in [test_spec, code_result, results]):
            print("\nAll examples completed successfully!")
            print("\nSummary:")
            print("1. Generated test specifications for the hypothesis")
            print("2. Generated and executed statistical test code")
            print("3. Ran sequential falsification testing")
            print("\nCheck the output above for detailed results of each step.")
        else:
            print("\nSome examples failed. Check the error messages above.")
            
    except ValueError as e:
        print(f"\nConfiguration Error: {str(e)}")
        print("\nPlease set up your API key and try again.")
    except Exception as e:
        print(f"\nUnexpected Error: {str(e)}")
        print("\nPlease check your configuration and try again.")

if __name__ == '__main__':
    main() 