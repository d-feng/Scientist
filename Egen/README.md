# E-Gen: Agentic AI Framework for Hypothesis Testing

E-Gen is a powerful framework that uses agentic AI to process information and perform hypothesis testing through falsification tests. It leverages LangGraph to orchestrate multiple AI agents that work together to generate and execute statistical tests.

## Features

- **Code Generator Agent**: Generates Python code for statistical tests based on hypotheses
- **Falsification Agent**: Runs falsification tests and analyzes results
- **Sequential Testing**: Orchestrates multiple tests with statistical power analysis
- **LLM Integration**: Uses language models for hypothesis analysis and test generation
- **Multiple Testing Methods**: Supports various statistical approaches including:
  - Fisher's Method
  - E-value calibration
  - Likelihood ratio tests

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from e_gen import SequentialFalsificationTest

# Initialize the framework
sft = SequentialFalsificationTest()

# Configure with your data and parameters
sft.configure(
    data=your_data,
    alpha=0.1,
    beta=0.1,
    aggregate_test='E-value',
    max_num_of_tests=10,
    domain="biology"
)

# Run tests for your hypothesis
results = sft.run("Your hypothesis here")
```

## Architecture

The framework uses a multi-agent architecture with LangGraph:

1. **Test Proposal Agent**: Designs appropriate falsification tests
2. **Code Generator Agent**: Implements the statistical tests
3. **Falsification Agent**: Executes tests and analyzes results
4. **Orchestrator**: Manages the sequential testing process

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt
- Access to a language model (e.g., Claude)

## License

MIT License 