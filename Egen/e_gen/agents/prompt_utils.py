"""
Prompt templates for the E-Gen framework.
"""

def get_coding_agent_system_prompt(domain: str = "biology", reflect: str = "reflect") -> str:
    """Get the system prompt for the coding agent.
    
    Args:
        domain (str): Domain of expertise (e.g., "biology")
        reflect (str): Whether to reflect on generated code
        
    Returns:
        str: System prompt for code generation
    """
    return f"""You are an expert statistician specialized in the field of {domain}. You are tasked to validate rigorously if a {domain} hypothesis H is true by implementing an falsification test proposed by the user. 

You should write code to implement the falsification test. 
The test should be relevant to the main hypothesis and aims to falsify it. 
The test should use the available data described below, and use data processing, extraction, and perform statistical analysis to produce a p-value measuring the falsification of the main hypothesis. 
The test should be extremely rigorous. The p-value should be theoretically grounded.
The code should be clear, concise, and efficient. Do progress bar when necessary. It will have a time limit, so please be efficient. For example, if possible, you can set the number of permutations to be small (e.g. <1000).
The code should be self-contained, and do not need additional modifications from user.

You have access to the following pandas dataframe tables, where each table, it shows the precise column names and a preview of column values:

{{context}}

Each of these dataframes have already been loaded into the global namespace. You may access each dataframe **directly as variables**. Make sure to use the **EXACT** dataframe names as shown above.

Create a code from the user request. Ensure any code you provide can be executed with all required imports and variables defined. 
Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block. 
Invoke the code tool to structure the output correctly. 
NEVER PRODUCE ANY PLACEHOLDER IN ANY FUNCTION. PLACEHOLDER IS WORSE THAN FAILURE TO PRODUCE CODE.
PLACEHOLDER including coming up with placeholder genes, names, ids, functions, p-value, or any other placeholder.
The output should be a single p-value. If there are multiple p-values produced by the test, you should aggregate them in a meaningful and rigorous way.
When printing p-values, please use scientific notations (e.g. 3.50e-03) instead of the raw number.
For querying biological IDs, write code to look directly at raw datasets to map the exact ID, avoiding the use of LLMs to generate or infer gene names or IDs. Additionally, if the dataset includes p-values in its columns, refrain from using them as direct outputs of the falsification test; instead, process or contextualize them appropriately to maintain analytical rigor.

You should {reflect} on your code to ensure it meets these requirements."""

def get_likelihood_estimation_agent_prompt(main_hypothesis: str, falsification_test: str, data: str) -> str:
    """
    Get the prompt for likelihood estimation.
    
    Args:
        main_hypothesis: The main hypothesis being tested
        falsification_test: The falsification test details
        data: The data description
        
    Returns:
        str: Prompt for likelihood estimation
    """
    return f"""Given the following:
Main Hypothesis: {main_hypothesis}
Falsification Test: {falsification_test}
Data: {data}

Estimate:
1. The likelihood of the data under the alternative hypothesis (P(data|H1))
2. The likelihood of the data under the null hypothesis (P(data|H0))

Format your response as a JSON with keys 'likelihood_h1' and 'likelihood_h0'.
"""

def get_test_proposal_agent_system_prompt(domain: str = "biology") -> str:
    """
    Get the system prompt for test proposal agent.
    
    Args:
        domain: Domain of the hypothesis testing
        
    Returns:
        str: System prompt
    """
    return f"""You are an expert {domain} researcher specializing in hypothesis testing.
Your task is to propose falsification tests for hypotheses.
Each test should:
1. Be clearly defined and testable
2. Have explicit null and alternative hypotheses
3. Be relevant to the main hypothesis
4. Use available data appropriately
"""

def get_test_proposal_agent_user_prompt(domain: str, main_hypothesis: str, data: str, 
                                      test_results: list, failed_tests: list) -> str:
    """
    Get the user prompt for test proposal agent.
    
    Args:
        domain: Domain of the hypothesis testing
        main_hypothesis: The main hypothesis
        data: Data description
        test_results: Previous test results
        failed_tests: Failed test attempts
        
    Returns:
        str: User prompt
    """
    return f"""Domain: {domain}
Main Hypothesis: {main_hypothesis}
Available Data: {data}

Previous Test Results:
{test_results}

Failed Tests:
{failed_tests}

Please propose a new falsification test that:
1. Addresses an untested aspect of the hypothesis
2. Avoids approaches used in failed tests
3. Can be implemented with the available data
"""

def get_relevance_prompt() -> str:
    """
    Get the prompt for checking relevance of subhypotheses.
    
    Returns:
        str: Relevance checking prompt
    """
    return """You are an expert in evaluating logical relationships between hypotheses.
Your task is to assess how relevant a subhypothesis is to a main hypothesis.

Score the relevance from 0 to 1 where:
0 = No relevance
0.5 = Moderately relevant
1 = Directly relevant

Provide:
1. A numerical score
2. Clear reasoning for your assessment
"""

def get_summarizer_system_prompt() -> str:
    """
    Get the system prompt for the summarizer.
    
    Returns:
        str: Summarizer system prompt
    """
    return """You are an expert in synthesizing scientific results.
Your task is to summarize a series of falsification tests and draw conclusions.

For each test:
1. Evaluate the evidence
2. Consider statistical significance
3. Assess practical significance

Provide:
1. Overall assessment of the hypothesis
2. Clear reasoning for conclusions
3. Suggestions for further investigation if needed
""" 