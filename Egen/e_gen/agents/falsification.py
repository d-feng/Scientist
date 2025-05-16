"""
Falsification agents for hypothesis testing.
"""

import contextlib
import io
import logging
import json
import traceback
import multiprocessing
from typing import Dict, List, Literal, Optional, TypedDict, Annotated
import re
import tempfile
import os

import numpy as np
from scipy.stats import chi2
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_core.messages.base import get_msg_title_repr
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .utils import get_llm, pretty_print
from .prompt_utils import (
    get_coding_agent_system_prompt,
    get_likelihood_estimation_agent_prompt,
    get_test_proposal_agent_system_prompt,
    get_test_proposal_agent_user_prompt,
    get_relevance_prompt,
    get_summarizer_system_prompt
)

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)

class code(BaseModel):
    """Code output"""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

def parse_output(solution):
    """When we add 'include_raw=True' to structured output,
    it will return a dict w 'raw', 'parsed', 'parsing_error'."""
    if not solution["parsed"]:
        print('code solution fail to produce')
        print(solution)
    return solution["parsed"]

class LogLikelihoodRatioInput(BaseModel):
    """Model for likelihood ratio input."""
    likelihood_h1: float = Field(description="probability of data given hypothesis is alternative, P(data|h1)")
    likelihood_h0: float = Field(description="probability of data given hypothesis is null, P(data|h0)")

class TestSpecification(BaseModel):
    """Model for test specification."""
    test_name: str = Field(description="name of the test")
    test_description: str = Field(description="test description")
    null_hypothesis: str = Field(description="null hypothesis")
    alternate_hypothesis: str = Field(description="alternate hypothesis")

class ParserYesNo(BaseModel):
    """Model for parsing p-value presence."""
    check_output_error: Optional[str] = Field(
        description="Does the given text contains a p-value? Yes if it has; No if does not."
    )
    p_val: Optional[str] = Field(description="p-value formatted in scientific notations")

class DataInputCheckResult(BaseModel):
    """Model for checking fake data entries."""
    fake_data_entries: str = Field(
        description="Does the code make up fake data entries? Yes if it does; No if does not."
    )

class RelevanceSubhypothesis(BaseModel):
    """Model for subhypothesis relevance."""
    relevance_reasoning: Optional[str] = Field(
        description="What is the reason behind this relevance score?"
    )
    relevance_score: Optional[str] = Field(description="relevance score")

class OutputSpecification(BaseModel):
    """Model for hypothesis testing output."""
    main_hypothesis: Optional[str] = Field(description="The main hypothesis under study")
    falsification_test_result: Optional[str] = Field(description="The result of the sequential falsification test")
    reasoning: Optional[str] = Field(description="Reasoning, summarizing, and analyzing these results")
    conclusion: Optional[bool] = Field(description="Conclusion on whether the hypothesis is true or false (True/False)")
    rationale: Optional[str] = Field(description="Rationale behind the conclusion")

class GraphState(TypedDict):
    """State for the falsification graph."""
    messages: Annotated[List, add_messages]
    cur_test_proposal: str

def likelihood_ratio_e_value(likelihood_ratio: List[float], alpha: float = 0.1) -> tuple[bool, float]:
    """Calculate e-value from likelihood ratios."""
    likelihood_ratio = np.array(likelihood_ratio)
    cum_e = 1/np.prod(likelihood_ratio)
    return cum_e < alpha, cum_e

def e_value_kappa_calibrator(p_values: List[float], alpha: float = 0.1, kappa: float = 0.5) -> tuple[bool, float]:
    """Calculate e-value using kappa calibrator."""
    p_values = np.array(p_values)
    e_values = kappa * p_values ** (kappa-1)
    cum_e = np.prod(e_values)
    return cum_e > 1/alpha, cum_e

def e_value_integral_calibrator(p_values: List[float], alpha: float = 0.1) -> tuple[bool, float]:
    """Calculate e-value using integral calibrator."""
    p_values = np.array(p_values)
    e_values = (1 - p_values + p_values * np.log(p_values))/(p_values * (-np.log(p_values))**2)
    cum_e = np.prod(e_values)
    return cum_e > 1/alpha, cum_e

def fishers_method(p_values: List[float], alpha: float = 0.1) -> tuple[bool, float]:
    """Implement Fisher's method for combining p-values."""
    p_values = np.array(p_values)
    chi_square_stat = -2 * np.sum(np.log(p_values))
    degrees_of_freedom = 2 * len(p_values)
    combined_p_value = 1 - chi2.cdf(chi_square_stat, degrees_of_freedom)
    return combined_p_value < alpha, combined_p_value

def _run_code_in_process(code_str, imports, data=None):
    """Execute code in a separate process.
    
    Args:
        code_str (str): The code to execute
        imports (str): Import statements
        data (dict): Data to make available during execution
        
    Returns:
        str: Output from code execution or error message
    """
    try:
        output_capture = io.StringIO()
        full_code = imports + '\n\n' + code_str
        
        # Create execution globals with builtins and data
        exec_globals = {}
        exec_globals.update(__builtins__)
        
        # Add data to globals if provided
        if data:
            exec_globals.update(data)
        
        # Execute the code with output capture
        with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
            # Redirect logging output
            if logging.getLogger().handlers:
                logging.getLogger().handlers[0].stream = output_capture
                
            exec(full_code, exec_globals)
            output = output_capture.getvalue()
            
            # If no output was captured, check if there's a p_value in globals
            if not output and 'p_value' in exec_globals:
                output = f"p-value: {exec_globals['p_value']:.6e}"
            elif not output:
                output = "No output was captured"
                
            return output
    except Exception as e:
        return f"ERROR: {str(e)}\n{traceback.format_exc()}"

def _run_code_wrapper(code, imports, data, output_file):
    """Wrapper function to run code and save output to a file."""
    try:
        output = _run_code_in_process(code, imports, data)
        with open(output_file, 'w') as f:
            json.dump({'output': output}, f)
    except Exception as e:
        with open(output_file, 'w') as f:
            json.dump({'error': str(e), 'traceback': traceback.format_exc()}, f)

class CodeGeneratorAgent:
    """Agent for generating code for falsification tests."""
    
    def __init__(self, data, llm="gpt-4o", max_retry=10, 
                 time_limit=10, reflect=True, verbose=True, llm_approx=False, 
                 domain="biology", port=None, api_key="EMPTY"):
        """Initialize the code generator agent."""
        self.data = data
        self.llm = get_llm(llm, temperature=0.0, port=port, api_key=api_key)
        print(llm)
        self.time_limit = time_limit
        self.llm_approx = llm_approx
        self.domain = domain
        self.max_iterations = max_retry
        self.flag = "reflect" if reflect else "do not reflect"
        self.verbose = verbose

        # Initialize prompts
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", "You are an evaluateGPT. Check if the output from a statistical test contains a p-value. If it does not have a p-value, then return No. If it return p value is nan, also return No. Otherwise Yes. Test output: ",
                ),
                ("placeholder", "{messages}"),
            ]
        )
    
        self.tool_404_parser_llm = self.format_check_prompt | self.llm.with_structured_output(ParserYesNo)
        
        self.data_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", "You are an evaluateGPT. Your task is to check if a LLM-generated code is hallucinating fake data entries. If code is directly using an existing datafrmae, then return No. However, if the code is making up new data entries such as `df = pd.DataFrame({{fake_data_entries}})`, return Yes.",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        self.data_checker = self.data_check_prompt | self.llm.with_structured_output(DataInputCheckResult)

        # Create data context string
        self.data_context = self._get_data_context()

        # Create system prompt with data context
        system_prompt = get_coding_agent_system_prompt(domain=self.domain, reflect=self.flag)
        print('system_prompt: ', system_prompt)
        self.code_gen_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", system_prompt,
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )

        structured_llm_claude = self.llm.with_structured_output(code, include_raw=True)
        self.code_gen_chain = self.code_gen_prompt | structured_llm_claude | parse_output

        # Set up the workflow graph
        self._setup_workflow()

    def _setup_workflow(self):
        """Set up the LangGraph workflow for code generation."""
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("generate", self.generate)  # Code generation
        workflow.add_node("check_code", self.code_check)  # Code checking
        workflow.add_node("reflect", self.reflect)  # Error reflection

        # Build graph
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "check_code")
        workflow.add_conditional_edges(
            "check_code",
            self.decide_next_step,
            {
                "end": END,
                "reflect": "reflect",
                "generate": "generate",
            },
        )
        workflow.add_edge("reflect", "generate")

        # Compile the workflow
        self.app = workflow.compile()

    def _get_data_context(self):
        """Generate context string describing available dataframes."""
        context = []
        for name, df in self.data.items():
            context.append(f"DataFrame '{name}':")
            context.append("Columns:")
            for col in df.columns:
                context.append(f"- {col} ({df[col].dtype})")
            context.append("\nSample values:")
            context.append(str(df.head()))
            context.append("\n")
        return "\n".join(context)

    def reflect(self, state: GraphState) -> GraphState:
        """
        Reflect on errors and generate improved solution.

        Args:
            state (dict): The current graph state containing:
                - messages: List of message tuples
                - iterations: Number of iterations so far
                - generation: Current code solution
                - error: Error status

        Returns:
            dict: Updated state with reflection added
        """
        if self.verbose:
            print("---REFLECTING ON ERRORS---")

        try:
            # Extract state components
            messages = state["messages"]
            iterations = state["iterations"]
            code_solution = state["generation"]

            # Generate reflection using the code generation chain
            reflection = self.code_gen_chain.invoke({
                "context": self.data_context,
                "messages": messages
            })

            # Add reflection to message history
            messages.append(("assistant", f"Here are reflections on the error and proposed improvements:\n{reflection}"))

            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations
            }
        except Exception as e:
            print(f"Error during reflection: {str(e)}")
            # If reflection fails, return original state
            return state

    def decide_next_step(self, state: GraphState) -> str:
        """
        Determines the next step in the code generation process.

        Args:
            state (dict): The current graph state containing:
                - error: Error status
                - iterations: Number of iterations so far

        Returns:
            str: Next step to take - 'end', 'reflect', or 'generate'
        """
        try:
            error = state.get("error", "yes")
            iterations = state.get("iterations", 0)

            if error == "no" or iterations >= self.max_iterations:
                if self.verbose:
                    print("---DECISION: FINISH---")
                    if iterations >= self.max_iterations:
                        print("Reached maximum iterations")
                    else:
                        print("Successfully generated valid code")
                return "end"
            else:
                if self.verbose:
                    print("---DECISION: RE-TRY SOLUTION---")
                    if self.flag == "reflect":
                        print("Will attempt reflection before regeneration")
                    else:
                        print("Will attempt direct regeneration")
                return "reflect" if self.flag == "reflect" else "generate"
        except Exception as e:
            print(f"Error in decision making: {str(e)}")
            # If decision making fails, default to end
            return "end"

    def generate(self, messages: List, iterations: int = 0) -> Dict:
        """Generate code for statistical test.
        
        Args:
            messages (List): List of message tuples for the LLM
            iterations (int): Number of iterations so far
            
        Returns:
            Dict: Generated code and test results
        """
        if iterations >= self.max_iterations:
            return {
                "error": "yes",
                "status": "Max retries reached"
            }
            
        try:
            # Generate code using the prompt template with context
            code_solution = self.code_gen_chain.invoke({
                "messages": messages,
                "context": self.data_context
            })
            
            # Check the generated code
            check_result = self.code_check({
                "messages": messages,
                "generation": code_solution,
                "iterations": iterations
            })
            
            if check_result.get("error") == "yes":
                # If error, try again with feedback
                messages.append(("assistant", str(code_solution)))
                messages.append(("user", f"Error: {check_result.get('status')}. Please fix and try again."))
                
                # Decide next step
                next_step = self.decide_next_step({
                    "error": "yes",
                    "iterations": iterations
                })
                
                if next_step == "reflect":
                    # Add reflection before regenerating
                    reflection_state = self.reflect({
                        "messages": messages,
                        "generation": code_solution,
                        "iterations": iterations
                    })
                    messages = reflection_state["messages"]
                
                return self.generate(messages, iterations + 1)
                
            return check_result
            
        except Exception as e:
            print(f"Error in code generation: {str(e)}")
            return {
                "error": "yes",
                "status": f"Generation failed: {str(e)}"
            }

    def code_check(self, state: GraphState):
        """
        Check code

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """
        if self.verbose:
            print("---CHECKING CODE---")

        # State
        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]

        # Get solution components
        imports = code_solution.imports
        code = code_solution.code

        print(imports + '\n\n' + code)

        # Check imports
        try:
            exec(imports)
        except Exception as e:
            if self.verbose:
                print("---CODE IMPORT CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the import test: {e}")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
                "status": "Failed test"
            }
        
        data_check = self.data_checker.invoke({ "messages": [("user", imports + '\n\n' + code)]}).dict()
        if data_check['fake_data_entries'].lower() == "yes":
            print("Data input check failed")
            messages += [("user", "Your solution failed the data input test: Do NOT make up fake data entries.")]
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
                "status": "Failed test"
            }

        try:
            # Create a temporary file for output
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                output_file = temp_file.name

            # Start process
            process = multiprocessing.Process(
                target=_run_code_wrapper,
                args=(code, imports, self.data, output_file)
            )
            process.start()
            process.join(timeout=self.time_limit * 60)

            if process.is_alive():
                process.terminate()
                process.join()
                if self.verbose:
                    print("---CODE BLOCK CHECK: FAILED---")
                    print("Execution surpassed time limit, please come up with a more efficient implementation.")
                error_message = [("user", "Your solution failed due to time limit: Execution surpassed time limit, please come up with a more efficient implementation.")]
                messages += error_message
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test"
                }

            # Read output from temporary file
            try:
                with open(output_file, 'r') as f:
                    result = json.load(f)
                
                if 'error' in result:
                    if self.verbose:
                        print("---CODE BLOCK CHECK: FAILED---")
                        print(result['traceback'])
                    messages += [("user", f"Your solution failed the code execution test: {result['error']}")]
                    return {
                        "generation": code_solution,
                        "messages": messages,
                        "iterations": iterations,
                        "error": "yes",
                        "status": "Failed test"
                    }
                    
                captured_output = result['output']
                if captured_output == "No output was captured":
                    print("No output was captured")
                    messages += [("user", "Your solution did not produce any output. Make sure to print the p-value.")]
                    return {
                        "generation": code_solution,
                        "messages": messages,
                        "iterations": iterations,
                        "error": "yes",
                        "status": "Failed test"
                    }
            finally:
                # Clean up temporary file
                try:
                    os.unlink(output_file)
                except:
                    pass

        except Exception as e:
            print(f"Process execution error: {str(e)}")
            messages += [("user", f"Your solution failed the code execution test: {str(e)}")]
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
                "status": "Failed test"
            }

        if self.llm_approx:
            if len(captured_output) == 0:
                if self.verbose:
                    print("---NO OUTPUT RETURNED---")
                error_message = [("user", f"Your solution failed the output test, which tests if the result returns any thing for falsification test: " + captured_output)]
                messages += error_message
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test"
                }
            else:
                if self.verbose:
                    print("---NO CODE TEST FAILURES---")

                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "no",
                    "status": "success",
                    "captured_output": captured_output
                }

        else:
            # Check if example produces reasonable output
            checker = self.tool_404_parser_llm.invoke({ "messages": [("user", captured_output)]}).dict()
            output = checker['check_output_error']
            try:
                p_val = float(checker['p_val'])
            except:
                print('Error parsing p_value')
                print(checker['p_val'])
                output = 'No'
                
            if output == 'No':
                if self.verbose:
                    print("---P-value OUTPUT CHECK: FAILED---")
                error_message = [("user", f"Your solution failed the output test, which tests if the result returns any p-value for falsification test: " + captured_output)]
                messages += error_message
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test"
                }

            p_val = float(checker['p_val'])

            if np.isnan(p_val):
                if self.verbose:
                    print("---P-value is nan: FAILED---")
                error_message = [("user", f"Your solution p-value for falsification test is nan: " + captured_output)]
                messages += error_message
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test"
                }

            if p_val == 0:
                if self.verbose:
                    print("---P-value is 0: FAILED---")
                error_message = [("user", f"Your solution p-value for falsification test is exact 0 - supposedly wrong: " + captured_output)]
                messages += error_message
                return {
                    "generation": code_solution,
                    "messages": messages,
                    "iterations": iterations,
                    "error": "yes",
                    "status": "Failed test"
                }

            # No errors
            if self.verbose:
                print("---NO CODE TEST FAILURES---")
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "no",
                "status": "success",
                "captured_output": captured_output,
                "p_val": checker['p_val']
            }

    def go(self, question: str, log: Optional[Dict] = None) -> Dict:
        """
        Execute the code generation workflow.

        Args:
            question (str): The coding task/question to solve
            log (Optional[Dict]): Dictionary for logging the process

        Returns:
            Dict: Result of the code generation process including:
                - generation: Generated code solution
                - messages: Conversation history
                - error: Error status
                - status: Status message
                - captured_output: Output from code execution (if successful)
        """
        if self.verbose:
            print(f"\nProcessing question: {question}")

        try:
            # Configure the graph execution
            config = {"recursion_limit": 500}

            # Initialize the state and run the graph
            initial_state = {
                "messages": [("user", question)],
                "iterations": 0
            }

            # Execute the workflow
            result = self.app.invoke(initial_state, config=config)

            if log is not None:
                # Add workflow execution details to log
                log["workflow"] = {
                    "question": question,
                    "final_state": result
                }

            return result

        except Exception as e:
            error_result = {
                "error": "yes",
                "status": f"Workflow execution failed: {str(e)}",
                "messages": [("user", question)],
                "iterations": 0
            }
            
            if log is not None:
                log["workflow"] = {
                    "question": question,
                    "error": str(e)
                }
            
            return error_result

class FalsificationAgent:
    """Agent for running falsification tests."""
    
    def __init__(self, llm="gpt-4o", port=None, api_key="EMPTY"):
        """Initialize the falsification agent."""
        self.llm = get_llm(llm, port=port, api_key=api_key)
        self.output_parser = self.llm.with_structured_output(OutputSpecification)
        
    def run_falsification_test(self, code: str, data: Dict) -> Dict:
        """Run a falsification test."""
        # Implementation details would go here
        pass

class SequentialFalsificationTest:
    """Orchestrator for running sequential falsification tests."""
    
    def __init__(self, llm="gpt-4o", is_local=False, port=None, api_key="EMPTY"):
        """Initialize the sequential falsification test orchestrator."""
        if is_local:
            assert port is not None, "A server port must be provided when using a locally served model."
        self.port = port
        self.api_key = api_key
        self.llm_use = llm
        self.llm = get_llm(llm, port=self.port, api_key=self.api_key)
        self.output_parser = self.llm.with_structured_output(OutputSpecification)
        
        # Initialize state tracking
        self.num_of_tests = 0
        self.res = False
        self.res_stat = None
        self.tracked_tests = []
        self.tracked_stat = []
        
        # Initialize logging
        self.log = {
            'designer': [],
            'executor': [],
            'relevance_checker': [],
            'summarizer': [],
            'sequential_testing': []
        }
        
    def configure(self, data, alpha=0.1, beta=0.1, aggregate_test='E-value',
                 llm_approx=False, max_num_of_tests=10, time_limit=10,
                 max_retry=10, domain="biology", max_failed_tests=10,
                 relevance_checker=False):
        """Configure the sequential falsification test."""
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.aggregate_test = aggregate_test
        self.llm_approx = llm_approx
        self.max_num_of_tests = max_num_of_tests
        self.domain = domain
        self.max_failed_tests = max_failed_tests
        self.relevance_checker = relevance_checker
        
        # Initialize agents
        self.test_proposer = FalsificationTestProposalAgent(
            data=self.data,
            llm=self.llm_use,
            domain=self.domain,
            port=self.port,
            api_key=self.api_key
        )
        
        self.code_generator = CodeGeneratorAgent(
            data=self.data,
            llm=self.llm_use,
            time_limit=time_limit,
            max_retry=max_retry,
            llm_approx=self.llm_approx,
            domain=self.domain,
            port=self.port,
            api_key=self.api_key
        )
        
        self.falsification_agent = FalsificationAgent(
            llm=self.llm_use,
            port=self.port,
            api_key=self.api_key
        )
        
        # Set up the graph
        self._setup_graph()
        
    def _setup_graph(self):
        """Set up the LangGraph for sequential testing."""
        # Graph setup would go here
        pass
        
    def run(self, hypothesis: str) -> tuple[Dict, str, Dict]:
        """Run the sequential falsification test.
        
        Args:
            hypothesis (str): The hypothesis to test
            
        Returns:
            tuple[Dict, str, Dict]: Results of the test, including:
                - Test statistics and decisions
                - Final conclusion
                - Logging information
        """
        if not hasattr(self, 'data'):
            raise ValueError("Test not configured. Call configure() first.")
            
        print("\nStarting sequential falsification test...")
        print(f"Testing hypothesis: {hypothesis}")
        
        # Initialize test tracking
        self.tracked_tests = []
        self.tracked_stat = []
        self.num_of_tests = 0
        failed_tests = 0
        reject = False
        stat = None
        
        while self.num_of_tests < self.max_num_of_tests and failed_tests < self.max_failed_tests:
            try:
                # Generate test specification
                test_spec = self.test_proposer.go(
                    main_hypothesis=hypothesis,
                    test_results=self.tracked_tests,
                    log=self.log
                )
                print(f"\nProposed test specification:\n{test_spec}")
                
                # Generate and run test code
                test_result = self.code_generator.generate(
                    messages=[("user", f"Generate a statistical test for this test specification:\n{test_spec}")],
                    iterations=0
                )
                
                if test_result.get("error") == "yes":
                    failed_tests += 1
                    self.test_proposer.add_to_failed_tests(test_spec)
                    print(f"Test {self.num_of_tests + 1} failed. Trying again...")
                    continue
                    
                # Extract p-value from test
                p_value = float(test_result.get("p_val", 1.0))
                self.tracked_stat.append(p_value)
                self.tracked_tests.append(test_spec)
                self.test_proposer.add_to_existing_tests(test_spec)
                self.num_of_tests += 1
                
                print(f"\nTest {self.num_of_tests} completed:")
                print(f"p-value: {p_value}")
                
                # Aggregate results based on chosen method
                if len(self.tracked_stat) > 0:
                    if self.aggregate_test == 'E-value':
                        reject, stat = e_value_integral_calibrator(self.tracked_stat, self.alpha)
                    elif self.aggregate_test == 'Fisher':
                        reject, stat = fishers_method(self.tracked_stat, self.alpha)
                    else:
                        raise ValueError(f"Unknown aggregate test method: {self.aggregate_test}")
                        
                    print(f"Current aggregate statistic ({self.aggregate_test}): {stat}")
                    
                    if reject:
                        print("\nNull hypothesis rejected!")
                        break
                    
            except Exception as e:
                print(f"Error in test {self.num_of_tests + 1}: {str(e)}")
                failed_tests += 1
                continue
                
        # Handle case where all tests failed
        if self.num_of_tests == 0:
            results = {
                'num_tests': 0,
                'p_values': [],
                'aggregate_method': self.aggregate_test,
                'aggregate_statistic': None,
                'reject_null': False
            }
            conclusion = """
            Sequential Falsification Test Results:
            -----------------------------------
            All tests failed to execute. Please check the error messages above.
            """
            return results, conclusion, self.log
            
        # Prepare final results
        results = {
            'num_tests': self.num_of_tests,
            'p_values': self.tracked_stat,
            'aggregate_method': self.aggregate_test,
            'aggregate_statistic': stat,
            'reject_null': reject
        }
        
        # Generate conclusion
        conclusion = f"""
        Sequential Falsification Test Results:
        -----------------------------------
        Number of tests run: {self.num_of_tests}
        Aggregate test method: {self.aggregate_test}
        Aggregate statistic: {stat}
        Significance level (alpha): {self.alpha}
        
        Individual p-values: {', '.join([f'{p:.6f}' for p in self.tracked_stat])}
        
        Final decision: {'Reject' if reject else 'Fail to reject'} the null hypothesis
        """
        
        return results, conclusion, self.log 

class FalsificationTestProposalAgent:
    """Agent for proposing falsification tests."""
    
    def __init__(self, data, llm='claude-3-5-sonnet-20241022', domain="biology", port=None, api_key="EMPTY"):
        """Initialize the falsification test proposal agent.
        
        Args:
            data: The data to test against
            llm (str): Language model to use
            domain (str): Domain of testing (e.g. "biology")
            port (Optional[int]): Port for local model
            api_key (str): API key for hosted models
        """
        self.data = data
        self.llm = get_llm(llm, port=port, api_key=api_key)
        self.domain = domain
        self.existing_tests = []
        self.failed_tests = []
        
        # Set up prompts and chains
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", get_test_proposal_agent_system_prompt(self.domain)), 
            ("human", "{input}")
        ])
        self.chain = self.system_prompt | self.llm.with_structured_output(TestSpecification)
        self.output_parser = self.llm.with_structured_output(TestSpecification)

    def go(self, main_hypothesis: str, test_results=None, log=None) -> str:
        """Generate a new falsification test proposal.
        
        Args:
            main_hypothesis (str): The main hypothesis to test
            test_results (Optional[List]): Previous test results
            log (Optional[Dict]): Logging dictionary
            
        Returns:
            str: Formatted test specification question
        """
        if not test_results:
            test_results = self.existing_tests
            
        # Get prompt for test proposal
        prompt_modifier = get_test_proposal_agent_user_prompt(
            self.domain, 
            main_hypothesis, 
            self.data, 
            test_results, 
            self.failed_tests
        )

        # Create and run agent
        self.app = create_react_agent(self.llm, [])
        config = {"recursion_limit": 500}
        inputs = {"messages": [("user", prompt_modifier)]}
        
        # Stream agent outputs
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            pattern = r"={32}\x1b\[1m (Ai|Human) Message \x1b\[0m={32}"
            clean_out = re.sub(pattern, '', out)
            if log is not None:
                log['designer'].append(clean_out)
        
        # Parse output with retries
        res = None
        for _ in range(10):
            try:
                res = self.output_parser.invoke(s["messages"][-1].content)
                if res:
                    break
            except Exception:
                continue
                
        if not res:
            raise ValueError("Failed to parse test specification after 10 retries")
        
        # Format question
        question = (
            f"Main hypothesis: {main_hypothesis}\n"
            f"Falsification Test name: {res.test_name}\n"
            f"Falsification Test description: {res.test_description}\n"
            f"Falsification Test Null sub-hypothesis: {res.null_hypothesis}\n"
            f"Falsification Test Alternate sub-hypothesis: {res.alternate_hypothesis}"
        )
        return question

    def add_to_existing_tests(self, test: str) -> None:
        """Add a test to the list of existing tests.
        
        Args:
            test (str): Test to add
        """
        self.existing_tests.append(test)

    def add_to_failed_tests(self, test: str) -> None:
        """Add a test to the list of failed tests.
        
        Args:
            test (str): Test to add
        """
        self.failed_tests.append(test) 