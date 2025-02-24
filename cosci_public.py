import streamlit as st
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import List, Literal, Dict, Tuple, Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Initialize LLM
API_KEY = ''
openai = ChatOpenAI(model="gpt-4o", openai_api_key=API_KEY)

# Define Agent State
class AgentState(TypedDict):
    expected_result: str
    research_topic: str
    expectation: str
    hypothesis: str
    feedback: str
    evaluation: float
    iteration: int
    continue_loop: bool

# Define Nodes
class SupervisorAgent:
    def __call__(self, state):
        topic = state['research_topic']
        expectation = state['expectation']
        state['expected_result'] = (
            f"Generate a scientifically testable hypothesis for the research topic: {topic} "
            f"that meets the following expectation: {expectation}."
        )
        return state

class HypothesisGenerationAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state):
        prompt = (
            f"Research Topic: {state['research_topic']}\n"
            f"Expectation: {state['expectation']}\n\n"
            "Formulate a clear, novel, and scientifically sound hypothesis."
        )
        response = self.llm(messages=[HumanMessage(role="user", content=prompt)])
        state['hypothesis'] = response.content
        return state

class ReviewerNode:
    def __init__(self, llm):
        self.llm = llm
    
    def __call__(self, state):
        prompt = (
            f"Expected Result: {state['expected_result']}\n"
            f"Generated Hypothesis: {state['hypothesis']}\n\n"
            "Please provide feedback and rate the hypothesis on a scale from 0 to 1."
        )
        response = self.llm(messages=[HumanMessage(role="user", content=prompt)])
        feedback, score = self.parse_feedback(response.content)
        state['feedback'] = feedback
        state['evaluation'] = score
        return state
    
    def parse_feedback(self, content):
        lines = content.splitlines()
        feedback = ''
        score = 0.0
        for line in lines:
            if line.startswith("Feedback:"):
                feedback = line.replace("Feedback:", "").strip()
            elif line.startswith("Evaluation Score:"):
                try:
                    score = float(line.replace("Evaluation Score:", "").strip())
                except ValueError:
                    score = 0.0
        return feedback, score

class LoopControlNode:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
    
    def __call__(self, state):
        state['iteration'] += 1
        state['continue_loop'] = state['iteration'] < self.max_iterations
        return state

# Graph Construction
graph_builder = StateGraph(AgentState)

graph_builder.add_node("supervisor", SupervisorAgent())
graph_builder.add_node("hypothesis_generator", HypothesisGenerationAgent(openai))
graph_builder.add_node("reviewer", ReviewerNode(openai))
graph_builder.add_node("loop_control", LoopControlNode(max_iterations=3))

# Define edges
graph_builder.add_edge(START, "supervisor")
graph_builder.add_edge("supervisor", "hypothesis_generator")
graph_builder.add_edge("hypothesis_generator", "reviewer")
graph_builder.add_edge("reviewer", "loop_control")

def next_node(state):
    return "supervisor" if state['continue_loop'] else END

graph_builder.add_conditional_edges("loop_control", next_node)

graph = graph_builder.compile()

# Streamlit App
st.title("Advanced Research Hypothesis Generator with LangGraph")
st.write("Enter your research topic and expectations to generate and refine hypotheses using an AI-driven multi-agent system.")

# User Inputs
research_topic = st.text_input("Research Topic:", placeholder="e.g., Effects of a novel drug on AML cell proliferation")
expectation = st.text_area("Research Expectation:", placeholder="e.g., The drug should inhibit cell growth in AML cell lines.")

# Generate Hypothesis
if st.button("Generate Hypothesis"):
    if research_topic and expectation:
        with st.spinner("Generating and refining hypothesis..."):
            initial_state = AgentState(
                research_topic=research_topic,
                expectation=expectation,
                hypothesis="",
                feedback="",
                evaluation=0.0,
                iteration=0,
                continue_loop=True
            )
            final_state = graph.invoke(initial_state)
            st.success("Final Hypothesis Generated:")
            st.write(final_state['hypothesis'])
            st.info(f"Final Evaluation Score: {final_state['evaluation']}")
            st.write("Feedback:", final_state['feedback'])
    else:
        st.error("Please provide both research topic and expectation.")

# Sidebar Configuration
st.sidebar.title("Configuration")
st.sidebar.write("API Key: [Hidden for Security]")
st.sidebar.write("Model: GPT-4o")
st.sidebar.write("Max Iterations: 3")
