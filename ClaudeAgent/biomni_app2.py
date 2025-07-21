import streamlit as st
import os
from biomni.agent import A1
import time
from datetime import datetime
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def clean_agent_output(response):
    """Clean and format the agent response for better readability"""
    # Convert to string if it's not already
    response_str = str(response)
    
    # Remove the outer tuple/list formatting if present
    if response_str.startswith("(['") and response_str.endswith("'])"):
        response_str = response_str[2:-2]
    elif response_str.startswith("(") and response_str.endswith(")"):
        # Handle tuple formatting
        response_str = response_str[1:-1]
        if response_str.startswith("'") and response_str.endswith("'"):
            response_str = response_str[1:-1]
    
    # Remove AI message headers and separators
    response_str = re.sub(r'={30,}\s*Ai Message\s*={30,}\\n\\n', '', response_str)
    response_str = re.sub(r'={30,}.*?={30,}', '', response_str)
    
    # Replace escaped newlines with actual newlines
    response_str = response_str.replace('\\n', '\n')
    
    # Replace escaped quotes
    response_str = response_str.replace("\\'", "'")
    response_str = response_str.replace('\\"', '"')
    
    # Remove duplicate content (sometimes the response is repeated)
    lines = response_str.split('\n')
    seen_lines = set()
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and line not in seen_lines:
            seen_lines.add(line)
            cleaned_lines.append(line)
    
    # Join back and clean up extra whitespace
    cleaned_response = '\n'.join(cleaned_lines)
    
    # Remove excessive blank lines
    cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response)
    
    return cleaned_response.strip()

# Page configuration
st.set_page_config(
    page_title="BiOmni Agent Chat",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    .agent-message {
        background-color: #f5f5f5;
        margin-right: auto;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
    .example-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'agent' not in st.session_state:
    st.session_state.agent = None

if 'agent_initialized' not in st.session_state:
    st.session_state.agent_initialized = False

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Data path configuration
    data_path = st.text_input(
        "Data Path",
        value="./data",
        help="Path where BiOmni data will be stored (~11GB will be downloaded on first run)"
    )
    
    # LLM model selection
    llm_model = st.selectbox(
        "LLM Model",
        options=[
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "gpt-4",
            "gpt-3.5-turbo"
        ],
        index=0,
        help="Choose the language model for the agent"
    )
    
    # API Key status
    st.subheader("🔑 API Configuration")
    if ANTHROPIC_API_KEY:
        st.markdown('<div class="status-box status-success">✅ API Key Loaded from .env</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-error">❌ API Key not found in .env file</div>', unsafe_allow_html=True)
        st.warning("Please add ANTHROPIC_API_KEY to your .env file")
    
    # Initialize/Reinitialize agent button
    if st.button("🚀 Initialize Agent", type="primary", disabled=not ANTHROPIC_API_KEY):
        with st.spinner("Initializing BiOmni Agent..."):
            try:
                # Initialize the agent
                st.session_state.agent = A1(path=data_path, llm=llm_model)
                st.session_state.agent_initialized = True
                st.success("✅ Agent initialized successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")
    
    # Agent status
    st.subheader("Agent Status")
    if st.session_state.agent_initialized:
        st.markdown('<div class="status-box status-success">🟢 Agent Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-warning">🟡 Agent Not Initialized</div>', unsafe_allow_html=True)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main content
st.markdown('<h1 class="main-header">🧬 BiOmni Agent Chat Interface</h1>', unsafe_allow_html=True)

# Example queries section
with st.expander("📋 Example Queries", expanded=False):
    st.markdown("""
    <div class="example-box">
    <h4>🧪 CRISPR Screen Planning</h4>
    <code>Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.</code>
    </div>
    
    <div class="example-box">
    <h4>🔬 scRNA-seq Analysis</h4>
    <code>Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis</code>
    </div>
    
    <div class="example-box">
    <h4>💊 ADMET Prediction</h4>
    <code>Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O</code>
    </div>
    
    <div class="example-box">
    <h4>🧬 General Biomedical Query</h4>
    <code>What are the key factors involved in cancer drug resistance mechanisms?</code>
    </div>
    """, unsafe_allow_html=True)

# Chat interface
st.subheader("💬 Chat with BiOmni Agent")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        timestamp = message.get('timestamp', '')
        
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
                <div class="timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Convert markdown content for proper rendering
            content = message["content"]
            st.markdown(f"""
            <div class="chat-message agent-message">
                <strong>🧬 BiOmni Agent:</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Use st.markdown for proper formatting of the content
            st.markdown(content)
            
            st.markdown(f"""
            <div style="text-align: left; margin-left: 1rem;">
                <div class="timestamp">{timestamp}</div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input(
    "Enter your biomedical query here...",
    disabled=not st.session_state.agent_initialized
)

# Process user input
if user_input and st.session_state.agent_initialized:
    # Add user message to chat history
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input,
        "timestamp": current_time
    })
    
    # Show user message immediately
    st.rerun()

# Process agent response (separate from input to avoid double processing)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.agent_initialized:
    user_query = st.session_state.messages[-1]["content"]
    
    with st.spinner("🤖 BiOmni Agent is processing your request..."):
        try:
            # Execute the agent query
            response = st.session_state.agent.go(user_query)
            
            # Clean and format the response
            cleaned_response = clean_agent_output(response)
            
            # Add agent response to chat history
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({
                "role": "agent", 
                "content": cleaned_response,
                "timestamp": current_time
            })
            
            st.rerun()
            
        except Exception as e:
            error_msg = f"An error occurred while processing your request: {str(e)}"
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.messages.append({
                "role": "agent", 
                "content": error_msg,
                "timestamp": current_time
            })
            st.rerun()

# Footer information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🧬 BiOmni Agent Chat Interface | Built with Streamlit</p>
    <p>⚠️ Note: The data lake (~11GB) will be downloaded automatically on first run</p>
</div>
""", unsafe_allow_html=True)

# Instructions for first-time users
if not st.session_state.agent_initialized:
    if not ANTHROPIC_API_KEY:
        st.error("""
        🔑 **API Key Required!**
        
        Please create a `.env` file in your project directory with:
        ```
        ANTHROPIC_API_KEY=your_api_key_here
        ```
        Then restart the application.
        """)
    else:
        st.info("""
        👋 **Welcome to BiOmni Agent Chat!**
        
        **To get started:**
        1. Configure your data path and LLM model in the sidebar
        2. Click "🚀 Initialize Agent" to set up the BiOmni agent
        3. Start chatting with biomedical queries!
        
        **Note:** The first initialization will download ~11GB of data for the BiOmni data lake.
        """)