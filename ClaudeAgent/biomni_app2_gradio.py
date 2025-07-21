import gradio as gr
import os
from biomni.agent import A1
from datetime import datetime
from dotenv import load_dotenv
import re

# Load API key
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Agent and state
agent = None
chat_history = []

# Available LLM models
llm_models = [
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "gpt-4",
    "gpt-3.5-turbo"
]

def clean_agent_output(response):
    response_str = str(response)
    if response_str.startswith("(['") and response_str.endswith("'])"):
        response_str = response_str[2:-2]
    elif response_str.startswith("(") and response_str.endswith(")"):
        response_str = response_str[1:-1]
        if response_str.startswith("'") and response_str.endswith("'"):
            response_str = response_str[1:-1]
    response_str = re.sub(r'={30,}\s*Ai Message\s*={30,}\\n\\n', '', response_str)
    response_str = re.sub(r'={30,}.*?={30,}', '', response_str)
    response_str = response_str.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
    lines = response_str.split('\n')
    seen = set()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            cleaned.append(line)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(cleaned)).strip()

def initialize_agent(data_path, llm_model):
    global agent
    try:
        agent = A1(path=data_path, llm=llm_model)
        return "🟢 Agent initialized successfully!", gr.update(interactive=True)
    except Exception as e:
        return f"❌ Initialization failed: {e}", gr.update(interactive=False)

def chat_with_agent(user_message):
    global agent, chat_history
    if not agent:
        return chat_history + [[user_message, "❗ Please initialize the agent first."]]
    chat_history.append([user_message, None])
    try:
        response = agent.go(user_message)
        cleaned = clean_agent_output(response)
        chat_history[-1][1] = cleaned
    except Exception as e:
        chat_history[-1][1] = f"Error: {str(e)}"
    return chat_history

def clear_chat():
    global chat_history
    chat_history = []
    return []

with gr.Blocks(title="🧬 BiOmni Agent Chat") as demo:
    gr.Markdown("## 🧬 BiOmni Agent Chat Interface")
    gr.Markdown("🚀 Initialize the agent and start chatting with biomedical queries.")

    with gr.Row():
        data_path_input = gr.Textbox(label="Data Path", value="./data")
        llm_model_input = gr.Dropdown(label="LLM Model", choices=llm_models, value=llm_models[0])
        init_btn = gr.Button("🚀 Initialize Agent")
    status = gr.Textbox(label="Agent Status", value="🟡 Not Initialized", interactive=False)

    with gr.Row():
        chatbot = gr.Chatbot()
        msg_input = gr.Textbox(placeholder="Enter your biomedical query here...", label="Chat Input")
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("🗑️ Clear Chat")

    # Actions
    init_btn.click(fn=initialize_agent, inputs=[data_path_input, llm_model_input], outputs=[status, msg_input])
    send_btn.click(fn=chat_with_agent, inputs=[msg_input], outputs=[chatbot])
    msg_input.submit(fn=chat_with_agent, inputs=[msg_input], outputs=[chatbot])
    clear_btn.click(fn=clear_chat, outputs=[chatbot])

    gr.Markdown("#### 💡 Example Queries")
    gr.Markdown("""
    - **CRISPR Planning**: `Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.`
    - **scRNA-seq Analysis**: `Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis`
    - **ADMET Prediction**: `Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O`
    - **Drug Resistance**: `What are the key factors involved in cancer drug resistance mechanisms?`
    """)

demo.launch()
