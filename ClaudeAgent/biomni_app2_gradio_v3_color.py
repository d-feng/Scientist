import gradio as gr
import os
from biomni.agent import A1
from dotenv import load_dotenv
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from datetime import datetime
import tempfile

# Load API key
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Global state
agent = None
chat_history = []

llm_models = [
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
    "gpt-4",
    "gpt-3.5-turbo"
]

example_queries = [
    "Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.",
    "Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis",
    "Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "What are the key factors involved in cancer drug resistance mechanisms?"
]

def clean_agent_output(response):
    response_str = str(response)
    if response_str.startswith("(['") and response_str.endswith("'])"):
        response_str = response_str[2:-2]
    elif response_str.startswith("(") and response_str.endswith(")"):
        response_str = response_str[1:-1]
        if response_str.startswith("'") and response_str.endswith("'"):
            response_str = response_str[1:-1]
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
    return '\n'.join(cleaned).strip()

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
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "❗ Please initialize the agent first."})
        return chat_history
    chat_history.append({"role": "user", "content": user_message})
    try:
        response = agent.go(user_message)
        cleaned = clean_agent_output(response)
        chat_history.append({"role": "assistant", "content": cleaned})
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
    return chat_history

def clear_chat():
    global chat_history
    chat_history = []
    return []

def save_chat_to_pdf(history):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica", 12)

    for entry in history:
        role = entry["role"].capitalize()
        content = entry["content"]
        for line in f"{role}: {content}".split('\n'):
            c.drawString(40, y, line.strip())
            y -= 15
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 12)

    c.save()
    buffer.seek(0)
    return buffer

def export_chat_history():
    buffer = save_chat_to_pdf(chat_history)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb") as f:
        f.write(buffer.getbuffer())
        return f.name

def clear_input(_):
    return ""

def handle_example_selection(prompt):
    return "", chat_with_agent(prompt)

# Custom CSS for buttons
custom_css = """
.green-btn {
    background-color: #00E47C !important;
    color: #2D2F45 !important;
    font-weight: bold;
    border-radius: 6px !important;
}
.green-btn:hover {
    background-color: #43FFAA !important;
}
"""

# Main interface
demo = gr.Blocks(css=custom_css)

with demo:
    gr.Markdown("## 🧬 BiOmni Agent Chat Interface")
    gr.Markdown("🚀 Initialize the agent and start chatting with biomedical queries.")

    with gr.Row():
        data_path_input = gr.Textbox(label="Data Path", value="./data")
        llm_model_input = gr.Dropdown(label="LLM Model", choices=llm_models, value=llm_models[0])
        init_btn = gr.Button("🚀 Initialize Agent")
    status = gr.Textbox(label="Agent Status", value="🟡 Not Initialized", interactive=False)

    gr.Markdown("### 💡 Choose an Example Prompt")
    example_selector = gr.Dropdown(label="Example Queries", choices=example_queries, value=None)

    with gr.Row():
        chatbot = gr.Chatbot(label="Chat", type="messages")
        msg_input = gr.Textbox(placeholder="Enter your biomedical query here...", label="Chat Input")

    with gr.Row():
        send_btn = gr.Button("Send", elem_classes=["green-btn"])
        clear_btn = gr.Button("🗑️ Clear Chat" )
        save_pdf_btn = gr.Button("💾 Save Chat as PDF")
    pdf_file = gr.File(label="Download PDF")

    # Actions
    init_btn.click(fn=initialize_agent, inputs=[data_path_input, llm_model_input], outputs=[status, msg_input])
    send_btn.click(fn=chat_with_agent, inputs=[msg_input], outputs=[chatbot])
    send_btn.click(fn=clear_input, inputs=[msg_input], outputs=[msg_input])
    msg_input.submit(fn=chat_with_agent, inputs=[msg_input], outputs=[chatbot])
    msg_input.submit(fn=clear_input, inputs=[msg_input], outputs=[msg_input])
    clear_btn.click(fn=clear_chat, outputs=[chatbot])
    save_pdf_btn.click(fn=export_chat_history, outputs=[pdf_file])
    example_selector.change(fn=handle_example_selection, inputs=example_selector, outputs=[msg_input, chatbot])

if __name__ == "__main__":
    demo.launch()
