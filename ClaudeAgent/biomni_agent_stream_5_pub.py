import gradio as gr
import os
from biomni.agent import A1
from dotenv import load_dotenv
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import threading
import queue
from datetime import datetime
import tempfile
from contextlib import redirect_stdout

# Load API key
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Global state
agent = None
chat_history = []
logs_content = ""

llm_models = [
    "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20241022", 
    "claude-3-opus-20240229",
    "gpt-4",
    "gpt-3.5-turbo"
]

example_queries = [
    {
        "title": "🧬 CRISPR Screen Design",
        "description": "Plan a CRISPR screen to identify genes that regulate T cell exhaustion",
        "prompt": "Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.",
        "category": "Genomics"
    },
    {
        "title": "🔬 scRNA-seq Analysis", 
        "description": "Perform single-cell RNA sequencing annotation and hypothesis generation",
        "prompt": "Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis",
        "category": "Transcriptomics"
    },
    {
        "title": "💊 ADMET Prediction",
        "description": "Predict drug properties for a given compound structure",
        "prompt": "Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "category": "Drug Discovery"
    },
    {
        "title": "🛡️ Drug Resistance",
        "description": "Analyze cancer drug resistance mechanisms",
        "prompt": "What are the key factors involved in cancer drug resistance mechanisms?",
        "category": "Oncology"
    },
    {
        "title": "🧪 Protein Analysis",
        "description": "Analyze protein structure and function relationships",
        "prompt": "Analyze the structure-function relationship of p53 tumor suppressor protein and its role in cancer.",
        "category": "Structural Biology"
    },
    {
        "title": "🦠 Pathway Analysis",
        "description": "Investigate metabolic pathway interactions",
        "prompt": "Analyze the glycolysis pathway and its regulation in cancer cells, including key enzymes and metabolic reprogramming.",
        "category": "Metabolism"
    }
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
    response_str = response_str.replace('\\n', '\n').replace("\\'", "'").replace('\\\"', '"')
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
    """Initialize the underlying agent and enable UI components accordingly."""
    global agent
    try:
        agent = A1(path=data_path, llm=llm_model)
        return (
            "🟢 Agent initialized successfully",
            gr.update(interactive=True),
            gr.update(visible=True),
            gr.update(interactive=True),
        )
    except Exception as e:
        return (
            f"❌ Initialization failed: {e}",
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(interactive=False),
        )


def chat_with_agent_stream(user_message):
    """Chat function that streams the assistant's reply and captures console logs."""
    global agent, chat_history
    if not agent:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "Please initialize the agent first by configuring your settings above."})
        yield chat_history, "", ""
        return

    chat_history.append({"role": "user", "content": user_message})
    yield chat_history, "", ""

    assistant_msg = {"role": "assistant", "content": ""}
    chat_history.append(assistant_msg)

    log_queue = queue.Queue()
    resp_holder = {}

    class QueueWriter(io.TextIOBase):
        def write(self, s):
            log_queue.put(s)
            return len(s)

    def run_agent():
        writer = QueueWriter()
        try:
            with redirect_stdout(writer):
                resp_holder['resp'] = agent.go(user_message)
        except Exception as exc:
            resp_holder['error'] = exc
        finally:
            log_queue.put(None)

    thread = threading.Thread(target=run_agent)
    thread.start()

    logs = ""
    while True:
        chunk = log_queue.get()
        if chunk is None:
            break
        logs += chunk
        yield chat_history, "", logs

    thread.join()
    if 'error' in resp_holder:
        assistant_msg['content'] = f"I encountered an error: {resp_holder['error']}"
    else:
        cleaned = clean_agent_output(resp_holder.get('resp', ''))
        assistant_msg['content'] = cleaned
    
    global logs_content
    logs_content = logs
    yield chat_history, "", logs


def clear_chat():
    global chat_history
    chat_history = []
    return []


def export_chat_history():
    if not chat_history:
        return None
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica", 12)
    for entry in chat_history:
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb") as f:
        f.write(buffer.getbuffer())
        return f.name


def export_logs():
    global logs_content
    content = logs_content or "No logs captured for this conversation."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as f:
        f.write(content.encode("utf-8"))
        return f.name


# Simplified CSS
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

.header-section {
    background: linear-gradient(135deg, #051D19 0%, #051D19 100%);
    padding: 1.5rem;    
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
}

.header-section h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(45deg, #00E47C, #00E47C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.example-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem;
    margin: 0.5rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.example-card:hover {
    background: #f1f5f9;
    border-color: #00E47C;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 228, 124, 0.2);
}

.example-title {
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.5rem;
    display: block;
}

.example-description {
    color: #64748b;
    font-size: 0.875rem;
    margin-bottom: 0.5rem;
}

.example-category {
    background: #00E47C;
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    display: inline-block;
    margin-bottom: 0.5rem;
}
"""

# Create the interface with simplified approach
with gr.Blocks(css=custom_css, title="Biomedical Agent") as demo:
    # Header
    gr.HTML("""
        <div class="header-section">
            <h1>🧬 Biomedical Agent</h1>
            <p>Advanced Biomedical AI assistant for research and analysis</p>
        </div>
    """)
    
    # Configuration Section
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚙️ Configuration")
            with gr.Row():
                with gr.Column(scale=3):
                    data_path_input = gr.Textbox(
                        label="Data Path",
                        value="./data",
                        placeholder="Enter path to your data directory"
                    )
                with gr.Column(scale=3):
                    llm_model_input = gr.Dropdown(
                        label="Language Model",
                        choices=llm_models,
                        value=llm_models[0]
                    )
                with gr.Column(scale=2):
                    init_btn = gr.Button("🚀 Initialize Agent", variant="primary")
            
            status = gr.Textbox(
                label="Status",
                value="⏳ Ready to initialize",
                interactive=False
            )
    
    # Example Prompts Section (simplified)
    with gr.Row(visible=False) as examples_row:
        with gr.Column():
            gr.Markdown("### 💡 Example Prompts")
            
            # Create example buttons in a more compatible way
            example_buttons = []
            with gr.Row():
                for i in range(0, len(example_queries), 2):
                    with gr.Column():
                        for j in range(2):
                            if i + j < len(example_queries):
                                example = example_queries[i + j]
                                btn = gr.Button(
                                    f"{example['title']}\n{example['description']}", 
                                    elem_classes=["example-card"]
                                )
                                example_buttons.append((btn, example['prompt']))
    
    # Chat Interface
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(
                label="Chat with Agent",
                type="messages",
                height=500,
                show_copy_button=True
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask me anything about biomedical research...",
                    label="",
                    scale=4,
                    interactive=False
                )
                send_btn = gr.Button("Send", scale=1, variant="primary", interactive=False)
            
            logs_box = gr.Textbox(
                label="Console Output",
                interactive=False,
                lines=10
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat")
                save_pdf_btn = gr.Button("📄 Export PDF")
                save_log_btn = gr.Button("📜 Export Logs")
                
    # Hidden file outputs
    pdf_file = gr.File(label="Download Chat History", visible=False)
    log_file = gr.File(label="Download Logs", visible=False)
    
    # Event handlers
    init_btn.click(
        fn=initialize_agent,
        inputs=[data_path_input, llm_model_input],
        outputs=[status, msg_input, examples_row, send_btn]
    )
    
    # Example button handlers
    for btn, prompt in example_buttons:
        btn.click(lambda p=prompt: p, outputs=[msg_input])
    
    send_btn.click(
        chat_with_agent_stream,
        inputs=[msg_input],
        outputs=[chatbot, msg_input, logs_box]
    )
    
    msg_input.submit(
        chat_with_agent_stream,
        inputs=[msg_input], 
        outputs=[chatbot, msg_input, logs_box]
    )
    
    clear_btn.click(fn=clear_chat, outputs=[chatbot])
    
    save_pdf_btn.click(fn=export_chat_history, outputs=[pdf_file]).then(
        lambda: gr.update(visible=True), outputs=[pdf_file]
    )
    
    save_log_btn.click(fn=export_logs, outputs=[log_file]).then(
        lambda: gr.update(visible=True), outputs=[log_file]
    )

# Enable queueing
demo.queue()

if __name__ == "__main__":
    demo.launch(share=False, server_port=7863, show_error=True)