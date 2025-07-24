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
        return "🟢 Agent initialized successfully", gr.update(interactive=True), gr.update(visible=True)
    except Exception as e:
        return f"❌ Initialization failed: {e}", gr.update(interactive=False), gr.update(visible=False)

def chat_with_agent(user_message):
    global agent, chat_history
    if not agent:
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": "Please initialize the agent first by configuring your settings above."})
        return chat_history, ""
    
    chat_history.append({"role": "user", "content": user_message})
    try:
        response = agent.go(user_message)
        cleaned = clean_agent_output(response)
        chat_history.append({"role": "assistant", "content": cleaned})
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"I encountered an error: {str(e)}"})
    return chat_history, ""

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
    if not chat_history:
        return None
    buffer = save_chat_to_pdf(chat_history)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode="wb") as f:
        f.write(buffer.getbuffer())
        return f.name

def handle_example_click(example_text):
    return example_text

# Enhanced CSS inspired by Claude's design
custom_css = """
/* Global theme */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Header styling */
.header-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}

.header-section h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(45deg, #ffffff, #e0e7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-section p {
    font-size: 1.1rem;
    opacity: 0.9;
    margin: 0;
}

/* Configuration section */
.config-section {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.config-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Status indicator */
.status-success {
    background: #dcfce7 !important;
    border: 1px solid #22c55e !important;
    color: #15803d !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

.status-error {
    background: #fef2f2 !important;
    border: 1px solid #ef4444 !important;
    color: #dc2626 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

.status-pending {
    background: #fef3c7 !important;
    border: 1px solid #f59e0b !important;
    color: #d97706 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* Primary button */
.primary-btn {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2) !important;
}

.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

/* Secondary buttons */
.secondary-btn {
    background: white !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.secondary-btn:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    transform: translateY(-1px) !important;
}

/* Danger button */
.danger-btn {
    background: #ef4444 !important;
    color: white !important;
    border: none !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.danger-btn:hover {
    background: #dc2626 !important;
    transform: translateY(-1px) !important;
}

/* Example prompts */
.examples-section {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.example-btn {
    background: #f1f5f9 !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin: 0.5rem !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    font-size: 0.9rem !important;
    line-height: 1.4 !important;
}

.example-btn:hover {
    background: #e2e8f0 !important;
    border-color: #cbd5e1 !important;
    transform: translateY(-1px) !important;
}

/* Chat interface */
.chat-container {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* Input styling */
.chat-input {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    font-size: 1rem !important;
    transition: border-color 0.2s ease !important;
}

.chat-input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Dropdown styling */
.dropdown {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    background: white !important;
}

/* Hide default gradio styling */
.gradio-container .prose {
    max-width: none !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 1rem !important;
    }
    
    .header-section h1 {
        font-size: 2rem !important;
    }
    
    .config-section, .examples-section {
        padding: 1rem !important;
    }
}
"""

# Create the interface
demo = gr.Blocks(css=custom_css, title="BiOmni Agent", theme=gr.themes.Default())

with demo:
    # Header
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="header-section">
                    <h1>🧬 BiOmni Agent</h1>
                    <p>Advanced biomedical AI assistant for research and analysis</p>
                </div>
            """)
    
    # Configuration Section
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="config-title">⚙️ Configuration</div>')
            with gr.Row():
                with gr.Column(scale=3):
                    data_path_input = gr.Textbox(
                        label="Data Path",
                        value="./data",
                        placeholder="Enter path to your data directory",
                        elem_classes=["chat-input"]
                    )
                with gr.Column(scale=3):
                    llm_model_input = gr.Dropdown(
                        label="Language Model",
                        choices=llm_models,
                        value=llm_models[0],
                        elem_classes=["dropdown"]
                    )
                with gr.Column(scale=2):
                    init_btn = gr.Button(
                        "🚀 Initialize Agent",
                        elem_classes=["primary-btn"],
                        size="lg"
                    )
            
            status = gr.Textbox(
                label="Status",
                value="⏳ Ready to initialize",
                interactive=False,
                elem_classes=["status-pending"]
            )
    
    # Example Prompts Section
    with gr.Row(visible=False) as examples_row:
        with gr.Column():
            gr.HTML('<div class="config-title">💡 Example Prompts</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    example_btn_1 = gr.Button(
                        example_queries[0],
                        elem_classes=["example-btn"]
                    )
                with gr.Column(scale=1):
                    example_btn_2 = gr.Button(
                        example_queries[1], 
                        elem_classes=["example-btn"]
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    example_btn_3 = gr.Button(
                        example_queries[2],
                        elem_classes=["example-btn"]
                    )
                with gr.Column(scale=1):
                    example_btn_4 = gr.Button(
                        example_queries[3],
                        elem_classes=["example-btn"]
                    )
    
    # Chat Interface
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(
                label="Chat with BiOmni Agent",
                type="messages",
                height=500,
                show_copy_button=True,
                elem_classes=["chat-container"]
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask me anything about biomedical research...",
                    label="",
                    scale=4,
                    interactive=False,
                    elem_classes=["chat-input"]
                )
                send_btn = gr.Button(
                    "Send",
                    scale=1,
                    elem_classes=["primary-btn"],
                    interactive=False
                )
            
            with gr.Row():
                clear_btn = gr.Button(
                    "🗑️ Clear Chat",
                    elem_classes=["secondary-btn"]
                )
                save_pdf_btn = gr.Button(
                    "📄 Export PDF",
                    elem_classes=["secondary-btn"]
                )
                
    # Hidden file output
    pdf_file = gr.File(label="Download Chat History", visible=False)
    
    # Event handlers
    init_btn.click(
        fn=initialize_agent,
        inputs=[data_path_input, llm_model_input],
        outputs=[status, msg_input, examples_row]
    )
    
    send_btn.click(
        fn=chat_with_agent,
        inputs=[msg_input],
        outputs=[chatbot, msg_input]
    )
    
    msg_input.submit(
        fn=chat_with_agent,
        inputs=[msg_input], 
        outputs=[chatbot, msg_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot]
    )
    
    save_pdf_btn.click(
        fn=export_chat_history,
        outputs=[pdf_file]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[pdf_file]
    )
    
    # Example button handlers
    example_btn_1.click(
        fn=handle_example_click,
        inputs=[gr.State(example_queries[0])],
        outputs=[msg_input]
    )
    example_btn_2.click(
        fn=handle_example_click,
        inputs=[gr.State(example_queries[1])],
        outputs=[msg_input]
    )
    example_btn_3.click(
        fn=handle_example_click,
        inputs=[gr.State(example_queries[2])],
        outputs=[msg_input]
    )
    example_btn_4.click(
        fn=handle_example_click,
        inputs=[gr.State(example_queries[3])],
        outputs=[msg_input]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        #server_name="0.0.0.0",
        server_port=7862,
        show_error=True
    )