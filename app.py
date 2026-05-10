import os
import gradio as gr
from transformers import pipeline, TextIteratorStreamer
from threading import Thread
import torch

hf_token = os.getenv("HF_TOKEN")
pipe = None
load_error = None

def load_model():
    global pipe, load_error
    if pipe is not None:
        return True
    if not hf_token:
        load_error = "HF_TOKEN secret is not set. Go to Space Settings > Secrets and add HF_TOKEN."
        return False
    try:
        print("Loading MedGemma 1.5 4B...")
        pipe = pipeline(
            "image-text-to-text",
            model="google/medgemma-1.5-4b-it",
            dtype=torch.float16,
            device_map="auto",
            token=hf_token,
            low_cpu_mem_usage=True,
        )
        print("Model loaded.")
        return True
    except Exception as e:
        load_error = str(e)
        print(f"Load error: {e}")
        return False

def analyze_xray(image):
    if not load_model():
        yield f"Model failed to load:\n{load_error}"
        return
    if image is None:
        yield "Please upload a chest X-ray image."
        return

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this chest X-ray."},
            ],
        }
    ]

    model = pipe.model
    processor = pipe.tokenizer

    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)

    thread = Thread(target=model.generate, kwargs={
        **inputs,
        "max_new_tokens": 150,
        "streamer": streamer,
        "do_sample": False,
    })
    thread.start()

    generated = ""
    for token in streamer:
        generated += token
        yield generated


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

body, .gradio-container {
    background: #f0f2f5 !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* ── TOP NAV BAR ── */
#navbar {
    background: #1c1f2e;
    padding: 0 32px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid #7b2fbe;
    margin-bottom: 0;
}

#navbar .nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #ffffff;
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: -0.3px;
}

#navbar .nav-logo span.accent {
    color: #a855f7;
}

#navbar .nav-links {
    display: flex;
    gap: 24px;
    align-items: center;
}

#navbar .nav-links a {
    color: #9ca3af;
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 500;
    transition: color 0.15s;
}

#navbar .nav-links a:hover { color: #ffffff; }

#navbar .nav-badge {
    background: #7b2fbe;
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.3px;
}

/* ── MAIN CONTENT AREA ── */
#main-content {
    background: #f0f2f5;
    padding: 20px 24px 16px;
}

/* ── METRICS HEADER ── */
#metrics-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px 20px;
    background: linear-gradient(135deg, #1c1f2e 0%, #2d1b69 100%);
    border-radius: 12px 12px 0 0;
    margin-bottom: 0;
}

#metrics-header .xray-logo {
    font-size: 2.8rem;
    line-height: 1;
    filter: drop-shadow(0 0 8px rgba(168,85,247,0.6));
}

#metrics-header .header-text h2 {
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 2px 0;
    letter-spacing: -0.3px;
}

#metrics-header .header-text p {
    font-size: 0.8rem;
    color: #9ca3af;
    margin: 0;
}

#metrics-header .tag-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-left: auto;
}

#metrics-header .tag {
    background: rgba(168,85,247,0.2);
    border: 1px solid rgba(168,85,247,0.4);
    border-radius: 6px;
    color: #c084fc;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
}

/* ── PANELS ── */
#upload-panel, #report-panel {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    padding: 20px;
}

#panel-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #7b2fbe;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── CUSTOM PANEL HEADERS ── */
.panel-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 16px 0;
    border-bottom: 2px solid #f0f2f5;
    margin-bottom: 14px;
}

.panel-icon {
    font-size: 2rem;
    line-height: 1;
    flex-shrink: 0;
}

.panel-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #1c1f2e;
    letter-spacing: -0.3px;
    line-height: 1.2;
}

.panel-sub {
    font-size: 0.78rem;
    color: #9ca3af;
    font-weight: 500;
    margin-top: 2px;
}

/* ── BUTTONS ── */
#analyze-btn {
    background: #7b2fbe !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    color: #ffffff !important;
    padding: 16px 0 !important;
    box-shadow: 0 4px 14px rgba(123,47,190,0.4) !important;
    transition: all 0.2s !important;
}

#analyze-btn:hover {
    background: #6d28d9 !important;
    box-shadow: 0 6px 22px rgba(123,47,190,0.55) !important;
    transform: translateY(-1px) !important;
}

#clear-btn {
    border-radius: 8px !important;
    border: 2px solid #7b2fbe !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #7b2fbe !important;
    background: #ffffff !important;
    padding: 16px 0 !important;
    transition: all 0.2s !important;
}

#clear-btn:hover {
    background: #f5f0ff !important;
    box-shadow: 0 4px 14px rgba(123,47,190,0.2) !important;
}

/* ── METRICS ROW ── */
#metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    margin-bottom: 20px;
    border: 1px solid #e2e8f0;
    border-top: none;
    border-radius: 0 0 12px 12px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}

#metrics-row .metric-card {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    padding: 18px 24px;
}

#metrics-row .metric-card:last-child {
    border-right: none;
}

#metrics-row .metric-card .metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #7b2fbe;
    line-height: 1;
    margin-bottom: 4px;
}

#metrics-row .metric-card .metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── FOOTER ── */
#footer {
    background: #1c1f2e;
    border-top: 1px solid #2e2e4a;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 0;
}

#footer .footer-left {
    font-size: 0.8rem;
    color: #6b7280;
}

#footer .footer-left strong { color: #9ca3af; }

#footer .footer-right a {
    color: #a855f7;
    text-decoration: none;
    font-size: 0.8rem;
    font-weight: 600;
}
"""

NAVBAR_HTML = """
<div id="navbar">
    <div class="nav-logo">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <rect x="2" y="12" width="4" height="10" rx="1" fill="#a855f7"/>
            <rect x="8" y="7" width="4" height="15" rx="1" fill="#c084fc"/>
            <rect x="14" y="3" width="4" height="19" rx="1" fill="#7b2fbe"/>
            <rect x="20" y="9" width="4" height="13" rx="1" fill="#a855f7"/>
        </svg>
        Med<span class="accent">Gemma</span>&nbsp;Studio
    </div>
    <div class="nav-links">
        <a href="#">Docs</a>
        <a href="#">Models</a>
        <a href="https://github.com/DrFarooqAi" target="_blank">GitHub</a>
        <span class="nav-badge">v1.5 · 4B</span>
    </div>
</div>
"""

METRICS_HTML = """
<div id="metrics-header">
    <div class="xray-logo">&#129753;</div>
    <div class="header-text">
        <h2>MedGemma X-ray Studio</h2>
        <p>AI-powered chest X-ray analysis &mdash; By Dr. Muhammad Farooq</p>
    </div>
    <div class="tag-row">
        <span class="tag">Google DeepMind</span>
        <span class="tag">Privacy First</span>
        <span class="tag">On-Device</span>
    </div>
</div>
<div id="metrics-row">
    <div class="metric-card">
        <div class="metric-value">4B</div>
        <div class="metric-label">Parameters</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">float16</div>
        <div class="metric-label">Precision</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">100%</div>
        <div class="metric-label">On-Device</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">0</div>
        <div class="metric-label">Data Sent to Cloud</div>
    </div>
</div>
"""

FOOTER_HTML = """
<div id="footer">
    <div class="footer-left">
        <strong>&#9888; Not for clinical use.</strong> Educational and research purposes only.
        Results must not be used for diagnosis or treatment.
    </div>
    <div class="footer-right">
        <a href="https://github.com/DrFarooqAi" target="_blank">Built by Dr. Muhammad Farooq &rarr;</a>
    </div>
</div>
"""

with gr.Blocks(title="MedGemma Studio — Chest X-ray Analysis") as demo:

    gr.HTML(NAVBAR_HTML)

    with gr.Column(elem_id="main-content"):

        gr.HTML(METRICS_HTML)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_id="upload-panel"):
                gr.HTML("""
                <div class="panel-header">
                    <div class="panel-icon">&#129481;</div>
                    <div>
                        <div class="panel-title">Upload Chest X-ray</div>
                        <div class="panel-sub">PNG or JPG &middot; Standard PA view recommended</div>
                    </div>
                </div>
                """)
                image_input = gr.Image(
                    type="pil",
                    label="",
                    height=320,
                    show_label=False,
                )
            with gr.Column(scale=1, elem_id="report-panel"):
                gr.HTML("""
                <div class="panel-header">
                    <div class="panel-icon">&#128203;</div>
                    <div>
                        <div class="panel-title">AI Radiology Report</div>
                        <div class="panel-sub">Generated by MedGemma 1.5 &middot; Not for clinical use</div>
                    </div>
                </div>
                """)
                report_output = gr.Textbox(
                    lines=15,
                    placeholder="Your AI-generated report will appear here...",
                    show_label=False,
                )

        with gr.Row():
            clear_btn = gr.ClearButton(
                components=[image_input, report_output],
                value="Clear",
                elem_id="clear-btn",
                scale=1,
            )
            analyze_btn = gr.Button(
                "Run Analysis",
                variant="primary",
                elem_id="analyze-btn",
                scale=3,
            )

    gr.HTML(FOOTER_HTML)

    analyze_btn.click(fn=analyze_xray, inputs=image_input, outputs=report_output)

demo.launch(theme=gr.themes.Base(), css=CSS)
