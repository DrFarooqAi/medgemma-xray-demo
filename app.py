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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

body, .gradio-container {
    background: #09090f !important;
    color: #ffffff !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* ── NAVBAR ── */
#navbar {
    background: rgba(9,9,15,0.95);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0 32px;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
}

#navbar .nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #ffffff;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: -0.3px;
}

#navbar .k-icon {
    width: 30px;
    height: 30px;
    background: linear-gradient(135deg, #7c3aed, #a855f7, #ec4899);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    font-weight: 900;
    color: white;
    box-shadow: 0 0 16px rgba(124,58,237,0.5);
}

#navbar .nav-center {
    display: flex;
    gap: 28px;
}

#navbar .nav-center a {
    color: rgba(255,255,255,0.55);
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 500;
    transition: color 0.15s;
}

#navbar .nav-center a:hover { color: #ffffff; }

#navbar .nav-right .nav-tag {
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.4);
    color: #c084fc;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 5px 14px;
    border-radius: 20px;
}

/* ── HERO ── */
#hero {
    background: #09090f;
    padding: 64px 32px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

#hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 300px;
    background: radial-gradient(ellipse, rgba(124,58,237,0.18) 0%, transparent 70%);
    pointer-events: none;
}

#hero .hero-glow-icon {
    font-size: 4rem;
    display: block;
    margin: 0 auto 20px;
    filter: drop-shadow(0 0 24px rgba(168,85,247,0.7)) drop-shadow(0 0 48px rgba(236,72,153,0.4));
}

#hero h1 {
    font-size: 3rem;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 16px 0;
    letter-spacing: -1.2px;
    line-height: 1.1;
}

#hero h1 span {
    background: linear-gradient(90deg, #a78bfa, #ec4899, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

#hero .hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.6);
    margin: 0 auto 28px;
    max-width: 480px;
    line-height: 1.6;
}

#hero .hero-pills {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
}

#hero .hero-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    color: rgba(255,255,255,0.7);
    font-size: 0.78rem;
    font-weight: 500;
    padding: 6px 16px;
}

/* ── STATS BAR ── */
#stats-bar {
    background: #0f0f18;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding: 0 32px;
    display: flex;
    align-items: center;
    gap: 0;
}

#stats-bar .stat-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 14px 28px 14px 0;
    margin-right: 28px;
    border-right: 1px solid rgba(255,255,255,0.07);
}

#stats-bar .stat-item:last-child { border-right: none; }

#stats-bar .stat-label {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.75);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

#stats-bar .stat-value {
    font-size: 0.95rem;
    font-weight: 800;
    color: #ffffff;
}

#stats-bar .stat-value.green { color: #4ade80; }
#stats-bar .stat-value.purple { color: #a78bfa; }

/* ── MAIN CONTENT ── */
#main-content {
    background: #09090f;
    padding: 28px 24px 32px;
}

/* ── PANELS — ClickUp card style ── */
#upload-panel, #report-panel {
    background: #111118;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow:
        0 0 0 1px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.05),
        0 32px 64px rgba(0,0,0,0.6);
    padding: 24px;
    transition: border-color 0.2s;
}

#upload-panel:hover { border-color: rgba(124,58,237,0.3); }
#report-panel:hover { border-color: rgba(236,72,153,0.25); }

.panel-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.chip-upload {
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.35);
    color: #a78bfa;
}

.chip-report {
    background: rgba(236,72,153,0.12);
    border: 1px solid rgba(236,72,153,0.3);
    color: #f472b6;
}

.panel-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -0.3px;
    margin-bottom: 4px;
}

.panel-sub {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.75);
    font-weight: 400;
    margin-bottom: 16px;
}

/* Gradio dark overrides */
.gradio-container textarea {
    background: #0d0d14 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
}

.gradio-container textarea::placeholder { color: rgba(255,255,255,0.55) !important; }

/* ── BUTTONS ── */
#analyze-btn {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    padding: 15px 0 !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.45) !important;
    transition: all 0.2s !important;
}

#analyze-btn:hover {
    box-shadow: 0 8px 28px rgba(124,58,237,0.65) !important;
    transform: translateY(-2px) !important;
}

#clear-btn {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,0.6) !important;
    padding: 15px 0 !important;
    transition: all 0.2s !important;
}

#clear-btn:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
}

/* ── FOOTER ── */
#footer {
    background: #0f0f18;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

#footer .footer-left {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.4);
}

#footer .footer-left strong { color: rgba(255,255,255,0.7); }

#footer .footer-right a {
    color: #a78bfa;
    text-decoration: none;
    font-size: 0.78rem;
    font-weight: 600;
}

#footer .footer-right a:hover { color: #c084fc; }
"""

NAVBAR_HTML = """
<div id="navbar">
    <div class="nav-logo">
        <div class="k-icon">M</div>
        MedGemma&nbsp;Studio
    </div>
    <div class="nav-center">
        <a href="#">Overview</a>
        <a href="#">Models</a>
        <a href="https://github.com/DrFarooqAi" target="_blank">GitHub</a>
        <a href="https://huggingface.co/farooqgenai" target="_blank">HuggingFace</a>
    </div>
    <div class="nav-right">
        <span class="nav-tag">MedGemma 1.5 &middot; 4B</span>
    </div>
</div>
"""

HERO_HTML = """
<div id="hero">
    <span class="hero-glow-icon">&#129753;</span>
    <h1>The AI that reads<br><span>your chest X-rays</span></h1>
    <p class="hero-sub">Upload a chest X-ray and get a streaming AI radiology report — instantly, on-device, privately.</p>
    <div class="hero-pills">
        <span class="hero-pill">&#128274; 100% On-Device</span>
        <span class="hero-pill">&#9889; Streaming Output</span>
        <span class="hero-pill">&#127775; MedGemma 1.5 &middot; 4B</span>
        <span class="hero-pill">&#128203; Google DeepMind</span>
    </div>
</div>
"""

STATS_BAR_HTML = """
<div id="stats-bar">
    <div class="stat-item">
        <span class="stat-label">Model</span>
        <span class="stat-value purple">MedGemma 1.5</span>
    </div>
    <div class="stat-item">
        <span class="stat-label">Parameters</span>
        <span class="stat-value">4B</span>
    </div>
    <div class="stat-item">
        <span class="stat-label">Precision</span>
        <span class="stat-value">float16</span>
    </div>
    <div class="stat-item">
        <span class="stat-label">On-Device</span>
        <span class="stat-value green">100%</span>
    </div>
    <div class="stat-item">
        <span class="stat-label">Data to Cloud</span>
        <span class="stat-value green">0 bytes</span>
    </div>
    <div class="stat-item">
        <span class="stat-label">Developer</span>
        <span class="stat-value">Google DeepMind</span>
    </div>
</div>
"""

FOOTER_HTML = """
<div id="footer">
    <div class="footer-left">
        <strong>&#9888; Not for clinical use.</strong> Educational and research purposes only.
    </div>
    <div class="footer-right">
        <a href="https://github.com/DrFarooqAi" target="_blank">Built by Dr. Muhammad Farooq &rarr;</a>
    </div>
</div>
"""

with gr.Blocks(title="MedGemma Studio — Chest X-ray Analysis") as demo:

    gr.HTML(NAVBAR_HTML)
    gr.HTML(HERO_HTML)
    gr.HTML(STATS_BAR_HTML)

    with gr.Column(elem_id="main-content"):

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_id="upload-panel"):
                gr.HTML("""
                <div>
                    <div class="panel-chip chip-upload">&#129481; X-ray Input</div>
                    <div class="panel-title">Upload Chest X-ray</div>
                    <div class="panel-sub">PNG or JPG &middot; Standard PA view recommended</div>
                </div>
                """)
                image_input = gr.Image(
                    type="pil",
                    height=300,
                    show_label=False,
                )
            with gr.Column(scale=1, elem_id="report-panel"):
                gr.HTML("""
                <div>
                    <div class="panel-chip chip-report">&#128203; AI Report</div>
                    <div class="panel-title">AI Radiology Report</div>
                    <div class="panel-sub">Streamed token by token &middot; Not for clinical use</div>
                </div>
                """)
                report_output = gr.Textbox(
                    lines=14,
                    placeholder="Your AI-generated report will stream here...",
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
