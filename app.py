import os
import gradio as gr
from PIL import Image
from transformers import pipeline
import torch

# Get token from environment (set via PowerShell)
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("❌ HF_TOKEN not found. Run in PowerShell: $env:HF_TOKEN='your_token'")

print("⏳ Loading MedGemma 1.5 on CPU...")
pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.float32,
    device="cpu",
    token=hf_token,
    low_cpu_mem_usage=True,
)

def analyze_xray(image):
    if image is None:
        return "⚠️ Please upload an X-ray."
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this chest X-ray."}]}]
    output = pipe(text=messages, max_new_tokens=300)
    return output[0]["generated_text"][-1]["content"] + "\n\n---\n⚠️ Demo only."

demo = gr.Interface(fn=analyze_xray, inputs=gr.Image(type="pil"), outputs=gr.Textbox())
demo.launch(inbrowser=True)