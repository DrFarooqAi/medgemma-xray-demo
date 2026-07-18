# MedGemma 1.5 — Chest X-ray Analysis

AI-powered chest X-ray reporting using Google's open-source **MedGemma 1.5 4B** model. Runs entirely on-device — no data sent to the cloud.

> **Not for clinical use. Educational and research purposes only.**

---

## Live Demo

**Try it instantly — no setup needed:**

[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg.svg)](https://huggingface.co/spaces/farooqgenai/medgemma-xray-demo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrFarooqAi/medgemma-xray-demo/blob/main/medgemma_colab_demo.ipynb)

**Direct link:** https://huggingface.co/spaces/farooqgenai/medgemma-xray-demo

**Run on a free GPU:** open [`medgemma_colab_demo.ipynb`](medgemma_colab_demo.ipynb) in Google Colab (T4) for fast, streaming reports.

---

## How to Use the Demo

1. **Open the Space** using the link above
2. **Wait for the model to load** — first load takes 3–5 minutes (progress shown in logs)
3. **Upload a chest X-ray** — click the upload area or drag and drop a PNG/JPG file
4. **Click "Run Analysis"**
5. **Read the report** — it streams word by word into the report box on the right
6. **Click "Clear"** to reset and analyze another image

> The report appears token by token (like ChatGPT) — you don't need to wait for the full response before reading.

---

## What It Does

- Upload any chest X-ray (PNG/JPG)
- Get an AI-generated radiology-style report
- Streaming output — results appear immediately as they generate
- 100% on-device — your images never leave the server
- Built with open-source MedGemma 1.5 4B (Google DeepMind)

---

## Tech Stack

| Component | Detail |
|---|---|
| Model | `google/medgemma-1.5-4b-it` |
| Precision | float16 |
| UI | Gradio 6 |
| Hosting | Hugging Face Spaces (CPU-basic) |
| Output | Streaming via `TextIteratorStreamer` |

---

## Run Locally

### Prerequisites

1. Request model access: [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
2. Get a Hugging Face **Read** token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Python 3.10+ with 16GB+ RAM (32GB recommended for float32)

### Steps

```bash
# Clone
git clone https://github.com/DrFarooqAi/medgemma-xray-demo.git
cd medgemma-xray-demo

# Install dependencies
pip install -r requirements.txt

# Set your HF token
# Windows PowerShell:
$env:HF_TOKEN="hf_your_token_here"

# Linux/Mac:
export HF_TOKEN="hf_your_token_here"

# Run
python app.py
```

Open your browser at **http://127.0.0.1:7860** after launch.

---

## Author

**Dr. Muhammad Farooq** — AI & Healthcare Technology Specialist

- GitHub: [DrFarooqAi](https://github.com/DrFarooqAi)
- HF Space: [farooqgenai/medgemma-xray-demo](https://huggingface.co/spaces/farooqgenai/medgemma-xray-demo)

---

## License

Apache 2.0 — open for educational and research use.
