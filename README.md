# 🩻 MedGemma 1.5 Chest X-ray Analyzer

A **local, offline AI demo** using Google's open-source **MedGemma 1.5** model to analyze chest X-rays — no cloud, no data leaves your machine.

![Demo Screenshot](demo.png)

> ⚠️ **For demonstration only — not for clinical diagnosis or medical use.**

---

## 🔍 What It Does
- Upload any chest X-ray (PNG/JPG)
- Get an AI-generated **radiology-style report**
- Runs entirely **on your laptop** (Intel i7 + 32GB RAM tested)
- Uses **open-source models** (free for research & commercial use)

---

## ▶️ How to Run Locally

### Prerequisites
1. Request access to the model:  
   👉 [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
2. Get your Hugging Face token:  
   👉 [Settings → Tokens](https://huggingface.co/settings/tokens) (role: **Read**)

### Steps
```bash
# Clone this repo
git clone https://github.com/your-username/medgemma-xray-demo.git
cd medgemma-xray-demo

# Install dependencies
pip install -r requirements.txt

# Create .env file with your token
# (copy from .env.example and replace with your real token)
cp .env.example .env
# Then edit .env and paste your token

# Run the app
python app.py