---
title: Thai News Classification with WangchanBERTa
emoji: 🧠
colorFrom: red
colorTo: orange
sdk: docker
app_port: 8501
tags:
  - streamlit
  - nlp
  - thai
  - transformers
  - text-classification
pinned: false
short_description: AI Engineer portfolio project — Thai news classification using a fine-tuned transformer model
env:
  TRANSFORMERS_CACHE: './.cache'
---

# Thai News Classification (WangchanBERTa Fine-tuning)

An end-to-end NLP project for **Thai news category classification**, built to demonstrate practical **AI Engineering** skills: data preparation, transformer fine-tuning, model evaluation, and deployment with Streamlit + Docker.

> ✅ Portfolio project for AI Engineer applications

---

## 1) Project Overview

This project fine-tunes a Thai pre-trained transformer (Wangchan family) to classify Thai news text into topic categories (e.g., politics, business, sports, entertainment).

The repository includes:
- Model inference pipeline
- Streamlit web app for interactive predictions
- Docker-ready deployment setup

---

## 2) Why This Project (AI Engineer Relevance)

This repository demonstrates core AI Engineer competencies:

- **Applied NLP**: Transformer-based text classification for Thai language
- **Model Adaptation**: Fine-tuning a domain-relevant pre-trained model
- **Experiment Mindset**: Metric-based validation and iteration
- **MLOps Basics**: Reproducible environment and containerized deployment
- **Product Thinking**: Simple UI for non-technical stakeholders (Streamlit)

---

## 3) Tech Stack

- **Language**: Python  
- **Deep Learning / NLP**: PyTorch, Hugging Face Transformers  
- **App Layer**: Streamlit  
- **Deployment**: Docker  

---

## 4) Repository Structure

```text
.
├── README.md
├── src/
│   └── streamlit_app.py         # Streamlit inference app
├── Dockerfile                   # Containerized deployment
├── requirements.txt             # Python dependencies
└── (training / model files)     # add your training and checkpoint paths
```

---

## 5) Problem Statement

News publishers and aggregators process large volumes of Thai content daily.  
Manual categorization is slow and inconsistent.  
This project automates categorization with a transformer model to improve speed and consistency.

---

## 6) Model Pipeline

1. **Input**: Thai news title/content  
2. **Preprocessing**: Tokenization compatible with selected Wangchan model  
3. **Inference**: Fine-tuned transformer predicts class probabilities  
4. **Output**: Predicted category + confidence score  

---

## 7) Key Engineering Highlights

- Designed for Thai-language NLP constraints
- Built a clean prediction interface for rapid demo/use
- Structured for reproducibility (dependency + container setup)
- Ready to extend with logging, API serving, and CI/CD

---

## 8) Run Locally

```bash
# 1) Create and activate virtual env
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run Streamlit app
streamlit run src/streamlit_app.py
```

Open: `http://localhost:8501`

---

## 9) Run with Docker

```bash
docker build -t thai-news-ai-engineer-portfolio .
docker run -p 8501:8501 thai-news-ai-engineer-portfolio
```

---

## 10) Evaluation (Fill with Your Real Results)

> Replace the placeholders below with your actual values.

- Accuracy: `XX.XX%`
- Macro F1: `XX.XX`
- Weighted F1: `XX.XX`
- Validation loss: `X.XXXX`

If available, also include:
- Confusion matrix
- Per-class precision/recall/F1
- Example misclassifications and error analysis

---

## 11) Example Predictions

| Input Text (Thai) | Predicted Class | Confidence |
|---|---|---|
| *(ใส่ตัวอย่างข้อความข่าวที่นี่)* | *(หมวดหมู่)* | *(0.xx)* |

---

## 12) What I Would Improve Next

- Add training script + experiment tracking (e.g., W&B/MLflow)
- Add REST API service (FastAPI) for production integration
- Add automated tests for inference pipeline
- Add CI/CD workflow for lint/test/build
- Optimize latency and model size for lower-cost serving

---

## 13) AI Engineer Skills Demonstrated

- Data-to-deployment workflow ownership
- Model-centric software engineering
- Practical NLP problem solving for non-English language
- Clear communication of technical outcomes for stakeholders

---

## 14) About Me

I built this project to showcase hands-on AI engineering skills in:
- NLP model fine-tuning
- Production-minded packaging
- User-facing AI application development

I am actively seeking an **AI Engineer** role and open to opportunities involving NLP/LLMs, ML systems, and applied AI product development.

---

## 15) License

Add your preferred license (e.g., MIT) in a `LICENSE` file.
