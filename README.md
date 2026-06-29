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

An end-to-end NLP project for **Thai news category classification**, built to demonstrate practical **AI Engineering** skills: data preparation, transformer fine-tuning, model evaluation, and deployment.

> ✅ Portfolio project for AI Engineer applications

---

## 1) Project Overview

This project fine-tunes a Thai pre-trained transformer (WangchanBERTa) to classify Thai news text into topic categories (politics, business, sports, entertainment, and more).

The repository includes:
- Model training & fine-tuning pipeline
- Model inference pipeline
- Streamlit web app for interactive predictions
- Docker-ready deployment setup

---

## 2) Why This Project (AI Engineer Relevance)

This repository demonstrates core AI Engineer competencies:

- **Applied NLP**: Transformer-based multi-label text classification for Thai language
- **Model Adaptation**: Fine-tuning a domain-relevant pre-trained model (WangchanBERTa)
- **Experiment Mindset**: Metric-based validation and iterative improvement across 5 epochs
- **MLOps Basics**: Reproducible environment, containerized deployment, checkpoint management
- **Product Thinking**: Simple UI for non-technical stakeholders (Streamlit web interface)

---

## 3) Tech Stack

- **Language**: Python  
- **Deep Learning / NLP**: PyTorch, Hugging Face Transformers  
- **Pre-trained Model**: WangchanBERTa (airesearch/wangchanberta-base-att-spm-uncased)
- **Training Framework**: Hugging Face Trainer API
- **App Layer**: Streamlit  
- **Deployment**: Docker  

---

## 4) Repository Structure

```text
.
├── README.md
├── src/
│   └── streamlit_app.py                      # Streamlit inference app
├── Dockerfile                                # Containerized deployment
├── requirements.txt                          # Python dependencies
├── Fine_tune_News_classification.ipynb       # Training notebook (Colab)
└── (training / model files)                  # Training checkpoints & model files
```

---

## 5) Problem Statement

News publishers and aggregators process large volumes of Thai content daily.  
Manual categorization is slow and inconsistent.  
This project **automates news categorization** with a fine-tuned transformer model to improve:
- **Speed**: Instant classification vs. manual review
- **Consistency**: Deterministic multi-label predictions
- **Scalability**: Can process thousands of articles

---

## 6) Model Pipeline

1. **Input**: Thai news title + body text  
2. **Preprocessing**: Tokenization using WangchanBERTa tokenizer (Thai SentencePiece)
3. **Model**: WangchanBERTa base model + classification head
4. **Output**: Multi-label predictions with confidence scores for 12 topic categories

---

## 7) Key Engineering Highlights

- ✅ **Thai-specific NLP**: Handles Thai language tokenization & encoding
- ✅ **Multi-label classification**: Supports articles tagged with multiple topics
- ✅ **Production-ready**: Clean inference interface, error handling
- ✅ **Reproducible**: Fixed seeds, version control, dependency management
- ✅ **Scalable**: Docker containerization for easy deployment

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

## 10) Model Performance (Training Results)

**Dataset**: Prachatai Thai news corpus (2006-2017)  
**Train/Val/Test split**: 70/10/20  
**Total articles**: 20,000+ annotated  
**Task**: Multi-label classification (12 categories)  
**Model**: WangchanBERTa (base, uncased)  
**Training**: 5 epochs, batch size 8, learning rate 2e-5

### Final Performance (Epoch 5/5):

| Metric | Value |
|--------|-------|
| **F1 Score (Micro)** | **76.13%** |
| **F1 Score (Macro)** | **69.76%** |
| **Precision (Micro)** | **79.21%** |
| **Recall (Micro)** | **73.28%** |
| **Training Loss** | 0.1194 |
| **Validation Loss** | 0.1715 |

### Per-Epoch Progress:

| Epoch | Train Loss | Val Loss | F1 Micro | F1 Macro | Precision | Recall |
|-------|-----------|----------|----------|----------|-----------|--------|
| 1 | 0.1794 | 0.1734 | 72.44% | 64.93% | 77.30% | 68.15% |
| 2 | 0.1502 | 0.1667 | 74.41% | 67.39% | 79.93% | 69.60% |
| 3 | 0.1269 | 0.1651 | 75.46% | 68.85% | 79.74% | 71.63% |
| 4 | 0.1300 | 0.1691 | 76.08% | 69.37% | 78.84% | 73.51% |
| 5 | 0.1194 | 0.1715 | **76.13%** | **69.76%** | **79.21%** | **73.28%** |

---

## 11) Topic Categories

The model classifies Thai news into 12 categories:

1. **Politics** - Political news, government, parliament
2. **Human Rights** - Rights advocacy, violations, justice
3. **Quality of Life** - Health, welfare, social services
4. **International** - Foreign affairs, global events
5. **Social** - Social issues, community, culture
6. **Environment** - Environmental protection, climate
7. **Economics** - Business, markets, trade
8. **Culture** - Arts, entertainment, heritage
9. **Labor** - Worker rights, employment, unions
10. **National Security** - Defense, military, security
11. **ICT** - Technology, digital, telecommunications
12. **Education** - Schools, universities, learning

---

## 12) Example Predictions

| Thai News Title | Predicted Categories | Confidence |
|---|---|---|
| "ประยุทธ์ประกาศนโยบายใหม่" | Politics, National Security | 0.78, 0.65 |
| "ราคาน้ำมันพุ่งสูงกดดันชาวนา" | Economics, Quality of Life | 0.82, 0.71 |
| "นักเรียนชุมนุมเรียกร้องสิทธิ" | Social, Human Rights, Education | 0.75, 0.73, 0.68 |

---

## 13) What I Would Improve Next

- [ ] Add experiment tracking (Weights & Biases / MLflow)
- [ ] REST API service (FastAPI) for production integration
- [ ] Automated tests for inference pipeline + CI/CD
- [ ] Class imbalance handling (SMOTE, weighted loss)
- [ ] Model optimization (quantization, distillation, ONNX export)
- [ ] Extended Thai news datasets (Thai PBS, Manager, Bangkok Post)
- [ ] Explainability layer (attention visualization, SHAP)
- [ ] Multi-language support (cross-lingual transfer learning)

---

## 14) AI Engineer Skills Demonstrated

✅ **End-to-end ML ownership** — from data to deployment  
✅ **Model fine-tuning** — optimizing pre-trained transformers  
✅ **Metric-driven development** — validation & performance tracking  
✅ **Production readiness** — containerization, versioning, reproducibility  
✅ **NLP for low-resource languages** — Thai language processing  
✅ **User-facing AI** — Streamlit app for stakeholder interaction  

---

## 15) Dataset & Attribution

**Primary Source**: [Prachatai](https://prachatai.com/) — Thai independent news organization  
**Dataset**: Annotated Thai news corpus (2006-2017)  
**Multi-label annotation**: 12 topic categories per article  
**Data split**: Train: 70%, Validation: 10%, Test: 20%

---

## 16) How to Train Your Own Model

See `Fine_tune_News_classification.ipynb` for the complete training pipeline in Google Colab:

1. Connect to Google Drive and load the dataset
2. Install dependencies (transformers, torch, pandas)
3. Tokenize articles using WangchanBERTa tokenizer
4. Fine-tune model with Hugging Face Trainer
5. Evaluate on test set
6. Export best checkpoint

---

## 17) About Me

I built this project to showcase hands-on **AI Engineering** skills:
- 🧠 **NLP Model Fine-tuning** — specialized for non-English languages
- 🚀 **Production-minded** — deployment-ready code & containerization
- 👥 **User-focused** — building AI interfaces for stakeholders
- 📊 **Data-driven** — metric-based experiments & iteration

**Actively seeking**: AI Engineer roles in NLP/LLMs, ML systems, and applied AI product development.

---

## 18) License

MIT License — Feel free to use this project for education & research.

For questions or collaboration, open an issue or reach out! 🙌
