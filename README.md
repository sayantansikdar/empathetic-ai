# 🤗 EmpaChat — Empathetic Response AI

An end-to-end emotion classification and empathetic response generation system powered by **GPT-2 XL** with **LoRA fine-tuning**: an emotion classifier trained on **GoEmotions** and a reply generator trained on real listener replies from **EmpatheticDialogues**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌐 Live demo

- **Results page:** [huggingface.co/spaces/sikdaarrr/EmpaChat](https://huggingface.co/spaces/sikdaarrr/EmpaChat) —
  a static page ([`space/`](space/)) that reads the measured metrics and sample replies live from the adapter repo
  [`sikdaarrr/empachat-gpt2xl-lora`](https://huggingface.co/sikdaarrr/empachat-gpt2xl-lora).
- **Chat with the model:** open [`EmpaChat_Chat_Colab.ipynb`](https://colab.research.google.com/github/sayantansikdar/empathetic-ai/blob/main/EmpaChat_Chat_Colab.ipynb)
  on a free T4 and **Run all** — it serves [`demo/gradio_app.py`](demo/gradio_app.py) through a public Gradio link.

## 📊 Results

Measured by the training run and published next to each adapter as `metrics.json`
(the Space reads them from there):

| Stage | Metric | Result |
|---|---|---|
| Emotion classifier | Test accuracy / macro-F1 (GoEmotions, 3-class) | *published after the first full training run* |
| Reply generator | Validation perplexity (EmpatheticDialogues replies) | *published after the first full training run* |
| LoRA | Trainable parameters (GPT-2 XL, r = 8) | ≈9.8M of ≈1.56B (≈99.4% fewer) |

## ☁️ Train on a free Colab GPU

Open [`EmpaChat_Colab_Training.ipynb`](EmpaChat_Colab_Training.ipynb) in Colab with a **T4** runtime and run it top to
bottom. Both stages train with 4-bit QLoRA, checkpoint to Google Drive (a disconnected session resumes where it left
off), and push their adapters + metrics to the Hub as soon as they finish.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Message                          │
│                         │                                │
│                    ┌────▼────┐                           │
│                    │  Clean  │  Text preprocessing       │
│                    └────┬────┘                           │
│                         │                                │
│              ┌──────────┼──────────┐                     │
│              ▼                     ▼                     │
│     ┌────────────────┐    ┌───────────────┐              │
│     │   Classifier   │    │   Generator   │              │
│     │  GPT-2 XL +    │    │  GPT-2 XL +   │              │
│     │  LoRA (SEQ_CLS)│    │  LoRA (CLM)   │              │
│     └───────┬────────┘    └───────┬───────┘              │
│             │                     │                      │
│     ┌───────▼────────┐    ┌───────▼───────┐              │
│     │ Emotion Label  │───▶│  Empathetic   │              │
│     │ + Confidence   │    │   Response    │              │
│     └────────────────┘    └───────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### LoRA Configuration
- **Rank (r)**: 8
- **Alpha**: 16
- **Dropout**: 0.1
- **Target Modules**: `c_attn`, `c_proj`, `c_fc`
- **Quantization**: 4-bit NF4 (QLoRA for GPU-constrained setups)

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/empathetic-ai.git
cd empathetic-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# One-liner training command (full pipeline)
python train.py --lora --epochs 3

# Just the classifier
python train.py --lora --epochs 3 --skip_generator --skip_ab_test

# With QLoRA quantization (for T4 GPU)
python train.py --lora --quantize --epochs 3 --batch_size 4

# With lighter model (for testing / CPU)
python train.py --lora --epochs 3 --model gpt2-medium --batch_size 16
```

### 3. Start the API

```bash
# One-liner API start
uvicorn src.api.main:app --reload --port 8000
```

Test the endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Classify emotion
curl -X POST http://localhost:8000/classify_emotion \
  -H "Content-Type: application/json" \
  -d '{"text": "I just got promoted and I am so happy!"}'

# Generate empathetic response
curl -X POST http://localhost:8000/generate_response \
  -H "Content-Type: application/json" \
  -d '{"text": "I lost my job today and feel terrible."}'
```

### 4. Launch Demo

```bash
python demo/gradio_app.py
# Open http://localhost:7860
```

### 5. Run A/B Test

```bash
# Rule-based judge (no API needed)
python -m src.evaluation.ab_simulator --num_samples 100

# With GPT-4 judge (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python -m src.evaluation.ab_simulator --num_samples 100 --use_gpt4
```

---

## 📁 Project Structure

```
empathetic-ai/
├── data/                          # GoEmotions cache (auto-downloaded)
├── notebooks/                     # EDA notebooks
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py        # Text cleaning + emotion mappings
│   │   └── data_loader.py         # GoEmotions loading + tokenization
│   ├── models/
│   │   ├── lora_config.py         # LoRA + quantization configs
│   │   ├── emotion_classifier.py  # GPT-2 XL emotion classifier
│   │   └── response_generator.py  # Empathetic response generator
│   ├── training/
│   │   ├── train_classifier.py    # Classifier training loop
│   │   ├── train_generator.py     # Generator training loop
│   │   └── ab_testing.py          # A/B test orchestrator
│   ├── evaluation/
│   │   ├── metrics.py             # Accuracy, F1, confusion matrix
│   │   └── ab_simulator.py        # A/B test simulation engine
│   └── api/
│       ├── main.py                # FastAPI endpoints
│       └── schemas.py             # Pydantic models
├── demo/
│   └── gradio_app.py              # Gradio chat interface
├── outputs/                       # Training outputs + LoRA adapters
├── train.py                       # Main training entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Details

### Dataset: GoEmotions
- **Source**: [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- **Size**: 58,000 Reddit comments
- **Labels**: 27 emotions → mapped to 3 sentiment classes (positive, negative, neutral)
- **Split**: Train/Validation/Test (pre-split from HuggingFace)

### Model: GPT-2 XL (1.5B Parameters)
- **Base**: `gpt2-xl` from HuggingFace
- **Classification**: `GPT2ForSequenceClassification` with left-padding
- **Generation**: `GPT2LMHeadModel` with emotion-conditioned prompts

### LoRA Fine-Tuning
| Component | Value |
|-----------|-------|
| Total Params | ~1,560,000,000 |
| Trainable Params | ~8,400,000 |
| Reduction | **99.4%** |
| Rank | 8 |
| Target Modules | c_attn, c_proj, c_fc |
| QLoRA (optional) | 4-bit NF4 |

### Training Hyperparameters
| Param | Value |
|-------|-------|
| Learning Rate | 2e-4 |
| Batch Size | 8 |
| Epochs | 3 |
| Warmup | 10% |
| Scheduler | Cosine |
| Weight Decay | 0.01 |
| Max Sequence Length | 128 |

### A/B Testing
- **Baseline**: Rule-based responses (minimal empathy)
- **Treatment**: LoRA-adapted empathetic responses
- **Judge**: GPT-4 API or rule-based heuristic
- **Metrics**: Empathy (1-5), Relevance (1-5), Helpfulness (1-5)
- **Statistics**: Paired t-test, Cohen's d, p-value

---

## 🐳 Docker Deployment

```bash
# Build and run both services
docker-compose up --build

# API: http://localhost:8000
# Demo: http://localhost:7860
# Docs: http://localhost:8000/docs
```

---

## 📈 Experiment Tracking

```bash
# View MLflow dashboard
mlflow ui --backend-store-uri ./outputs/mlruns
# Open http://localhost:5000
```

Tracked metrics:
- Total / trainable parameters
- Training loss curves
- Accuracy and F1 scores
- Inference latency
- GPU memory usage

---

## 🛠️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMPACHAT_MODEL` | `gpt2-xl` | Base model name |
| `EMPACHAT_CLASSIFIER_ADAPTER` | `./outputs/classifier/classifier_lora` | Classifier LoRA path |
| `EMPACHAT_GENERATOR_ADAPTER` | `./outputs/generator/generator_lora` | Generator LoRA path |
| `OPENAI_API_KEY` | — | For GPT-4 A/B test judge |

---

## ⚙️ Hardware Requirements

| Setup | GPU | RAM | Training Time |
|-------|-----|-----|---------------|
| Full (QLoRA) | T4 16GB | 16GB+ | ~1-2 hours |
| Full (FP16) | A100 40GB | 32GB+ | ~30 min |
| Testing | CPU | 8GB+ | N/A (inference only) |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [GoEmotions](https://arxiv.org/abs/2005.00547) — Demszky et al., 2020
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [HuggingFace](https://huggingface.co/) — Transformers, PEFT, Datasets
