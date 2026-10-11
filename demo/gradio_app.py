"""
EmpaChat — chat demo.

Run locally (`python demo/gradio_app.py`) or on a free Colab GPU via
EmpaChat_Chat_Colab.ipynb (GRADIO_SHARE=1 gives a public link).

GPT-2 XL + two LoRA adapters trained with this repo's pipeline:
  classifier/  GoEmotions → positive / negative / neutral
  generator/   EmpatheticDialogues listener replies, conditioned on that label
Adapters and their measured metrics are loaded from ADAPTER_REPO on the Hub.
Nothing here is canned: if the adapters aren't available, the app says so.
"""

import json
import os
import re
from threading import Thread

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextIteratorStreamer,
)

BASE_MODEL = os.getenv("BASE_MODEL", "gpt2-xl")
ADAPTER_REPO = os.getenv("ADAPTER_REPO", "sikdaarrr/empachat-gpt2xl-lora")
# Set LOCAL_ADAPTER_DIR to a folder holding classifier/ and generator/ to run offline.
LOCAL_ADAPTER_DIR = os.getenv("LOCAL_ADAPTER_DIR")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# fp16 on GPU; bf16 on CPU keeps two 1.5B models within ~7 GB of RAM
DTYPE = getattr(torch, os.getenv("DTYPE", "float16" if DEVICE == "cuda" else "bfloat16"))
torch.set_num_threads(max(1, os.cpu_count() or 1))

LABELS = ["positive", "negative", "neutral"]
PROMPT = "Speaker ({emotion}): {user_text}\nListener:"  # same template as training

# Same normalisation the classifier saw in training (src/preprocessing/text_cleaner.py)
_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_SUB = re.compile(r"r/\w+")
_HTML = re.compile(r"&\w+;")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    for pat in (_URL, _USER, _SUB, _HTML):
        text = pat.sub(" ", text)
    return _WS.sub(" ", text.lower()).strip()


def _adapter_source(sub: str):
    if LOCAL_ADAPTER_DIR:
        return os.path.join(LOCAL_ADAPTER_DIR, sub), {}
    return ADAPTER_REPO, {"subfolder": sub}


def _metrics(sub: str) -> dict:
    try:
        if LOCAL_ADAPTER_DIR:
            path = os.path.join(LOCAL_ADAPTER_DIR, sub, "metrics.json")
        else:
            path = hf_hub_download(ADAPTER_REPO, "metrics.json", subfolder=sub)
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


CLS_METRICS, GEN_METRICS = _metrics("classifier"), _metrics("generator")
READY = bool(CLS_METRICS and GEN_METRICS)
classifier = generator = tokenizer = None

if READY:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    src, kw = _adapter_source("classifier")
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), torch_dtype=DTYPE, pad_token_id=tokenizer.eos_token_id
    )
    classifier = PeftModel.from_pretrained(base, src, **kw).merge_and_unload().to(DEVICE).eval()

    src, kw = _adapter_source("generator")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)
    generator = PeftModel.from_pretrained(base, src, **kw).merge_and_unload().to(DEVICE).eval()
    NEWLINE = tokenizer.encode("\n")[0]


@torch.inference_mode()
def tidy(reply: str) -> str:
    """First line only; drop a trailing half-sentence if the token budget cut it."""
    reply = reply.split("\n")[0].strip()
    if reply and reply[-1] not in ".!?\"')":
        cut = max(reply.rfind(p) for p in (". ", "! ", "? "))
        if cut > 0:
            reply = reply[: cut + 1]
    return reply


def classify(text: str) -> dict:
    ids = tokenizer(clean_text(text), return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
    probs = torch.softmax(classifier(**ids).logits.float(), dim=-1)[0]
    return {label: float(p) for label, p in zip(LABELS, probs)}


def respond(message: str, history: list):
    """Classify the message, then stream the generator's reply."""
    message = (message or "").strip()
    if not message:
        yield history, gr.update(), ""
        return
    if not READY:
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ The trained adapters aren't on the Hub yet, so there's no model to answer with."},
        ]
        yield history, gr.update(), ""
        return

    scores = classify(message)
    emotion = max(scores, key=scores.get)
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    yield history, scores, ""

    prompt = PROMPT.format(emotion=emotion, user_text=message)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=[tokenizer.eos_token_id, NEWLINE],
        pad_token_id=tokenizer.eos_token_id,
    )
    Thread(target=lambda: torch.inference_mode()(generator.generate)(**kwargs), daemon=True).start()

    reply = ""
    for chunk in streamer:
        reply += chunk
        history[-1]["content"] = reply.split("\n")[0].strip()
        yield history, scores, ""
    history[-1]["content"] = tidy(reply) or "…"
    yield history, scores, ""


def _header() -> str:
    if not READY:
        return (
            "### ⏳ Model not published yet\n"
            f"Waiting for trained adapters at `{ADAPTER_REPO}`. "
            "Run the training notebook in the GitHub repo to create them."
        )
    acc = CLS_METRICS.get("test_accuracy", 0) * 100
    f1 = CLS_METRICS.get("test_f1_macro", 0)
    ppl = GEN_METRICS.get("val_perplexity")
    trainable = GEN_METRICS.get("trainable_params", 0)
    total = GEN_METRICS.get("total_params", 0) or 1
    return (
        f"**Emotion classifier** — {acc:.1f}% test accuracy · macro-F1 {f1:.3f} "
        f"on {CLS_METRICS.get('n_test', 0):,} held-out GoEmotions comments (3-class)  \n"
        f"**Reply generator** — trained on {GEN_METRICS.get('n_train', 0):,} real EmpatheticDialogues "
        f"listener replies · validation perplexity {ppl}  \n"
        f"**LoRA** — {trainable/1e6:.1f}M trainable of {total/1e9:.2f}B parameters "
        f"({(1 - trainable/total)*100:.1f}% fewer) · base `{BASE_MODEL}`"
    )


EXAMPLES = [
    "I just got promoted at work and I'm so excited about it!",
    "My grandmother passed away last week and I miss her so much.",
    "I failed my exam even though I studied really hard.",
    "I've been thinking about my old school friends lately.",
    "I feel completely overwhelmed and don't know what to do anymore.",
]

with gr.Blocks(title="EmpaChat — Empathetic Response AI") as demo:
    gr.Markdown(
        "# 🤗 EmpaChat\n"
        "GPT-2 XL with two LoRA adapters: one reads the emotion in your message, "
        "the other writes a reply the way a supportive listener would.\n\n" + _header()
    )
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=420, label="Conversation")
            box = gr.Textbox(placeholder="Tell EmpaChat how you're feeling…", show_label=False, lines=1, max_lines=4)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
        with gr.Column(scale=2):
            emotion = gr.Label(label="Detected emotion", num_top_classes=3)
            gr.Markdown(
                ("Running on a GPU — replies stream in within a few seconds.  \n" if DEVICE == "cuda" else
                 "Running on CPU, so a reply takes roughly 20–60 seconds and streams in as it's written.  \n")
                + "This is a research demo, not a support service. If you're struggling, please reach out "
                "to someone you trust or a local helpline."
            )
    gr.Examples(EXAMPLES, inputs=box)

    send.click(respond, [box, chat], [chat, emotion, box])
    box.submit(respond, [box, chat], [chat, emotion, box])
    clear.click(lambda: ([], None, ""), outputs=[chat, emotion, box])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        theme=gr.themes.Soft(primary_hue="violet"),
        share=os.getenv("GRADIO_SHARE") == "1",
    )
