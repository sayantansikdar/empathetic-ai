---
title: EmpaChat
emoji: 🤗
colorFrom: purple
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Emotion-aware empathetic replies — GPT-2 XL + LoRA
models:
  - openai-community/gpt2-xl
  - sikdaarrr/empachat-gpt2xl-lora
datasets:
  - google-research-datasets/go_emotions
  - Estwld/empathetic_dialogues_llm
---

# EmpaChat

GPT-2 XL (1.5B) with two LoRA adapters, trained with the pipeline in
[sayantansikdar/empathetic-ai](https://github.com/sayantansikdar/empathetic-ai):

- **Emotion classifier** — GoEmotions mapped to positive / negative / neutral.
- **Reply generator** — real listener replies from EmpatheticDialogues, conditioned on the detected emotion.

This static page reads the measured metrics and sample replies live from
[`sikdaarrr/empachat-gpt2xl-lora`](https://huggingface.co/sikdaarrr/empachat-gpt2xl-lora).
To chat with the model, open the
[chat notebook](https://colab.research.google.com/github/sayantansikdar/empathetic-ai/blob/main/EmpaChat_Chat_Colab.ipynb)
on a free Colab GPU.
