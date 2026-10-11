---
title: EmpaChat
emoji: 🤗
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
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
- **Reply generator** — real listener replies from EmpatheticDialogues,
  conditioned on the detected emotion; loss on the reply only.

The metrics shown in the app are read from the adapters' `metrics.json`,
written by the training run itself. The Space runs on a free CPU, so replies
take a little while and stream in as they are generated.

This is a research demo, not a mental-health service.
