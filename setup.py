from setuptools import setup, find_packages

setup(
    name="empathetic-ai",
    version="1.0.0",
    description="Empathetic Response AI — Emotion classification and empathetic response generation using LoRA-fine-tuned GPT-2 XL",
    author="Sayantan Sikdar",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.108.0",
        "uvicorn>=0.25.0",
        "gradio>=4.10.0",
    ],
)
