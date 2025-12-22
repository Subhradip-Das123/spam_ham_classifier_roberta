import torch
from transformers import pipeline

MODEL_ID = "subhradip-nlp-labs/roberta-spam-classifier"

@torch.inference_mode()
def load_model():
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        framework="pt",   # 🔥 force PyTorch
        device=-1         # 🔥 CPU only
    )
    return classifier


