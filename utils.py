import torch
from transformers import pipeline

MODEL_ID = "subhradip-nlp-labs/roberta-spam-classifier"

@torch.inference_mode()
def load_model():
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device=device
    )
    return classifier


