import torch
from transformers import pipeline

MODEL_PATH = "models/roberta_spam_classifier"

@torch.inference_mode()
def load_model():
    device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=device
    )

    return classifier
