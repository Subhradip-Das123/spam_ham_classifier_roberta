import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "subhradip-nlp-labs/roberta-spam-classifier"

@torch.inference_mode()
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        ignore_mismatched_sizes=True  # 🔥 FIX
    )

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )
    return classifier
