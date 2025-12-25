import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Spam / Ham Classifier",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Spam / Ham Message Classifier")
st.write("Fine-tuned **RoBERTa** model deployed using Streamlit")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="subhradip-nlp-labs/roberta-spam-classifier"
    )

classifier = load_model()

text = st.text_area(
    "Enter message text",
    height=120,
    placeholder="Type SMS or email content here..."
)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = classifier(text)[0]
        label = result["label"]
        score = result["score"]

        if label.endswith("1") or label.upper() == "SPAM":
            st.error(f"🚫 SPAM ({score:.2%} confidence)")
        else:
            st.success(f"✅ HAM ({score:.2%} confidence)")


