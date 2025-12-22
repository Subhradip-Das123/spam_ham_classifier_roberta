import streamlit as st
from utils import load_model

st.set_page_config(
    page_title="Spam–Ham Classifier (RoBERTa)",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam–Ham Classifier (RoBERTa)")
st.write(
    "This app uses a **fine-tuned RoBERTa model** to classify messages as "
    "**Spam** or **Ham (Not Spam)**."
)

@st.cache_resource
def get_classifier():
    return load_model()

classifier = get_classifier()
st.success("✅ Model loaded successfully")

message = st.text_area(
    "✉️ Enter your message",
    height=150,
    placeholder="Congratulations! You've won a free prize..."
)

if st.button("Classify"):
    if not message.strip():
        st.warning("⚠️ Please enter a message")
    else:
        result = classifier(message, truncation=True, max_length=128)[0]
        label = result["label"]
        score = result["score"]

        if label.endswith("1"):
            st.error(f"🚫 **Spam Detected**\n\nConfidence: {score*100:.2f}%")
        else:
            st.success(f"✅ **Ham (Not Spam)**\n\nConfidence: {score*100:.2f}%")
