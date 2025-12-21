import streamlit as st
from utils import load_model

# ---------- Page config ----------
st.set_page_config(
    page_title="Spam–Ham Classifier (RoBERTa)",
    page_icon="📧",
    layout="centered"
)

# ---------- Title ----------
st.title("📧 Spam–Ham Classifier (RoBERTa)")
st.write(
    "This app uses a **fine-tuned RoBERTa model** to classify messages as "
    "**Spam** or **Ham (Not Spam)**."
)

# ---------- Load model ----------
@st.cache_resource
def get_classifier():
    return load_model()

classifier = get_classifier()
st.success("✅ Model loaded successfully")

# ---------- Input ----------
message = st.text_area(
    "✉️ Enter your message",
    height=150,
    placeholder="Congratulations! You've won a free prize..."
)

# ---------- Predict ----------
if st.button("Classify"):
    if not message.strip():
        st.warning("⚠️ Please enter a message")
    else:
        result = classifier(message, truncation=True, max_length=128)[0]

        label = result["label"]
        score = result["score"]

        # Handle LABEL_0 / LABEL_1
        if label.endswith("1"):
            st.error(f"🚫 **Spam Detected**\n\nConfidence: {score*100:.2f}%")
        else:
            st.success(f"✅ **Ham (Not Spam)**\n\nConfidence: {score*100:.2f}%")

# ---------- Footer ----------
# st.markdown("---")
# st.caption("🚀 Powered by RoBERTa • Hugging Face • Streamlit")
