import streamlit as st
from transformers import pipeline

# Load FinBERT model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)

sentiment_model = load_model()

# Streamlit UI
st.set_page_config(page_title="FinBERT Sentiment Analyzer", page_icon="💹", layout="centered")

st.title("💹 FinBERT Financial Sentiment Analyzer")
st.write("Enter financial news, tweets, or text snippets to analyze their sentiment using **FinBERT**.")

# Input area
text_input = st.text_area("Enter text to analyze:", height=150, placeholder="e.g. The company's quarterly earnings exceeded expectations...")

if st.button("Analyze Sentiment"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            result = sentiment_model(text_input)[0]
        st.success(f"**Sentiment:** {result['label']}  \n**Confidence:** {result['score']:.2f}")
    else:
        st.warning("Please enter some text before analyzing.")

# Footer
st.markdown("---")
st.markdown("Model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)  |  Built with [Streamlit](https://streamlit.io/)")

