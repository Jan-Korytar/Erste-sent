import streamlit as st
from transformers import pipeline

# Load FinBERT model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)

sentiment_model = load_model()

# Streamlit UI
st.set_page_config(page_title="FinBERT Sentiment Analyzer", layout="centered")

st.title("FinBERT Analyzer")
st.write("Enter any text, output is classified sentiment (negative, positive, neutral) and confidence level.")

# Input area
text_input = st.text_area("Enter text to analyze:", height=200, value="Net interest income increased to EUR 5,761 million (+3.0%; EUR 5,591 million), primarily in the Czech Republic, Romania and Slovakia, on the back of loan growth and lower interest expenses on customer deposits. Net fee and commission income rose to EUR 2,340 million (+8.4%; EUR 2,158 million). Growth was registered across all core markets and income categories. Net trading result declined to EUR 231 million (EUR 428 million); the line item gains/losses from financial instruments measured at fair value through profit or loss rose to EUR 58 million (EUR -70 million). The development of both line items was mostly attributable to valuation effects. Operating income increased to EUR 8,587 million (+3.2%; EUR 8,319 million). General administrative expenses were up at EUR 4,068 million (+6.8%; EUR 3,809 million). Personnel expenses increased to EUR 2,449 million (+5.6%; EUR 2,318 million) driven by collectively agreed salary increases. Other administrative expenses were higher at EUR 1,206 million (+11.1%; EUR 1,086 million). While contributions to deposit insurance schemes included in other administrative expenses – mostly already posted upfront for the full year of 2025 – declined to EUR 59 million (EUR 72 million), IT expenses increased to EUR 530 million (EUR 451 million). Amortisation and depreciation amounted to EUR 413 million (+2.0%; EUR 405 million). Overall, the operating result increased moderately to EUR 4,519 million (+0.2%; EUR 4,510 million), the cost/income ratio stood at 47.4% (45.8%).")

if st.button("Analyze"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            result = sentiment_model(text_input)[0]
        st.success(f"**Sentiment:** {result['label']}  \n**Confidence:** {result['score']:.2f}")
    else:
        st.warning("Please enter some text before analyzing.")

# Footer
st.markdown("---")
st.markdown("Model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)")

