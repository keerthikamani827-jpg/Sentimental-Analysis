import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

st.title("AI Sentiment Analyzer")
st.write("Analyze the sentiment of your text using Hugging Face Transformers.")

@st.cache_resource
def load_model():
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return sentiment_pipeline

sentiment_pipeline = load_model()

text = st.text_area(
    "Enter your text:",
    placeholder="Example: I really enjoyed this movie!"
)

if st.button("Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        with st.spinner("Analyzing sentiment..."):

            result = sentiment_pipeline(text)[0]

        label = result["label"]
        score = result["score"]

        st.subheader("Sentiment Result")

        if label == "POSITIVE":
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")

        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence:** {score:.2%}")

        st.progress(float(score))