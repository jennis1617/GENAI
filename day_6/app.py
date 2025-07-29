import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

st.set_page_config(page_title="RAG-lite Evaluation", layout="wide")
st.title("🔍 RAG-lite: Recursive vs Sliding Chunking")
st.markdown("Upload chunked CSVs, ask a question, and manually evaluate the best-matched chunks.")

# Upload CSVs
file1 = st.file_uploader("📄 Upload Chunked CSV 1 (Recursive)", type="csv")
file2 = st.file_uploader("📄 Upload Chunked CSV 2 (Sliding)", type="csv")

def generate_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

if file1 and file2:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    if 'text' not in df1.columns or 'text' not in df2.columns:
        st.error("Both CSVs must contain a 'text' column named `text`.")
        st.stop()

    with st.spinner("🔄 Generating embeddings..."):
        texts1 = df1['text'].tolist()
        texts2 = df2['text'].tolist()

        emb1 = generate_embeddings(texts1)
        emb2 = generate_embeddings(texts2)

    st.success("✅ Embeddings generated!")

    question = st.text_input("❓ Ask a question:")

    if question:
        query_embedding = model.encode(question, convert_to_tensor=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧩 Recursive Chunking")
            sim_scores = util.cos_sim(query_embedding, emb1)[0]
            top_idx = np.argmax(sim_scores)
            top_chunk = texts1[top_idx]
            top_score = sim_scores[top_idx].item()

            st.markdown(f"**🔹 Best Match Score:** `{top_score:.4f}`")
            st.text_area("📄 Retrieved Answer", top_chunk, height=200, key="recursive_text")
            is_correct = st.radio("Is the answer correct?", ["Yes", "No"], key="recursive_eval")
            f1_recursive = 1.0 if is_correct == "Yes" else 0.0
            st.metric("F1 Score", f"{f1_recursive:.2f}")

        with col2:
            st.subheader("📦 Sliding Chunking")
            sim_scores = util.cos_sim(query_embedding, emb2)[0]
            top_idx = np.argmax(sim_scores)
            top_chunk = texts2[top_idx]
            top_score = sim_scores[top_idx].item()

            st.markdown(f"**🔹 Best Match Score:** `{top_score:.4f}`")
            st.text_area("📄 Retrieved Answer", top_chunk, height=200, key="sliding_text")
            is_correct = st.radio("Is the answer correct?", ["Yes", "No"], key="sliding_eval")
            f1_sliding = 1.0 if is_correct == "Yes" else 0.0
            st.metric("F1 Score", f"{f1_sliding:.2f}")
else:
    st.info("👆 Please upload both Recursive and Sliding chunked CSV files.")
