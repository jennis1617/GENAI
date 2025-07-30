import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load better model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")  # upgraded for better performance

model = load_model()

st.set_page_config(page_title="RAG-lite Evaluation", layout="wide")
st.title("🔍 RAG-lite: Recursive vs Sliding Chunking (Auto Evaluation)")
st.markdown("Upload your chunked `.csv` files and enter a question. We’ll auto-evaluate using semantic similarity!")

# Upload files
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

        # ---- Recursive
        with col1:
            st.subheader("🧩 Recursive Chunking")
            sim_scores = util.cos_sim(query_embedding, emb1)[0]
            top_idx = np.argmax(sim_scores)
            recursive_answer = texts1[top_idx]
            top_score = sim_scores[top_idx].item()

            st.markdown(f"**🔹 Best Match Score:** `{top_score:.4f}`")
            st.text_area("📄 Retrieved Answer", recursive_answer, height=200, key="recursive_text")

        # ---- Sliding
        with col2:
            st.subheader("📦 Sliding Chunking")
            sim_scores = util.cos_sim(query_embedding, emb2)[0]
            top_idx = np.argmax(sim_scores)
            sliding_answer = texts2[top_idx]
            top_score = sim_scores[top_idx].item()

            st.markdown(f"**🔹 Best Match Score:** `{top_score:.4f}`")
            st.text_area("📄 Retrieved Answer", sliding_answer, height=200, key="sliding_text")

        # ---- Evaluation
        st.divider()
        st.subheader("📊 Auto Evaluation using Semantic Similarity")

        emb_recursive = model.encode(recursive_answer, convert_to_tensor=True)
        emb_sliding = model.encode(sliding_answer, convert_to_tensor=True)

        # Semantic Precision, Recall, F1
        precision = util.pytorch_cos_sim(emb_sliding, emb_recursive).item()
        recall = util.pytorch_cos_sim(emb_recursive, emb_sliding).item()
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)

        # Summary Table
        metrics_df = pd.DataFrame({
            "Chunking Method": ["Recursive", "Sliding"],
            "Answer Text": [recursive_answer[:100] + "...", sliding_answer[:100] + "..."],
            "Precision (vs each other)": [1.0, round(precision, 4)],
            "Recall (vs each other)": [1.0, round(recall, 4)],
            "F1 Score": [1.0, round(f1, 4)]
        })

        st.table(metrics_df)
else:
    st.info("👆 Please upload both Recursive and Sliding chunked CSV files.")
