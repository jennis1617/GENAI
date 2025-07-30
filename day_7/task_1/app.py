import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Streamlit config
st.set_page_config(page_title="📚 RAG Comparison App", layout="wide")
st.title("📚 RAG: Fixed vs Recursive Chunking (Auto Evaluation, No API Key)")
st.caption("Upload a `.txt` file, ask questions, and compare answers + evaluation metrics.")

# Sidebar model config
st.sidebar.header("LLM Settings")
model_name = st.sidebar.selectbox("Choose a HuggingFace model", ["google/flan-t5-base", "tiiuae/falcon-rw-1b"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
max_new_tokens = st.sidebar.slider("Max Tokens", 50, 512, 200)

# Load evaluation embedding model
@st.cache_resource
def load_eval_model():
    return SentenceTransformer("all-mpnet-base-v2")  # better semantic similarity

eval_model = load_eval_model()

# Load LLM for answering
@st.cache_resource
def load_llm():
    text2text = pipeline("text2text-generation", model=model_name, temperature=temperature, max_new_tokens=max_new_tokens)
    return HuggingFacePipeline(pipeline=text2text)

# Chunk splitter
def split_docs(docs, strategy="fixed"):
    if strategy == "fixed":
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Build RetrievalQA
def build_qa(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    llm = load_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Upload and ask
uploaded_file = st.file_uploader("📄 Upload a `.txt` file:", type="txt")
query = st.text_input("🔍 Ask a question based on the uploaded document:")

if uploaded_file and query:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(uploaded_file.read())
            doc_path = temp_file.name

        docs = TextLoader(doc_path).load()
        fixed_chunks = split_docs(docs, "fixed")
        recursive_chunks = split_docs(docs, "recursive")

        # Fixed Chunking Answer
        with st.spinner("Processing with Fixed Chunking..."):
            fixed_qa = build_qa(fixed_chunks)
            fixed_result = fixed_qa.run(query)

        st.subheader("🧩 Answer using Fixed Chunking")
        st.success(fixed_result)

        # Recursive Chunking Answer
        with st.spinner("Processing with Recursive Chunking..."):
            recursive_qa = build_qa(recursive_chunks)
            recursive_result = recursive_qa.run(query)

        st.subheader("🌀 Answer using Recursive Chunking")
        st.success(recursive_result)

        # Evaluation Metrics
        st.divider()
        st.subheader("📊 Automatic Evaluation (Semantic Similarity)")

        emb_fixed = eval_model.encode(fixed_result, convert_to_tensor=True)
        emb_recursive = eval_model.encode(recursive_result, convert_to_tensor=True)

        precision = util.pytorch_cos_sim(emb_fixed, emb_recursive).item()
        recall = util.pytorch_cos_sim(emb_recursive, emb_fixed).item()
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)

        # Display Evaluation Table
        metrics_df = pd.DataFrame({
            "Chunking Method": ["Fixed", "Recursive (Reference)"],
            "Answer": [fixed_result[:100] + "...", recursive_result[:100] + "..."],
            "Precision (vs Recursive)": [round(precision, 4), 1.0],
            "Recall (vs Recursive)": [round(recall, 4), 1.0],
            "F1 Score": [round(f1, 4), 1.0]
        })

        st.table(metrics_df)
        st.caption("✅ Recursive is used as the reference for semantic overlap scoring.")

    except Exception as e:
        st.error(f"⚠️ Failed to process file: {e}")




