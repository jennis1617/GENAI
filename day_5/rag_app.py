import streamlit as st
import tempfile
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

st.set_page_config(page_title="RAG Evaluation: All Chunking Methods", layout="wide")
st.title("📊 Compare Chunking Methods: Fixed-size vs Recursive vs SentenceSplitter")

uploaded_file = st.file_uploader("📄 Upload a `.txt` file", type=["txt"])
question = st.text_input("❓ Enter your question:")

if uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        tmp.write(content)
        temp_file_path = tmp.name

    try:
        loader = TextLoader(temp_file_path, encoding="utf-8")
        docs = loader.load()
        st.success("✅ File loaded and question received.")
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

    # Common LLM & embedding setup
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pipe = pipeline("text2text-generation", model="google/flan-t5-small")
    llm = HuggingFacePipeline(pipeline=pipe)

    # Layout with 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📦 Fixed-size Chunking")
        fixed_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        fixed_chunks = fixed_splitter.split_documents(docs)
        st.write(f"Chunks created: {len(fixed_chunks)}")

        fixed_db = FAISS.from_documents(fixed_chunks, embeddings)
        fixed_chain = RetrievalQA.from_chain_type(llm=llm, retriever=fixed_db.as_retriever(), return_source_documents=True)

        with st.spinner("🔍 Answering using Fixed-size Chunking..."):
            fixed_result = fixed_chain.invoke(question)
        st.success("Answer (Fixed-size):")
        st.write(fixed_result['result'])

        correct_fixed = st.radio("Is the answer correct?", ["Yes", "No"], key="fixed_eval")
        f1_fixed = 1.0 if correct_fixed == "Yes" else 0.0
        st.metric("F1 Score", f"{f1_fixed:.2f}")

    with col2:
        st.subheader("🧩 Recursive Chunking")
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        recursive_chunks = recursive_splitter.split_documents(docs)
        st.write(f"Chunks created: {len(recursive_chunks)}")

        recursive_db = FAISS.from_documents(recursive_chunks, embeddings)
        recursive_chain = RetrievalQA.from_chain_type(llm=llm, retriever=recursive_db.as_retriever(), return_source_documents=True)

        with st.spinner("🔍 Answering using Recursive Chunking..."):
            recursive_result = recursive_chain.invoke(question)
        st.success("Answer (Recursive):")
        st.write(recursive_result['result'])

        correct_recursive = st.radio("Is the answer correct?", ["Yes", "No"], key="recursive_eval")
        f1_recursive = 1.0 if correct_recursive == "Yes" else 0.0
        st.metric("F1 Score", f"{f1_recursive:.2f}")

    with col3:
        st.subheader("✂️ Sentence Splitter Chunking")
        sentence_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=128, chunk_overlap=20)
        sentence_chunks = sentence_splitter.split_documents(docs)
        st.write(f"Chunks created: {len(sentence_chunks)}")

        sentence_db = FAISS.from_documents(sentence_chunks, embeddings)
        sentence_chain = RetrievalQA.from_chain_type(llm=llm, retriever=sentence_db.as_retriever(), return_source_documents=True)

        with st.spinner("🔍 Answering using Sentence Splitter..."):
            sentence_result = sentence_chain.invoke(question)
        st.success("Answer (Sentence Splitter):")
        st.write(sentence_result['result'])

        correct_sentence = st.radio("Is the answer correct?", ["Yes", "No"], key="sentence_eval")
        f1_sentence = 1.0 if correct_sentence == "Yes" else 0.0
        st.metric("F1 Score", f"{f1_sentence:.2f}")

