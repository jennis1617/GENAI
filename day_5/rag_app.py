import streamlit as st
import tempfile
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

st.set_page_config(page_title="RAG with TXT File", layout="centered")
st.title("📄 RAG App - Ask Questions from a Text File")

# Upload text file
uploaded_file = st.file_uploader("Upload a `.txt` file", type=["txt"])

chunk_method = st.selectbox(
    "Choose a chunking method",
    ["Fixed-size Chunking", "SentenceSplitter", "Recursive Chunking"]
)

query = st.text_input("Enter your question here:")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        tmp.write(content)
        temp_file_path = tmp.name

    try:
        loader = TextLoader(temp_file_path, encoding="utf-8")
        docs = loader.load()
        st.success("✅ File loaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

    # Apply chunking
    if chunk_method == "Fixed-size Chunking":
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    elif chunk_method == "SentenceSplitter":
        splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=20, tokens_per_chunk=128)
    elif chunk_method == "Recursive Chunking":
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = splitter.split_documents(docs)
    st.info(f"📦 Total chunks created: {len(chunks)}")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever()

    # Use local HuggingFace pipeline (flan-t5-small)
    pipe = pipeline("text2text-generation", model="google/flan-t5-small")
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    if query:
        with st.spinner("🔍 Searching for the answer..."):
            result = qa_chain.invoke(query)
        st.subheader("🧠 Answer")
        st.write(result['result'])

        with st.expander("📚 Source Chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source {i+1}**: {doc.page_content[:300]}...")













