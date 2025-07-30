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
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Streamlit app config
st.set_page_config(page_title="RAG Chunking Evaluation", layout="wide")
st.title("🔍 Chunking Methods Evaluation with Semantic Metrics")

# Upload & input
uploaded_file = st.file_uploader("📄 Upload a `.txt` file", type=["txt"])
question = st.text_input("❓ Enter your question:")

if uploaded_file and question:
    # Save uploaded content temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        tmp.write(content)
        temp_file_path = tmp.name

    # Load and prepare
    loader = TextLoader(temp_file_path, encoding="utf-8")
    docs = loader.load()
    
    # Embeddings
    rag_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")  # Needed for semantic similarity

    # LLM pipeline
    llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # Define chunking strategies
    chunking_strategies = {
        "Fixed-size": CharacterTextSplitter(chunk_size=500, chunk_overlap=50),
        "Recursive": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
        "Sentence Splitter": SentenceTransformersTokenTextSplitter(tokens_per_chunk=128, chunk_overlap=20)
    }

    # Run all methods
    answers = {}
    st.info("⏳ Generating answers for each chunking method...")

    for label, splitter in chunking_strategies.items():
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, rag_embeddings)
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), return_source_documents=True)
        response = chain.invoke(question)
        answer = response['result']
        answers[label] = answer

    # Display answers
    st.subheader("📝 Answers from Each Chunking Method")
    col1, col2, col3 = st.columns(3)
    for label, col in zip(answers.keys(), [col1, col2, col3]):
        with col:
            st.markdown(f"**{label}**")
            st.write(answers[label])

    # -----------------------------------
    # Auto Evaluation (Semantic Metrics)
    # -----------------------------------
    st.divider()
    st.subheader("📊 Automatic Evaluation Summary (Semantic Similarity)")

    reference_label = "Recursive"  # Use Recursive as the base reference
    reference_vector = semantic_model.encode(answers[reference_label], convert_to_tensor=True)

    evaluation = {
        "Chunking Method": [],
        "Avg Similarity to Others": [],
        "Precision (vs Recursive)": [],
        "Recall (vs Recursive)": [],
        "F1 Score (vs Recursive)": []
    }

    for label in answers:
        current_vector = semantic_model.encode(answers[label], convert_to_tensor=True)

        # Avg similarity to all other methods
        other_vectors = [
            semantic_model.encode(answers[other], convert_to_tensor=True)
            for other in answers if other != label
        ]
        avg_sim = sum([util.pytorch_cos_sim(current_vector, v).item() for v in other_vectors]) / len(other_vectors)

        # Precision, Recall, F1 (vs Recursive)
        precision = util.pytorch_cos_sim(current_vector, reference_vector).item()
        recall = util.pytorch_cos_sim(reference_vector, current_vector).item()
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)

        evaluation["Chunking Method"].append(label)
        evaluation["Avg Similarity to Others"].append(round(avg_sim, 4))
        evaluation["Precision (vs Recursive)"].append(round(precision, 4))
        evaluation["Recall (vs Recursive)"].append(round(recall, 4))
        evaluation["F1 Score (vs Recursive)"].append(round(f1, 4))

    df = pd.DataFrame(evaluation)
    st.table(df)
