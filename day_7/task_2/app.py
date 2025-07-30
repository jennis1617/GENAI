import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import tempfile
import pandas as pd

# Streamlit setup
st.set_page_config(page_title="RAG Prompt Evaluation", layout="wide")
st.title("🔍 RAG + Prompting Techniques Evaluation")
st.markdown("Compare prompt strategies using semantic similarity metrics (Precision, Recall, F1).")

# Upload document
uploaded_file = st.file_uploader("📄 Upload a `.txt` file", type="txt")
query = st.text_input("💬 Enter your query")

# Load sentence transformer for semantic evaluation
@st.cache_resource
def load_eval_model():
    return SentenceTransformer("all-mpnet-base-v2")

eval_model = load_eval_model()

# Load LLM
@st.cache_resource
def load_local_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")

local_llm = load_local_llm()

# Prompt templates
def chain_of_thought_prompt(context, query):
    return f"""Use chain-of-thought reasoning.\nContext:\n{context}\nQuestion: {query}\nAnswer (step-by-step):"""

def tree_of_thought_prompt(context, query):
    return f"""Break the answer into logical parts.\nContext:\n{context}\nQuestion: {query}\nAnswer (tree-like reasoning):"""

# Query rephrasing prompt for retrieval
def rephrase_query_prompt(query):
    return f"""You are a helpful assistant. Reformulate the following question to improve retrieval of relevant academic or technical context from a document.\n\nOriginal Question: {query}\n\nRewritten Question:"""

# Rephrase the query using the LLM
def rephrase_query(query):
    prompt = rephrase_query_prompt(query)
    try:
        response = local_llm(prompt, max_new_tokens=50, temperature=0.3)
        return response[0]["generated_text"].strip()
    except Exception as e:
        return query  # fallback to original query

# Answer generation with prompting in retrieval and generation
def generate_answer(vectorstore, prompt_fn, query):
    # 🔁 Rephrase the query using prompting (used in retrieval)
    reformulated_query = rephrase_query(query)

    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(reformulated_query)
    context = "\n".join([doc.page_content for doc in docs[:3]])
    prompt = prompt_fn(context, query)  # Use original query in generation

    try:
        response = local_llm(prompt, max_new_tokens=200, temperature=0.3)
        return response[0]["generated_text"], reformulated_query
    except Exception as e:
        return f"⚠️ Failed to generate response: {str(e)}", reformulated_query

# Main logic
if uploaded_file and query:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        loader = TextLoader(file_path)
        documents = loader.load()

        # Embedding model for RAG
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Split documents
        recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        fixed_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        recursive_chunks = recursive_splitter.split_documents(documents)
        fixed_chunks = fixed_splitter.split_documents(documents)

        recursive_vector = FAISS.from_documents(recursive_chunks, embeddings)
        fixed_vector = FAISS.from_documents(fixed_chunks, embeddings)

        # Generate all answers
        st.subheader("📘 Generating Answers...")
        with st.spinner("Generating all answers using Chain & Tree prompting..."):

            answers = {}
            reformulations = {}

            for label, (vector, prompt_fn) in {
                "Chain + Recursive": (recursive_vector, chain_of_thought_prompt),
                "Chain + Fixed": (fixed_vector, chain_of_thought_prompt),
                "Tree + Recursive": (recursive_vector, tree_of_thought_prompt),
                "Tree + Fixed": (fixed_vector, tree_of_thought_prompt),
            }.items():
                answer, rephrased = generate_answer(vector, prompt_fn, query)
                answers[label] = answer
                reformulations[label] = rephrased

        # Display answers and reformulated query
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔗 Chain-of-Thought + Recursive")
            st.write(answers["Chain + Recursive"])
            st.caption(f"Rephrased Query: `{reformulations['Chain + Recursive']}`")

            st.markdown("#### 🌳 Tree-of-Thought + Recursive")
            st.write(answers["Tree + Recursive"])
            st.caption(f"Rephrased Query: `{reformulations['Tree + Recursive']}`")

        with col2:
            st.markdown("#### 🔗 Chain-of-Thought + Fixed")
            st.write(answers["Chain + Fixed"])
            st.caption(f"Rephrased Query: `{reformulations['Chain + Fixed']}`")

            st.markdown("#### 🌳 Tree-of-Thought + Fixed")
            st.write(answers["Tree + Fixed"])
            st.caption(f"Rephrased Query: `{reformulations['Tree + Fixed']}`")

        # Semantic Evaluation
        st.divider()
        st.subheader("📊 Semantic Evaluation (vs Chain + Recursive)")

        reference_label = "Chain + Recursive"
        reference_emb = eval_model.encode(answers[reference_label], convert_to_tensor=True)

        evaluation = {
            "Prompting Method": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": []
        }

        for label, response in answers.items():
            current_emb = eval_model.encode(response, convert_to_tensor=True)
            precision = util.pytorch_cos_sim(current_emb, reference_emb).item()
            recall = util.pytorch_cos_sim(reference_emb, current_emb).item()
            f1 = (2 * precision * recall) / (precision + recall + 1e-8)

            evaluation["Prompting Method"].append(label)
            evaluation["Precision"].append(round(precision, 4))
            evaluation["Recall"].append(round(recall, 4))
            evaluation["F1 Score"].append(round(f1, 4))

        df = pd.DataFrame(evaluation)
        st.table(df)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
else:
    st.info("👆 Upload a `.txt` file and enter a query to start.")


