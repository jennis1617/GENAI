import streamlit as st
from rag_module import RAGModule
import time

st.set_page_config(page_title="DocuMentor – Chat RAG UI", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

st.title("📘 DocuMentor: Your AI-Powered RAG Chat (Enhanced)")

# ---- Sidebar: settings ----
# ---- Sidebar: settings ----
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("🔐 Groq API Key:", type="password")
    model_choice = st.selectbox("LLM Model (Groq)", options=["openai/gpt-oss-20b"], index=0)
    st.markdown("---")
    st.subheader("Answer Style")
    answer_style = st.radio("Choose style:", ["concise", "paragraph"], index=0)   # NEW
    st.markdown("---")
    st.subheader("Generation options (no pipeline change)")
    use_summarization = st.checkbox("Use context summarization (compress top chunks)", value=True)
    self_consistency_n = st.slider("Self-consistency samples (N)", min_value=1, max_value=5, value=3)
    grounding_threshold = st.slider("Grounding similarity threshold", min_value=30, max_value=80, value=55)
    grounding_threshold = grounding_threshold / 100.0


# if no key entered yet show info and halt
if not groq_api_key:
    st.info("Please enter your Groq API Key in the sidebar to start.")
    st.stop()

# init RAGModule
rag = RAGModule(groq_api_key)
rag.model_name = model_choice

# ---- Ingest Documents ----
with st.expander("📚 Ingest Documents (paste URLs or raw text)"):
    urls_input = st.text_area("Paste up to 10 URLs (one per line):", height=140)
    raw_text_input = st.text_area("Or paste raw documents (separate with === on new line):", height=140)
    if st.button("Ingest"):
        docs = []
        if urls_input.strip():
            urls = [u.strip() for u in urls_input.splitlines() if u.strip()][:10]
            docs.extend(rag.scrape_urls(urls))
        if raw_text_input.strip():
            parts = [p.strip() for p in raw_text_input.split("===") if p.strip()]
            docs.extend(parts)
        if docs:
            rag.chunk_and_store(docs)
            st.success(f"Ingested {len(docs)} documents / fragments.")
        else:
            st.warning("No input provided to ingest.")

# ---- Chat Controls ----
st.header("Ask a question about ingested content")
question = st.text_input("Ask a question:", value="", key="question_input")
if st.button("Generate Answer") and question.strip():
    with st.spinner("Retrieving context and generating answer..."):
        try:
            # retrieve chunks
            chunks = rag.retrieve(question, top_k=10)
            st.session_state.last_chunks = chunks

            # generate answer (with summarization, self-consistency and grounding options)
            answer, meta = rag.generate_answer(
                question,
                chunks,
                use_summarization=use_summarization,
                self_consistency_n=self_consistency_n,
                grounding_threshold=grounding_threshold
            )

            # evaluate
            metrics = rag.evaluate_answer(question, answer, chunks)

            # store chat
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "metrics": metrics,
                "chunks": chunks,
                "meta": meta
            })
        except Exception as e:
            st.error(f"Error during generation: {e}")

# show history
st.markdown("---")
st.header("💬 Chat History")
if not st.session_state.chat_history:
    st.info("No conversations yet.")
else:
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer']}")
        with st.expander("🔍 View Evaluation Metrics"):
            for k, v in entry["metrics"].items():
                st.write(f"**{k}**: {v}")
        with st.expander("📄 Retrieved Chunks & Meta"):
            st.write("**Used chunks (top):**")
            for i, c in enumerate(entry["chunks"][:8]):
                st.markdown(f"- Chunk {i+1}:")
                st.code(c[:800] + ("..." if len(c) > 800 else ""))
            st.write("**Generation meta:**")
            st.json(entry.get("meta", {}))





