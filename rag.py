# rag_module.py
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import chromadb
from chromadb.config import Settings
import uuid
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import numpy as np
from typing import List, Tuple


class RAGModule:
    """
    RAGModule keeps your pipeline (SentenceTransformer -> Chroma -> CrossEncoder -> LLM)
    but adds: query expansion, context summarization, multi-sample self-consistency,
    grounding/pruning, improved judge parsing and safer Groq handling.
    """

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(name="rag_chunks")
        self.groq_client = Groq(api_key=groq_api_key)

        # Default LLM
        self.model_name = "openai/gpt-oss-20b"

        self.splitter_kwargs = dict(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )

    # --------------------------
    # Utilities: scraping & chunking
    # --------------------------
    def scrape_urls(self, urls: List[str]) -> List[str]:
        docs = []
        for url in urls:
            try:
                res = requests.get(url, timeout=10)
                soup = BeautifulSoup(res.text, "html.parser")
                paragraphs = soup.find_all("p")
                text = "\n".join(p.get_text() for p in paragraphs)
                docs.append(text)
            except Exception as e:
                print(f"Error scraping {url}: {e}")
        return docs

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.splitter_kwargs["separators"],
            length_function=self.splitter_kwargs["length_function"]
        )
        return splitter.split_text(text)

    def chunk_and_store(self, docs: List[str]):
        for doc in docs:
            chunks = self.chunk_text(doc)
            if not chunks:
                continue
            embeddings = self.embedder.encode(chunks).tolist()
            ids = [str(uuid.uuid4()) for _ in chunks]
            self.collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    # --------------------------
    # Query expansion
    # --------------------------
    def expand_query(self, query: str) -> List[str]:
        q = query.strip()
        expansions = [
            q,
            f"Explain {q}",
            f"What is {q}?",
            f"How does {q} work",
            f"{q} fundamentals",
            f"{q} basics and overview",
            f"Benefits and use cases of {q}",
        ]
        seen, out = set(), []
        for s in expansions:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # --------------------------
    # Retrieval
    # --------------------------
    def retrieve(self, query: str, top_k: int = 10, per_expansion_k: int = 5) -> List[str]:
        expanded = self.expand_query(query)
        all_docs = []
        for q in expanded:
            q_emb = self.embedder.encode(q).tolist()
            res = self.collection.query(query_embeddings=[q_emb], n_results=per_expansion_k)
            docs = res.get("documents", [[]])[0]
            all_docs.extend(docs)

        seen, dedup_docs = set(), []
        for d in all_docs:
            if d not in seen:
                seen.add(d)
                dedup_docs.append(d)

        candidate_docs = dedup_docs[: max(len(dedup_docs), top_k)]

        if candidate_docs:
            pairs = [(query, doc) for doc in candidate_docs]
            try:
                scores = self.cross_encoder.predict(pairs)
                reranked = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
                docs_sorted = [d for d, _ in reranked][:top_k]
            except Exception:
                docs_sorted = candidate_docs[:top_k]
        else:
            docs_sorted = []

        return docs_sorted

    # --------------------------
    # Summarization
    # --------------------------
    def summarize_chunks(self, chunks: List[str], max_output_tokens: int = 400) -> str:
        if not chunks:
            return ""

        joined = "\n\n".join(chunks[:10])
        prompt = f"""
You are an assistant that creates a concise, factual summary (5–8 concise bullet points)
of the following extracted document fragments.
Only include facts present in the fragments. Do not add external knowledge.

Fragments:
{joined}

Summary (5–8 bullets; each bullet one clear, short sentence):
"""
        try:
            resp = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_output_tokens
            )
            text = ""
            if hasattr(resp, "choices") and resp.choices:
                c = resp.choices[0]
                if getattr(c, "message", None) is not None:
                    text = getattr(c.message, "content", "") or ""
                else:
                    text = getattr(c, "text", "") or ""
            else:
                data = resp if isinstance(resp, dict) else {}
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") if data else ""
            return (text or "").strip()
        except Exception as e:
            print(f"Warning: summarization failed: {e}")
            return "\n".join(chunks[:5])

    # --------------------------
    # Self-consistency
    # --------------------------
    def _generate_multiple_answers(self, prompt: str, n: int = 3, temps=[0.2, 0.5, 0.8]) -> List[str]:
        answers = []
        for i in range(n):
            temp = temps[i % len(temps)]
            try:
                resp = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=float(temp),
                    max_tokens=1024
                )
                text = ""
                if hasattr(resp, "choices") and resp.choices:
                    c = resp.choices[0]
                    if getattr(c, "message", None) is not None:
                        text = getattr(c.message, "content", "") or ""
                    else:
                        text = getattr(c, "text", "") or ""
                else:
                    data = resp if isinstance(resp, dict) else {}
                    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") if data else ""
                if text:
                    answers.append(text.strip())
            except Exception as e:
                print(f"Warning: generation sample {i} failed: {e}")
        return list(dict.fromkeys(answers))

    def _pick_most_central_answer(self, answers: List[str]) -> str:
        if not answers:
            return ""
        if len(answers) == 1:
            return answers[0]
        emb = self.embedder.encode(answers, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(emb, emb).cpu().numpy()
        mean_sims = [(sims[i].sum() - 1.0) / (len(answers)-1) for i in range(len(answers))]
        return answers[int(np.argmax(mean_sims))]

    # --------------------------
    # Grounding
    # --------------------------
    def prune_ungrounded_sentences(self, answer: str, chunks: List[str], threshold: float = 0.55, min_sentences: int = 2) -> str:
        if not answer or not chunks:
            return answer

        chunk_embs = self.embedder.encode(chunks, convert_to_tensor=True)
        sentences = re.split(r'(?<=[\.\?\!]\s)|\n', answer)
        kept = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_emb = self.embedder.encode(s_clean, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(s_emb, chunk_embs).cpu().numpy().squeeze()
            max_sim = float(np.max(sims)) if sims.size else 0.0
            if max_sim >= threshold:
                kept.append(s_clean)
        if len(kept) >= min_sentences:
            return " ".join(kept)
        return answer  # fallback if pruning makes answer too short

    # --------------------------
    # Main answer generation
    # --------------------------
    def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        use_summarization: bool = True,
        self_consistency_n: int = 3,
        grounding_threshold: float = 0.55,
        answer_style: str = "concise"
    ) -> Tuple[str, dict]:
        if use_summarization and context_chunks:
            compressed_ctx = self.summarize_chunks(context_chunks)
            ctx_for_model = compressed_ctx + "\n\n" + "\n\n".join(context_chunks[:3])
        else:
            ctx_for_model = "\n\n".join(context_chunks[:6])

        if answer_style == "concise":
            style_instruction = (
                "Answer in 3–5 concise bullet points. "
                "Each bullet should be a complete, clear sentence (max ~25 words)."
            )
        elif answer_style == "paragraph":
            style_instruction = (
                "Answer in 1–2 short, clear paragraphs. "
                "Cover all main points from the context."
            )
        else:
            style_instruction = "Answer briefly and clearly."

        prompt = f"""
You are a helpful assistant. Use ONLY the provided context to answer the user's question.
If the answer cannot be found in the context, say "Not found in the provided documents".

Context:
{ctx_for_model}

Question: {question}

{style_instruction}
"""

        samples = self._generate_multiple_answers(prompt, n=self_consistency_n)
        selected = self._pick_most_central_answer(samples) if samples else ""
        pruned = self.prune_ungrounded_sentences(selected, context_chunks, threshold=grounding_threshold)

        metadata = {
            "raw_samples": samples,
            "selected_sample": selected,
            "pruned_answer": pruned,
            "used_context_preview": ctx_for_model,
            "answer_style": answer_style
        }

        return pruned.strip() or selected.strip(), metadata

    # --------------------------
    # Evaluation
    # --------------------------
    def evaluate_answer(self, query: str, generated_answer: str, context_chunks: List[str]) -> dict:
        reference = "\n".join(context_chunks[:5]) if context_chunks else ""

        y_true, y_pred = reference.split(), generated_answer.split()
        common = set(y_true) & set(y_pred)
        f1 = (2 * len(common)) / (len(y_true) + len(y_pred) + 1e-8) if (len(y_true) + len(y_pred)) > 0 else 0.0

        try:
            rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_l = rouge.score(reference, generated_answer)["rougeL"].fmeasure if reference and generated_answer else 0.0
        except Exception:
            rouge_l = 0.0

        try:
            P, R, F1 = bert_score([generated_answer], [reference], lang="en", verbose=False)
            bert = float(F1[0].item())
        except Exception:
            bert = 0.0

        try:
            emb_ref = self.embedder.encode(reference or ["."], convert_to_tensor=True)
            emb_ans = self.embedder.encode(generated_answer or ".", convert_to_tensor=True)
            cosine = float(util.pytorch_cos_sim(emb_ref, emb_ans).item())
        except Exception:
            cosine = 0.0

        grounded_ratio = 0.0
        try:
            sentences = [s.strip() for s in re.split(r'(?<=[\.\?\!]\s)|\n', generated_answer) if s.strip()]
            if sentences and context_chunks:
                chunk_embs = self.embedder.encode(context_chunks, convert_to_tensor=True)
                counts = 0
                for s in sentences:
                    s_emb = self.embedder.encode(s, convert_to_tensor=True)
                    sims = util.pytorch_cos_sim(s_emb, chunk_embs).cpu().numpy().squeeze()
                    if sims.size and float(np.max(sims)) >= 0.55:
                        counts += 1
                grounded_ratio = counts / len(sentences)
        except Exception:
            grounded_ratio = 0.0

        judge_prompt = f"""
You are an impartial judge. Rate the quality of the following AI-generated answer from 1 to 10 with respect to:
- correctness vs the provided context
- faithfulness to context (no hallucinations)
- concision and clarity

Return:
Score: <integer 1-10>
Explanation: <short paragraph>

Context:
{reference}

Answer:
{generated_answer}
"""
        llm_score, llm_explanation = None, ""
        try:
            resp = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an impartial AI judge evaluating answer quality."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            text = ""
            if hasattr(resp, "choices") and resp.choices:
                c = resp.choices[0]
                if getattr(c, "message", None) is not None:
                    text = getattr(c.message, "content", "") or ""
                else:
                    text = getattr(c, "text", "") or ""
            else:
                data = resp if isinstance(resp, dict) else {}
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") if data else ""

            llm_explanation = text.strip()
            m = re.search(r'(\b[1-9]\b|\b10\b)', llm_explanation)
            if m:
                llm_score = int(m.group(0))
            else:
                m2 = re.search(r'score\s*[:\-]?\s*(\d{1,2})', llm_explanation, flags=re.IGNORECASE)
                if m2:
                    val = int(m2.group(1))
                    if 1 <= val <= 10:
                        llm_score = val
        except Exception as e:
            llm_explanation = f"Judge failed: {e}"

        return {
            "F1 Score": round(float(f1), 4),
            "ROUGE-L": round(float(rouge_l), 4),
            "BERTScore": round(float(bert), 4),
            "Cosine Similarity": round(float(cosine), 4),
            "Grounded Ratio": round(float(grounded_ratio), 4),
            "LLM-as-a-Judge (out of 10)": (llm_score if llm_score is not None else 0),
            "LLM-Judge Explanation": llm_explanation
        }






