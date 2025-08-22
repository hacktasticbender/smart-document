
import io
import os
import time
import math
import regex as re
import numpy as np
import pdfplumber
import streamlit as st



from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

# -------------------- Config & Setup --------------------

st.set_page_config(page_title="Smart Document QA (RAG)", page_icon="📄", layout="wide")
load_dotenv()  # load .env if present

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class DocChunk:
    text: str
    source: str
    page: int
    chunk_id: str

# Cache the encoder and tokenizer heavy objects
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    return SentenceTransformer(model_name)

def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    # normalized -> cosine similarity equals inner product with IndexFlatIP
    return vecs.astype("float32")

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    text = text.strip()
    if not text:
        return chunks

    # Process in smaller slices to avoid memory issues
    for start in range(0, len(text), chunk_size - overlap):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
    return chunks


def extract_pdf_pages(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    """Return list of (page_number, text) starting at 1."""
    out = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            out.append((i, txt))
    return out

def build_faiss(index_vectors: np.ndarray) -> faiss.IndexFlatIP:
    d = index_vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(index_vectors)  # vectors must be normalized for cosine
    return index

def search(index: faiss.IndexFlatIP, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(query_vec, k)
    return D, I

def format_citations(results: List[DocChunk]) -> str:
    # Inline short citations [filename p.X]
    uniq = []
    seen = set()
    for r in results:
        tag = f"{os.path.basename(r.source)} p.{r.page}"
        if tag not in seen:
            uniq.append(tag)
            seen.add(tag)
    return " | ".join(f"[{u}]" for u in uniq)

# -------------------- OpenAI (Optional) --------------------
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "No OPENAI_API_KEY found. Using extractive mode."
    try:
        # New SDK style
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"OpenAI client error: {e}"

def llm_answer_with_openai(client, question: str, contexts: List[DocChunk]) -> str:
    # Compose a grounded prompt
    context_block = "\n\n---\n\n".join(
        f"[{i+1}] Source: {os.path.basename(c.source)} (page {c.page})\n{c.text}"
        for i, c in enumerate(contexts)
    )
    system_msg = (
        "You are a precise research assistant. Answer the user's question ONLY using the provided sources. "
        "If the answer is not contained in the sources, say you don't know. "
        "Always keep answers concise and include inline citations like [filename p.X]."
    )
    user_msg = f"Question: {question}\n\nSources:\n{context_block}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(OpenAI error: {e})"

def extractive_answer(question: str, contexts: List[DocChunk]) -> str:
    # Simple extractive response: show top snippets with citations
    header = f"Top relevant excerpts for: **{question}**"
    parts = []
    for i, c in enumerate(contexts, 1):
        parts.append(
            f"**{i}. {os.path.basename(c.source)} – p.{c.page}**\n"
            f"{c.text[:700]}{'…' if len(c.text) > 700 else ''}"
        )
    return header + "\n\n" + "\n\n".join(parts)

# -------------------- Streamlit App --------------------

st.title("📄 Smart Document QA (RAG)")
st.caption("Upload PDFs → Embed with SentenceTransformers → Search with FAISS → (Optional) Ask via OpenAI → Answers with citations.")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk size (characters)", 600, 2000, 1200, 50)
    overlap = st.slider("Chunk overlap (characters)", 50, 400, 150, 10)
    top_k = st.slider("Top-K passages", 2, 10, 5, 1)
    provider = st.selectbox("LLM Provider", ["OPENAI", "NONE"], index=0 if os.getenv("LLM_PROVIDER","OPENAI")=="OPENAI" else 1)
    st.divider()
    clear = st.button("Clear Knowledge Base", use_container_width=True)

if "doc_chunks" not in st.session_state or clear:
    st.session_state.doc_chunks: List[DocChunk] = []
    st.session_state.embeddings = None
    st.session_state.index = None
    st.session_state.embedder = None
    st.session_state.history = []

# Upload
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        new_chunks: List[DocChunk] = []
        for up in uploaded_files:
            file_bytes = up.read()
            pages = extract_pdf_pages(file_bytes, up.name)
            for page_no, txt in pages:
                for j, ch in enumerate(chunk_text(txt, chunk_size=chunk_size, overlap=overlap)):
                    chunk_id = f"{up.name}-p{page_no}-c{j}"
                    new_chunks.append(DocChunk(text=ch, source=up.name, page=page_no, chunk_id=chunk_id))
        # Merge with existing
        st.session_state.doc_chunks.extend(new_chunks)

    st.success(f"Added {len(new_chunks)} chunks from {len(uploaded_files)} file(s).")

# Build / rebuild index when we have chunks and no index
if st.session_state.doc_chunks and (st.session_state.index is None or st.session_state.embeddings is None):
    with st.spinner("Building embeddings & FAISS index..."):
        if st.session_state.embedder is None:
            st.session_state.embedder = get_embedder()
        texts = [c.text for c in st.session_state.doc_chunks]
        X = embed_texts(st.session_state.embedder, texts)  # normalized
        st.session_state.embeddings = X
        st.session_state.index = build_faiss(X)
    st.success("Index ready. Ask away!")

# Chat UI
st.subheader("Ask a question")
query = st.chat_input("Type your question about the uploaded documents...")

if query:
    if not st.session_state.index:
        st.warning("Please upload PDFs first so I can build an index.")
    else:
        # Embed query and search
        qvec = embed_texts(st.session_state.embedder, [query])
        D, I = search(st.session_state.index, qvec, k=min(top_k, len(st.session_state.doc_chunks)))
        hits = [st.session_state.doc_chunks[int(idx)] for idx in I[0] if idx != -1]

        # Decide answer mode
        answer_text = ""
        citations_inline = format_citations(hits)
        if provider == "OPENAI":
            client, warn = get_openai_client()
            if warn:
                st.info(warn)
            if client:
                answer_text = llm_answer_with_openai(client, query, hits)
            else:
                answer_text = extractive_answer(query, hits)
        else:
            answer_text = extractive_answer(query, hits)

        # Persist chat history
        st.session_state.history.append({"role":"user","content":query})
        st.session_state.history.append({"role":"assistant","content":answer_text})

        # Render
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            st.write(answer_text)
            with st.expander("Sources"):
                for i, h in enumerate(hits, 1):
                    st.markdown(f"**{i}. {os.path.basename(h.source)} – p.{h.page}**")
                    st.write(h.text[:1500] + ("…" if len(h.text) > 1500 else ""))

# Show prior history
if st.session_state.history:
    st.divider()
    st.subheader("Conversation History")
    for turn in st.session_state.history[-8:]:
        with st.chat_message(turn["role"] if turn["role"] in ("user","assistant") else "assistant"):
            st.write(turn["content"])
            from openai import OpenAI

