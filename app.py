import html
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from core.answer import AnswerResult
from core.query_understanding import analyze_query, normalize_text
from documents.chunker import chunk_document
from documents.docx_reader import read_docx
from documents.pdf_reader import read_pdf
from documents.text_reader import read_text
from qa_engine import answer_question
from web.search import perform_live_web_search

st.set_page_config(
    page_title="InfoQuery — Smart Q&A Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg,#0b1020 0%,#111936 52%,#0d2442 100%); }
    [data-testid="stSidebar"] { background: #17172b; }
    [data-testid="stSidebar"] * { color:#f3f4f6; }
    .hero { padding: 18px 28px 8px; }
    .hero h1 { margin:0; font-size:2.6rem; letter-spacing:-1px; }
    .hero p { color:#b9c2d8; font-size:1.04rem; margin-top:7px; }
    .badge { display:inline-block; padding:6px 11px; border-radius:999px; background:#252c57; color:#cbd5ff; font-size:.82rem; margin-right:6px; }
    .answer { background:rgba(25,33,66,.94); border-left:4px solid #7c6cff; border-radius:12px; padding:18px 20px; }
    .muted { color:#9ca8c2; }
    .small { font-size:.86rem; }
    .live { border:1px solid #394579; border-radius:12px; padding:10px 14px; background:rgba(28,38,78,.7); }
    .source-pill { display:inline-block; padding:3px 8px; border-radius:6px; background:#1e293b; color:#93c5fd; font-size:.78rem; margin-right:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_PATH = Path(__file__).parent / "data" / "knowledge_base.json"


@st.cache_data
def load_kb():
    if not DATA_PATH.exists():
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def read_uploaded_file(uploaded_file) -> str:
    """Reads PDF, DOCX, or TXT file into raw text."""
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()
    if name.endswith(".txt"):
        return read_text(raw_bytes)
    if name.endswith(".pdf"):
        return read_pdf(raw_bytes)
    if name.endswith(".docx"):
        return read_docx(raw_bytes)
    raise ValueError("Supported file types are: TXT, PDF and DOCX")


base_records = load_kb()
if "extra_records" not in st.session_state:
    st.session_state.extra_records = []
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("### Q&A Algorithm")
method = st.sidebar.selectbox(
    "Choose retrieval method",
    [
        "Best — Hybrid + Reranking",
        "Hybrid (TF-IDF + BM25)",
        "TF-IDF + Cosine Similarity",
        "BM25",
    ],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("### Knowledge Source")
source_mode = st.sidebar.radio(
    "Choose source",
    ["Built-in Knowledge", "Live Web Search", "My Uploaded Document"],
    label_visibility="collapsed",
)
top_k = st.sidebar.slider("Retrieved evidence", 1, 5, 3)
threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.60, 0.12, 0.01)

st.sidebar.divider()
st.sidebar.markdown("### 📌 About")
st.sidebar.write(
    "**InfoQuery** is an explainable retrieval-based Q&A system using TF-IDF, "
    "cosine similarity, Okapi BM25, and a multi-signal ensemble reranker with character n-grams, "
    "exact phrase matching, date-awareness, and query-coverage signals. Live Web mode performs "
    "multi-query discovery across Google News RSS, high-reputation direct feeds (BBC, NDTV, The Hindu), "
    "and Wikipedia, extracts article bodies, and reranks passages with zero hallucination."
)
st.sidebar.caption("Wheeltrix internship ML project • Streamlit • scikit-learn")

# Hero Header
st.markdown(
    '<div class="hero">'
    '<span class="badge">🤖 ML Retrieval</span>'
    '<span class="badge">TF-IDF</span>'
    '<span class="badge">BM25</span>'
    '<span class="badge">🏆 Best Reranker</span>'
    '<span class="badge">🌐 Live Web Search</span>'
    '<h1>💬 InfoQuery</h1>'
    '<p>Ask about sports, current news, technology, science, general knowledge or your own documents.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Knowledge Source Management
records = base_records + st.session_state.extra_records

if source_mode == "My Uploaded Document":
    uploaded = st.file_uploader("Upload a TXT, PDF or DOCX knowledge document", type=["txt", "pdf", "docx"])
    if uploaded:
        try:
            raw_text = read_uploaded_file(uploaded)
            st.session_state.extra_records = chunk_document(raw_text, uploaded.name)
            records = base_records + st.session_state.extra_records
            st.success(f"Loaded {len(st.session_state.extra_records)} searchable chunks from {uploaded.name}.")
        except Exception as exc:
            st.error(f"Document error: {exc}")
    elif not st.session_state.extra_records:
        st.info("Upload a document to ask questions directly from your own content.")

elif source_mode == "Live Web Search":
    st.markdown(
        '<div class="live">🌐 <b>Live Web Search:</b> Multi-query search retrieves articles '
        'from Google News RSS, verified direct feeds (BBC, NDTV, The Hindu, Times of India), '
        'and Wikipedia. Articles are cleaned, deduplicated, and ranked using the selected ML algorithm.</div>',
        unsafe_allow_html=True,
    )
else:
    st.caption("Built-in knowledge is a curated demo dataset. Use Live Web Search for live news or current events.")

st.divider()

# Question Input Section
left, right = st.columns([2.4, 1])
with left:
    st.markdown("### 💬 Ask your question")
    question = st.text_area(
        "Question",
        placeholder="Example: What happened on 25 October 2026? or Who won the FIFA World Cup 2026?",
        height=135,
        label_visibility="collapsed",
    )
    ask = st.button("🚀 Get Answer", type="primary")

with right:
    st.markdown("### Try an example")
    examples = [
        "What happened on 25 October 2026?",
        "latest news about Ukraine",
        "Who won the FIFA World Cup 2026?",
        "Who won the 2024 ICC Men's T20 World Cup?",
        "What is BM25 and why is it useful?",
        "What is machine learning?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            question = ex
            ask = True

# Answering Logic
if ask:
    if not question or len(question.strip()) < 3:
        st.warning("Please enter a meaningful question.")
    else:
        search_records = records

        if source_mode == "Live Web Search":
            with st.spinner("Analyzing question & fetching live web evidence..."):
                analysis = analyze_query(question)
                try:
                    live_passages = perform_live_web_search(question, analysis)
                except Exception as e:
                    live_passages = []

            if live_passages:
                search_records = live_passages
                st.success(
                    f"Retrieved and extracted {len(live_passages)} live web passages across "
                    f"{len(analysis.search_queries)} focused searches. Ranking with {method}."
                )
            else:
                st.warning(
                    "Live search could not retrieve web results or network is unreachable. "
                    "Falling back to built-in knowledge base."
                )

        if not search_records:
            st.error("No searchable knowledge is available.")
        else:
            answer_res, results = answer_question(
                question,
                search_records,
                method=method,
                top_k=top_k,
                threshold=threshold,
                return_details=True
            )
            answer = answer_res.answer
            score = answer_res.confidence

            st.session_state.history.insert(
                0,
                {
                    "question": question,
                    "answer": answer,
                    "score": score,
                    "method": method,
                    "source": source_mode,
                },
            )

            st.markdown("### ✨ Answer")
            if score < threshold or not answer_res.is_confident:
                st.warning(
                    "Low retrieval confidence. Try rephrasing the question, switching source/method, "
                    "or uploading a document containing the answer."
                )

            # Render Grounded Answer Box
            # Convert newlines to breaks or render markdown
            st.markdown(
                f'<div class="answer"><div style="font-size:1.08rem;line-height:1.65">{html.escape(answer).replace(chr(10), "<br>")}</div></div>',
                unsafe_allow_html=True,
            )

            # Metrics row
            c1, c2, c3 = st.columns(3)
            c1.metric("Retrieval confidence", f"{score:.0%}")
            c2.metric("Evidence returned", len(results))
            c3.metric("Searchable passages", len(search_records))

            # Display Grounded Sources
            if answer_res.sources_formatted:
                st.markdown("### 📚 Grounded Sources")
                for s in answer_res.sources_formatted:
                    pub = s.get("publisher", "Source")
                    title = s.get("title", "Article")
                    date = s.get("pub_date", "")
                    url = s.get("url", "")
                    date_str = f" • {date}" if date else ""
                    if url:
                        st.markdown(f"- **{pub}** — [{title}]({url}){date_str}")
                    else:
                        st.markdown(f"- **{pub}** — {title}{date_str}")

            # Retrieved Evidence Expander Section
            st.markdown("### 🔎 Retrieved evidence")
            for i, (record, result_score) in enumerate(results, start=1):
                label = f"{i}. {record['title']} • relevance {result_score:.0%}"
                with st.expander(label):
                    st.caption(f"Source: {record.get('publisher', record.get('category', 'Knowledge'))}")
                    if record.get("pub_date"):
                        st.caption(f"Date: {record['pub_date']}")
                    st.write(record["content"])
                    if record.get("url"):
                        st.markdown(f"[Read full article]({record['url']})")

st.divider()

# Recent Questions History
st.markdown("### 🕘 Recent questions")
if st.session_state.history:
    for item in st.session_state.history[:5]:
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")
        st.caption(f"Method: {item['method']} • Source: {item['source']} • Confidence: {item['score']:.0%}")
        st.markdown("---")
else:
    st.caption("Your recent questions will appear here after you run a query.")

st.markdown(
    "<div class='muted small'>InfoQuery • Wheeltrix Internship ML Project • Explainable Q&A with TF-IDF + BM25 + Multi-Signal Reranker</div>",
    unsafe_allow_html=True,
)
