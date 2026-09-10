"""
app.py
------
Streamlit UI for the Multi-Document RAG Research Assistant.

PIPELINE (end-to-end):
  1. User uploads PDF(s) in the sidebar
  2. On "Process Documents": load_pdf → chunk_documents → embed_and_store
  3. User types a question in the chat input
  4. On submit: retrieve_reranked(query) → generate_answer(query, chunks) → display

STREAMLIT STATE MODEL:
  Streamlit re-runs the entire script on every user interaction.
  `st.session_state` is a dict that persists across re-runs.
  We use it to store:
    - messages:          Chat history [{role, content, sources, is_grounded}]
    - processed_files:   Set of filenames already embedded
    - total_chunks:      Running count of chunks in ChromaDB

  WHY NOT RE-EMBED ON EVERY RUN?
    Because embed_and_store uses `upsert`, re-embedding the same file
    is safe — but it wastes ~5-10 seconds per file. We track processed
    filenames in session_state to skip already-embedded files.

RUN THE APP:
  cd rag-pdf-assistant
  streamlit run app.py
  Then open http://localhost:8501 in your browser.
"""

# ── Streamlit Cloud secrets → os.environ bridge ───────────────────────────────
# On Streamlit Cloud, API keys are stored in st.secrets (set via the dashboard).
# Our utils/config.py reads from os.environ via os.getenv().
# This block copies st.secrets into os.environ BEFORE any RAG module is imported,
# so config.py sees the correct values on first initialization.
#
# Locally: st.secrets is empty or doesn't exist — this block is a no-op.
# On Streamlit Cloud: st.secrets contains GROQ_API_KEY etc. from the dashboard.
import os
try:
    import streamlit as _st_bootstrap
    for _k, _v in _st_bootstrap.secrets.items():
        if _k not in os.environ:          # don't override .env values locally
            os.environ[_k] = str(_v)
except Exception:
    pass  # running locally without secrets — config.py will read from .env

import tempfile
from pathlib import Path

import streamlit as st

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store, get_collection_count
from retrieval.retriever import retrieve_reranked
from generation.llm_chain import generate_answer, GroundedAnswer
from utils.exceptions import (
    DocumentLoadError,
    EmptyDocumentError,
    EmbeddingError,
    VectorStoreError,
    RetrievalError,
    RerankingError,
    GenerationError,
)
from utils.logger import get_logger

log = get_logger(__name__)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Research Assistant",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
# Color palette (4 solid colors + neutrals — nothing more)
#   BG:       #111318  very dark charcoal, main canvas
#   SURFACE:  #1a1d24  card/panel backgrounds
#   BORDER:   #2a2d38  all borders
#   ACCENT:   #4361ee  one strong blue-indigo, used sparingly
#   TEXT:     #dde1ee  primary text
#   MUTED:    #6b7080  secondary / metadata text
#   OK:       #2e7d52  grounded indicator (solid green, not neon)
#   WARN:     #8b2e2e  not-found indicator (solid red, not neon)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 15px;
}

.stApp {
    background-color: #111318;
    color: #dde1ee;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0e1015 !important;
    border-right: 1px solid #2a2d38 !important;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    color: #6b7080;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Sidebar buttons ──────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background-color: #4361ee;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.88rem;
    letter-spacing: 0.01em;
    padding: 0.55rem 1rem;
    transition: background-color 0.15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #3451d1;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background-color: transparent;
    color: #6b7080;
    border: 1px solid #2a2d38;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    border-color: #4361ee;
    color: #dde1ee;
}

/* ── Metrics ──────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
}
[data-testid="stMetricLabel"] { color: #6b7080; font-size: 0.78rem; }
[data-testid="stMetricValue"] { color: #dde1ee; font-size: 1.4rem; font-weight: 600; }

/* ── Slider ───────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background-color: #4361ee;
}

/* ── File uploader ────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed #2a2d38;
    border-radius: 8px;
    padding: 0.5rem;
}

/* ── Page header ──────────────────────────────────────────────── */
.page-header {
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid #2a2d38;
    margin-bottom: 1.4rem;
}
.page-header h1 {
    font-size: 1.35rem;
    font-weight: 600;
    color: #dde1ee;
    margin: 0;
    letter-spacing: -0.2px;
}
.page-header p {
    color: #6b7080;
    font-size: 0.85rem;
    margin: 0.3rem 0 0;
}

/* ── Chat message rows ────────────────────────────────────────── */
.chat-row {
    display: flex;
    gap: 0.65rem;
    margin: 0.85rem 0;
    align-items: flex-start;
}
.chat-row.user-row {
    flex-direction: row-reverse;
}

/* ── Chat bubbles ─────────────────────────────────────────────── */
.chat-bubble {
    border-radius: 8px;
    padding: 0.8rem 1.05rem;
    max-width: 82%;
    line-height: 1.7;
    font-size: 0.92rem;
}

/* ── Source badges ────────────────────────────────────────────── */
.sources-row {
    margin-top: 0.6rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
}
.tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 4px;
    padding: 0.18rem 0.5rem;
    border: 1px solid;
}
.tag-grounded {
    background-color: #152b1e;
    border-color: #2e7d52;
    color: #4caf77;
}
.tag-notfound {
    background-color: #2b1515;
    border-color: #8b2e2e;
    color: #e06060;
}
.source-chip {
    display: inline-block;
    background-color: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 4px;
    padding: 0.18rem 0.55rem;
    font-size: 0.75rem;
    color: #8b95b5;
    font-family: 'JetBrains Mono', monospace;
    word-break: break-all;
}
.source-chip .score {
    color: #4361ee;
    font-weight: 600;
    margin-left: 0.35rem;
}

/* ── Welcome state ────────────────────────────────────────────── */
.welcome-state {
    margin: 3.5rem auto;
    max-width: 500px;
    text-align: center;
    color: #6b7080;
}
.welcome-state h2 {
    color: #dde1ee;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.welcome-state p {
    font-size: 0.88rem;
    line-height: 1.7;
    margin: 0;
}
.welcome-hint {
    display: inline-block;
    margin-top: 1.2rem;
    background-color: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    font-size: 0.82rem;
    color: #6b7080;
    font-style: italic;
}

/* ── Spinner / input ──────────────────────────────────────────── */
.stChatInput textarea {
    background-color: #1a1d24 !important;
    border: 1px solid #2a2d38 !important;
    color: #dde1ee !important;
    border-radius: 8px !important;
}
.stChatInput textarea:focus {
    border-color: #4361ee !important;
}

/* ── Expander (citation cards) ────────────────────────────────── */
.streamlit-expanderHeader {
    background-color: #1a1d24 !important;
    border: 1px solid #2a2d38 !important;
    border-radius: 6px !important;
    color: #6b7080 !important;
    font-size: 0.82rem !important;
}

/* ── Dividers ─────────────────────────────────────────────────── */
hr { border-color: #2a2d38 !important; }

/* ── Native st.chat_message styling ──────────────────────────── */
/* Streamlit renders chat messages as [data-testid="stChatMessage"].
   We style them here so they match the design system instead of
   using Streamlit's default avatar-based bubbles. */
[data-testid="stChatMessage"] {
    background-color: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    gap: 0.65rem;
}
/* Remove the default avatar circle */
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}
/* Tighter line height and font for answer text */
[data-testid="stChatMessage"] p {
    line-height: 1.75;
    margin-bottom: 0.5rem;
    font-size: 0.93rem;
    color: #dde1ee;
}
/* Bullet and numbered list spacing */
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol {
    padding-left: 1.3rem;
    margin: 0.4rem 0 0.7rem;
}
[data-testid="stChatMessage"] li {
    line-height: 1.7;
    margin-bottom: 0.25rem;
    color: #dde1ee;
    font-size: 0.93rem;
}
/* Bold terms */
[data-testid="stChatMessage"] strong {
    color: #c5cbdf;
    font-weight: 600;
}
/* User message — slightly different background */
[data-testid="stChatMessage"]:has(.user-label) {
    background-color: #1c2140;
    border-color: #2d3260;
}
/* Sender label inside chat message */
.msg-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
    display: block;
}
.msg-label.user-label  { color: #4361ee; }
.msg-label.rag-label   { color: #6b7080; }

/* ── Hide Streamlit chrome (desktop) ─────────────────────────── */
#MainMenu, footer { visibility: hidden; }

/* ── Onboarding / Upload banner ──────────────────────────────── */
@keyframes pulse-ring {
    0%   { transform: scale(1);   opacity: 1; }
    70%  { transform: scale(1.55); opacity: 0; }
    100% { transform: scale(1.55); opacity: 0; }
}
@keyframes bounce-left {
    0%, 100% { transform: translateX(0); }
    50%       { transform: translateX(-5px); }
}
.upload-banner {
    background: linear-gradient(135deg, #1a1f35 0%, #1a1d24 100%);
    border: 1px solid #2d3260;
    border-left: 4px solid #4361ee;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.upload-banner-icon {
    position: relative;
    width: 36px;
    height: 36px;
    flex-shrink: 0;
}
.upload-banner-icon .dot {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background-color: #4361ee;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    color: white;
    position: relative;
    z-index: 2;
}
.upload-banner-icon .ring {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 2px solid #4361ee;
    animation: pulse-ring 1.6s ease-out infinite;
}
.upload-banner-text strong {
    display: block;
    color: #dde1ee;
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.upload-banner-text span {
    color: #6b7080;
    font-size: 0.82rem;
    line-height: 1.5;
}
.upload-banner-arrow {
    margin-left: auto;
    color: #4361ee;
    font-size: 1.3rem;
    animation: bounce-left 1.2s ease-in-out infinite;
    flex-shrink: 0;
}
/* Onboarding steps card */
.onboard-card {
    background-color: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 12px;
    padding: 1.8rem 1.6rem;
    margin: 1.5rem auto;
    max-width: 520px;
}
.onboard-card h2 {
    color: #dde1ee;
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0 0 0.4rem;
    letter-spacing: -0.2px;
}
.onboard-card .subtitle {
    color: #6b7080;
    font-size: 0.85rem;
    margin: 0 0 1.4rem;
    line-height: 1.5;
}
.onboard-steps {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
}
.onboard-step {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    background-color: #111318;
    border: 1px solid #2a2d38;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    transition: border-color 0.15s;
}
.step-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background-color: #4361ee;
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.step-body strong {
    display: block;
    color: #dde1ee;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.step-body span {
    color: #6b7080;
    font-size: 0.8rem;
    line-height: 1.5;
}
.step-body code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    background-color: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 3px;
    padding: 0.1rem 0.35rem;
    color: #8b95b5;
}
.onboard-hint {
    margin-top: 1.2rem;
    text-align: center;
    font-size: 0.8rem;
    color: #6b7080;
    font-style: italic;
}
@media (max-width: 768px) {
    .onboard-card {
        max-width: 100%;
        padding: 1.2rem 1rem;
        margin: 0.8rem auto;
    }
    .onboard-card h2 { font-size: 1rem; }
    .upload-banner { padding: 0.8rem 0.9rem; gap: 0.75rem; }
    .upload-banner-text strong { font-size: 0.86rem; }
    .upload-banner-text span { font-size: 0.76rem; }
}
/* Hide header on desktop — it's just clutter there */
@media (min-width: 769px) {
    header { visibility: hidden; }
}
/* On mobile: show header so the sidebar ▶ toggle is accessible */
@media (max-width: 768px) {
    header {
        visibility: visible !important;
        background-color: #111318 !important;
        border-bottom: 1px solid #2a2d38 !important;
    }
    /* Hide the deploy / GitHub / top-right action buttons inside the header */
    header [data-testid="stMainMenu"],
    header .stDeployButton,
    header [data-testid="baseButton-header"],
    header [data-testid="stAppViewBlockContainer"] > div:last-child {
        display: none !important;
    }
}
/* The ▶ collapsed-control button must ALWAYS be visible on every screen */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    z-index: 9999 !important;
}


/* ── Layout: Desktop (>1024px) ────────────────────────────────── */
.block-container {
    padding-top: 1rem !important;
    max-width: 860px;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ── Responsive: Tablet (≤1024px) ────────────────────────────── */
@media (max-width: 1024px) {
    .block-container {
        max-width: 100% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    .chat-bubble { max-width: 88%; }
    .welcome-state { max-width: 420px; }
}

/* ── Responsive: Mobile (≤768px) ─────────────────────────────── */
@media (max-width: 768px) {
    html, body, [class*="css"] { font-size: 14px; }

    .block-container {
        max-width: 100% !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 0.5rem !important;
        padding-bottom: 5rem !important;
    }

    .page-header { padding: 0.8rem 0 0.7rem; margin-bottom: 0.8rem; }
    .page-header h1 { font-size: 1.1rem; }
    .page-header p  { font-size: 0.78rem; }

    .chat-bubble {
        max-width: 96%;
        font-size: 0.88rem;
        padding: 0.65rem 0.85rem;
    }
    [data-testid="stChatMessage"] {
        padding: 0.65rem 0.8rem;
        border-radius: 6px;
    }
    [data-testid="stChatMessage"] p  { font-size: 0.88rem; }
    [data-testid="stChatMessage"] li { font-size: 0.88rem; }

    .source-chip {
        font-size: 0.68rem;
        padding: 0.14rem 0.4rem;
        max-width: 100%;
        word-break: break-all;
    }
    .sources-row { gap: 0.3rem; }
    .tag { font-size: 0.65rem; padding: 0.14rem 0.4rem; }

    .welcome-state {
        max-width: 100%;
        margin: 2rem auto;
        padding: 0 0.5rem;
    }
    .welcome-state h2   { font-size: 1rem; }
    .welcome-state p    { font-size: 0.82rem; }
    .welcome-hint {
        font-size: 0.76rem;
        padding: 0.4rem 0.7rem;
        display: block;
        text-align: center;
    }

    [data-testid="stMetricValue"] { font-size: 1.15rem; }
    [data-testid="stMetricLabel"] { font-size: 0.72rem; }

    .msg-label { font-size: 0.62rem; }

    .stChatInput textarea {
        font-size: 0.9rem !important;
        min-height: 44px !important;
    }
    .streamlit-expanderHeader { font-size: 0.78rem !important; }

    [data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 85vw !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        font-size: 0.85rem;
        padding: 0.6rem 0.9rem;
        min-height: 44px;
    }
}

/* ── Responsive: Small Mobile (≤480px) ───────────────────────── */
@media (max-width: 480px) {
    html, body, [class*="css"] { font-size: 13px; }

    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    .page-header h1 { font-size: 1rem; }

    .chat-bubble {
        max-width: 100%;
        font-size: 0.85rem;
        padding: 0.55rem 0.7rem;
    }
    .source-chip { font-size: 0.62rem; }
    .welcome-state h2 { font-size: 0.95rem; }
    .welcome-state p  { font-size: 0.78rem; }

    /* Stack chips vertically on very small screens */
    .sources-row {
        flex-direction: column;
        align-items: flex-start;
    }
}

/* ── Overflow prevention on all screens ───────────────────────── */
img, pre, code, table { max-width: 100%; }
pre, code {
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
}
[data-testid="stChatMessage"] pre { overflow-x: auto; max-width: 100%; }

/* ── Touch-friendly tap targets (mobile & tablet) ─────────────── */
@media (hover: none) and (pointer: coarse) {
    .stButton > button {
        min-height: 44px !important;
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
    }
    [data-testid="stFileUploader"] { padding: 0.8rem !important; }
    [data-testid="stSlider"] input[type="range"] { height: 28px; }
}
</style>
""", unsafe_allow_html=True)



# ── Session State Initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    """Initialise session state keys that don't yet exist."""
    defaults = {
        "messages": [],           # [{role, content, sources, is_grounded}]
        "processed_files": set(), # filenames already embedded
        "total_chunks": 0,        # live count from ChromaDB
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── Helper: refresh chunk count from ChromaDB ─────────────────────────────────
def _refresh_chunk_count() -> None:
    try:
        st.session_state.total_chunks = get_collection_count()
    except Exception:
        pass  # Don't crash the UI if ChromaDB isn't initialised yet


_refresh_chunk_count()


# ── Helper: Process a single uploaded PDF ─────────────────────────────────────
def _process_pdf(uploaded_file) -> str:
    """
    Run the full ingestion pipeline on one uploaded PDF file.

    Steps:
      1. Save the in-memory upload to a temp file (PyMuPDF needs a file path)
      2. load_pdf  → extract text per page
      3. chunk_documents → split into overlapping Chunk objects
      4. embed_and_store → encode + upsert into ChromaDB

    Returns:
        A status message string for display.

    Raises:
        DocumentLoadError, EmptyDocumentError, EmbeddingError, VectorStoreError
    """
    filename = uploaded_file.name

    # Streamlit's UploadedFile is an in-memory buffer — PyMuPDF needs a
    # real file path. We save to a temp file and pass its path.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        pages = load_pdf(tmp_path)
        chunks = chunk_documents(pages, source_file=filename)
        count = embed_and_store(chunks)
        st.session_state.processed_files.add(filename)
        _refresh_chunk_count()
        return f"**{filename}** — {len(pages)} pages, {count} chunks indexed"
    finally:
        # Always delete the temp file, even if processing failed.
        Path(tmp_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the knowledge base.",
        label_visibility="collapsed",
    )

    if st.button("Process Documents", type="primary", use_container_width=True,
                 disabled=not uploaded_files):
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.processed_files
        ]
        if not new_files:
            st.info("All uploaded files have already been processed.")
        else:
            with st.spinner(f"Processing {len(new_files)} file(s)..."):
                for f in new_files:
                    try:
                        msg = _process_pdf(f)
                        st.success(msg)
                    except EmptyDocumentError as e:
                        st.error(f"**{f.name}** — empty document: {e}")
                    except DocumentLoadError as e:
                        st.error(f"**{f.name}** — load error: {e}")
                    except (EmbeddingError, VectorStoreError) as e:
                        st.error(f"**{f.name}** — storage error: {e}")

    st.divider()

    # ── Knowledge Base Status ─────────────────────────────────────────────────
    st.markdown("## Knowledge Base")
    n_docs = len(st.session_state.processed_files)
    n_chunks = st.session_state.total_chunks

    col1, col2 = st.columns(2)
    col1.metric("Documents", n_docs)
    col2.metric("Chunks", n_chunks)

    if st.session_state.processed_files:
        st.markdown("**Indexed files**")
        for fname in sorted(st.session_state.processed_files):
            st.markdown(
                f'<div style="font-size:0.8rem;color:#6b7080;padding:0.15rem 0;'
                f'font-family:\'JetBrains Mono\',monospace;word-break:break-all;">'
                f'{fname}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Retrieval Settings ────────────────────────────────────────────────────
    st.markdown("## Settings")
    top_k = st.slider(
        "Results to retrieve",
        min_value=1, max_value=10, value=5,
        help="Number of document chunks retrieved per question. "
             "Higher = more context, slightly slower. Recommended: 3–5.",
    )

    st.divider()

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════════

# Page header — plain text, no decoration
st.markdown("""
<div class="page-header">
    <h1>DocMind</h1>
    <p>Ask questions about your documents. Answers are sourced and verifiable.</p>
</div>
""", unsafe_allow_html=True)


# ── Chat History Display ──────────────────────────────────────────────────────
def _render_sources(sources: list[dict], is_grounded: bool = True) -> str:
    """Render grounding tag + source chips. No emoji — text labels only."""
    parts = []

    if is_grounded and sources:
        parts.append('<span class="tag tag-grounded">verified</span>')
    elif not is_grounded:
        parts.append('<span class="tag tag-notfound">not in documents</span>')

    for s in sources:
        score = s.get("rerank_score", s.get("score", 0))
        parts.append(
            f'<span class="source-chip">'
            f'{s["source_file"]} &middot; p.{s["page_number"]}'
            f'<span class="score">{score:.2f}</span>'
            f'</span>'
        )

    if not parts:
        return ""
    return f'<div class="sources-row">{"".join(parts)}</div>'


# ── Floating upload banner (shown whenever no docs are loaded) ────────────────
# Shown at the top of the main area regardless of chat state.
# Tells mobile users exactly where to go to upload files.
if not st.session_state.processed_files:
    st.markdown("""
    <div class="upload-banner">
        <div class="upload-banner-icon">
            <div class="ring"></div>
            <div class="dot">&#9776;</div>
        </div>
        <div class="upload-banner-text">
            <strong>No documents uploaded yet</strong>
            <span>
                Tap the <b style="color:#dde1ee">&#9776; menu</b> button
                (top-left) &rarr; upload your PDF &rarr; tap
                <b style="color:#dde1ee">Process Documents</b>.
            </span>
        </div>
        <div class="upload-banner-arrow">&#8592;</div>
    </div>
    """, unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="onboard-card">
            <h2>Welcome to DocMind &#128216;</h2>
            <p class="subtitle">Ask questions about any PDF — every answer is cited
            directly from your document.</p>
            <div class="onboard-steps">
                <div class="onboard-step">
                    <div class="step-num">1</div>
                    <div class="step-body">
                        <strong>Open the sidebar</strong>
                        <span>Tap the <b style="color:#dde1ee">&#9776;</b> menu icon
                        in the top-left corner (or the
                        <b style="color:#dde1ee">&#9654;</b> arrow if sidebar is closed).</span>
                    </div>
                </div>
                <div class="onboard-step">
                    <div class="step-num">2</div>
                    <div class="step-body">
                        <strong>Upload your PDF</strong>
                        <span>Drag &amp; drop or tap <b style="color:#dde1ee">Browse files</b>
                        in the sidebar. Multiple PDFs are supported.</span>
                    </div>
                </div>
                <div class="onboard-step">
                    <div class="step-num">3</div>
                    <div class="step-body">
                        <strong>Process &amp; Ask</strong>
                        <span>Tap <code>Process Documents</code> in the sidebar,
                        then type any question in the chat box below.</span>
                    </div>
                </div>
            </div>
            <p class="onboard-hint">Try: &ldquo;What is the main argument of this paper?&rdquo;</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                # Use st.chat_message so markdown is processed natively.
                # The sender label is a small HTML span above the text.
                with st.chat_message("user", avatar=None):
                    st.markdown(
                        '<span class="msg-label user-label">You</span>',
                        unsafe_allow_html=True,
                    )
                    # st.markdown renders the text — bold, bullets, etc. work here
                    st.markdown(msg["content"])
            else:
                msg_sources  = msg.get("sources", [])
                msg_grounded = msg.get("is_grounded", True)
                sources_html = _render_sources(msg_sources, msg_grounded)

                with st.chat_message("assistant", avatar=None):
                    st.markdown(
                        '<span class="msg-label rag-label">DocMind</span>',
                        unsafe_allow_html=True,
                    )
                    # KEY FIX: st.markdown() processes the LLM's markdown output.
                    # Before this change, the text was injected as raw HTML content,
                    # so **bold** showed as "**bold**" and - bullets stayed literal.
                    st.markdown(msg["content"])

                    # Source badges below the answer text
                    if sources_html:
                        st.markdown(sources_html, unsafe_allow_html=True)

                    # Expandable citation cards
                    if msg_sources and msg_grounded:
                        with st.expander(
                            f"View {len(msg_sources)} source excerpt(s)", expanded=False
                        ):
                            for i, src in enumerate(msg_sources, 1):
                                score = src.get("rerank_score", src.get("score", 0))
                                st.markdown(
                                    f"**Excerpt {i}** &nbsp;&middot;&nbsp; "
                                    f"`{src['source_file']}` &nbsp;&middot;&nbsp; "
                                    f"Page {src['page_number']} &nbsp;&middot;&nbsp; "
                                    f"score: `{score:.3f}`"
                                )
                                text = src.get("text", "")
                                st.caption(
                                    text[:500] + ("..." if len(text) > 500 else "")
                                )
                                if i < len(msg_sources):
                                    st.divider()


# ── Chat Input ────────────────────────────────────────────────────────────────
question = st.chat_input(
    "Ask a question about your documents...",
    disabled=(st.session_state.total_chunks == 0),
)

if question:
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})

    # 2. Retrieve → Generate
    with st.spinner("Searching documents..."):
        try:
            chunks = retrieve_reranked(question, top_n=top_k)
        except (RetrievalError, RerankingError) as e:
            st.error(f"Retrieval failed: {e}")
            chunks = []

    if chunks:
        with st.spinner("Generating answer..."):
            try:
                grounded: GroundedAnswer = generate_answer(question, chunks)
            except GenerationError as e:
                # Store error as assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Generation failed: {e}",
                    "sources": [],
                    "is_grounded": False,
                })
                st.rerun()

        # Build the sources list from GroundedAnswer.citations
        # Citations already have text, source_file, page_number, rerank_score
        sources = [
            {
                "source_file": c["source_file"],
                "page_number":  c["page_number"],
                "text":         c.get("text", ""),
                "rerank_score": c.get("rerank_score", c.get("score", 0)),
                "score":        c.get("score", 0),
            }
            for c in grounded.citations
        ]
        answer_text = grounded.answer
        is_grounded = grounded.is_grounded
    else:
        answer_text = (
            "I don't know based on the provided documents. "
            "No relevant excerpts were found."
        )
        sources     = []
        is_grounded = False

    # 3. Add assistant message and rerun to render it
    st.session_state.messages.append({
        "role":        "assistant",
        "content":     answer_text,
        "sources":     sources,
        "is_grounded": is_grounded,
    })
    st.rerun()
