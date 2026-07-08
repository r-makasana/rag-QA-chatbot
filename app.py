import os
import time
import tempfile
import validators
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0E0F11 !important;
    color: #E8E9EB !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background: #0E0F11 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #16181C !important;
    border-right: 1px solid #2A2D35 !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main header ── */
.rag-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 28px 0 20px 0;
    border-bottom: 1px solid #2A2D35;
    margin-bottom: 24px;
}

.rag-logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #6C8EFF 0%, #A78BFA 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}

.rag-title {
    font-size: 20px;
    font-weight: 600;
    color: #E8E9EB;
    letter-spacing: -0.3px;
}

.rag-subtitle {
    font-size: 12px;
    color: #6B7280;
    font-weight: 400;
    margin-top: 2px;
}

/* ── Source badge ── */
.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: #1E2028;
    border: 1px solid #2A2D35;
    border-radius: 20px;
    font-size: 11px;
    color: #9CA3AF;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 20px;
}

.source-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6C8EFF;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Chat messages ── */
.msg-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    align-items: flex-start;
}

.msg-row.user { flex-direction: row-reverse; }

.msg-avatar {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
    font-weight: 600;
}

.msg-avatar.user-av {
    background: #6C8EFF22;
    color: #6C8EFF;
    border: 1px solid #6C8EFF44;
}

.msg-avatar.ai-av {
    background: #A78BFA22;
    color: #A78BFA;
    border: 1px solid #A78BFA44;
}

.msg-bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.65;
    color: #E8E9EB;
}

.msg-bubble.user-bubble {
    background: #1C1F26;
    border: 1px solid #2A2D35;
    border-top-right-radius: 4px;
}

.msg-bubble.ai-bubble {
    background: #16181C;
    border: 1px solid #2A2D35;
    border-top-left-radius: 4px;
}

.msg-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
    font-size: 10px;
    color: #4B5563;
    font-family: 'JetBrains Mono', monospace;
}

.msg-meta .cite {
    color: #6C8EFF;
    font-size: 10px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #4B5563;
}

.empty-icon {
    font-size: 40px;
    margin-bottom: 16px;
    opacity: 0.5;
}

.empty-title {
    font-size: 16px;
    font-weight: 500;
    color: #6B7280;
    margin-bottom: 8px;
}

.empty-desc {
    font-size: 13px;
    color: #4B5563;
    line-height: 1.6;
}

/* ── Sidebar elements ── */
.sidebar-section {
    padding: 16px 0;
    border-bottom: 1px solid #2A2D35;
    margin-bottom: 4px;
}

.sidebar-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    color: #6B7280;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* ── Metrics log ── */
.metrics-log {
    background: #16181C;
    border: 1px solid #2A2D35;
    border-radius: 10px;
    padding: 16px;
    margin-top: 8px;
}

.metrics-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #1E2028;
    font-size: 12px;
}

.metrics-row:last-child { border-bottom: none; }

.metrics-q {
    color: #9CA3AF;
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 70%;
    font-family: 'Inter', sans-serif;
}

.metrics-t {
    color: #6C8EFF;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    flex-shrink: 0;
}

.metrics-summary {
    display: flex;
    gap: 16px;
    padding: 12px 0 4px 0;
    font-family: 'JetBrains Mono', monospace;
}

.metric-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.metric-val {
    font-size: 18px;
    font-weight: 500;
    color: #E8E9EB;
}

.metric-lbl {
    font-size: 10px;
    color: #4B5563;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input {
    background: #16181C !important;
    border: 1px solid #2A2D35 !important;
    color: #E8E9EB !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #6C8EFF !important;
    box-shadow: 0 0 0 2px #6C8EFF22 !important;
}

.stFileUploader > div {
    background: #16181C !important;
    border: 1px dashed #2A2D35 !important;
    border-radius: 8px !important;
}

.stButton > button {
    background: #1E2028 !important;
    border: 1px solid #2A2D35 !important;
    color: #9CA3AF !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    border-color: #6C8EFF !important;
    color: #6C8EFF !important;
}

.stRadio > div { gap: 4px !important; }

.stRadio label {
    color: #9CA3AF !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
}

.stSuccess {
    background: #0F2720 !important;
    border: 1px solid #064E3B !important;
    color: #6EE7B7 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}

.stSpinner > div { border-top-color: #6C8EFF !important; }

[data-testid="stChatInput"] > div {
    background: #16181C !important;
    border: 1px solid #2A2D35 !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #E8E9EB !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}

[data-testid="stChatMessageContent"] {
    background: transparent !important;
}

/* Override Streamlit chat bubbles */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

div[class*="stChatMessage"] {
    background: transparent !important;
}

/* Divider */
hr { border-color: #2A2D35 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0E0F11; }
::-webkit-scrollbar-thumb { background: #2A2D35; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3A3D45; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "metrics" not in st.session_state:
    st.session_state.metrics = []
if "source_name" not in st.session_state:
    st.session_state.source_name = None

# ── Helpers ─────────────────────────────────────────────────────────────────────
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def build_chain(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.1-8b-instant")

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given chat history and the latest question, rephrase as a standalone question. Don't answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the context below. Be concise and clear.\n\nContext: {context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

def reset_state():
    st.session_state.chain = None
    st.session_state.chat_history = []
    st.session_state.store = {}
    st.session_state.metrics = []
    st.session_state.source_name = None

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-label">Source</div>', unsafe_allow_html=True)
    source_type = st.radio("", ["PDF", "Website URL"], label_visibility="collapsed")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if source_type == "PDF":
        uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")
        if uploaded_file and st.session_state.chain is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            with st.spinner("Indexing PDF..."):
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                st.session_state.chain = build_chain(docs)
                st.session_state.source_name = f"📄 {uploaded_file.name}"
            st.success("Ready")
    else:
        url = st.text_input("", placeholder="https://...", label_visibility="collapsed")
        if url and st.session_state.chain is None:
            if validators.url(url):
                with st.spinner("Loading page..."):
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    st.session_state.chain = build_chain(docs)
                    short = url.replace("https://", "").replace("http://", "")[:40]
                    st.session_state.source_name = f"🌐 {short}"
                st.success("Ready")
            else:
                st.error("Include https://")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if st.session_state.chain:
        if st.button("↺  New source"):
            reset_state()
            st.rerun()

    # ── Metrics (hidden in expander) ──
    if st.session_state.metrics:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">Session Log</div>', unsafe_allow_html=True)

        total = len(st.session_state.metrics)
        avg = sum(m["time"] for m in st.session_state.metrics) / total
        fastest = min(m["time"] for m in st.session_state.metrics)

        st.markdown(f"""
        <div class="metrics-summary">
            <div class="metric-item">
                <span class="metric-val">{total}</span>
                <span class="metric-lbl">queries</span>
            </div>
            <div class="metric-item">
                <span class="metric-val">{avg:.1f}s</span>
                <span class="metric-lbl">avg</span>
            </div>
            <div class="metric-item">
                <span class="metric-val">{fastest:.1f}s</span>
                <span class="metric-lbl">best</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("View query log", expanded=False):
            log_html = '<div class="metrics-log">'
            for m in st.session_state.metrics:
                q = m["question"][:45] + "..." if len(m["question"]) > 45 else m["question"]
                log_html += f'''
                <div class="metrics-row">
                    <span class="metrics-q">{q}</span>
                    <span class="metrics-t">{m["time"]:.2f}s</span>
                </div>'''
            log_html += '</div>'
            st.markdown(log_html, unsafe_allow_html=True)

# ── Main area ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <div class="rag-logo">⬡</div>
    <div>
        <div class="rag-title">RAG Chat</div>
        <div class="rag-subtitle">Retrieval-Augmented Generation</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Source indicator
if st.session_state.source_name:
    st.markdown(f"""
    <div class="source-badge">
        <div class="source-dot"></div>
        {st.session_state.source_name}
    </div>
    """, unsafe_allow_html=True)

# ── Chat display ─────────────────────────────────────────────────────────────────
if not st.session_state.chain:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⬡</div>
        <div class="empty-title">No source loaded</div>
        <div class="empty-desc">Upload a PDF or paste a website URL<br>in the sidebar to start chatting.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="msg-avatar user-av">U</div>
                <div>
                    <div class="msg-bubble user-bubble">{msg["content"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            cite_html = ""
            if msg.get("sources"):
                cite_html = f'<span class="cite">⬡ {msg["sources"]}</span>'
            st.markdown(f"""
            <div class="msg-row">
                <div class="msg-avatar ai-av">R</div>
                <div>
                    <div class="msg-bubble ai-bubble">{msg["content"]}</div>
                    <div class="msg-meta">
                        {cite_html}
                        <span>{msg.get("time", "")}s</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Chat input ──
    question = st.chat_input("Ask anything about your source...")

    if question:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner(""):
            start = time.time()
            response = st.session_state.chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": "rag_session"}}
            )
            elapsed = round(time.time() - start, 2)
            answer = response["answer"]

            # Extract sources
            sources = response.get("context", [])
            source_labels = []
            for doc in sources:
                meta = doc.metadata
                if "page" in meta:
                    source_labels.append(f"p.{meta['page']+1}")
                elif "source" in meta:
                    s = meta["source"].replace("https://", "")[:30]
                    source_labels.append(s)
            source_str = "  ·  ".join(set(source_labels)) if source_labels else ""

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "time": elapsed,
            "sources": source_str
        })
        st.session_state.metrics.append({"question": question, "time": elapsed})
        st.rerun()