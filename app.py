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
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

st.set_page_config(page_title="RAG Chat", layout="centered")
st.title("Chat with PDF or Website")

# Session state
if "store" not in st.session_state:
    st.session_state.store = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "metrics" not in st.session_state:
    st.session_state.metrics = []

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
        ("system", "Given the chat history and latest user question, "
                   "rephrase it as a standalone question. Don't answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the question using only the context below.
         At the end, always cite the source page or URL.

         Context: {context}"""),
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

# --- Sidebar: Input source ---
with st.sidebar:
    st.header("Input Source")
    source_type = st.radio("Choose source", ["PDF", "Website URL"])

    if source_type == "PDF":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file and st.session_state.chain is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            with st.spinner("Processing PDF..."):
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                st.session_state.chain = build_chain(docs)
                st.session_state.source_name = uploaded_file.name
            st.success("PDF ready!")

    else:
        url = st.text_input("Enter website URL")
        if url and st.session_state.chain is None:
            if validators.url(url):
                with st.spinner("Loading website..."):
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    st.session_state.chain = build_chain(docs)
                    st.session_state.source_name = url
                st.success("Website loaded!")
            else:
                st.error("Invalid URL. Include https://")

    # Reset button
    if st.button("Reset / Load new source"):
        st.session_state.chain = None
        st.session_state.chat_history = []
        st.session_state.store = {}
        st.session_state.metrics = []
        st.rerun()

    # Metrics panel
    if st.session_state.metrics:
        st.divider()
        st.subheader("Performance Metrics")
        total = len(st.session_state.metrics)
        avg_time = sum(m["time"] for m in st.session_state.metrics) / total
        st.metric("Questions asked", total)
        st.metric("Avg response time", f"{avg_time:.2f}s")
        st.metric("Fastest response", f"{min(m['time'] for m in st.session_state.metrics):.2f}s")
        st.metric("Slowest response", f"{max(m['time'] for m in st.session_state.metrics):.2f}s")

# --- Main chat area ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "time" in message:
            st.caption(f"Response time: {message['time']:.2f}s")

if st.session_state.chain:
    question = st.chat_input("Ask a question...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.time()
                response = st.session_state.chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": "rag_session"}}
                )
                elapsed = round(time.time() - start, 2)
                answer = response["answer"]

                # Show sources
                sources = response.get("context", [])
                source_info = []
                for doc in sources:
                    meta = doc.metadata
                    if "page" in meta:
                        source_info.append(f"Page {meta['page']+1}")
                    elif "source" in meta:
                        source_info.append(meta["source"])

                st.write(answer)
                if source_info:
                    st.caption(f"Sources: {', '.join(set(source_info))}")
                st.caption(f"Response time: {elapsed}s")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "time": elapsed
        })
        st.session_state.metrics.append({
            "question": question,
            "time": elapsed
        })
else:
    st.info("Upload a PDF or enter a website URL in the sidebar to get started.")