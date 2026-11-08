import streamlit as st
from pathlib import Path
import logging
from src.retrieval import  get_hybrid_retriever, get_router
from src.chain import (
    get_rag_chain,
    get_query_rewriter,
    get_faithfulness_checker,
    prepare_answer,
    check_answer,
)
from src.ingestion import (load_documents,split_documents,add_to_vectorstore,get_vectorstore,get_indexed_files,save_indexed_files,
    calculate_file_hash,
    delete_document_chunks,
)
# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Doc RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <div style="text-align: center; padding: 1.5rem 0 0.5rem 0;">
        <h1 style="margin-bottom: 0.2rem;">📄 Multi-Doc RAG Assistant</h1>
        <p style="color: #9CA3AF; font-size: 0.95rem;">
            Ask questions across multiple documents, grounded and cited.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .source-card {
        background-color: #1A1D24;
        border: 1px solid #2D3139;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .source-card .source-label {
        color: #6366F1;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }
    .source-card .source-text {
        color: #9CA3AF;
        font-size: 0.85rem;
        line-height: 1.4;
    }
    .strategy-badge {
        display: inline-block;
        background-color: #2D3139;
        color: #9CA3AF;
        font-size: 0.75rem;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Data directory
# --------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
import uuid

def new_conversation():
    cid = str(uuid.uuid4())
    st.session_state.conversations[cid] = {"title": "New chat", "messages": []}
    st.session_state.current_id = cid

def switch_conversation(cid):
    st.session_state.current_id = cid

if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    new_conversation()

# Point `messages` at the open conversation's list.
# It is the same list object, so appends are saved to the conversation.
st.session_state.messages = (
    st.session_state.conversations[st.session_state.current_id]["messages"]
)
# --------------------------------------------------
# Sidebar - Document Upload
# --------------------------------------------------

with st.sidebar:
    st.button("➕ New chat", on_click=new_conversation, use_container_width=True)
    st.caption("Conversations")
    conversation_list = st.container()
    st.divider()
    st.markdown(
        """
        <div style="padding-bottom: 0.5rem;">
            <h3 style="margin-bottom: 0;">📁 Documents</h3>
            <p style="color: #9CA3AF; font-size: 0.85rem; margin-top: 0.2rem;">
                Upload files to build your knowledge base
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or CSV files",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.caption(f"**{len(uploaded_files)} file(s) selected**")
        for f in uploaded_files:
            st.markdown(f"&nbsp;&nbsp;📄 {f.name}")
    else:
        st.info("Upload one or more documents to get started.")

    st.divider()

    if st.button("🗑️ Clear chat"):
        st.session_state.messages.clear()
        st.session_state.conversations[st.session_state.current_id]["title"] = "New chat"
        st.rerun()

# --------------------------------------------------
# Index Documents
# --------------------------------------------------

if uploaded_files and st.sidebar.button("Index documents"):

    indexed_files = get_indexed_files()
    vectorstore = get_vectorstore()  # Initialize the vector store
    files_to_index = []

    for file in uploaded_files:
        save_path = DATA_DIR / file.name

        # Save the uploaded file
        with open(save_path, "wb") as out:
            out.write(file.getbuffer())

        # Calculate its content hash
        file_hash = calculate_file_hash(save_path)

        old_hash = indexed_files.get(file.name)

        # Skip if the same file content is already indexed
        if old_hash == file_hash:
            st.sidebar.info(f"{file.name} is already indexed.")
            continue

        # Check whether identical content exists under another filename
        duplicate_exists = file_hash in indexed_files.values()

        if duplicate_exists:
            # Remove the previous version if this filename was modified
            if old_hash is not None:
                
                if vectorstore:
                    delete_document_chunks(vectorstore, file.name)

            indexed_files[file.name] = file_hash
            st.sidebar.info(f"Duplicate content detected: {file.name}")
            continue

        # If the same filename has changed, remove its old chunks
        if old_hash is not None:
            vectorstore = get_vectorstore()
            if vectorstore:
                delete_document_chunks(vectorstore, file.name)

        

            files_to_index.append(file.name)

    if files_to_index:
        with st.spinner(f"Indexing {len(files_to_index)} file(s)..."):

            docs = load_documents(
                str(DATA_DIR),
                only_files=set(files_to_index)
            )

            chunks = split_documents(docs)
            vectorstore = add_to_vectorstore(chunks, vectorstore=vectorstore)

            for filename in files_to_index:
                indexed_files[filename] = calculate_file_hash(DATA_DIR / filename)

            if vectorstore:
                vectorstore.save_local("faiss_index")

        st.sidebar.success(
            f"Indexed {len(chunks)} chunks from {len(files_to_index)} file(s)."
        )

    save_indexed_files(indexed_files)

    # Clear cached resources
    st.cache_resource.clear()
        

# --------------------------------------------------
# Load RAG Pipeline
# --------------------------------------------------

@st.cache_resource
def load_pipeline():
    vectorstore = get_vectorstore()

    if vectorstore is None:
        return None

    return {
        "vectorstore": vectorstore,
        "hybrid_retriever": get_hybrid_retriever(vectorstore),
        "router": get_router(),
        "rag_chain": get_rag_chain(),
        "rewriter": get_query_rewriter(),
        "checker": get_faithfulness_checker(),
    }

# --------------------------------------------------
# Chat UI
# --------------------------------------------------

st.divider()



AVATARS = {"user": "🧑", "assistant": "🤖"}
MAX_HISTORY_MESSAGES = 6  # last 3 question/answer pairs

for role, text in st.session_state.messages:
    with st.chat_message(role, avatar=AVATARS[role]):
        st.write(text)

# --------------------------------------------------
# User Input
# --------------------------------------------------

user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Store user message
    st.session_state.messages.append(("user", user_input))
    convo = st.session_state.conversations[st.session_state.current_id]
    if convo["title"] == "New chat":
        convo["title"] = user_input[:30] + ("..." if len(user_input) > 30 else "")
    
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.write(user_input)

    # Load pipeline
    pipeline = load_pipeline()

    # --------------------------------------------------
    # No documents indexed
    # --------------------------------------------------

    if pipeline is None:
        assistant_text = "Please upload and index at least one document first."

        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.write(assistant_text)

        st.session_state.messages.append(("assistant", assistant_text))

    # --------------------------------------------------
    # Documents available
    # --------------------------------------------------

    else:
        try:
            with st.chat_message("assistant", avatar=AVATARS["assistant"]):

                # 1. Rewrite the follow-up and retrieve (no streaming yet)
                with st.spinner("Searching documents..."):
                    question, docs, strategy = prepare_answer(
                        user_input,
                        st.session_state.messages[:-1][-MAX_HISTORY_MESSAGES:],
                        pipeline["rewriter"],
                        pipeline["vectorstore"],
                        pipeline["router"],
                        pipeline["hybrid_retriever"],
                    )

                # 2. Stream the answer as it is written
                answer = st.write_stream(
                    pipeline["rag_chain"].stream(
                        {"docs": docs, "question": question}
                    )
                )

                st.caption(f"🔎 Searched for: {question}")

                # 3. Faithfulness check on the finished answer
                check = check_answer(pipeline["checker"], docs, answer)

                if not check.is_faithful:
                    st.warning(f"⚠️ Possible hallucination: {check.reason}")

                st.markdown(
                    f'<span class="strategy-badge">🔍 {strategy}</span>',
                    unsafe_allow_html=True,
                )

                with st.expander("📚 Sources"):
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page")
                        label = f"{source}, page {page + 1}" if page is not None else source
                        snippet = doc.page_content[:200].replace("\n", " ") + "..."
                        st.markdown(
                            f'<div class="source-card"><div class="source-label">📄 {label}</div>'
                            f'<div class="source-text">{snippet}</div></div>',
                            unsafe_allow_html=True,
                        )

            # Save the finished answer to the conversation
            st.session_state.messages.append(("assistant", answer))

        # --------------------------------------------------
        # Error handling
        # --------------------------------------------------

        except Exception as e:
            logging.exception("Answer generation failed")
            error_text = str(e)
            

            if "rate limit" in error_text.lower() or "429" in error_text:
                friendly = (
                    "I'm getting a lot of requests right now (free-tier rate limit). "
                    "Please wait about a minute and try again."
                )
            else:
                friendly = (
                    "Something went wrong while generating the answer. "
                    "Please try again."
                )

            st.session_state.messages.append(("assistant", friendly))

            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                st.write(friendly)

with conversation_list:
    for cid, convo in reversed(list(st.session_state.conversations.items())):
        is_current = cid == st.session_state.current_id
        st.button(
            ("👉 " if is_current else "💬 ") + convo["title"],
            key=f"conv_{cid}",
            on_click=switch_conversation,
            args=(cid,),
            use_container_width=True,
            type="primary" if is_current else "secondary",
        )