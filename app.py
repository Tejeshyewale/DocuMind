import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_loader import load_pdf
from utils.vector_store import (
    split_documents,
    build_vector_store,
    load_vector_store,
    merge_vector_stores,
    calculate_md5,
    get_faiss_index_path,
)
from utils.rag_pipeline import answer_question

# Load environment configuration
load_dotenv()

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="DocuMind — Multi-Doc RAG Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enforce clean dark layout and style styling via CSS injection
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global font configuration */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: #f1f5f9;
    }
    
    /* Styled Title with gradient */
    .main-title {
        background: linear-gradient(135deg, #6366f1, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    /* Document card style in sidebar */
    .doc-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .doc-title {
        font-weight: 600;
        color: #f8fafc;
        font-size: 0.9rem;
        word-break: break-all;
    }
    .doc-meta {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Chat history message custom styles */
    .chat-bubble {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .chat-user {
        background-color: #1e293b;
        border-left: 4px solid #6366f1;
    }
    .chat-assistant {
        background-color: #0f172a;
        border-left: 4px solid #10b981;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Verify API key configuration
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY not found in configuration. Copy .env.example to .env and add your "
        "free key from https://console.groq.com/keys to get started."
    )
    st.stop()

# Initialize Session States
if "loaded_docs" not in st.session_state:
    # Key: MD5 file hash -> Value: dict(filename, page_count, chunk_count)
    st.session_state.loaded_docs = {}
if "vector_stores" not in st.session_state:
    # Key: MD5 file hash -> Value: FAISS vector_store instance
    st.session_state.vector_stores = {}
if "combined_vector_store" not in st.session_state:
    # Merged FAISS vector_store instance for the current session
    st.session_state.combined_vector_store = None
if "history" not in st.session_state:
    # List of tuple: (question, answer, citations, retrieved_docs)
    st.session_state.history = []


def rebuild_combined_index():
    """Merge all loaded individual vector stores into one session-wide index."""
    stores_list = list(st.session_state.vector_stores.values())
    if stores_list:
        st.session_state.combined_vector_store = merge_vector_stores(stores_list)
    else:
        st.session_state.combined_vector_store = None


def remove_document(file_hash: str):
    """Remove a document from the session and rebuild the active vector index."""
    if file_hash in st.session_state.loaded_docs:
        doc_info = st.session_state.loaded_docs[file_hash]
        del st.session_state.loaded_docs[file_hash]
        if file_hash in st.session_state.vector_stores:
            del st.session_state.vector_stores[file_hash]
        rebuild_combined_index()
        st.success(f"Removed '{doc_info['filename']}' successfully.")
        st.rerun()


# --- SIDEBAR PANEL ---
with st.sidebar:
    st.markdown("### 🛠️ DocuMind Dashboard")
    st.caption("Manage documents, configure vector caches, and clean sessions.")
    st.divider()

    # Multi-file Upload Form
    st.markdown("#### 📂 Load PDF Documents")
    uploaded_files = st.file_uploader(
        "Upload one or multiple PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        process_clicked = st.button("Process Selected PDFs", type="primary", use_container_width=True)
        if process_clicked:
            for uploaded_file in uploaded_files:
                try:
                    # Read bytes to verify content hash
                    file_bytes = uploaded_file.read()
                    file_hash = calculate_md5(file_bytes)
                    filename = uploaded_file.name

                    # Avoid redundant parsing if it's already in the active session list
                    if file_hash in st.session_state.loaded_docs:
                        st.info(f"'{filename}' is already loaded in the active session.")
                        continue

                    index_path = get_faiss_index_path(file_hash)

                    # Check Disk Cache
                    if os.path.exists(index_path):
                        with st.spinner(f"Loading '{filename}' from cache index..."):
                            store = load_vector_store(index_path)
                            st.session_state.vector_stores[file_hash] = store

                            # NOTE: docstore._dict is a dict of {id: Document}.
                            # Document objects are NOT hashable, so we must NOT
                            # wrap .values() in set(). Just count dict entries
                            # directly to get the number of cached chunks.
                            chunk_count = len(store.docstore._dict)

                            st.session_state.loaded_docs[file_hash] = {
                                "filename": filename,
                                "pages": "Cached",
                                "chunks": chunk_count
                            }
                            st.toast(f"⚡ Loaded '{filename}' from cache!", icon="💾")
                    else:
                        # Process fresh PDF
                        with st.spinner(f"Parsing & Chunking '{filename}'..."):
                            # Save temporarily to parse page structures
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(file_bytes)
                                tmp_path = tmp.name

                            try:
                                pages = load_pdf(tmp_path, original_filename=filename)
                                chunks = split_documents(pages)
                            finally:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)

                        with st.spinner(f"Generating Embeddings for '{filename}'..."):
                            store = build_vector_store(chunks, index_path)
                            st.session_state.vector_stores[file_hash] = store
                            st.session_state.loaded_docs[file_hash] = {
                                "filename": filename,
                                "pages": len(pages),
                                "chunks": len(chunks)
                            }
                            st.toast(f"✅ Indexed '{filename}' successfully!", icon="📝")

                except Exception as e:
                    st.error(f"Failed to process '{uploaded_file.name}': {str(e)}")

            # Update session-wide FAISS index
            rebuild_combined_index()
            st.rerun()

    st.divider()

    # Active Documents List
    st.markdown("#### 📄 Active Documents")
    if st.session_state.loaded_docs:
        for f_hash, doc_details in list(st.session_state.loaded_docs.items()):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.markdown(
                    f"""
                    <div class="doc-card">
                        <div class="doc-title">{doc_details['filename']}</div>
                        <div class="doc-meta">Pages: {doc_details['pages']} | Chunks: {doc_details['chunks']}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with col2:
                # Top offset alignment for button
                st.write("")
                if st.button("❌", key=f"del_{f_hash}", help=f"Remove {doc_details['filename']}"):
                    remove_document(f_hash)
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # Reset Buttons
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.history = []
        st.success("Chat history cleared!")
        st.rerun()

    if st.button("🚨 Reset All Session Data", type="secondary", use_container_width=True):
        st.session_state.loaded_docs = {}
        st.session_state.vector_stores = {}
        st.session_state.combined_vector_store = None
        st.session_state.history = []
        st.success("Session fully reset.")
        st.rerun()


# --- MAIN PANEL ---
st.markdown('<div class="main-title">📄 DocuMind</div>', unsafe_allow_html=True)
st.caption("A production-ready Multi-Document Q&A RAG pipeline powered by Groq & local HuggingFace sentence-transformers.")

st.divider()

# Verify if we have documents loaded
if not st.session_state.combined_vector_store:
    st.info("⬆️ Load one or more PDFs in the sidebar panel to begin asking questions!")
else:
    # Q&A Form
    with st.form("qa_form", clear_on_submit=True):
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            question = st.text_input(
                "Ask a question across all loaded documents:",
                placeholder="What passcode is mentioned in page 1?",
                label_visibility="collapsed"
            )
        with col2:
            submit_button = st.form_submit_button("Ask Question", type="primary", use_container_width=True)

    if submit_button and question:
        with st.spinner("Analyzing document contexts & calling Groq LLM..."):
            try:
                # Query index and get response plus citations and chunks
                answer, citations, retrieved_docs = answer_question(
                    st.session_state.combined_vector_store,
                    question
                )
                # Save into history (insert at start so newest Q&A is displayed at the top)
                st.session_state.history.insert(0, (question, answer, citations, retrieved_docs))
            except Exception as e:
                st.error(
                    f"⚠️ An error occurred while retrieving answers: {str(e)}. "
                    f"Please verify your API limits or key validity."
                )

    # Display Conversation History
    if st.session_state.history:
        st.markdown("### 💬 Conversation Thread")
        for q, a, citations, retrieved_docs in st.session_state.history:
            # User Message
            st.markdown(
                f"""
                <div class="chat-bubble chat-user">
                    <strong>Question:</strong><br>{q}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Assistant Message
            st.markdown(
                f"""
                <div class="chat-bubble chat-assistant">
                    <strong>Answer:</strong><br>{a}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Citations List
            if citations:
                st.markdown(f"**Citations:** *{', '.join(citations)}*")
            else:
                st.markdown("**Citations:** *None generated.*")
                
            # Collapsible Retrieved Context Sections
            with st.expander("🔍 Show Retrieved Grounding Chunks (GenAI Interview Context)"):
                for idx, doc in enumerate(retrieved_docs):
                    st.markdown(
                        f"""
                        **Chunk {idx + 1}** | Source: `{doc['source']}` | Page: `{doc['page']}`
                        ```text
                        {doc['content']}
                        ```
                        ---
                        """
                    )
            st.divider()
