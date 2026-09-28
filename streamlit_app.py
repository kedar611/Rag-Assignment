import os
import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine

# Load local .env if available
load_dotenv()

st.set_page_config(
    page_title="PDF RAG Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished look
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .badge-rag {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "indexed_file_name" not in st.session_state:
    st.session_state.indexed_file_name = None
if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- SIDEBAR: Configuration ---
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    
    provider = st.selectbox(
        "Model Provider",
        options=["Groq (Recommended - Free & Fast)", "Google Gemini", "Offline Demo (No Key Needed)", "OpenAI"],
        index=0,
        help="Select the backend provider. Groq is free and fast. Offline Demo mode requires no API key.",
    )
    if provider == "Groq (Recommended - Free & Fast)":
        provider_key = "groq"
    elif provider == "Google Gemini":
        provider_key = "gemini"
    elif provider == "Offline Demo (No Key Needed)":
        provider_key = "demo"
    else:
        provider_key = "openai"

    api_key = ""
    if provider_key == "demo":
        st.info("💡 **Offline Demo Mode Active:** No API key or internet model access required. Uses local text similarity to retrieve excerpts.")
    elif provider_key == "groq":
        default_key = os.getenv("GROQ_API_KEY") or ""
        st.caption("💡 [Get a free Groq API Key](https://console.groq.com/keys)")
        api_key = st.text_input(
            "Groq API Key",
            value=default_key,
            type="password",
            placeholder="gsk_...",
            help="Must start with gsk_... from Groq Console",
        )
        if api_key and not api_key.startswith("gsk_"):
            st.error("❌ **Invalid Key Format:** Groq keys must start with `gsk_...`")
    elif provider_key == "gemini":
        default_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        st.caption("💡 [Get a free Gemini API Key](https://aistudio.google.com/app/apikey)")
        api_key = st.text_input(
            "Google Gemini API Key",
            value=default_key,
            type="password",
            placeholder="AIzaSy...",
            help="Must start with AIzaSy... from Google AI Studio",
        )
        if api_key and not api_key.startswith("AIzaSy"):
            st.error(
                "❌ **Invalid Key Format:** Your key starts with `" + api_key[:5] + "...`.\n\n"
                "Google Gemini strictly requires a key starting with **`AIzaSy...`**.\n\n"
                "👉 [Click here to create a free key](https://aistudio.google.com/app/apikey) or select **'Offline Demo (No Key Needed)'** above to test right now!"
            )
    else:
        default_key = os.getenv("OPENAI_API_KEY") or ""
        st.caption("💡 [Get an OpenAI API Key](https://platform.openai.com/api-keys)")
        api_key = st.text_input(
            "OpenAI API Key",
            value=default_key,
            type="password",
            placeholder="sk-...",
        )

    st.markdown("---")
    st.subheader("🎯 RAG Parameters")
    chunk_size = st.slider("Chunk Size (characters)", min_value=300, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=50, max_value=400, value=150, step=25)
    top_k = st.slider("Top-k Relevant Chunks", min_value=1, max_value=8, value=4, step=1)

    st.markdown("---")
    if st.button("🧹 Reset Application & Index", use_container_width=True):
        st.session_state.rag_engine = None
        st.session_state.indexed_file_name = None
        st.session_state.num_pages = 0
        st.session_state.num_chunks = 0
        st.session_state.chat_history = []
        st.rerun()


# --- MAIN HEADER ---
st.markdown('<div class="badge-rag">⚡ FAISS + LLM Grounded Q&A</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">📄 PDF Question-Answering with RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload a PDF document, chunk and embed it into a vector store, and ask questions with precise source citations.</div>',
    unsafe_allow_html=True,
)

# --- STEP 1: FILE UPLOADER & INDEXING ---
st.subheader("1. Upload PDF Document")
uploaded_file = st.file_uploader(
    "Choose a PDF file to index",
    type=["pdf"],
    help="Upload research papers, manuals, reports, or contracts in PDF format.",
)

# Trigger indexing when a new file is uploaded or when configuration changes
if uploaded_file is not None:
    needs_reindex = (
        st.session_state.rag_engine is None
        or st.session_state.indexed_file_name != uploaded_file.name
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"📁 Selected File: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
    with col2:
        reindex_clicked = st.button("🔄 (Re)Index Document", use_container_width=True)

    if needs_reindex or reindex_clicked:
        if not api_key and provider_key != "demo":
            st.error(f"⚠️ Please enter a valid **{provider} API Key** in the sidebar before indexing, or choose 'Offline Demo' mode.")
        else:
            with st.spinner("Extracting text, chunking, and embedding..."):
                try:
                    file_bytes = uploaded_file.read()
                    engine = RAGEngine(
                        provider=provider_key,
                        api_key=api_key,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    # 1. Extract
                    pages_data = engine.extract_text_from_pdf(file_bytes)
                    if not pages_data:
                        st.error("No readable text found in this PDF. It might be scanned or image-only.")
                    else:
                        # 2. Chunk
                        chunks = engine.chunk_pdf(pages_data)
                        # 3. Vector Index
                        engine.build_vector_index(chunks)

                        st.session_state.rag_engine = engine
                        st.session_state.indexed_file_name = uploaded_file.name
                        st.session_state.num_pages = len(pages_data)
                        st.session_state.num_chunks = len(chunks)
                        st.session_state.chat_history = []
                        st.success(
                            f"✅ Successfully indexed **{len(chunks)} chunks** across **{len(pages_data)} pages** into FAISS!"
                        )
                except Exception as e:
                    st.error(f"Error during document processing: {str(e)}")

# Display Index Status Banner
if st.session_state.rag_engine is not None:
    st.markdown(
        f"""
        <div class="status-card">
            📊 <b>Active Index:</b> {st.session_state.indexed_file_name} &nbsp;|&nbsp;
            📄 <b>Pages:</b> {st.session_state.num_pages} &nbsp;|&nbsp;
            🧩 <b>Chunks:</b> {st.session_state.num_chunks} &nbsp;|&nbsp;
            🤖 <b>Provider:</b> {provider}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Optional Preview of Extracted Chunks
    with st.expander("🔍 Preview Document Chunks & Metadata"):
        for chunk in st.session_state.rag_engine.chunks[:5]:
            st.markdown(f"**Chunk #{chunk.chunk_id} (Page {chunk.page_number})**")
            st.code(chunk.content[:300] + ("..." if len(chunk.content) > 300 else ""))
        if len(st.session_state.rag_engine.chunks) > 5:
            st.caption(f"Showing 5 of {len(st.session_state.rag_engine.chunks)} chunks.")

st.markdown("---")

# --- STEP 2: QUESTION ANSWERING SECTION ---
st.subheader("2. Ask Questions")

if st.session_state.rag_engine is None:
    st.warning("👈 Please upload a PDF above and provide an API key to enable question answering.")
else:
    # Let the user choose between "Ask 2-3 Questions Mode" or "Interactive Chat Mode"
    tab_multi, tab_chat = st.tabs(["📋 Ask 2-3 Questions (Batch Mode)", "💬 Interactive Chat Mode"])

    # TAB 1: 2-3 Questions Form
    with tab_multi:
        st.markdown("Ask up to 3 questions about your document simultaneously and get grounded answers side by side:")
        with st.form("multi_questions_form"):
            q1 = st.text_input("Question 1", placeholder="e.g. What is the main objective or summary of this document?")
            q2 = st.text_input("Question 2 (Optional)", placeholder="e.g. What are the key findings or results?")
            q3 = st.text_input("Question 3 (Optional)", placeholder="e.g. What recommendations or conclusions are mentioned?")
            
            submit_multi = st.form_submit_button("🚀 Get Answers for All Questions", use_container_width=True)

        if submit_multi:
            questions = [q.strip() for q in [q1, q2, q3] if q.strip()]
            if not questions:
                st.info("Please enter at least one question.")
            else:
                for idx, q in enumerate(questions, 1):
                    with st.spinner(f"Retrieving and answering Question #{idx}: '{q}'..."):
                        try:
                            res = st.session_state.rag_engine.query(q, top_k=top_k)
                            
                            st.markdown(f"### ❓ Question {idx}: {res['question']}")
                            st.markdown(res["answer"])

                            with st.expander(f"📚 View Retrieved Sources ({len(res['sources'])} chunks) for Q{idx}"):
                                for s in res["sources"]:
                                    st.markdown(
                                        f"""
                                        <div class="source-box">
                                            <b>Page {s['page']}</b> &bull; Cosine Similarity: <code>{s['score']:.4f}</code>
                                            <p style="margin-top: 5px; white-space: pre-wrap;">{s['content']}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                            st.markdown("---")
                        except Exception as e:
                            st.error(f"Error answering Question {idx}: {str(e)}")

    # TAB 2: Interactive Single/Multi-turn Chat
    with tab_chat:
        st.markdown("Ask questions one at a time and inspect citations:")

        # Render conversation history
        for item in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(item["question"])
            with st.chat_message("assistant"):
                st.markdown(item["answer"])
                with st.expander("📚 Sources & Citations"):
                    for s in item["sources"]:
                        st.markdown(
                            f"**Page {s['page']}** (Similarity: `{s['score']:.4f}`):\n> {s['content']}"
                        )

        # Chat input box
        user_query = st.chat_input("Ask a question about the uploaded document...")
        if user_query:
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)

            # Generate and stream response
            with st.chat_message("assistant"):
                with st.spinner("Searching document & generating answer..."):
                    try:
                        res = st.session_state.rag_engine.query(user_query, top_k=top_k)
                        st.markdown(res["answer"])

                        with st.expander("📚 Sources & Citations"):
                            for s in res["sources"]:
                                st.markdown(
                                    f"**Page {s['page']}** (Similarity: `{s['score']:.4f}`):\n> {s['content']}"
                                )

                        st.session_state.chat_history.append(res)
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
