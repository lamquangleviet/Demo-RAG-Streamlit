"""
Complete RAG Application with Streamlit UI
Author: RAG Team
Description: A user-friendly Streamlit interface for a RAG (Retrieval-Augmented Generation) system
             that supports both web URLs and PDF file uploads with Cohere embeddings and LLM generation.
"""

import os
import streamlit as st
import cohere
from io import BytesIO
from typing import List
import tempfile
import time
import uuid

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere

# Community loaders
from langchain_community.document_loaders import WebBaseLoader
import bs4

# PDF and web utilities
from PyPDF2 import PdfReader
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', message='.*__fields__.*', category=DeprecationWarning)

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up API keys from environment
os.environ["COHERE_API_KEY"] = st.secrets.get("COHERE_API_KEY", "")
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")

# Initialize Cohere client
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])


# ============================================================================
# CUSTOM EMBEDDINGS CLASS
# ============================================================================

class CohereEmbeddings(Embeddings):
    """Custom Cohere embeddings wrapper for LangChain."""
    
    def __init__(self, cohere_client, model="embed-v4.0"):
        self.client = cohere_client
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        response = self.client.embed(
            input_type="search_document",
            model=self.model,
            texts=texts
        )
        return response.embeddings.float_
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = self.client.embed(
            input_type="search_query",
            model=self.model,
            texts=[text]
        )
        return response.embeddings.float_[0]


# ============================================================================
# DOCUMENT LOADING FUNCTIONS
# ============================================================================

def load_pdf(pdf_files) -> List[Document]:
    """
    Extract text from uploaded PDF files and return as LangChain Document objects.
    
    Args:
        pdf_files: List of uploaded PDF file objects
    
    Returns:
        List of Document objects with metadata, with each page of the PDF treated as 1 document.
    """
    docs = []
    for pdf_file in pdf_files:
        try:
            # Read PDF from BytesIO stream
            reader = PdfReader(pdf_file)
            pdf_name = pdf_file.name if hasattr(pdf_file, 'name') else "uploaded_pdf"
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                page_content = f"--- {pdf_name} | Page {page_num + 1} ---\n{text}"
                docs.append(Document(
                    page_content=page_content,
                    metadata={
                        "source": pdf_name,
                        "file_name": pdf_name,
                        "page": page_num + 1,
                        "type": "pdf"
                    }
                ))
            st.success(f"✓ Extracted {len(reader.pages)} pages from {pdf_name}")
        except Exception as e:
            st.error(f"✗ Error reading {pdf_file.name}: {e}")
    
    return docs


def load_webpage_flexible(web_url: str) -> List[Document]:
    """
    Load webpage with fallback strategies to handle various HTML structures.
    
    Args:
        web_url: URL of the webpage to load
    
    Returns:
        List of Document objects
    """
    # Strategy 1: Try common article/content class selectors
    common_selectors = [
        {
            "name": 'class_=("post-content", "post-title", "post-header", "article-content", "main-content")',
            "kwargs": dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header", "article-content", "main-content")))
        },
        {
            "name": 'class_=("content", "article", "entry", "post")',
            "kwargs": dict(parse_only=bs4.SoupStrainer(class_=("content", "article", "entry", "post")))
        },
        {
            "name": 'name=["article", "main", "p", "h1", "h2", "h3"]',
            "kwargs": dict(parse_only=bs4.SoupStrainer(name=["article", "main", "p", "h1", "h2", "h3"]))
        },
    ]
    
    documents = None
    for i, strategy in enumerate(common_selectors):
        try:
            loader = WebBaseLoader(web_path=(web_url), bs_kwargs=strategy["kwargs"])
            documents = loader.load()
            if documents and any(len(doc.page_content.strip()) > 100 for doc in documents):
                print(f"✓ Successfully loaded using selector strategy: {strategy['name']}")
                return documents
        except Exception as e:
            print(f"  Strategy {strategy['name']} failed: {str(e)[:50]}")
            continue
    
    # Strategy 2: Load all text without specific selectors (last resort)
    try:
        st.info("Attempting fallback: loading all page text...")
        loader = WebBaseLoader(web_path=web_url)
        documents = loader.load()
        if documents:
            documents = [doc for doc in documents if len(doc.page_content.strip()) > 50]
            st.success(f"✓ Fallback successful: loaded {len(documents)} document(s)")
            return documents
    except Exception as e:
        st.warning(f"Fallback failed: {e}")
    
    # Strategy 3: If still no documents, raise informative error
    raise ValueError(
        f"Could not extract content from {web_url}. "
        "Possible reasons:\n"
        "- The URL may not be accessible\n"
        "- The page may require JavaScript to load\n"
        "- The page structure is not compatible\n"
        "Try a different URL or ensure it's a public article/blog post."
    )


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for display in prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# STREAMLIT UI LAYOUT
# ============================================================================

st.title("🤖 RAG System - Retrieval-Augmented Generation")
st.markdown("---")

# Create two columns: sidebar for inputs, main for outputs
with st.sidebar:
    st.header("📋 Input Configuration")
    
    # Document Type Selection
    doc_type = st.radio(
        "Choose document source type:",
        options=["Web URL", "PDF Files"],
        key="doc_type"
    )
    
    # Input based on document type
    if doc_type == "Web URL":
        web_url = st.text_input(
            "Enter webpage URL:",
            placeholder="https://example.com/article",
            key="web_url"
        )
        pdf_files = None
    else:  # PDF Files
        pdf_files = st.file_uploader(
            "Upload PDF file(s):",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_files"
        )
        web_url = None
    
    # Question Input
    question = st.text_area(
        "Enter your question:",
        placeholder="What is the main topic?",
        height=100,
        key="question"
    )
    
    # Model Selection
    ai_model = st.selectbox(
        "Select AI Model (from Cohere):",
        options=[
            "command-a-03-2025",
            "command-r7b-12-2024",
            "command-a-translate-08-2025",
            "command-a-reasoning-08-2025"
        ],
        index=0,
        key="ai_model"
    )
    
    # Start Button
    st.markdown("---")
    start_button = st.button(
        "🚀 Start RAG Process",
        use_container_width=True,
        type="primary"
    )


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

# Initialize session state variables
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "full_prompt" not in st.session_state:
    st.session_state.full_prompt = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = None


# ============================================================================
# INPUT CHANGE DETECTION & VALIDATION
# ============================================================================

def get_current_inputs():
    """Get current input values for change detection."""
    if doc_type == "Web URL":
        return {
            "type": "web",
            "web_url": web_url,
            "pdf_files": None,
            "question": question,
            "ai_model": ai_model
        }
    else:
        # For PDF files, create a hashable representation
        pdf_hashes = tuple(sorted([f.name for f in pdf_files])) if pdf_files else None
        return {
            "type": "pdf",
            "web_url": None,
            "pdf_files": pdf_hashes,
            "question": question,
            "ai_model": ai_model
        }


def inputs_changed(current, last):
    """Check if any inputs have changed."""
    if last is None:
        return True
    return current != last


# ============================================================================
# MAIN RAG PIPELINE
# ============================================================================

def run_rag_pipeline(current_inputs):
    """
    Run the complete RAG pipeline: load documents, create vectorstore,
    retrieve context, generate answer.
    """
    
    # Placeholder for loading documents
    with st.spinner("📚 Loading documents..."):
        try:
            if current_inputs["type"] == "web":
                if not current_inputs["web_url"]:
                    st.error("❌ Please enter a valid URL")
                    return False
                documents = load_webpage_flexible(current_inputs["web_url"])
            else:  # PDF
                if not current_inputs["pdf_files"]:
                    st.error("❌ Please upload at least one PDF file")
                    return False
                documents = load_pdf(st.session_state.pdf_files_temp)
            
            st.session_state.documents_loaded = len(documents)
            st.success(f"✓ Loaded {len(documents)} document(s)")
        except Exception as e:
            st.error(f"❌ Error loading documents: {e}")
            return False
    
    # Split documents into chunks
    with st.spinner("✂️ Splitting documents into chunks..."):
        try:
            # Automatically compute chunk_size and chunk_overlap so we get ~90 chunks
            def compute_chunk_params(docs, target_chunks=70, min_overlap=50, max_overlap=300, step=10):
                """Search for chunk_size and chunk_overlap that yield target_chunks when splitting.

                Returns (chunk_size, chunk_overlap, splits)
                """
                import math

                # Combine content length as a heuristic
                combined = "\n\n".join(getattr(d, 'page_content', '') for d in docs)
                total_chars = len(combined)

                # Edge case: no content
                if total_chars == 0:
                    return 800, 100, []

                best_candidate = None
                best_diff = None

                # Try range of overlaps to find an exact or closest match
                for overlap in range(min_overlap, max_overlap + 1, step):
                    # heuristic chunk size solving S = O + (T - O)/N
                    chunk_size = int(math.ceil((total_chars - overlap) / target_chunks + overlap))
                    if chunk_size <= overlap:
                        chunk_size = overlap + 10

                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                    splits_try = splitter.split_documents(docs)
                    n = len(splits_try)
                    diff = abs(n - target_chunks)

                    if diff == 0:
                        return chunk_size, overlap, splits_try

                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_candidate = (chunk_size, overlap, splits_try)

                # If no exact match found, try refining chunk_size around best candidate
                if best_candidate is None:
                    # fallback
                    return 800, 100, RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)

                chunk_size, overlap, splits_try = best_candidate
                n = len(splits_try)
                # refine by adjusting chunk_size up or down until we approach target (bounded iterations)
                iterations = 0
                while n != target_chunks and iterations < 1000:
                    iterations += 1
                    # if we have too many chunks, increase chunk_size to reduce count
                    if n > target_chunks:
                        chunk_size += 10
                    # if too few chunks, decrease chunk_size but keep > overlap
                    else:
                        if chunk_size - 10 > overlap:
                            chunk_size -= 10
                        else:
                            break

                    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
                    splits_try = splitter.split_documents(docs)
                    n = len(splits_try)

                return chunk_size, overlap, splits_try

            chunk_size, chunk_overlap, splits = compute_chunk_params(documents, target_chunks=30)
            st.info(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            st.success(f"✓ Split into {len(splits)} chunks")
        except Exception as e:
            st.error(f"❌ Error splitting documents: {e}")
            return False
    
    # Create embeddings and vectorstore
    with st.spinner("🔗 Creating embeddings and vectorstore..."):
        try:
            # Initialize embeddings
            cohere_embeddings = CohereEmbeddings(cohere_client=co, model="embed-v4.0")
            
            # Create fresh collection
            collection_name = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=cohere_embeddings,
                collection_name=collection_name,
                persist_directory=None
            )
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 10, "lambda_mult": 0}
            )
            
            # Store in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = retriever
            st.success(f"✓ Created vectorstore: {collection_name}")
        except Exception as e:
            st.error(f"❌ Error creating vectorstore: {e}")
            return False
    
    # Retrieve and generate answer
    with st.spinner("🔍 Retrieving context and generating answer..."):
        try:
            # Get context (used to show the full prompt, not necessary if run RAG only)
            docs = st.session_state.retriever.invoke(current_inputs["question"])
            context_text = format_docs(docs)
            
            # Create prompt template
            prompt_template = PromptTemplate(
                input_variables=["question", "context"],
                template=(
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise. "
                    "Do not take previous/old context or answer and only consider the new current context.\n"
                    "Question: {question}\n"
                    "Context: {context}"
                )
            )
            
            # Format full prompt
            full_prompt = prompt_template.format(
                question=current_inputs["question"],
                context=context_text
            )
            st.session_state.full_prompt = full_prompt
            
            # Create and invoke RAG chain
            llm = ChatCohere(model=current_inputs["ai_model"], temperature=0.75)
            rag_chain = (
                {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            answer = rag_chain.invoke(current_inputs["question"])
            st.session_state.answer = answer
            
            st.success("✓ Answer generated successfully")
            return True
        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")
            return False


# ============================================================================
# MAIN EXECUTION LOGIC
# ============================================================================

# Get current inputs
current_inputs = get_current_inputs()

# Store PDF files temporarily if in PDF mode
if doc_type == "PDF Files" and pdf_files:
    st.session_state.pdf_files_temp = pdf_files

# Check if inputs are valid
has_valid_inputs = (
    question.strip() != "" and (
        (doc_type == "Web URL" and web_url.strip() != "") or
        (doc_type == "PDF Files" and pdf_files is not None and len(pdf_files) > 0)
    )
)

if not has_valid_inputs:
    st.info("👈 Please fill in all required fields in the sidebar to begin.")
else:
    # Check if inputs have changed
    inputs_have_changed = inputs_changed(current_inputs, st.session_state.last_inputs)
    
    # Determine if we should run the pipeline
    should_run = False
    if start_button:
        should_run = True
    elif inputs_have_changed:
        # Inputs changed, clear results and require new button click
        st.session_state.full_prompt = None
        st.session_state.answer = None
    
    if should_run and has_valid_inputs:
        # Update last inputs
        st.session_state.last_inputs = current_inputs
        
        # Run pipeline
        success = run_rag_pipeline(current_inputs)
        
        if success:
            st.markdown("---")
            
            # Display results
            st.header("📊 Results")
            
            # Display Answer
            with st.expander("💡 Generated Answer", expanded=True):
                st.markdown("### Answer from LLM:")
                st.success(st.session_state.answer)
            
            # Display Full Prompt
            with st.expander("📝 Full Prompt Sent to LLM", expanded=True):
                st.markdown("### Question:")
                st.write(current_inputs["question"])
                st.markdown("### Context (Retrieved Documents):")
                st.write(st.session_state.full_prompt.split("Context: ")[1] if "Context: " in st.session_state.full_prompt else "No context retrieved")
                st.markdown("### Full Formatted Prompt:")
                st.code(st.session_state.full_prompt, language="text")
            
            # Display metadata
            with st.expander("📌 Metadata", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents Loaded", st.session_state.documents_loaded or 0)
                with col2:
                    st.metric("AI Model", current_inputs["ai_model"])
                with col3:
                    st.metric("Document Type", current_inputs["type"].upper())
    
    elif inputs_have_changed:
        st.warning("⚠️ Inputs have changed. Press 🚀 Start RAG Process to run with new inputs.")

