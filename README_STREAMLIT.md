# Complete RAG Streamlit Application

A user-friendly, modern web interface for a Retrieval-Augmented Generation (RAG) system powered by Cohere embeddings and language models.

## Features

✅ **Dual Document Support**
- Load content from a single web URL with intelligent HTML parsing and fallback strategies
- Upload one or multiple PDF files for processing
- Automatic document chunking with configurable parameters

✅ **Smart State Management**
- Preserves input state across Streamlit reruns
- Only rebuilds vectorstore when inputs change
- Efficient session-based caching of embeddings and retrievers

✅ **Comprehensive Output Display**
- Full formatted prompt sent to the LLM
- Retrieved context snippets
- Generated answers with proper formatting
- Metadata about the current session

✅ **Modern UI/UX**
- Clean, intuitive sidebar for all inputs
- Expandable sections for detailed inspection
- Real-time status updates and error handling
- Professional styling with Streamlit components

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements_streamlit.txt
```

2. **Set up API keys:**
Create a `.streamlit/secrets.toml` file in your project directory:
```toml
COHERE_API_KEY = "your_cohere_api_key"
LANGCHAIN_API_KEY = "your_langchain_api_key"
```

**Alternative:** Use environment variables:
```bash
export COHERE_API_KEY="your_key"
export LANGCHAIN_API_KEY="your_key"
```

## Usage

Run the Streamlit application:
```bash
streamlit run complete_rag_streamlit.py
```

Then:
1. Choose document source type (Web URL or PDF Files)
2. Enter the URL or upload PDF(s)
3. Type your question
4. Select an AI model
5. Click "🚀 Start RAG Process"
6. View the full prompt and generated answer

## Architecture

### Key Components

- **CohereEmbeddings**: Custom LangChain embeddings wrapper for Cohere's embed API
- **load_pdf()**: Extracts text from PDF files with page metadata
- **load_webpage_flexible()**: Intelligent HTML content extraction with multiple fallback strategies
- **Session State Management**: Tracks inputs and regenerates vectorstore only when needed
- **RAG Chain**: LangChain runnable composition with retrieval, prompt templating, and generation

### Data Flow

```
User Input (Document + Question)
    ↓
Load Documents (Web/PDF)
    ↓
Split into Chunks
    ↓
Create Embeddings (Cohere)
    ↓
Build Vectorstore (Chroma)
    ↓
Retrieve Context (MMR)
    ↓
Format Prompt
    ↓
Generate Answer (Cohere LLM)
    ↓
Display Results
```

## Configuration

### Text Splitter Settings
- **chunk_size**: 800 characters (adjust for longer/shorter chunks)
- **chunk_overlap**: 100 characters (prevents context loss at boundaries)

### Retriever Settings
- **search_type**: MMR (Maximal Marginal Relevance)
- **k**: 10 (number of documents to retrieve)
- **lambda_mult**: 0 (diversity parameter)

### LLM Settings
- **temperature**: 0.75 (controls randomness; higher = more creative)
- **Models available**: command-a-03-2025, command-r7b-12-2024, and others

## Troubleshooting

### API Key Issues
- Ensure API keys are in `.streamlit/secrets.toml`
- Check that keys are valid and have proper permissions
- Verify no extra whitespace in secrets file

### Vectorstore Not Clearing
- Session state should automatically clear vectorstore per run
- If issues persist, clear browser cache and restart Streamlit

### PDF Upload Fails
- Ensure PDF is not encrypted
- Check file size (typically < 50MB recommended)
- Verify PDF contains extractable text (not scanned images)

### Webpage Loading Issues
- Try a different URL if parsing fails
- Ensure the website is publicly accessible
- JavaScript-heavy sites may not work with BeautifulSoup

## Advanced Usage

### Modify Chunk Size
Edit in the code:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for longer chunks
    chunk_overlap=150
)
```

### Add More Models
Update the selectbox options:
```python
ai_model = st.selectbox(
    "Select AI Model:",
    options=["model1", "model2", "model3"],
)
```

### Customize the System Prompt
Modify the prompt template in the `run_rag_pipeline()` function:
```python
template="Your custom system prompt here with {question} and {context}"
```

## Performance Tips

- Use MMR search for better diversity in retrieved documents
- Adjust `chunk_size` based on your domain (technical docs may need larger chunks)
- Reduce `k` (number of retrieved docs) if generation is slow
- Lower `temperature` for more consistent answers

## Files Included

- `complete_rag_streamlit.py` - Main Streamlit application
- `requirements_streamlit.txt` - Python dependencies
- `.streamlit/secrets.toml` - API key configuration template
- `README.md` - This documentation file

## License

MIT License - Feel free to modify and use as needed.

## Support

For issues or questions, refer to:
- [Streamlit Docs](https://docs.streamlit.io/)
- [LangChain Docs](https://python.langchain.com/)
- [Cohere Docs](https://docs.cohere.com/)
