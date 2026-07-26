# RAG-Based Document Q&A Bot

Ask questions about a PDF and get answers grounded in its actual content, using Retrieval-Augmented Generation instead of relying on the LLM's own memory.

## How it works

1. `build_index.py` — loads a PDF, splits it into overlapping chunks, converts each chunk into an embedding (using a free local sentence-transformers model), and stores everything in a FAISS vector index on disk.
2. `app.py` — takes a question, embeds it the same way, finds the most similar chunks from the index, and passes those chunks + the question to an LLM (Groq's free API) so the answer is grounded in the actual document instead of the model guessing.

## Why RAG instead of just asking the LLM directly

LLMs don't know anything about your specific document and can hallucinate if you just ask them questions about it. RAG fixes this by retrieving the actual relevant text first and forcing the model to answer based on that, rather than whatever it "remembers."

## Running it yourself

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Get a free API key from [Groq](https://console.groq.com), copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste your key in.
3. Put the PDF you want to query in this folder, update `PDF_PATH` in `build_index.py`.
4. Build the index:
   ```
   python build_index.py
   ```
5. Run the app:
   ```
   streamlit run app.py
   ```

## Limitations

- Answer quality depends heavily on chunk size/overlap — too small and you lose context, too big and retrieval gets noisy.
- Doesn't handle scanned/image-only PDFs since there's no OCR step here.
- Only searches within the one indexed document, no multi-document support yet.

## Possible improvements

- Add support for multiple documents at once
- Try a re-ranking step after retrieval to improve answer relevance
- Swap in a bigger/different embedding model and compare retrieval quality
