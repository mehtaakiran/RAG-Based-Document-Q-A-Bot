from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "my_document.pdf"  # swap this for whatever pdf you want to ask questions about

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# chunk_overlap so we don't lose context that gets cut off at a chunk boundary
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(pages)

print(f"split into {len(chunks)} chunks")

# free, runs locally, no API key needed for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("faiss_index")

print("index saved to ./faiss_index")
