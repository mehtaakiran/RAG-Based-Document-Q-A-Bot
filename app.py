import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

st.title("Ask My Document")
st.write("Upload a PDF's worth of knowledge already indexed? Just ask a question about it below.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# groq's free tier - fast + no cost, put your key in .streamlit/secrets.toml as GROQ_API_KEY
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=st.secrets["GROQ_API_KEY"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

question = st.text_input("Ask something about the document")

if question:
    with st.spinner("thinking..."):
        result = qa_chain.invoke({"query": question})

    st.write(result["result"])

    with st.expander("Where this answer came from"):
        for doc in result["source_documents"]:
            st.write(doc.page_content[:300] + "...")
