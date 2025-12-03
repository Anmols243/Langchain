import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

st.set_page_config(page_title="Objectbox", layout="wide")
st.title("ObjectBox Demo")
st.info("Loading us cesus docs from pdf -> Objectbox -> Groq")
st.write("Running file:", os.path.abspath(__file__))

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5",
        model_kwargs = {"device":"cpu"},
        encode_kwargs = {"normalize_embeddings":True}
    )

@st.cache_data
def load_and_split_docs():
    loader = PyPDFDirectoryLoader("us_census")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    
    return text_splitter.split_documents(docs)

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def vector_embeddings():
    if "vectorstore" not in st.session_state:
        with st.spinner("Loading & Indexing Documents..."):
            embeddings = get_embeddings()
            split = load_and_split_docs()
            try:
                st.session_state.vectorstore = ObjectBox.from_documents(
                    split,
                    embeddings,
                    embedding_dimensions=768
                )
            except Exception as e:
                st.error(f"Vectorstore init failed: {e}")
                st.write("session keys:", list(st.session_state.keys()))
                raise
        st.rerun()

vector_embeddings()

if "vectorstore" in st.session_state:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k":4})
else:
    retriever = None
    st.warning("Vectorstore not initialized.")

prompt = ChatPromptTemplate.from_template(
"""
Answer the question based ONLY on the provided context. Think step-by-step.

Context: {context}

Question: {question}

Final Answer (detailed but concise):
"""
)

llm = get_llm()

def rag_chain(retriever,llm):
    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

if retriever:
    chain = rag_chain(retriever,llm)
    query = st.text_input("Enter your question from the documents")
    if query:
        start = time.process_time()
        try:
            response = chain.invoke(query)
            st.write(response)
            st.caption(f"Response time: {time.process_time()-start:.2f}s")

        except Exception as e:
            st.error(f"Error: {str(e)}")
    


