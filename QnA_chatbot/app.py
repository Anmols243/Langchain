import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from operator import itemgetter
import streamlit as st
import time
from langchain_core.runnables import RunnablePassthrough,  RunnableLambda
load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

if "vectors" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
         model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device":"cpu"},
        encode_kwargs={'normalize_embeddings': True}
    )
    st.session_state.loader = PyPDFLoader("us_census")
    st.session_state.docs = st.session_state.loader.load()
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

    st.title("Application Demo")
llm = ChatGroq(groq_api_key,model="llama-3.3-70b-versatile")

prompt =ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

Context:
{context}   

Question:
{input}                                                                            
"""
)

retriever = st.session_state.vectors.as_retriever()

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    RunnablePassthrough.assign(
        context = itemgetter("input")
        | retriever
        | RunnableLambda(format_docs)
    )
    | prompt
    | llm
)

query = st.text_input("Write your question here..")

if query:
    start = time.process_time()
    response = rag_chain.invoke({"input": query})
    print("Response Time", time.process_time() - start)
    print(response.content if hasattr(response,"content") else response)