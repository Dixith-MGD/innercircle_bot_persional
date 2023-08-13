import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

os.environ["OPENAI_API_KEY"] = "sk-**************************"

st.title('INNERCIRCLE BOT')
st.sidebar.title('Developed by UNORTHODOX')
st.sidebar.info('Chat')
pdf_path = 'dataset.pdf'
pdf_reader = PdfReader(pdf_path)
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text=text)

if os.path.exists(f"{pdf_path[:-4]}.pkl"):
    with open(f"{pdf_path[:-4]}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
else:
    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    with open(f"{pdf_path[:-4]}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)

query = st.text_input("Ask a question about Innercircle")

if query:
    embeddings = OpenAIEmbeddings()
    docs = VectorStore.similarity_search(query=query, k=3)
    llm = OpenAI()
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
    st.write(response)
