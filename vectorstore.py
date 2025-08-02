from dotenv import load_dotenv 
load_dotenv()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


pdf_read = PyPDFLoader("/Users/apple/Documents/TeachAIToFamily/tata-motor-IAR-2024-25.pdf")
data = pdf_read.load()
    
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, 
    chunk_overlap = 512,
    length_function=len
)
splitted_documents = splitter.split_documents(data)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = Chroma.from_documents(documents=splitted_documents, embedding=embeddings,
                                 persist_directory="chromadb",collection_name="rag")
