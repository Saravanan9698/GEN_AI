# Imports
import os
from pathlib import Path
import logging
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File Paths
MODEL_FILE = Path("D:/Projects/GEN-AI/Model/llama-2-7b-chat.Q4_K_M.gguf")
DOC_FILE = Path("D:/Projects/GEN-AI/Documents/Saravanan Data Science Resume ATS match without emoji.pdf")
EMBED_MODEL_PATH = Path("D:/Projects/GEN-AI/models/all-MiniLM-L6-v2")
VECTOR_DB_PATH = Path("db/faiss")

# Load PDF and Split
def load_and_split_docs(pdf_path):
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# Embedding and Vector DB
def create_or_load_vector_store(docs, embedding_path, db_path):
    embedding = HuggingFaceEmbeddings(
        model_name=str(embedding_path),
        model_kwargs={"device": "cpu"}
    )

    if db_path.exists():
        logger.info("Loading existing vector store...")
        vect = FAISS.load_local(str(db_path), embedding, allow_dangerous_deserialization=True)
    else:
        logger.info("Creating new vector store...")
        vect = FAISS.from_documents(docs, embedding)
        vect.save_local(str(db_path))

    return vect

# Prompt Template
def get_prompt_template():
    prompt = """
You are an intelligent assistant. Use the provided context to answer the question accurately.

- If the answer is found in the context, reply concisely.
- If the answer is not found, respond with "Not found".
- Do not add any extra commentary.

Context: {context}

Question: {question}

Answer:
"""
    return PromptTemplate(template=prompt, input_variables=["context", "question"])

# Load LLM Model
def load_llm_model(model_file_path):
    return CTransformers(
        model=str(model_file_path),
        model_type="llama",
        config={
            "max_new_tokens": 500,
            "temperature": 0.1
        }
    )

# QA Pipeline Setup
def setup_retrieval_qa(llm, prompt_template, vector_store):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type_kwargs={"prompt": prompt_template},
        retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    )

# Main Logic
def main():
    logger.info("Loading and processing documents...")
    docs = load_and_split_docs(DOC_FILE)

    logger.info("Setting up embeddings and vector database...")
    vect_db = create_or_load_vector_store(docs, EMBED_MODEL_PATH, VECTOR_DB_PATH)

    logger.info("Loading language model...")
    llm = load_llm_model(MODEL_FILE)

    logger.info("Building prompt and QA system...")
    prompt_template = get_prompt_template()
    qa_pipeline = setup_retrieval_qa(llm, prompt_template, vect_db)

    # Example Questions
    questions = [
        "what is the Name ?",
        "what is the email id ?",
        "which city is mentioned ?"
    ]

    for q in questions:
        result = qa_pipeline({"query": q})
        print(f"Q: {q}\nA: {result['result']}\n")

if __name__ == "__main__":
    main()
