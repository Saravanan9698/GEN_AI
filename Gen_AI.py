# --- Imports ---
import os
import fitz  # PyMuPDF
import base64
import docx2txt
import streamlit as st
import nltk
import json
from sentence_transformers import SentenceTransformer
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import numpy as np

# --- Downloads ---
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# --- Path Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "sentence_transformers", "all-MiniLM-L6-v2")
MODEL_PATH = os.path.join(BASE_DIR, "Model", "llama-2-7b-chat.Q4_K_M.gguf")
IMAGE_PATH = os.path.join(BASE_DIR, "Images", "ChatGPT_Image.png")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
EVAL_LOG = os.path.join(BASE_DIR, "evaluation_log.json")

# --- Embeddings ---
@st.cache_resource
def get_embeddings(device):
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        st.error(f"Embedding model not found at: {EMBEDDING_MODEL_PATH}")
        return None
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": device}
    )



INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
EVAL_LOG = os.path.join(BASE_DIR, "evaluation_log.json")

# --- Page Config ---
st.set_page_config(page_title="💬 ChatBot AI", layout="wide")

# --- Session State Setup ---
defaults = {
    "llm": None,
    "vectorstore": None,
    "documents_processed": False,
    "ready_to_ask": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- Background Image ---
def img_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        st.warning(f"Could not load background image: {e}")
        return None

img_base64 = img_to_base64(IMAGE_PATH)
if img_base64:
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.5)), url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

# --- File Extraction ---
def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    try:
        if ext == ".pdf":
            return "".join([page.get_text() for page in fitz.open("pdf", file.read())])
        elif ext == ".docx":
            return docx2txt.process(file)
        elif ext == ".txt":
            return file.getvalue().decode("utf-8")
        else:
            st.error(f"Unsupported file type: {ext}")
            return ""
    except Exception as e:
        st.warning(f"Error processing {file.name}: {e}")
        return ""

def process_documents(files):
    with ThreadPoolExecutor() as executor:
        return "\n\n".join([t for t in executor.map(extract_text, files) if t])

# --- Embeddings ---
@st.cache_resource
def get_embeddings(device):
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": device}
    )

# --- Load LLM ---
@st.cache_resource
def load_llm():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None
    return CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        config={"max_new_tokens": 256, "temperature": 0.1}
    )

# --- Prompt Template ---
def get_qa_prompt_template():
    return PromptTemplate(
        template="You are an assistant answering questions based on document content.\nContext: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )

# --- Evaluation ---
def calculate_metrics(pred, truth):
    smoother = SmoothingFunction().method1
    bleu = sentence_bleu([truth.split()], pred.split(), smoothing_function=smoother)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(truth, pred)
    return {
        "BLEU": round(bleu, 4),
        "ROUGE-1": round(rouge['rouge1'].fmeasure, 4),
        "ROUGE-L": round(rouge['rougeL'].fmeasure, 4)
    }

def log_evaluation(q, a, t, m):
    with open(EVAL_LOG, "a", encoding="utf-8") as f:
        json.dump({
            "timestamp": str(datetime.now()),
            "question": q, "answer": a, "truth": t, "metrics": m
        }, f)
        f.write("\n")

def get_evaluation_stats():
    if not os.path.exists(EVAL_LOG):
        return None
    with open(EVAL_LOG, "r", encoding="utf-8") as f:
        logs = [json.loads(line) for line in f.readlines()]
    if not logs:
        return None
    avg = lambda k: round(sum(log["metrics"][k] for log in logs) / len(logs), 4)
    return {
        "Average BLEU": avg("BLEU"),
        "Average ROUGE-1": avg("ROUGE-1"),
        "Average ROUGE-L": avg("ROUGE-L"),
        "Total Evaluations": len(logs)
    }

def generate_optimization_suggestions(stats):
    tips = []
    if stats["Average BLEU"] < 0.4:
        tips.append("Consider fine-tuning the LLM on domain-specific QA datasets.")
    if stats["Average ROUGE-1"] < 0.5:
        tips.append("Split documents into smaller, context-rich chunks.")
    if stats["Average ROUGE-L"] < 0.5:
        tips.append("Use a stronger embedding model or context retriever.")
    return tips or ["Current configuration is performing well."]

# --- Title & Sidebar ---
st.title("ChatBot AI")
st.subheader("From documents to decisions — powered by AI, secured locally.")

with st.sidebar:
    st.header("Upload your Document!")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files and st.button("Start The Fun!"):
        with st.spinner("Processing documents..."):
            full_text = process_documents(uploaded_files)
            if full_text:
                splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
                chunks = splitter.split_text(full_text)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                embeddings = get_embeddings(device)
                vectordb = FAISS.from_texts(chunks, embedding=embeddings)
                vectordb.save_local(INDEX_PATH)
                st.session_state.vectorstore = vectordb
                st.session_state.documents_processed = True
                st.success("Documents processed and vector DB created.")
            else:
                st.error("Failed to extract text from documents.")

# --- Load FAISS if exists ---
if not st.session_state.documents_processed and os.path.exists(INDEX_PATH):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = get_embeddings(device)
    try:
        vectordb = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        st.session_state.vectorstore = vectordb
        st.session_state.documents_processed = True
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")

# --- Q&A Section ---
if st.session_state.documents_processed and st.session_state.vectorstore:
    st.subheader("Ask a Question Based on the Documents 📄💡")
    question = st.text_input("Enter your question:")

    if question:
        if st.session_state.llm is None:
            st.session_state.llm = load_llm()

        if st.session_state.llm:
            retriever = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.5})
            qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": get_qa_prompt_template()}
            )

            with st.spinner("Thinking... 🤔"):
                result = qa_chain.invoke({"query": question})
                answer = result.get("result") if isinstance(result, dict) else result
                st.session_state.ready_to_ask = True

            st.markdown("### 🤖 Answer:")
            st.success(answer)

            if st.checkbox("Enable Evaluation Mode"):
                ground_truth = st.text_area("Enter the Ground Truth Answer")
                if ground_truth and st.button("Evaluate"):
                    metrics = calculate_metrics(answer, ground_truth)
                    log_evaluation(question, answer, ground_truth, metrics)
                    st.subheader("📊 Evaluation Metrics")
                    st.json(metrics)

                    stats = get_evaluation_stats()
                    if stats:
                        st.subheader("📈 Aggregated Stats")
                        st.json(stats)

                        suggestions = generate_optimization_suggestions(stats)
                        st.subheader("🛠 Optimization Suggestions")
                        for tip in suggestions:
                            st.markdown(f"- {tip}")

# --- Sidebar Footer ---
with st.sidebar:
    st.markdown("---")
    st.markdown("""
        ### How to use:
        1. Upload PDF, DOCX, or TXT documents
        2. Click 'Start The Fun!'
        3. Ask questions about your documents
        4. (Optional) Enable evaluation mode for answer quality

        ### About:
        This app uses a locally hosted LLaMA 2 model via CTransformers with FAISS for retrieval.
        It evaluates responses using BLEU and ROUGE metrics.
    """)
