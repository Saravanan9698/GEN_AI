import os
import fitz
import base64
import docx2txt
import streamlit as st
import nltk
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import numpy as np


# Initial downloads
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Session state variables
for key in ['llm', 'vectorstore', 'documents_processed', 'ready_to_ask']:
    if key not in st.session_state:
        st.session_state[key] = None if key == 'llm' else False

st.set_page_config(page_title="💬 AskDocs AI", layout="wide")

# Background image

def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_path = "D:/Projects/GEN-AI/Images/ChatGPT Image.png"
if os.path.exists(image_path):
    img_base64 = img_to_base64(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.5)), url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-position: center;
        }}
        </style>
        """, unsafe_allow_html=True
    )

# File extract

def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == '.pdf':
        return "".join([p.get_text() for p in fitz.open(stream=file.read(), filetype="pdf")])
    elif ext == '.docx':
        return docx2txt.process(file)
    elif ext == '.txt':
        return file.getvalue().decode('utf-8')
    else:
        st.error(f"Unsupported file type: {ext}")
        return None

# Process documents

def process_documents(files):
    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(extract_text, files))
    return "\n\n".join([t for t in texts if t])

# Load LLM
@st.cache_resource
def load_llm():
    model_path = r"D:\Projects\GEN-AI\Model\llama-2-7b-chat.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}.")
        return None
    return CTransformers(
        model=model_path,
        model_type="llama",
        config={"max_new_tokens": 256, "temperature": 0.1}
    )

# Prompt template

def get_qa_prompt_template():
    return PromptTemplate(
        template="You are an assistant answering questions based on document content.\nContext: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )

# Evaluation functions

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
    with open("evaluation_log.json", "a", encoding="utf-8") as f:
        json.dump({"timestamp": str(datetime.now()), "question": q, "answer": a, "truth": t, "metrics": m}, f)
        f.write("\n")

def get_evaluation_stats():
    if not os.path.exists("evaluation_log.json"):
        return None
    with open("evaluation_log.json", "r", encoding="utf-8") as f:
        logs = [json.loads(line) for line in f.readlines()]
    if not logs:
        return None
    avg = lambda key: round(sum([log['metrics'][key] for log in logs]) / len(logs), 4)
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
        tips.append("Try using a more powerful embedding model.")
    return tips if tips else ["Current configuration is performing well."]

# UI layout
st.title("AskDocs AI")
st.subheader("*From documents to decisions — powered by AI, secured locally.*")

# Upload sidebar
with st.sidebar:
    st.header("*Upload your Document!*")
    uploaded_files = st.file_uploader("*Upload PDF, DOCX, or TXT files*", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if uploaded_files and st.button("Start The Fun!"):
        with st.spinner("Processing documents..."):
            all_text = process_documents(uploaded_files)
            if all_text:
                splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
                chunks = splitter.split_text(all_text)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                embeddings = HuggingFaceEmbeddings(
                    model_name=r"D:\Projects\GEN-AI\models\all-MiniLM-L6-v2",
                    model_kwargs={"device": device}
                )
                vectordb = FAISS.from_texts(chunks, embedding=embeddings)
                vectordb.save_local("faiss_index")
                st.session_state['vectorstore'] = vectordb
                st.session_state['documents_processed'] = True
                st.success("Documents processed and vector DB created.")

# Load FAISS if exists
if not st.session_state['documents_processed'] and os.path.exists("faiss_index"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=r"D:\Projects\GEN-AI\models\all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    st.session_state['vectorstore'] = vectordb
    st.session_state['documents_processed'] = True

# QA interaction
if st.session_state['documents_processed'] and st.session_state['vectorstore']:
    st.subheader("Ask a Question Based on the Documents 📄💡")
    question = st.text_input("Enter your question:")

    if question:
        if st.session_state.llm is None:
            st.session_state.llm = load_llm()

        if st.session_state.llm:
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 2, "lambda_mult": 0.5}
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": get_qa_prompt_template()}
            )

            with st.spinner("Thinking... 🤔"):
                result = qa_chain.invoke({"query": question})
                answer = result.get("result", result)
                st.session_state['ready_to_ask'] = True

            st.markdown("### 🤖 Answer:")
            st.success(answer)

            ground_truth = st.text_area("Optional: Provide Ground Truth Answer for Evaluation", "")
            if st.button("Evaluate Answer") and ground_truth:
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