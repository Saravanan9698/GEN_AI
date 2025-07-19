# 💬 ChatBot AI

> *From documents to decisions — powered by AI, secured locally.*

## 🔍 What is ChatBot AI?

**ChatBot AI** is a early stage startup comprising a powerful and private document question-answering app powered by **LLaMA 2** and **LangChain**. Upload your PDF, DOCX, or TXT files, and ask intelligent questions. All data is processed locally to maintain maximum privacy.

---

## 🎯 Features

- 📄 Supports PDF, DOCX, and TXT files
- 🧠 Local LLaMA 2 model inference (no cloud dependencies)
- 🔍 Intelligent search & similarity-based retrieval
- 📦 Built with Streamlit for an interactive UI
- 🖼️ Custom dark-themed background support
- 🧵 Efficient multi-threaded document parsing

---

## 🚀 Getting Started

### 📦 Prerequisites

- Python 3.9+
- Download the LLaMA 2 model from [HERE!](https://huggingface.co/Saravanan2003/llama-2-7b-chat.Q4_K_S.gguf)  

### 📁 Folder Structure  
  
```  
GEN_AI_FINAL_PROJECT/
├── Gen_AI.py                          # Main application script
├── HuggingFaceEmbeddings.ipynb        # Jupyter notebook for embedding exploration
├── requirements.txt                   # Project dependencies
├── evaluation_log.json                # Evaluation log file
│
├── Model/                             # LLaMA model folder
│   └── llama-2-7b-chat.Q4_K_M.gguf    # Quantized LLaMA model file
│
├── models/
│   └── sentence_transformers/
│       └── all-MiniLM-L6-v2/
│           ├── 1_Pooling/
│           ├── onnx/
│           ├── openvino/
│           ├── .cache/
│           ├── config_sentence_transformers.json
│           ├── config.json
│           ├── data_config.json
│           ├── model.safetensors
│           ├── modules.json
│           ├── pytorch_model.bin
│           ├── rust_model.ot
│           ├── sentence_bert_config.json
│           ├── special_tokens_map.json
│           ├── tf_model.h5
│           ├── tokenizer_config.json
│           ├── tokenizer.json
│           ├── vocab.txt
│           ├── README.md
│           └── train_script.py        # Optional training script
│
├── db/
│   └── faiss/
│       ├── index.faiss                # FAISS index file
│       └── index.pkl                  # Serialized metadata or index data
│
├── faiss_index/                       # FAISS vector store (auto-generated)
│   ├── index.faiss
│   └── index.pkl
│
├── evaluation_logs/                  # Auto-generated logs
│   └── evaluation_results.json
│
├── Images/
│   ├── 767.jpg
│   ├── ChatGPT_Image.png
│   └── freepik__adjust__9850.jpeg
│
├── Documents/
│   ├── Advanced_Facts_Octopus.pdf
│   ├── Advanced_Velociraptor_Text.txt
│   ├── Dire_Wolf.docx
│   └── Saravanan Data Science Resume ATS match without emoji.pdf
│
└── env/                               # Python virtual environment
    ├── Include/
    ├── Lib/
    ├── Scripts/
    └── pyvenv.cfg

```
---

## 🧰 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/AskDocs_GEN-AI.git
cd AskDocs_GEN-AI

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```  
## 🧠 Usage  

```bash
# Run
streamlit run app.py
```

1.  Upload one or more PDF, DOCX, or TXT documents.  
2.  Click Start The Fun!  
3.  Ask your question in natural language.  
4.  Get answers with cited document sources.

## 📦 Dependencies  
 
- streamlit  
- PyMuPDF  
- python-docx  
- docx2txt  
- requests  
- numpy  
- langchain  
- sentence-transformers  
- nltk  
- rouge-score  
- ctransformers  
- faiss-cpu  
- huggingface-hub  

## 📎 Notes  

- You need to download and manually place the LLaMA model.
- Ensure GPU support is configured if needed (set gpu_layers accordingly).
- Image background is customizable — make sure the image path is valid.
