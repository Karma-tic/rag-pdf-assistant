# 📚 RAG PDF Assistant

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/Built%20with-LangChain-orange)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Free%20Tier-purple)](https://groq.com)

> **Chat with your PDFs using AI.** Upload any PDF, ask questions in plain English, get cited answers grounded in your documents. No OpenAI billing, no local GPU needed.

---

## ✨ What Makes This Different

| Feature | Most RAG Tutorials | This Repo |
|---------|-------------------|-----------|
| **Cost** | Requires OpenAI credits ($$$) | **100% free** via Groq |
| **Setup** | Complex cloud infra | **Runs on your MacBook** |
| **Privacy** | Sends docs to third parties | **Local embeddings**, only queries go to LLM |
| **Dependencies** | Heavy local models (2GB+) | **Lightweight** (~22MB embedder) |
| **Learning Curve** | Assumes ML knowledge | **Step-by-step**, every line explained |

---

## 🚀 Quick Start (3 Minutes)

### 1. Clone & Setup
```bash
git clone https://github.com/Karma-tic/rag-pdf-assistant.git
cd rag-pdf-assistant

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2. Add Your API Key
bash
Copy
# Get free key at https://console.groq.com/keys
echo "GROQ_API_KEY=your_key_here" > .env
3. Add PDFs
bash
Copy
# Drop any PDFs into this folder (ignored by git for privacy)
cp ~/Downloads/your-document.pdf data/
4. Ingest & Chat
bash
Copy
# Build the knowledge base (one-time per PDF batch)
python ingest.py

# Ask questions
python rag_chain.py "What are the key findings?"
python rag_chain.py "Summarize the methodology section"

# Or start interactive chat
python chat.py
🏗️ Architecture
plain
Copy
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Your PDFs  │────▶│  PyPDFLoader │────▶│  Text Chunks │
│  (data/*.pdf)│     │  (1 page/doc)│     │ (500 chars)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌─────▼──────┐
│  Chroma DB   │◀────│  Embeddings  │◀────│  Local HF  │
│  (vectors)   │     │  (384-dim)   │     │  MiniLM    │
└──────┬──────┘     └──────────────┘     └────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Retriever   │────▶│  LLM Prompt  │────▶│  Groq LLM   │
│  (top-4)     │     │  + context   │     │  (free)     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                         ┌──────▼──────┐
                                         │  Cited      │
                                         │  Answer     │
                                         └─────────────┘
Key Design Decisions:
Local embeddings (all-MiniLM-L6-v2) = privacy + zero cost
Groq LLM = fast inference, no local GPU needed
Chroma vector DB = pure Python, zero setup
Recursive chunking = preserves sentence boundaries
📁 Project Structure
Table
File	Purpose	Key Concept
document_loader.py	Extract text from PDFs	Document parsing
chunker.py	Split text intelligently	Chunking strategies
embedder.py	Text → vector conversion	Semantic embeddings
vector_store.py	Store & search vectors	Similarity search
rag_chain.py	Full Q&A pipeline	Retrieval-Augmented Generation
ingest.py	One-command rebuild	ETL pipeline
chat.py	Interactive CLI	Conversational interface
🎯 Use Cases
Legal: Query contracts, find clauses, check compliance
Research: Summarize papers, compare methodologies
Business: Analyze reports, extract KPIs, audit documents
Education: Study guides, textbook Q&A, thesis review
Personal: Resume analysis, medical records, tax documents
🛠️ Customization
Change Chunk Size
Python
Copy
# chunker.py
chunk_size=1000,    # Larger = more context, fewer chunks
chunk_overlap=100   # Higher = less boundary loss
Use Different LLM
Python
Copy
# rag_chain.py
model="mixtral-8x7b-32768"   # More creative
model="llama-3.3-70b-versatile"  # More factual (default)
Adjust Retrieved Chunks
Python
Copy
# rag_chain.py
search_kwargs={"k": 6}   # More context = longer answers
search_kwargs={"k": 2}   # Less context = focused answers
🔒 Privacy & Security
Table
What	Where It Happens	Data Sent
PDF text extraction	Your MacBook	Nothing
Text chunking	Your MacBook	Nothing
Embedding (text → vectors)	Your MacBook	Nothing
Vector storage	Your MacBook	Nothing
LLM generation	Groq servers	Only your question + relevant chunks
Your original PDFs never leave your machine.
🧠 Learning Path
This repo is designed as a progressive tutorial. Check the git history:
bash
Copy
git log --oneline
Each commit = one concept mastered:
Project setup & version control
Python environment isolation
Secure API key management
Document ingestion & parsing
Semantic chunking strategies
Text embeddings (the "magic")
Vector databases & similarity search
RAG pipeline construction
Prompt engineering for accuracy
Interactive chat interfaces
🐛 Troubleshooting
Table
Error	Fix
ModuleNotFoundError	Run pip install -r requirements.txt
No vector database found	Run python ingest.py first
API key invalid	Check .env file has GROQ_API_KEY=
Rate limit exceeded	Wait 1 min (Groq free tier: ~20 req/min)
Model decommissioned	Update model= in rag_chain.py to latest from Groq docs
🤝 Contributing
Found a bug? Want to add features?
Fork the repo
Create a branch: git checkout -b feature-name
Commit your changes
Push and open a Pull Request
Ideas for contributions:
Add support for .docx, .txt, .md files
Implement hybrid search (BM25 + vectors)
Add reranking with cross-encoders
Build a web UI with Streamlit
Add conversation memory for follow-up questions
📜 License
MIT License — free for personal and commercial use.
🙏 Acknowledgments
LangChain for the orchestration framework
HuggingFace for open-source embeddings
Groq for fast, free LLM inference
Chroma for the vector database
⭐ Star History
If this helped you learn RAG, please star the repo — it helps others find it!
https://star-history.com/#Karma-tic/rag-pdf-assistant&Date
Built with ❤️ for learners who want to understand RAG from the ground up.