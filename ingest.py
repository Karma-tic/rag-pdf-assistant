#!/usr/bin/env python3
"""
One-command ingestion: PDFs → Chunks → Vector DB
Run this whenever you add new PDFs to the data/ folder.
"""
from document_loader import load_pdfs_from_folder
from chunker import chunk_documents
from vector_store import create_vector_store

def main():
    print("🚀 Starting ingestion pipeline...")
    
    # Step 1: Load
    docs = load_pdfs_from_folder()
    if not docs:
        print("❌ No PDFs found in data/ folder!")
        return
    
    # Step 2: Chunk
    chunks = chunk_documents(docs)
    
    # Step 3: Embed & Store
    store = create_vector_store(chunks)
    
    print("\n✅ Ingestion complete! You can now run: python rag_chain.py")

if __name__ == "__main__":
    main()
