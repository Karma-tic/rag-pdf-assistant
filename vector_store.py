from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # <-- FIXED
from typing import List
from embedder import get_embedding_model
import os

CHROMA_DIR = "chroma_db"

def create_vector_store(chunks: List[Document]) -> Chroma:
    """
    Creates a Chroma vector database from document chunks.
    
    Chroma stores:
    - The original text (for retrieval)
    - The embedding vectors (for search)
    - Metadata (source, page, chunk_index)
    
    It uses HNSW (Hierarchical Navigable Small World) algorithm for fast
    approximate nearest neighbor search - think of it as a smart index.
    """
    # Remove old DB if exists (for clean rebuilds)
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("Cleared old vector database")
    
    embedding_model = get_embedding_model()
    
    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name="pdf_documents"
    )
    
    print(f"Vector store created with {len(chunks)} documents")
    print(f"Stored in: {os.path.abspath(CHROMA_DIR)}")
    
    return vector_store

def load_vector_store() -> Chroma:
    """Load existing vector store from disk."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError("No vector database found. Run ingestion first!")
    
    embedding_model = get_embedding_model()
    
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_name="pdf_documents"
    )

def search_similar(query: str, k: int = 4) -> List[Document]:
    """
    Search for chunks most similar to the query.
    k=4 means "return top 4 matches"
    """
    store = load_vector_store()
    results = store.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    # Full pipeline test
    from document_loader import load_pdfs_from_folder
    from chunker import chunk_documents
    
    print("=== Building Vector Store ===")
    docs = load_pdfs_from_folder()
    chunks = chunk_documents(docs)
    store = create_vector_store(chunks)
    
    print("\n=== Testing Search ===")
    query = "What is this document about?"
    results = search_similar(query, k=2)
    
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Source: {doc.metadata['source_file']}, Page: {doc.metadata['page']}")
        print(f"Content: {doc.page_content[:200]}...")
