from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document  # <-- FIXED: was langchain.schema
from typing import List
import os

def load_pdfs_from_folder(folder_path: str = "data") -> List[Document]:
    """
    Loads all PDFs from a folder and returns LangChain Document objects.
    Each Document contains: page_content (text) and metadata (source, page number)
    """
    documents = []
    
    # Walk through the data folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Loading: {filename}")
            
            # PyPDFLoader reads the PDF and splits by page
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Each 'page' is a Document with metadata
            for page in pages:
                # Add filename to metadata for traceability
                page.metadata["source_file"] = filename
            
            documents.extend(pages)
            print(f"  → Extracted {len(pages)} pages")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents

if __name__ == "__main__":
    docs = load_pdfs_from_folder()
    # Show first 200 chars of first page
    if docs:
        print(f"\nSample content:\n{docs[0].page_content[:200]}...")
