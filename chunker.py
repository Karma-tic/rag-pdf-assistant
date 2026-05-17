from langchain_text_splitters import RecursiveCharacterTextSplitter  # <-- FIXED
from langchain_core.documents import Document  # <-- Also fixed for consistency
from typing import List

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into smaller chunks with overlap.
    
    RecursiveCharacterTextSplitter tries to split at natural boundaries:
    1. First tries paragraphs (\n\n)
    2. Then lines (\n)
    3. Then sentences (.)
    4. Finally words (space)
    This keeps semantic meaning intact.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Each chunk: ~500 characters
        chunk_overlap=50,      # Overlap: 50 characters (continuity)
        length_function=len,   # Count characters, not tokens
        separators=["\n\n", "\n", ".", " ", ""]  # Priority order
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk index to metadata for debugging
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
    
    return chunks

if __name__ == "__main__":
    from document_loader import load_pdfs_from_folder
    docs = load_pdfs_from_folder()
    chunks = chunk_documents(docs)
    
    # Show a sample chunk
    if chunks:
        print(f"\n--- Sample Chunk #{chunks[0].metadata['chunk_index']} ---")
        print(f"Source: {chunks[0].metadata['source_file']}, Page: {chunks[0].metadata['page']}")
        print(f"Content:\n{chunks[0].page_content[:300]}...")
