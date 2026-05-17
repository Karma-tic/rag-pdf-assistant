from langchain_huggingface import HuggingFaceEmbeddings  # Local, free embeddings
from typing import List

def get_embedding_model():
    """
    Uses a free, local embedding model instead of OpenAI.
    
    'all-MiniLM-L6-v2' is a 22MB model that runs entirely on your MacBook.
    It creates 384-dimensional vectors (smaller than OpenAI's 1536, but still good).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use 'mps' if you have Apple Silicon GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

if __name__ == "__main__":
    embedder = get_embedding_model()
    
    # Demo: embed two sentences
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn canine leaps above the sleepy hound"
    ]
    
    vectors = embedder.embed_documents(texts)
    
    print(f"Vector dimensions: {len(vectors[0])}")
    print(f"First vector (first 5 numbers): {vectors[0][:5]}")
    print(f"Second vector (first 5 numbers): {vectors[1][:5]}")
    
    # Calculate similarity
    import numpy as np
    dot_product = np.dot(vectors[0], vectors[1])
    norm1 = np.linalg.norm(vectors[0])
    norm2 = np.linalg.norm(vectors[1])
    similarity = dot_product / (norm1 * norm2)
    
    print(f"\nSimilarity between the two sentences: {similarity:.4f}")
    print("(1.0 = identical meaning, 0.0 = completely different)")
    print("\n✅ Local embeddings working! No OpenAI required.")
