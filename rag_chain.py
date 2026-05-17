from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough  # <-- FIXED
from langchain_core.output_parsers import StrOutputParser  # <-- Also fixed
from vector_store import load_vector_store
from config import OPENAI_API_KEY

def create_rag_chain():
    """
    Builds the complete RAG pipeline using LangChain's LCEL (LangChain Expression Language).
    
    The chain flow:
    1. Retrieve relevant chunks from vector store
    2. Format them into a context string
    3. Combine with user question and system prompt
    4. Send to LLM
    5. Parse and return answer
    """
    
    # Initialize components
    vector_store = load_vector_store()
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # Cheap and fast for testing
        temperature=0,          # 0 = deterministic, no creativity/hallucination
        openai_api_key=OPENAI_API_KEY
    )
    
    # The retriever: converts vector store into a function
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Fetch top 4 chunks
    )
    
    # The prompt: engineering the LLM behavior
    template = """You are a helpful assistant that answers questions based on provided documents.
    
    Follow these rules STRICTLY:
    1. Answer using ONLY the information in the provided context
    2. If the answer isn't in the context, say "I don't have enough information to answer this"
    3. Cite the source file and page number for each fact you use
    4. Be concise but complete
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Helper: format retrieved documents into a string
    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source_file', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            content = doc.page_content
            formatted.append(f"[Source: {source}, Page: {page}]\n{content}")
        return "\n\n---\n\n".join(formatted)
    
    # Build the chain using | operator (pipe)
    # RunnablePassthrough passes the input through unchanged
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def ask_question(question: str) -> str:
    """Simple interface to ask questions."""
    chain = create_rag_chain()
    print(f"\nQuestion: {question}")
    print("=" * 50)
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    return answer

if __name__ == "__main__":
    # Interactive mode
    import sys
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask_question(question)
    else:
        # Demo questions
        questions = [
            "What is this document about?",
            "Summarize the main points"
        ]
        for q in questions:
            ask_question(q)
            print("\n" + "="*50 + "\n")
