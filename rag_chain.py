from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector_store import load_vector_store
from config import GROQ_API_KEY

def create_rag_chain():
    vector_store = load_vector_store()
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
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
    
    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get('source_file', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            content = doc.page_content
            formatted.append(f"[Source: {source}, Page: {page}]\n{content}")
        return "\n\n---\n\n".join(formatted)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def ask_question(question: str) -> str:
    chain = create_rag_chain()
    print(f"\nQuestion: {question}")
    print("=" * 50)
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    return answer

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask_question(question)
    else:
        questions = [
            "What is this document about?",
            "Summarize the main points"
        ]
        for q in questions:
            ask_question(q)
            print("\n" + "="*50 + "\n")
