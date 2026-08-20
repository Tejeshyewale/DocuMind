"""
Steps 6 and 7 of the RAG pipeline: retrieve the most relevant chunks for a
question, hand them to Groq as grounding context, and return the answer
plus detailed source citations and retrieved contexts.
"""

import os
from langchain_groq import ChatGroq

PROMPT_TEMPLATE = """You are a helpful assistant answering questions strictly \
based on the context extracted from a document. If the answer is not present \
in the context, say clearly that the document does not contain this information \
— do not make anything up.

Context:
{context}

Question:
{question}

Answer:"""


def get_llm():
    """
    Configure and load the Groq chat LLM.
    Model is read from GROQ_CHAT_MODEL env var, and api key from GROQ_API_KEY.
    """
    model_name = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(
        model=model_name,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


def answer_question(vector_store, question: str, k: int = 4):
    """
    Run the full retrieve -> augment -> generate flow for one question.
    Returns:
        answer_text (str): The response from the LLM.
        citations (list of str): List of formatted source citations (e.g. ["sample.pdf, page 1"]).
        retrieved_docs (list of dict): List of dictionaries containing page_content and metadata for showing context.
    """
    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    llm = get_llm()
    response = llm.invoke(prompt)

    citation_set = set()
    retrieved_docs = []
    
    for d in docs:
        source = d.metadata.get("source", "Unknown Document")
        page = d.metadata.get("page", "?")
        
        # PyPDFLoader pages are 0-indexed; convert to human-friendly 1-indexed page numbers
        page_num = page + 1 if isinstance(page, int) else page
        citation_set.add(f"{source}, page {page_num}")
        
        retrieved_docs.append({
            "content": d.page_content,
            "source": source,
            "page": page_num
        })
        
    citations = sorted(list(citation_set))

    return response.content, citations, retrieved_docs
