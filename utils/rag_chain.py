"""
rag_chain.py
Builds the RAG (Retrieval-Augmented Generation) pipeline.
Connects retrieval (ChromaDB) with generation (Gemini LLM) using LangChain.
"""

import os
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# LLM Configuration
LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-1.5-flash")
LLM_TEMPERATURE = 0.1            # Low temperature = more factual, less creative


# Strict RAG prompt template
# This ensures the model ONLY answers from provided context
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions STRICTLY based on the provided document context.

INSTRUCTIONS:
- Answer ONLY using the information from the context below
- If the answer is not found in the context, respond EXACTLY: "The answer is not available in the uploaded document."
- Do not use outside knowledge, make assumptions, or hallucinate information
- Be concise but complete
- If the context contains relevant information, cite the specific details

DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

ANSWER:"""


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize the Gemini chat model.

    Returns:
        Configured Gemini LLM instance

    Note:
        Requires GOOGLE_API_KEY environment variable.
        Uses gemini-1.5-flash for fast, affordable responses.
    """
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        convert_system_message_to_human=True  # Gemini doesn't support system messages directly
    )


def format_docs(docs: List[Document]) -> str:
    """
    Format a list of Document chunks into a single context string.

    Each chunk is separated by a divider for clear separation.
    Page metadata is included to help the LLM locate information.

    Args:
        docs: Retrieved document chunks from similarity search

    Returns:
        Formatted context string for the prompt
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        # Include page number if available in metadata
        page_info = f"[Page {doc.metadata.get('page', 'N/A')}]"
        formatted.append(f"--- Chunk {i} {page_info} ---\n{doc.page_content}")

    return "\n\n".join(formatted)


def create_rag_chain(retriever):
    """
    Build the complete RAG pipeline using LangChain's LCEL (LangChain Expression Language).

    Pipeline flow:
        1. User question → retriever fetches relevant chunks
        2. Chunks are formatted into context string
        3. Context + question are fed into the prompt template
        4. Gemini LLM generates answer based ONLY on context
        5. Output is parsed to clean string

    Args:
        retriever: A retriever object (e.g., vector_store.as_retriever())

    Returns:
        Compiled RAG chain ready for invocation
    """
    # Initialize components
    llm = get_llm()
    prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    output_parser = StrOutputParser()

    # Build the chain using LCEL pipe operator (|)
    # This creates a clear, composable data flow
    rag_chain = (
        {
            # "context": retrieve relevant docs and format them
            "context": retriever | format_docs,
            # "question": pass through the original question unchanged
            "question": RunnablePassthrough()
        }
        | prompt           # Fill the template with context + question
        | llm              # Send to Gemini for generation
        | output_parser    # Extract clean string from LLM response
    )

    return rag_chain


def answer_question(rag_chain, question: str) -> str:
    """
    Execute the RAG chain to answer a user question.

    Args:
        rag_chain: Compiled RAG pipeline
        question: User's natural language question

    Returns:
        Generated answer string
    """
    # The chain handles retrieval, formatting, and generation automatically
    response = rag_chain.invoke(question)
    return response
