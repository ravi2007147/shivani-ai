"""RAG pipeline for querying with context."""

import sys
from pathlib import Path
from typing import List, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import RAG_PROMPT_TEMPLATE, RETRIEVAL_K, MAX_CONTEXT_LENGTH


class RAGPipeline:
    """Manages the RAG pipeline for querying with context across multiple knowledge bases."""
    
    def __init__(
        self,
        vectorstores: List,  # Can be a single vectorstore or list of vectorstores
        llm_model: str,
        base_url: str,
    ):
        """Initialize the RAG pipeline.
        
        Args:
            vectorstores: Single Chroma vectorstore or list of vectorstores
            llm_model: Name of the LLM model to use
            base_url: Base URL for Ollama API
        """
        # Handle both single vectorstore and list of vectorstores
        if isinstance(vectorstores, list):
            self.vectorstores = vectorstores
            self.vectorstore = vectorstores[0] if vectorstores else None
        else:
            self.vectorstores = [vectorstores]
            self.vectorstore = vectorstores
        
        # Create retriever from first vectorstore (for compatibility)
        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever()
        else:
            self.retriever = None
        
        self.llm = OllamaLLM(
            model=llm_model,
            base_url=base_url,
        )
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=RAG_PROMPT_TEMPLATE,
        )
    
    def add_vectorstore(self, vectorstore) -> None:
        """Add a vectorstore to the active set.
        
        Args:
            vectorstore: Chroma vectorstore to add
        """
        if vectorstore not in self.vectorstores:
            self.vectorstores.append(vectorstore)
            # Update main vectorstore to first one if it was None
            if not self.vectorstore:
                self.vectorstore = vectorstore
                self.retriever = vectorstore.as_retriever()
    
    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents from all active knowledge bases.
        Optimized for large knowledge bases with efficient retrieval.
        
        Args:
            query: Query string
            k: Number of documents to retrieve per vectorstore (defaults to RETRIEVAL_K)
            
        Returns:
            List of relevant documents from all vectorstores
        """
        if k is None:
            k = RETRIEVAL_K
        
        all_docs = []
        
        # Search across all vectorstores (parallel processing potential)
        for vectorstore in self.vectorstores:
            try:
                # Use similarity_search_with_score for better ranking
                try:
                    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
                    # Sort by score (lower is better for distance)
                    docs_with_scores.sort(key=lambda x: x[1])
                    docs = [doc for doc, score in docs_with_scores]
                except Exception:
                    # Fallback to regular similarity search
                    docs = vectorstore.similarity_search(query, k=k)
                
                all_docs.extend(docs)
            except Exception:
                # Fallback to retriever if similarity_search fails
                try:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
                    try:
                        docs = retriever.invoke(query)
                    except Exception:
                        docs = retriever.get_relevant_documents(query)
                    all_docs.extend(docs)
                except Exception:
                    pass
        
        # Remove duplicates based on page_content
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content[:100])  # Use hash for faster comparison
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Limit total documents to prevent context overflow
        max_docs = k * len(self.vectorstores) if self.vectorstores else k
        return unique_docs[:max_docs]
    
    def query_with_context(self, query: str) -> dict:
        """Query with RAG context.
        Optimized for large knowledge bases with context length management.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with 'answer' and 'context_documents'
        """
        # Retrieve relevant documents
        docs = self.retrieve_documents(query)
        
        # Build context while respecting max length
        context_parts = []
        current_length = 0
        
        for doc in docs:
            doc_content = doc.page_content
            doc_length = len(doc_content)
            
            # Check if adding this doc would exceed limit
            if current_length + doc_length > MAX_CONTEXT_LENGTH and context_parts:
                break
            
            context_parts.append(doc_content)
            current_length += doc_length
        
        # Combine document contents as context
        context = "\n\n".join(context_parts)
        
        # Format prompt with context and question
        formatted_prompt = self.prompt_template.format(
            context=context,
            question=query
        )

        print("formatted_prompt",formatted_prompt)
        
        # Get answer from LLM
        answer = self.llm.invoke(formatted_prompt)
        
        return {
            "answer": answer,
            "context_documents": docs[:len(context_parts)],  # Only return used docs
        }
    
    def query_direct(self, query: str) -> str:
        """Query LLM directly without RAG context.
        
        Args:
            query: Query string
            
        Returns:
            Answer string
        """
        prompt = f"Answer the following question directly and naturally, without any meta-commentary or introductory phrases.\n\nQuestion: {query}\n\nAnswer:"
        answer = self.llm.invoke(prompt)
        return answer

