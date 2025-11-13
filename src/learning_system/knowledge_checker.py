"""LAYER 2: Knowledge Checker - Vector DB retrieval to check for existing knowledge.

Before involving the LLM, this layer checks the vector database to see if
we already have knowledge about the topic stored.
"""

import logging
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class KnowledgeChecker:
    """Checks vector database for existing knowledge before involving LLM.
    
    This is Layer 2 of the learning system. It:
    1. Performs vector search to check if knowledge exists
    2. If knowledge exists, loads it as context
    3. If no knowledge exists, can trigger LLM memory check or Auto-Discovery
    """
    
    def __init__(self, rag_pipeline=None, vectorstores=None, retriever=None):
        """Initialize the Knowledge Checker.
        
        Args:
            rag_pipeline: Optional RAGPipeline instance for retrieval
            vectorstores: Optional list of vectorstores for direct access
            retriever: Optional retriever for document retrieval
        """
        self.rag_pipeline = rag_pipeline
        self.vectorstores = vectorstores or []
        self.retriever = retriever
        
        # Minimum similarity score to consider knowledge as "existing"
        self.min_similarity_threshold = 0.3
        
        # Minimum number of documents to consider knowledge as "sufficient"
        self.min_documents_threshold = 1
    
    def check_knowledge(self, query: str, k: int = 5) -> Dict[str, any]:
        """Check if knowledge exists in vector database.
        
        Args:
            query: Query string to check for knowledge
            k: Number of documents to retrieve (default: 5)
            
        Returns:
            Dictionary with:
                - has_knowledge: True if knowledge exists, False otherwise
                - documents: List of relevant documents
                - context: Combined context from documents
                - similarity_scores: List of similarity scores
                - confidence: Confidence score (0.0 to 1.0)
        """
        if not query or not query.strip():
            return {
                'has_knowledge': False,
                'documents': [],
                'context': '',
                'similarity_scores': [],
                'confidence': 0.0,
                'message': 'Empty query'
            }
        
        query = query.strip()
        logger.info(f"Checking knowledge for query: {query}")
        
        # Try to retrieve documents
        try:
            documents = self._retrieve_documents(query, k)
            
            if not documents or len(documents) == 0:
                logger.info(f"No documents found for query: {query}")
                return {
                    'has_knowledge': False,
                    'documents': [],
                    'context': '',
                    'similarity_scores': [],
                    'confidence': 0.0,
                    'message': 'No documents found in vector database'
                }
            
            # Extract similarity scores if available
            similarity_scores = self._extract_similarity_scores(documents)
            
            # Check if we have sufficient knowledge
            has_knowledge = self._has_sufficient_knowledge(documents, similarity_scores)
            
            # Build context from documents
            context = self._build_context(documents)
            
            # Calculate confidence
            confidence = self._calculate_confidence(documents, similarity_scores)
            
            logger.info(f"Knowledge check result: has_knowledge={has_knowledge}, confidence={confidence:.2f}, documents={len(documents)}")
            
            return {
                'has_knowledge': has_knowledge,
                'documents': documents,
                'context': context,
                'similarity_scores': similarity_scores,
                'confidence': confidence,
                'message': 'Knowledge found' if has_knowledge else 'Insufficient knowledge'
            }
            
        except Exception as e:
            logger.error(f"Error checking knowledge: {str(e)}", exc_info=True)
            return {
                'has_knowledge': False,
                'documents': [],
                'context': '',
                'similarity_scores': [],
                'confidence': 0.0,
                'message': f'Error checking knowledge: {str(e)}'
            }
    
    def _retrieve_documents(self, query: str, k: int) -> List[Document]:
        """Retrieve documents from vector database.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of Document objects
        """
        documents = []
        
        # Try using RAG pipeline first
        if self.rag_pipeline:
            try:
                docs = self.rag_pipeline.retrieve_documents(query, k=k)
                if docs:
                    documents.extend(docs)
            except Exception as e:
                logger.warning(f"Error retrieving from RAG pipeline: {str(e)}")
        
        # Try using retriever directly
        if not documents and self.retriever:
            try:
                docs = self.retriever.invoke(query)
                if docs:
                    documents.extend(docs)
            except Exception as e:
                logger.warning(f"Error retrieving from retriever: {str(e)}")
        
        # Try using vectorstores directly
        if not documents and self.vectorstores:
            try:
                for vectorstore in self.vectorstores:
                    try:
                        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
                        docs = retriever.invoke(query)
                        if docs:
                            documents.extend(docs)
                    except Exception as e:
                        logger.warning(f"Error retrieving from vectorstore: {str(e)}")
                        continue
            except Exception as e:
                logger.warning(f"Error retrieving from vectorstores: {str(e)}")
        
        # Remove duplicates based on content
        unique_documents = []
        seen_content = set()
        for doc in documents:
            content_hash = hash(doc.page_content[:100])  # Hash first 100 chars
            if content_hash not in seen_content:
                unique_documents.append(doc)
                seen_content.add(content_hash)
        
        return unique_documents[:k]
    
    def _extract_similarity_scores(self, documents: List[Document]) -> List[float]:
        """Extract similarity scores from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of similarity scores
        """
        scores = []
        for doc in documents:
            # Try to get score from metadata
            if hasattr(doc, 'metadata') and doc.metadata:
                score = doc.metadata.get('score', doc.metadata.get('similarity_score', None))
                if score is not None:
                    scores.append(float(score))
                else:
                    scores.append(1.0)  # Default score if not available
            else:
                scores.append(1.0)  # Default score if no metadata
        return scores
    
    def _has_sufficient_knowledge(self, documents: List[Document], similarity_scores: List[float]) -> bool:
        """Check if we have sufficient knowledge.
        
        Args:
            documents: List of Document objects
            similarity_scores: List of similarity scores
            
        Returns:
            True if sufficient knowledge exists, False otherwise
        """
        # Check minimum document count
        if len(documents) < self.min_documents_threshold:
            return False
        
        # Check similarity scores
        if similarity_scores:
            # Check if any score is above threshold
            max_score = max(similarity_scores) if similarity_scores else 0.0
            if max_score < self.min_similarity_threshold:
                return False
        
        # Check document content quality
        total_content_length = sum(len(doc.page_content) for doc in documents)
        if total_content_length < 100:  # Minimum content length
            return False
        
        return True
    
    def _build_context(self, documents: List[Document], max_length: int = 5000) -> str:
        """Build context string from documents.
        
        Args:
            documents: List of Document objects
            max_length: Maximum context length (default: 5000)
            
        Returns:
            Combined context string
        """
        context_parts = []
        current_length = 0
        
        for doc in documents:
            content = doc.page_content
            content_length = len(content)
            
            # Check if adding this document would exceed max length
            if current_length + content_length > max_length and context_parts:
                # Add truncated version if space allows
                remaining = max_length - current_length
                if remaining > 100:
                    context_parts.append(content[:remaining] + "...")
                break
            
            context_parts.append(content)
            current_length += content_length
        
        return "\n\n".join(context_parts)
    
    def _calculate_confidence(self, documents: List[Document], similarity_scores: List[float]) -> float:
        """Calculate confidence score.
        
        Args:
            documents: List of Document objects
            similarity_scores: List of similarity scores
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not documents:
            return 0.0
        
        # Base confidence on number of documents
        doc_confidence = min(len(documents) / 5.0, 1.0)  # Max at 5 documents
        
        # Base confidence on similarity scores
        score_confidence = 0.0
        if similarity_scores:
            avg_score = sum(similarity_scores) / len(similarity_scores)
            max_score = max(similarity_scores)
            score_confidence = (avg_score + max_score) / 2.0
        else:
            score_confidence = 0.7  # Default if scores not available
        
        # Base confidence on content length
        total_length = sum(len(doc.page_content) for doc in documents)
        length_confidence = min(total_length / 2000.0, 1.0)  # Max at 2000 chars
        
        # Combined confidence
        confidence = (doc_confidence * 0.3 + score_confidence * 0.5 + length_confidence * 0.2)
        
        return min(confidence, 1.0)
    
    def check_knowledge_with_llm_memory(self, query: str, llm=None) -> Dict[str, any]:
        """Check knowledge with LLM memory check.
        
        This is used when vector DB search doesn't find knowledge.
        It asks the LLM to check its memory/knowledge about the topic.
        
        Args:
            query: Query string
            llm: Optional LLM instance for memory check
            
        Returns:
            Dictionary with:
                - has_knowledge: True if LLM has knowledge, False otherwise
                - response: LLM response
                - confidence: Confidence score
        """
        if not llm:
            return {
                'has_knowledge': False,
                'response': '',
                'confidence': 0.0,
                'message': 'No LLM provided'
            }
        
        try:
            prompt = f"""Do we have existing knowledge about: "{query}"?

Answer with ONLY one of these options:
- "YES" if we have knowledge about this topic
- "NO" if we don't have knowledge about this topic

Answer:"""
            
            response = llm.invoke(prompt).strip().upper()
            
            has_knowledge = "YES" in response or "HAVE" in response or "KNOW" in response
            
            # Check for explicit NO
            if "NO" in response or "DON'T" in response or "DO NOT" in response:
                has_knowledge = False
            
            logger.info(f"LLM memory check for '{query}': has_knowledge={has_knowledge}")
            
            return {
                'has_knowledge': has_knowledge,
                'response': response,
                'confidence': 0.7 if has_knowledge else 0.3,
                'message': 'LLM memory check completed'
            }
            
        except Exception as e:
            logger.error(f"Error in LLM memory check: {str(e)}", exc_info=True)
            return {
                'has_knowledge': False,
                'response': '',
                'confidence': 0.0,
                'message': f'Error in LLM memory check: {str(e)}'
            }

