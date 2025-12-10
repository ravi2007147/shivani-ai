"""Relation Extractor - Extracts structured relations from scraped content using LLM.

This module provides functionality to:
1. Retrieve related facts from vector DB
2. Use LLM to extract structured relations between new content and existing knowledge
3. Clean and filter content based on relevance
4. Return structured JSON metadata along with cleaned text
"""

import logging
import json
from typing import List, Dict, Optional, Tuple, Any
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RelationExtractor:
    """Extracts structured relations from scraped content using LLM.
    
    For every scraped text chunk:
    1. Retrieves related facts from vector DB
    2. Sends to LLM for relation extraction:
       - known facts (from vector DB)
       - new scraped text
    3. Outputs:
       - structured relations
       - cleaned relevant text
    4. If related = true:
       - Returns cleaned text and structured JSON
    5. If related = false:
       - Returns None (content discarded)
    """
    
    def __init__(
        self,
        ollama_model: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text"
    ):
        """Initialize the Relation Extractor.
        
        Args:
            ollama_model: Name of the Ollama LLM model to use
            ollama_base_url: Base URL for Ollama API
            embedding_model: Name of the embedding model (for similarity search)
        """
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=ollama_model,
            base_url=ollama_base_url
        )
        
        logger.info(f"RelationExtractor initialized with model: {ollama_model}")
    
    def _get_llm(self) -> OllamaLLM:
        """Get or create LLM instance."""
        if not self.llm:
            self.llm = OllamaLLM(
                model=self.ollama_model,
                base_url=self.ollama_base_url
            )
        return self.llm
    
    def retrieve_related_facts(
        self,
        vectorstore,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """Retrieve related facts from vector DB using similarity search.
        
        Args:
            vectorstore: Chroma vectorstore instance
            query: Query text to search for related content
            k: Number of similar documents to retrieve (default: 5)
            
        Returns:
            List of related Document objects from vector DB
        """
        if not vectorstore:
            logger.warning("No vectorstore provided, returning empty list")
            return []
        
        try:
            # Perform similarity search
            related_docs = vectorstore.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(related_docs)} related facts from vector DB")
            return related_docs
        except Exception as e:
            logger.error(f"Error retrieving related facts: {str(e)}")
            return []
    
    def extract_relations(
        self,
        new_content: str,
        related_facts: List[Document],
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract structured relations using LLM.
        
        Args:
            new_content: New scraped content to analyze
            related_facts: List of related facts from vector DB
            topic: Optional topic/term being searched
            
        Returns:
            Dictionary with:
            - 'is_related': bool - Whether content is related to existing facts
            - 'cleaned_text': str - Cleaned relevant text
            - 'structured_relations': dict - Structured relation data
            - 'confidence': float - Confidence score (0.0 to 1.0)
        """
        llm = self._get_llm()
        
        # Prepare related facts summary
        related_facts_text = ""
        if related_facts:
            facts_list = []
            for i, doc in enumerate(related_facts, 1):
                content = doc.page_content[:500]  # Limit each fact
                metadata = doc.metadata or {}
                source = metadata.get('source', 'Unknown')
                
                facts_list.append(f"{i}. {content}\n   (Source: {source})")
            
            related_facts_text = "\n\n".join(facts_list)
        else:
            related_facts_text = "No existing related facts found in knowledge base."
        
        # Truncate new content for prompt (keep it manageable)
        content_preview = new_content[:3000] if len(new_content) > 3000 else new_content
        
        # Build prompt
        topic_context = f"\nTopic/Search Term: {topic}\n" if topic else ""
        
        prompt = f"""You are a Relation Extractor AI. Your task is to analyze new scraped content and determine if it's related to existing knowledge, then extract structured relations.

{topic_context}
=== EXISTING KNOWLEDGE FROM VECTOR DB ===
{related_facts_text}

=== NEW SCRAPED CONTENT ===
{content_preview}

=== INSTRUCTIONS ===
1. Analyze the new scraped content in relation to the existing knowledge
2. Determine if the new content is RELATED to the existing knowledge:
   - Related: Content discusses the same topic, entity, concept, or adds information that connects to existing knowledge
   - Not Related: Content is completely different, off-topic, or has no connection to existing knowledge
3. If RELATED:
   - Extract structured relations (entities, concepts, relationships, facts)
   - Clean the text to keep only relevant parts
   - Create a structured JSON with relations
4. If NOT RELATED:
   - Return is_related: false
   - Discard the content

=== OUTPUT FORMAT (JSON) ===
{{
  "is_related": true/false,
  "confidence": 0.0-1.0,
  "cleaned_text": "cleaned and relevant text only",
  "structured_relations": {{
    "entities": ["entity1", "entity2"],
    "concepts": ["concept1", "concept2"],
    "relationships": [
      {{"subject": "entity1", "predicate": "relates to", "object": "entity2"}},
      {{"subject": "concept1", "predicate": "describes", "object": "entity1"}}
    ],
    "facts": ["fact1", "fact2"],
    "connections": ["connection to existing knowledge"]
  }},
  "reasoning": "brief explanation of why related or not"
}}

Answer ONLY with valid JSON. Do not include any text outside the JSON.
"""
        
        try:
            logger.info("🔍 Extracting relations using LLM...")
            response = llm.invoke(prompt).strip()
            
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                # Fallback: try parsing entire response
                result = json.loads(response)
            
            # Validate and process result
            is_related = result.get('is_related', False)
            confidence = float(result.get('confidence', 0.0))
            cleaned_text = result.get('cleaned_text', '')
            structured_relations = result.get('structured_relations', {})
            reasoning = result.get('reasoning', '')
            
            # If not related, clear cleaned_text
            if not is_related:
                cleaned_text = ''
            
            logger.info(f"✅ Relation extraction complete: is_related={is_related}, confidence={confidence:.2f}")
            
            return {
                'is_related': is_related,
                'confidence': confidence,
                'cleaned_text': cleaned_text,
                'structured_relations': structured_relations,
                'reasoning': reasoning,
                'original_length': len(new_content),
                'cleaned_length': len(cleaned_text)
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM response as JSON: {str(e)}")
            logger.error(f"Response: {response[:500]}")
            # Fallback: return as not related if JSON parsing fails
            return {
                'is_related': False,
                'confidence': 0.0,
                'cleaned_text': '',
                'structured_relations': {},
                'reasoning': f'JSON parsing error: {str(e)}',
                'original_length': len(new_content),
                'cleaned_length': 0
            }
        except Exception as e:
            logger.error(f"❌ Error in relation extraction: {str(e)}")
            return {
                'is_related': False,
                'confidence': 0.0,
                'cleaned_text': '',
                'structured_relations': {},
                'reasoning': f'Error: {str(e)}',
                'original_length': len(new_content),
                'cleaned_length': 0
            }
    
    def process_content(
        self,
        content: str,
        vectorstore,
        topic: Optional[str] = None,
        k_facts: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Process content through the full relation extraction pipeline.
        
        Args:
            content: Scraped content to process
            vectorstore: Chroma vectorstore instance (can be None)
            topic: Optional topic/term being searched
            k_facts: Number of related facts to retrieve (default: 5)
            
        Returns:
            Dictionary with processed content and relations if related, None otherwise.
            Contains:
            - 'cleaned_text': str - Cleaned relevant text for embedding
            - 'structured_relations': dict - Structured JSON metadata
            - 'confidence': float - Confidence score
            - 'is_related': bool - Whether content is related
            - 'metadata': dict - Additional metadata for storage
        """
        if not content or len(content.strip()) < 50:
            logger.warning("Content too short, skipping relation extraction")
            return None
        
        # Step 1: Retrieve related facts from vector DB
        related_facts = []
        if vectorstore:
            # Use topic or content as query
            query = topic if topic else content[:500]
            related_facts = self.retrieve_related_facts(vectorstore, query, k=k_facts)
        else:
            logger.info("No vectorstore provided, skipping fact retrieval")
        
        # Step 2: Extract relations using LLM
        result = self.extract_relations(content, related_facts, topic=topic)
        
        # Step 3: If not related, discard (return None)
        if not result.get('is_related', False):
            logger.info(f"Content not related to existing knowledge (confidence: {result.get('confidence', 0.0):.2f}) - discarding")
            return None
        
        # Step 4: If related, prepare output with cleaned text and structured relations
        structured_json = json.dumps(result.get('structured_relations', {}), indent=2)
        
        return {
            'cleaned_text': result.get('cleaned_text', ''),
            'structured_relations': result.get('structured_relations', {}),
            'structured_json': structured_json,
            'confidence': result.get('confidence', 0.0),
            'is_related': True,
            'reasoning': result.get('reasoning', ''),
            'original_length': result.get('original_length', 0),
            'cleaned_length': result.get('cleaned_length', 0),
            'metadata': {
                'relation_extractor': True,
                'confidence': result.get('confidence', 0.0),
                'related_facts_count': len(related_facts),
                'structured_relations': result.get('structured_relations', {})
            }
        }
    
    def process_content_chunks(
        self,
        content_chunks: List[str],
        vectorstore,
        topic: Optional[str] = None,
        k_facts: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple content chunks through relation extraction.
        
        Args:
            content_chunks: List of content strings to process
            vectorstore: Chroma vectorstore instance
            topic: Optional topic/term being searched
            k_facts: Number of related facts to retrieve per chunk
            
        Returns:
            List of processed content dictionaries (only related chunks are included)
        """
        processed_chunks = []
        
        for i, chunk in enumerate(content_chunks, 1):
            logger.info(f"Processing chunk {i}/{len(content_chunks)}...")
            result = self.process_content(chunk, vectorstore, topic=topic, k_facts=k_facts)
            
            if result:
                processed_chunks.append(result)
            else:
                logger.info(f"Chunk {i} discarded (not related)")
        
        logger.info(f"✅ Processed {len(processed_chunks)}/{len(content_chunks)} chunks as related")
        return processed_chunks




