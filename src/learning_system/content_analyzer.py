"""Content Analyzer - Main orchestrator for the smart learning system.

This class combines IntentRouter (Layer 1) and KnowledgeChecker (Layer 2)
to analyze content and determine the best course of action.
"""

import logging
from typing import Dict, Optional, List, Any
from langchain_ollama import OllamaLLM

from .intent_router import IntentRouter
from .knowledge_checker import KnowledgeChecker
from .auto_discovery_agent import AutoDiscoveryAgent

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Main content analyzer that orchestrates the learning system layers.
    
    This class:
    1. Routes input using IntentRouter (Layer 1)
    2. Checks knowledge using KnowledgeChecker (Layer 2)
    3. Determines next actions (e.g., trigger Auto-Discovery)
    """
    
    def __init__(
        self,
        rag_pipeline=None,
        vectorstores=None,
        retriever=None,
        ollama_model: str = "mistral",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """Initialize the Content Analyzer.
        
        Args:
            rag_pipeline: Optional RAGPipeline instance for knowledge checking
            vectorstores: Optional list of vectorstores for knowledge checking
            retriever: Optional retriever for knowledge checking
            ollama_model: Ollama model name for LLM operations
            ollama_base_url: Ollama base URL
        """
        self.intent_router = IntentRouter()
        self.knowledge_checker = KnowledgeChecker(
            rag_pipeline=rag_pipeline,
            vectorstores=vectorstores,
            retriever=retriever
        )
        self.auto_discovery_agent = AutoDiscoveryAgent(
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            headless=False,  # Set to False for testing/debugging - can see browser
            search_engine="duckduckgo"
        )
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.llm = None
        self.rag_pipeline = rag_pipeline
        self.vectorstores = vectorstores
        self.retriever = retriever
    
    def _get_llm(self) -> OllamaLLM:
        """Get or create LLM instance.
        
        Returns:
            OllamaLLM instance
        """
        if self.llm is None:
            self.llm = OllamaLLM(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=0.2
            )
        return self.llm
    
    def analyze(self, input_text: str, check_knowledge: bool = True) -> Dict[str, Any]:
        """Analyze input content and determine actions.
        
        This is the main entry point for content analysis.
        It combines Layer 1 (Intent Router) and Layer 2 (Knowledge Check).
        
        Args:
            input_text: Input text to analyze
            check_knowledge: Whether to check knowledge in vector DB (default: True)
            
        Returns:
            Dictionary with:
                - intent: Detected intent (url, question, command, etc.)
                - has_knowledge: Whether knowledge exists in vector DB
                - knowledge_context: Context from vector DB if available
                - next_action: Recommended next action
                - metadata: Additional analysis information
        """
        logger.info(f"Analyzing content: {input_text[:100]}...")
        
        # Layer 1: Route intent
        routing_result = self.intent_router.route(input_text)
        intent = routing_result.get('intent')
        processor = routing_result.get('processor')
        metadata = routing_result.get('metadata', {})
        
        result = {
            'intent': intent,
            'processor': processor,
            'has_knowledge': False,
            'knowledge_context': '',
            'knowledge_documents': [],
            'next_action': 'process',
            'metadata': metadata,
            'routing_result': routing_result
        }
        
        # Layer 2: Check knowledge (if enabled and applicable)
        if check_knowledge:
            # Determine what to search for in knowledge base
            knowledge_query = self._build_knowledge_query(input_text, intent, metadata)
            
            if knowledge_query:
                # Check vector DB for existing knowledge
                knowledge_result = self.knowledge_checker.check_knowledge(knowledge_query, k=5)
                
                result['has_knowledge'] = knowledge_result.get('has_knowledge', False)
                result['knowledge_context'] = knowledge_result.get('context', '')
                result['knowledge_documents'] = knowledge_result.get('documents', [])
                result['knowledge_confidence'] = knowledge_result.get('confidence', 0.0)
                result['knowledge_message'] = knowledge_result.get('message', '')
                
                # If no knowledge found, check LLM memory
                if not result['has_knowledge']:
                    llm = self._get_llm()
                    llm_memory_result = self.knowledge_checker.check_knowledge_with_llm_memory(
                        knowledge_query,
                        llm=llm
                    )
                    
                    result['llm_memory_check'] = llm_memory_result.get('has_knowledge', False)
                    result['llm_memory_response'] = llm_memory_result.get('response', '')
                    
                    # Determine next action
                    if not result['llm_memory_check']:
                        # No knowledge in vector DB and no memory in LLM
                        # Trigger Auto-Discovery (to be implemented)
                        result['next_action'] = 'auto_discovery'
                        result['auto_discovery_trigger'] = True
                        result['auto_discovery_reason'] = 'No existing knowledge found in vector DB or LLM memory'
                    else:
                        # LLM has memory but not in vector DB
                        # Could store this in vector DB for future use
                        result['next_action'] = 'process_with_llm_memory'
                        result['llm_memory_available'] = True
                else:
                    # Knowledge exists in vector DB
                    result['next_action'] = 'process_with_context'
                    result['context_available'] = True
        
        # Determine processor-specific actions
        # Only override next_action if auto-discovery is not triggered
        if not result.get('auto_discovery_trigger', False):
            if intent == 'url':
                result['next_action'] = 'website_processor'
                result['url'] = routing_result.get('url')
                result['domain'] = routing_result.get('domain')
            elif intent == 'question':
                if result.get('has_knowledge'):
                    result['next_action'] = 'answer_with_context'
                else:
                    result['next_action'] = 'answer_or_discover'
            elif intent == 'command':
                result['next_action'] = 'execute_command'
        else:
            # Auto-discovery is triggered, but still set URL/domain info if available
            if intent == 'url':
                result['url'] = routing_result.get('url')
                result['domain'] = routing_result.get('domain')
        
        logger.info(f"Analysis complete: intent={intent}, has_knowledge={result.get('has_knowledge')}, next_action={result.get('next_action')}")
        
        return result
    
    def _build_knowledge_query(self, input_text: str, intent: str, metadata: Dict) -> Optional[str]:
        """Build knowledge query from input.
        
        Args:
            input_text: Original input text
            intent: Detected intent
            metadata: Routing metadata
            
        Returns:
            Knowledge query string or None
        """
        if intent == 'url':
            # For URLs, extract domain or topic
            domain = metadata.get('domain')
            if domain:
                # Remove common TLDs for cleaner query
                domain_parts = domain.split('.')
                if len(domain_parts) >= 2:
                    # Get main domain (e.g., "upwork" from "upwork.com")
                    main_domain = domain_parts[-2] if domain_parts[-2] != 'www' else domain_parts[-3] if len(domain_parts) >= 3 else domain_parts[-1]
                    return f"what is {main_domain}"
                return f"what is {domain}"
            return f"general knowledge about {input_text}"
        
        elif intent == 'question':
            # For questions, use the question itself
            return input_text
        
        elif intent == 'command':
            # For commands, extract the topic
            command_type = metadata.get('command_type', '')
            if command_type:
                return f"{command_type} {input_text}"
            return input_text
        
        else:
            # For general input, use as-is
            return input_text
    
    def get_analysis_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Get human-readable summary of analysis result.
        
        Args:
            analysis_result: Analysis result dictionary
            
        Returns:
            Human-readable summary string
        """
        intent = analysis_result.get('intent', 'unknown')
        has_knowledge = analysis_result.get('has_knowledge', False)
        next_action = analysis_result.get('next_action', 'unknown')
        knowledge_confidence = analysis_result.get('knowledge_confidence', 0.0)
        
        summary_parts = []
        
        # Intent summary
        summary_parts.append(f"**Intent:** {intent.capitalize()}")
        
        # Knowledge summary
        if has_knowledge:
            summary_parts.append(f"**Knowledge:** Found in vector DB (confidence: {knowledge_confidence:.2f})")
            summary_parts.append(f"**Next Action:** {next_action}")
        else:
            llm_memory = analysis_result.get('llm_memory_check', False)
            if llm_memory:
                summary_parts.append(f"**Knowledge:** Not in vector DB, but LLM has memory")
                summary_parts.append(f"**Next Action:** {next_action}")
            else:
                summary_parts.append(f"**Knowledge:** Not found in vector DB or LLM memory")
                summary_parts.append(f"**Next Action:** {next_action} (Auto-Discovery needed)")
        
        # Additional info
        if analysis_result.get('url'):
            summary_parts.append(f"**URL:** {analysis_result.get('url')}")
        if analysis_result.get('domain'):
            summary_parts.append(f"**Domain:** {analysis_result.get('domain')}")
        
        return "\n".join(summary_parts)
    
    def should_trigger_auto_discovery(self, analysis_result: Dict[str, Any]) -> bool:
        """Check if auto-discovery should be triggered.
        
        Args:
            analysis_result: Analysis result dictionary
            
        Returns:
            True if auto-discovery should be triggered, False otherwise
        """
        return (
            not analysis_result.get('has_knowledge', False) and
            not analysis_result.get('llm_memory_check', False) and
            analysis_result.get('auto_discovery_trigger', False)
        )
    
    def trigger_auto_discovery(
        self,
        topic: str,
        knowledge_template: Optional[str] = None,
        vectorstore_manager=None,
        embedding_model: str = "nomic-embed-text",
        profile_id: str = "default",
        max_search_results: int = 10,
        max_urls_to_extract: int = 3,
        is_from_url: bool = False
    ) -> Dict[str, Any]:
        """Trigger auto-discovery for a topic.
        
        This method activates Layer 3 (Auto-Discovery Agent) to:
        1. Search the web for information about the topic
        2. Extract content from search results
        3. Verify content relevance using LLM
        4. Summarize verified knowledge using LLM
        5. Store in vector DB as long-term memory
        
        Args:
            topic: Topic to discover (e.g., "Upwork")
            knowledge_template: Optional template for knowledge structure
            vectorstore_manager: VectorStoreManager instance for storing
            embedding_model: Embedding model name
            profile_id: Profile ID to store knowledge under
            max_search_results: Maximum number of search results
            max_urls_to_extract: Maximum number of URLs to extract content from
            is_from_url: Whether the topic is from a URL (affects search query format)
            
        Returns:
            Dictionary with auto-discovery results:
                - success: True if successful
                - knowledge: Summarized knowledge
                - vectorstore: Created vectorstore
                - persist_dir: Persistence directory
                - kb_id: Knowledge base ID
                - error: Error message if failed
        """
        logger.info(f"Triggering auto-discovery for topic: {topic} (from_url: {is_from_url})")
        
        try:
            # Use Auto-Discovery Agent to discover and store knowledge
            discovery_result = self.auto_discovery_agent.discover_and_store(
                topic=topic,
                knowledge_template=knowledge_template,
                max_search_results=max_search_results,
                max_urls_to_extract=max_urls_to_extract,
                vectorstore_manager=vectorstore_manager,
                embedding_model=embedding_model,
                profile_id=profile_id,
                is_from_url=is_from_url
            )
            
            if discovery_result.get('success'):
                logger.info(f"Auto-discovery successful for topic: {topic}")
                
                # If vectorstore was created, add it to the RAG pipeline
                if discovery_result.get('vectorstore') and self.rag_pipeline:
                    try:
                        self.rag_pipeline.add_vectorstore(discovery_result['vectorstore'])
                        if self.vectorstores:
                            self.vectorstores.append(discovery_result['vectorstore'])
                        logger.info(f"Added discovered knowledge to RAG pipeline")
                    except Exception as e:
                        logger.warning(f"Failed to add vectorstore to RAG pipeline: {str(e)}")
                
                # Register knowledge base if KB ID was created
                if discovery_result.get('kb_id') and vectorstore_manager:
                    try:
                        from src.utils.kb_manager import KnowledgeBaseManager
                        kb_manager = KnowledgeBaseManager(profile_id=profile_id)
                        kb_manager.register_knowledge_base(
                            kb_id=discovery_result['kb_id'],
                            persist_dir=discovery_result['persist_dir'],
                            text_preview=discovery_result['knowledge'][:1000],
                            chunk_count=0,  # Will be calculated automatically
                            title=f"Auto-discovered: {topic}"
                        )
                        logger.info(f"Registered knowledge base: {discovery_result['kb_id']}")
                    except Exception as e:
                        logger.warning(f"Failed to register knowledge base: {str(e)}")
            
            return discovery_result
            
        except Exception as e:
            logger.error(f"Error triggering auto-discovery: {str(e)}", exc_info=True)
            return {
                'success': False,
                'knowledge': '',
                'vectorstore': None,
                'persist_dir': None,
                'kb_id': None,
                'error': f"Error in auto-discovery: {str(e)}"
            }
    
    def extract_topic_from_input(self, input_text: str, intent: str, metadata: Dict) -> Optional[str]:
        """Extract topic from input for auto-discovery.
        
        Args:
            input_text: Original input text
            intent: Detected intent
            metadata: Routing metadata
            
        Returns:
            Topic string or None
        """
        if intent == 'url':
            # For URLs, extract domain name
            domain = metadata.get('domain')
            if domain:
                # Remove TLD and get main domain (e.g., "upwork" from "upwork.com")
                domain_parts = domain.split('.')
                if len(domain_parts) >= 2:
                    main_domain = domain_parts[-2] if domain_parts[-2] != 'www' else domain_parts[-3] if len(domain_parts) >= 3 else domain_parts[-1]
                    return main_domain.capitalize()
                return domain.capitalize()
        
        elif intent == 'question':
            # For questions, try to extract the topic
            # Simple heuristic: look for "what is X" or "who is X"
            import re
            patterns = [
                r'what is (.+?)\?',
                r'who is (.+?)\?',
                r'what are (.+?)\?',
                r'who are (.+?)\?',
            ]
            for pattern in patterns:
                match = re.search(pattern, input_text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            # Fallback: use first few words
            words = input_text.split()[:5]
            return ' '.join(words)
        
        elif intent == 'command':
            # For commands, extract the object/topic
            command_type = metadata.get('command_type', '')
            # Remove command verb and get the rest
            words = input_text.split()
            if len(words) > 1:
                return ' '.join(words[1:])
        
        # Default: use input text (first 50 chars)
        return input_text[:50].strip() if input_text else None

