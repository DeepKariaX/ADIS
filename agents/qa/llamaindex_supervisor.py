"""
LlamaIndex-powered Supervisor Agent using Agent Framework
"""
import time
from typing import List, Dict, Any, Optional
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.core.llms import ChatMessage, MessageRole
from agents.base_agent import BaseAgent
from agents.qa.text_rag_agent import TextRAGAgent
from agents.qa.table_analysis_agent import TableAnalysisAgent
from agents.qa.image_analysis_agent import ImageAnalysisAgent
from database.models import AgentResponse, UserQuery, QueryIntent, QueryResponse
from database.mongodb_client import mongodb_client
from utils.logger import logger


class LlamaIndexSupervisorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="LlamaIndexSupervisor",
            description="Advanced supervisor agent using LlamaIndex ReAct framework for orchestrating specialized sub-agents"
        )
        
        # Initialize sub-agents
        self.text_agent = TextRAGAgent()
        self.table_agent = TableAnalysisAgent()
        self.image_agent = ImageAnalysisAgent()
        
        # Initialize source tracking
        self._last_text_sources = []
        self._last_table_sources = []
        self._last_image_sources = []
        self._last_detailed_sources = []
        
        # Create tools for each sub-agent
        self.tools = self._create_agent_tools()
        
        # Initialize ReAct agent
        self.react_agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=10
        )
    
    def _create_agent_tools(self) -> List[FunctionTool]:
        """Create tools for each sub-agent that the supervisor can use."""
        
        async def search_text_content(query: str) -> str:
            """Search and answer questions using textual content from documents."""
            try:
                self._agents_used.add("TextRAGAgent")
                response = await self.text_agent.process(query)
                if response.response_type == "text_rag_response":
                    content = response.content
                    # Store both aggregated and detailed sources for later retrieval
                    self._last_text_sources = response.sources or []
                    if isinstance(content, dict):
                        # Extract detailed sources if available
                        detailed_sources = content.get('detailed_sources', [])
                        self._last_detailed_sources = detailed_sources
                        
                        answer = content.get('answer', str(content))
                        sources_info = f"\n[Sources: {', '.join(self._last_text_sources)}]" if self._last_text_sources else ""
                        return f"{answer}{sources_info}"
                    return str(content)
                return f"Text search failed: {response.content}"
            except Exception as e:
                return f"Text search error: {str(e)}"
        
        async def analyze_table_data(query: str) -> str:
            """Analyze and query tabular data from documents."""
            try:
                self._agents_used.add("TableAnalysisAgent")
                response = await self.table_agent.process(query)
                if response.response_type in ["table_analysis_response", "table_query_response"]:
                    content = response.content
                    # Store table sources for later retrieval
                    self._last_table_sources = response.sources or []
                    if isinstance(content, dict):
                        answer = content.get('answer', str(content))
                        sources_info = f"\n[Sources: {', '.join(self._last_table_sources)}]" if self._last_table_sources else ""
                        return f"{answer}{sources_info}"
                    return str(content)
                return f"Table analysis failed: {response.content}"
            except Exception as e:
                return f"Table analysis error: {str(e)}"
        
        async def analyze_images(query: str) -> str:
            """Analyze and answer questions about images, figures, and charts."""
            try:
                self._agents_used.add("ImageAnalysisAgent")
                response = await self.image_agent.process(query)
                if response.response_type == "image_analysis_response":
                    content = response.content
                    if isinstance(content, dict):
                        return content.get('answer', str(content))
                    return str(content)
                return f"Image analysis failed: {response.content}"
            except Exception as e:
                return f"Image analysis error: {str(e)}"
        
        async def get_document_overview() -> str:
            """Get an overview of available documents and their content types."""
            try:
                documents = await mongodb_client.list_documents()
                if not documents:
                    return "No documents are currently available in the system."
                
                overview = []
                max_docs = self.settings.max_documents_overview
                for doc in documents[:max_docs]:  # Limit to configured number of documents
                    # Handle both dict and Document object types
                    if hasattr(doc, 'metadata'):
                        filename = getattr(doc.metadata, 'filename', 'Unknown') if doc.metadata else 'Unknown'
                        elements_count = len(getattr(doc, 'elements', []))
                    else:
                        filename = doc.get('metadata', {}).get('filename', 'Unknown')
                        elements_count = len(doc.get('elements', []))
                    
                    doc_info = f"Document: {filename}"
                    if elements_count > 0:
                        doc_info += f" ({elements_count} elements)"
                    overview.append(doc_info)
                
                return "Available documents:\\n" + "\\n".join(overview)
            except Exception as e:
                return f"Error getting document overview: {str(e)}"
        
        tools = [
            FunctionTool.from_defaults(
                fn=search_text_content,
                name="search_text_content",
                description="Search and answer questions using textual content from processed documents. Use this for questions about concepts, explanations, summaries, and general text-based information."
            ),
            FunctionTool.from_defaults(
                fn=analyze_table_data,
                name="analyze_table_data", 
                description="Analyze tabular data and answer questions about numbers, statistics, comparisons, and data trends. Use this for questions about specific data points, calculations, or table contents."
            ),
            FunctionTool.from_defaults(
                fn=analyze_images,
                name="analyze_images",
                description="Analyze images, figures, charts, and diagrams. Use this for questions about visual content, chart descriptions, or figure explanations."
            ),
            FunctionTool.from_defaults(
                fn=get_document_overview,
                name="get_document_overview",
                description="Get an overview of available documents in the system. Use this when the user asks what documents are available or wants to know what content exists."
            )
        ]
        
        return tools
    
    async def process(self, query: UserQuery, **kwargs) -> AgentResponse:
        start_time = time.time()
        
        try:
            self.log_activity(f"Processing query with LlamaIndex ReAct agent: {query.query_text}")
            
            # Clear previous sources and tracking
            self._last_text_sources = []
            self._last_table_sources = []
            self._last_image_sources = []
            self._last_detailed_sources = []
            self._agents_used = set()
            
            # Prepare the query with context
            enhanced_query = self._prepare_enhanced_query(query.query_text)
            
            # Use ReAct agent to process the query
            response = await self.react_agent.achat(enhanced_query)
            
            # Extract the final answer
            final_answer = str(response)
            
            # Determine intent from the query (simplified)
            intent = await self._analyze_query_intent(query.query_text)
            
            # Collect all sources from tool calls and format them properly
            all_sources = []
            all_sources.extend(self._last_text_sources)
            all_sources.extend(self._last_table_sources)
            all_sources.extend(self._last_image_sources)
            # Remove duplicates while preserving order
            raw_sources = list(dict.fromkeys(all_sources))
            
            # Convert raw document IDs to human-readable names
            formatted_sources = await self._format_sources_for_storage(raw_sources)
            
            processing_time = time.time() - start_time
            
            # Create response
            response_obj = self.create_response(
                content={
                    "answer": final_answer,
                    "intent": intent.value,
                    "reasoning": str(response),
                    "agent_used": "ReActAgent",
                    "detailed_sources": self._last_detailed_sources
                },
                response_type="supervisor_response",
                sources=formatted_sources,
                metadata={
                    "query_id": query.query_id,
                    "intent": intent.value,
                    "reasoning_steps": len(response.sources) if hasattr(response, 'sources') else 0
                },
                processing_time=processing_time
            )
            
            # Update query intent
            query.intent = intent
            # Create clean response using the new format
            clean_response = await mongodb_client.create_clean_response(
                query_id=query.query_id,
                query_text=query.query_text,
                answer=final_answer,
                intent=query.intent.value,
                primary_agent="LlamaIndexSupervisor",
                sources=formatted_sources,
                detailed_sources=self._last_detailed_sources,
                reasoning=f"Query processed using ReAct agent with {len(formatted_sources)} sources",
                sub_agents=list(self._agents_used),
                processing_time=processing_time,
                model_used=self.settings.llm_model
            )
            
            await mongodb_client.insert_query(query)
            await mongodb_client.insert_response(clean_response)
            
            self.log_activity(f"Completed query processing in {processing_time:.2f}s")
            return response_obj
            
        except Exception as e:
            self.log_activity(f"ReAct agent processing failed: {e}", "error")
            processing_time = time.time() - start_time
            
            return self.create_response(
                content=f"I apologize, but I encountered an error while processing your query: {str(e)}",
                response_type="error",
                processing_time=processing_time
            )
    
    def _prepare_enhanced_query(self, query_text: str) -> str:
        """Enhance the query with instructions for the ReAct agent."""
        
        # Analyze the intent to provide better guidance
        query_lower = query_text.lower()
        
        # Detect query characteristics to guide tool selection
        guidance = ""
        if any(keyword in query_lower for keyword in ['compare', 'performance', 'vs', 'versus', 'better', 'accuracy', 'score', 'result', 'number', 'statistic', 'data', 'metric']):
            guidance = "\nIMPORTANT: This query involves numerical comparisons or performance metrics. Consider using analyze_table_data tool to search for structured data and tables."
        elif any(keyword in query_lower for keyword in ['figure', 'image', 'chart', 'diagram', 'graph', 'visualization']):
            guidance = "\nIMPORTANT: This query involves visual content. Consider using analyze_images tool."
        
        enhanced_query = f"""You are an expert document intelligence assistant with access to specialized tools for analyzing different types of content.

User Query: {query_text}{guidance}

Instructions:
1. Analyze the query to determine what type of information is needed
2. Use the appropriate tools to gather information:
   - search_text_content: For questions about textual content, concepts, explanations
   - analyze_table_data: For questions about numbers, data, statistics, comparisons
   - analyze_images: For questions about figures, charts, diagrams
   - get_document_overview: When user asks what documents are available
3. If the query seems to require multiple types of information, use multiple tools
4. Synthesize the information from different tools into a comprehensive answer
5. Always cite your sources and be clear about what information comes from where
6. If you can't find relevant information, say so clearly

Please provide a comprehensive answer to the user's query."""

        return enhanced_query
    
    
    async def _analyze_query_intent(self, query_text: str) -> QueryIntent:
        """Simplified intent analysis."""
        query_lower = query_text.lower()
        
        # Table-related keywords
        table_keywords = ['table', 'data', 'number', 'statistic', 'revenue', 'price', 'value', 'count', 'performance', 'compare', 'comparison', 'versus', 'vs', 'mnist', 'tfd', 'model', 'accuracy', 'score']
        if any(keyword in query_lower for keyword in table_keywords):
            return QueryIntent.TABLE_QUERY
        
        # Image-related keywords  
        image_keywords = ['figure', 'image', 'chart', 'diagram', 'picture', 'graph', 'visualization']
        if any(keyword in query_lower for keyword in image_keywords):
            return QueryIntent.IMAGE_QUERY
        
        # Multi-modal keywords
        multi_keywords = ['compare', 'relationship', 'correlation', 'both', 'and']
        if any(keyword in query_lower for keyword in multi_keywords):
            return QueryIntent.MULTI_MODAL
        
        # Default to text search
        return QueryIntent.TEXT_SEARCH
    
    async def _format_sources_for_storage(self, raw_sources: List[str]) -> List[str]:
        """Convert raw document IDs to human-readable source names."""
        formatted_sources = []
        
        for source in raw_sources:
            try:
                # Skip empty sources
                if not source or source == "":
                    continue
                
                # If it looks like a document ID (timestamp-like), look it up
                if source.replace('.', '').replace('_', '').isdigit():
                    document = await mongodb_client.get_document(source)
                    if document and document.metadata and document.metadata.filename:
                        filename = document.metadata.filename
                        # Remove file extension for cleaner display
                        base_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
                        formatted_sources.append(base_filename)
                    else:
                        # Fallback to shortened ID
                        formatted_sources.append(f"Document_{source[:10]}")
                else:
                    # Already formatted or readable name
                    formatted_sources.append(source)
                    
            except Exception as e:
                self.log_activity(f"Error formatting source {source}: {e}", "warning")
                # Fallback to original source
                formatted_sources.append(source)
        
        return formatted_sources