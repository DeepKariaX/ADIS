import time
import pandas as pd
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
from database.models import AgentResponse, ContentType
from database.mongodb_client import mongodb_client
from database.vector_store import vector_store
from config.llm_factory import get_langchain_llm
from config.settings import get_settings
from utils.logger import logger

class TableAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="TableAnalysis",
            description="Analyzes and queries tabular data from documents"
        )
        self.settings = get_settings()
        self.llm = get_langchain_llm()
    
    async def process(self, query_text: str, **kwargs) -> AgentResponse:
        start_time = time.time()
        
        try:
            self.log_activity(f"Processing table query: {query_text}")
            
            # Step 1: Find relevant tables
            relevant_tables = await self._find_relevant_tables(query_text)
            
            if not relevant_tables:
                return self.create_response(
                    content="I couldn't find any relevant tables to answer your question.",
                    response_type="no_results",
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Analyze tables and generate answer
            answer = await self._analyze_tables(query_text, relevant_tables)
            
            # Step 3: Extract sources
            sources = [table['document_id'] for table in relevant_tables]
            
            processing_time = time.time() - start_time
            
            response = self.create_response(
                content={
                    "answer": answer,
                    "tables_analyzed": len(relevant_tables),
                    "sources": sources
                },
                response_type="table_analysis_response",
                sources=sources,
                metadata={
                    "tables_found": len(relevant_tables),
                    "analysis_method": "llm_table_analysis"
                },
                processing_time=processing_time
            )
            
            self.log_activity(f"Analyzed {len(relevant_tables)} tables")
            return response
            
        except Exception as e:
            self.log_activity(f"Table analysis failed: {e}", "error")
            processing_time = time.time() - start_time
            
            return self.create_response(
                content=f"Table analysis failed: {str(e)}",
                response_type="error",
                processing_time=processing_time
            )
    
    async def _find_relevant_tables(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            # Search for table content in vector embeddings
            table_chunks = vector_store.search_similar(
                query_text=query_text,
                top_k=top_k,
                content_types=[ContentType.TABLE]
            )
            
            # Process table chunks directly from embeddings
            relevant_tables = []
            
            for chunk in table_chunks:
                chunk_id = chunk.get('element_id')
                text_content = chunk.get('text_content', '')
                metadata = chunk.get('metadata', {})
                document_id = metadata.get('document_id', 'unknown')
                
                if text_content and '[TABLE]' in text_content:
                    table_info = {
                        'chunk_id': chunk_id,
                        'document_id': document_id,
                        'table_text': text_content,
                        'metadata': metadata,
                        'similarity_score': 1.0 - chunk.get('distance', 0.5)
                    }
                    relevant_tables.append(table_info)
            
            # Sort by similarity score
            relevant_tables.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            self.log_activity(f"Found {len(relevant_tables)} relevant tables")
            return relevant_tables
            
        except Exception as e:
            self.log_activity(f"Table search failed: {e}", "error")
            return []
    
    async def _analyze_tables(self, query_text: str, relevant_tables: List[Dict[str, Any]]) -> str:
        try:
            # Prepare table data for analysis
            tables_context = []
            
            for i, table_info in enumerate(relevant_tables[:3]):  # Limit to 3 tables to avoid token limits
                table_text = table_info.get('table_text', '')
                if not table_text or '[TABLE]' not in table_text:
                    continue
                
                document_id = table_info['document_id']
                chunk_id = table_info['chunk_id']
                
                # Format table data with document context
                formatted_table = f"Table {i+1} from document '{document_id}' (chunk: {chunk_id}):\n"
                formatted_table += table_text + "\n"
                
                tables_context.append(formatted_table)
            
            if not tables_context:
                return "I found relevant table embeddings but couldn't extract readable table content for analysis."
            
            # Create analysis prompt
            system_prompt = """You are an expert data analyst that answers questions about tabular data.

            Instructions:
            1. Analyze the provided table data to answer the user's question
            2. Look for specific values, perform calculations if needed (sums, averages, comparisons, etc.)
            3. Be precise with numbers and cite specific table rows/columns when relevant
            4. Pay attention to the table format - content may be in [TABLE] blocks with rows and columns
            5. If the data doesn't fully answer the question, explain what's missing
            6. Format numerical answers clearly with proper units and error margins if provided
            7. Reference specific tables when using data from multiple tables
            """
            
            context = "\n".join(tables_context)
            
            user_prompt = f"""Table data:
            {context}

            Question: {query_text}

            Please analyze the table data and provide a comprehensive answer:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            answer = response.generations[0][0].text.strip()
            
            return answer
            
        except Exception as e:
            self.log_activity(f"Table analysis failed: {e}", "error")
            return f"I encountered an error while analyzing the tables: {str(e)}"
    
