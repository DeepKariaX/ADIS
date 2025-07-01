import time
from typing import List, Dict, Any, Optional
from agents.base_agent import BaseAgent
from database.models import AgentResponse, ContentType
from database.vector_store import vector_store
from database.mongodb_client import mongodb_client
from config.llm_factory import get_llm
from config.settings import get_settings
from utils.logger import logger

class TextRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="TextRAG",
            description="Enhanced RAG agent for answering questions using intelligent text chunks"
        )
        self.settings = get_settings()
        self.llm = get_llm()
    
    async def process(self, query_text: str, **kwargs) -> AgentResponse:
        start_time = time.time()
        
        try:
            self.log_activity(f"Processing text query: {query_text}")
            
            # Step 1: Retrieve relevant text chunks
            relevant_chunks = await self._retrieve_relevant_text(query_text)
            
            if not relevant_chunks:
                return self.create_response(
                    content="I couldn't find relevant textual information to answer your question.",
                    response_type="no_results",
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Generate answer using RAG
            answer = await self._generate_answer(query_text, relevant_chunks)
            
            # Step 3: Extract sources
            sources = await self._extract_sources(relevant_chunks)
            
            processing_time = time.time() - start_time
            
            # Extract detailed chunk sources
            detailed_sources = await self._extract_detailed_sources(relevant_chunks)
            
            response = self.create_response(
                content={
                    "answer": answer,
                    "retrieved_chunks": len(relevant_chunks),
                    "sources": sources,
                    "detailed_sources": detailed_sources
                },
                response_type="text_rag_response",
                sources=sources,
                metadata={
                    "chunks_retrieved": len(relevant_chunks),
                    "query_method": "vector_similarity"
                },
                processing_time=processing_time
            )
            
            self.log_activity(f"Generated answer using {len(relevant_chunks)} text chunks")
            return response
            
        except Exception as e:
            self.log_activity(f"Text RAG processing failed: {e}", "error")
            processing_time = time.time() - start_time
            
            return self.create_response(
                content=f"Text analysis failed: {str(e)}",
                response_type="error",
                processing_time=processing_time
            )
    
    async def _retrieve_relevant_text(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            if top_k is None:
                top_k = self.settings.rag_top_k
                
            # Search for relevant text chunks (intelligent chunks can contain multiple content types)
            text_types = [ContentType.TEXT, ContentType.HEADER, ContentType.LIST]
            
            relevant_chunks = vector_store.search_similar(
                query_text=query_text,
                top_k=top_k * 2,  # Get more candidates for filtering
                content_types=text_types
            )
            
            # Enhanced filtering and ranking
            filtered_chunks = self._filter_and_rank_chunks(relevant_chunks, query_text)
            
            # Limit to top_k results
            final_chunks = filtered_chunks[:top_k]
            
            self.log_activity(f"Retrieved {len(final_chunks)} relevant text chunks from {len(relevant_chunks)} candidates")
            return final_chunks
            
        except Exception as e:
            self.log_activity(f"Text retrieval failed: {e}", "error")
            return []
    
    def _filter_and_rank_chunks(self, chunks: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        """Enhanced filtering and ranking of chunks based on multiple criteria."""
        if not chunks:
            return []
        
        # Calculate enhanced relevance scores
        scored_chunks = []
        
        for chunk in chunks:
            score = 0.0
            metadata = chunk.get('metadata', {})
            
            # Base similarity score (lower distance = higher relevance)
            distance = chunk.get('distance', 1.0)
            similarity_score = max(0, 1.0 - distance)
            score += similarity_score * 0.5
            
            # Boost for intelligent chunks
            if metadata.get('chunk_type') == 'intelligent_chunk':
                score += 0.2
            
            # Boost for chunks with multiple content types (more context)
            content_types = metadata.get('content_types', [])
            if len(content_types) > 1:
                score += 0.1
            
            # Boost for longer chunks (more information)
            chunk_length = metadata.get('chunk_length', 0)
            if chunk_length > 1500:
                score += 0.1
            elif chunk_length > 800:
                score += 0.05
            
            # Boost based on chunk metadata quality
            if metadata.get('chunk_type') == 'intelligent_chunk':
                score += 0.05
            
            # Boost for chunks containing headers (likely important sections)
            if 'header' in content_types:
                score += 0.15
            
            # Text length relevance (penalize very short chunks)
            text_content = chunk.get('text_content', '')
            min_text_length = self.settings.thresholds.get("min_text_length", 200.0)
            if len(text_content) < min_text_length:
                score -= 0.2
            
            scored_chunks.append((chunk, score))
        
        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out chunks below a minimum threshold
        min_threshold = self.settings.thresholds.get("rag_similarity_threshold", 0.3)
        filtered = [chunk for chunk, score in scored_chunks if score >= min_threshold]
        
        # If nothing passes the threshold, take top 3 anyway
        if not filtered and scored_chunks:
            filtered = [chunk for chunk, score in scored_chunks[:3]]
        
        return filtered
    
    async def _generate_answer(self, query_text: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        try:
            # Prepare context from relevant chunks
            context_parts = []
            
            context_limit = self.settings.rag_context_limit
            for i, chunk in enumerate(relevant_chunks[:context_limit]):  # Limit to avoid token limits
                metadata = chunk.get('metadata', {})
                text_content = chunk.get('text_content', '')
                
                # Add chunk metadata for context
                chunk_info = []
                if 'document_id' in metadata:
                    chunk_info.append(f"Source: {metadata['document_id']}")
                if 'page_numbers' in metadata:
                    pages = metadata['page_numbers']
                    if pages:
                        chunk_info.append(f"Page: {pages}")
                if 'content_types' in metadata:
                    types = metadata['content_types']
                    if types:
                        chunk_info.append(f"Content: {', '.join(types)}")
                
                chunk_header = f"[Chunk {i+1}" + (f" - {', '.join(chunk_info)}" if chunk_info else "") + "]"
                context_parts.append(f"{chunk_header}\n{text_content}")
            
            context = "\n\n".join(context_parts)
            
            # Create enhanced RAG prompt
            prompt = f"""You are an AI assistant that answers questions based on provided document content. Use the following context to answer the user's question accurately and comprehensively.

CONTEXT:
{context}

QUESTION: {query_text}

Instructions:
- Answer based solely on the provided context
- If the context doesn't contain enough information to fully answer the question, say so
- Cite specific sources when possible (mention document names or page numbers)
- Provide detailed answers when the context supports it
- If multiple sources provide different perspectives, acknowledge this
- Be precise and factual

ANSWER:"""

            # Generate answer using the LLM
            if hasattr(self.llm, 'acomplete'):
                # For async LLMs
                response = await self.llm.acomplete(prompt)
                answer = response.text.strip()
            else:
                # For sync LLMs
                response = self.llm.complete(prompt)
                answer = response.text.strip()
            
            return answer
            
        except Exception as e:
            self.log_activity(f"Answer generation failed: {e}", "error")
            # Fallback to simple concatenation if LLM fails
            try:
                context_texts = [chunk.get('text_content', '') for chunk in relevant_chunks[:3]]
                fallback_answer = f"Based on the available information: {' '.join(context_texts[:500])}..."
                return fallback_answer
            except:
                return f"I encountered an error while generating the answer: {str(e)}"
    
    async def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        sources = set()
        
        for chunk in chunks:
            try:
                metadata = chunk.get('metadata', {})
                document_id = metadata.get('document_id')
                
                if document_id:
                    # Look up the actual document to get filename
                    document = await mongodb_client.get_document(document_id)
                    
                    if document and document.metadata and document.metadata.filename:
                        filename = document.metadata.filename
                        # Remove file extension for cleaner display
                        base_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
                    else:
                        # Fallback to document_id if lookup fails
                        base_filename = f"Document_{document_id[:10]}"
                    
                    # Extract page information if available
                    page_info = metadata.get('page_numbers', metadata.get('page_number'))
                    if page_info:
                        if isinstance(page_info, list) and page_info:
                            page_num = page_info[0] + 1 if page_info[0] >= 0 else 1
                        elif isinstance(page_info, int):
                            page_num = page_info + 1 if page_info >= 0 else 1
                        else:
                            page_num = 1
                    else:
                        page_num = 1
                    
                    # Create a descriptive source reference
                    source_ref = f"{base_filename} (Page {page_num})"
                    sources.add(source_ref)
                    
            except Exception as e:
                self.log_activity(f"Error extracting source info: {e}", "warning")
                # Fallback to basic info
                document_id = chunk.get('metadata', {}).get('document_id', 'Unknown')
                sources.add(f"Document_{document_id[:10]}")
        
        return list(sources)
    
    async def _extract_detailed_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract detailed source information for each chunk."""
        detailed_sources = []
        
        for i, chunk in enumerate(chunks):
            try:
                metadata = chunk.get('metadata', {})
                document_id = metadata.get('document_id')
                
                if document_id:
                    # Extract page information if available
                    page_info = metadata.get('page_numbers', metadata.get('page_number'))
                    if page_info:
                        if isinstance(page_info, list) and page_info:
                            page_num = page_info[0] + 1 if page_info[0] >= 0 else 1
                        elif isinstance(page_info, int):
                            page_num = page_info + 1 if page_info >= 0 else 1
                        else:
                            page_num = 1
                    else:
                        page_num = None
                    
                    # Create detailed source info in the format expected by MongoDB
                    chunk_detail = {
                        "element_id": chunk.get('element_id', 'unknown'),
                        "document_id": document_id,  # Use actual document_id from metadata
                        "content_type": metadata.get('content_type'),
                        "page_number": page_num,
                        "chunk_type": metadata.get('chunk_type', 'intelligent_chunk')
                    }
                    detailed_sources.append(chunk_detail)
                    
            except Exception as e:
                self.log_activity(f"Error extracting detailed source info: {e}", "warning")
                # Fallback to basic info
                chunk_detail = {
                    "element_id": chunk.get('element_id', 'unknown'),
                    "document_id": metadata.get('document_id', 'unknown'),
                    "content_type": metadata.get('content_type'),
                    "page_number": None,
                    "chunk_type": metadata.get('chunk_type', 'unknown')
                }
                detailed_sources.append(chunk_detail)
        
        return detailed_sources
    
