import time
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
from database.models import AgentResponse, ContentType
from database.mongodb_client import mongodb_client
from database.vector_store import vector_store
from config.llm_factory import get_langchain_llm
from config.settings import get_settings
from utils.logger import logger

class ImageAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ImageAnalysis",
            description="Analyzes and answers questions about images and figures in documents"
        )
        self.settings = get_settings()
        self.llm = get_langchain_llm()
    
    async def process(self, query_text: str, **kwargs) -> AgentResponse:
        start_time = time.time()
        
        try:
            self.log_activity(f"Processing image query: {query_text}")
            
            # Step 1: Find relevant images
            relevant_images = await self._find_relevant_images(query_text)
            
            if not relevant_images:
                return self.create_response(
                    content="I couldn't find any relevant images or figures to answer your question.",
                    response_type="no_results",
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Analyze images and generate answer
            answer = await self._analyze_images(query_text, relevant_images)
            
            # Step 3: Extract sources
            sources = [img['document_id'] for img in relevant_images]
            
            processing_time = time.time() - start_time
            
            response = self.create_response(
                content={
                    "answer": answer,
                    "images_analyzed": len(relevant_images),
                    "sources": sources
                },
                response_type="image_analysis_response",
                sources=sources,
                metadata={
                    "images_found": len(relevant_images),
                    "analysis_method": "caption_and_metadata_analysis"
                },
                processing_time=processing_time
            )
            
            self.log_activity(f"Analyzed {len(relevant_images)} images")
            return response
            
        except Exception as e:
            self.log_activity(f"Image analysis failed: {e}", "error")
            processing_time = time.time() - start_time
            
            return self.create_response(
                content=f"Image analysis failed: {str(e)}",
                response_type="error",
                processing_time=processing_time
            )
    
    async def _find_relevant_images(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            # Search for images based on captions and descriptions
            # Note: In a full implementation, we might use image embeddings
            # For now, we'll search based on text content (captions, descriptions)
            
            self.log_activity(f"Searching for images with query: {query_text}")
            
            image_chunks = vector_store.search_similar(
                query_text=query_text,
                top_k=top_k * 2,  # Search more broadly
                content_types=[ContentType.IMAGE]
            )
            
            self.log_activity(f"Vector search returned {len(image_chunks)} image chunks")
            
            # Get detailed image information from database
            relevant_images = []
            
            for chunk in image_chunks:
                element_id = chunk.get('element_id')
                metadata = chunk.get('metadata', {})
                document_id = metadata.get('document_id')
                
                if element_id and document_id:
                    # Get full image element from database
                    elements = await mongodb_client.get_elements_by_document(document_id)
                    
                    for element in elements:
                        if element.element_id == element_id and element.content_type == ContentType.IMAGE:
                            image_info = {
                                'element_id': element_id,
                                'document_id': document_id,
                                'image_content': element.image_content,
                                'metadata': element.metadata,
                                'bounding_box': element.bounding_box,
                                'similarity_score': 1.0 - chunk.get('distance', 0.5)
                            }
                            relevant_images.append(image_info)
                            break
                else:
                    # Log for debugging
                    self.log_activity(f"Missing data: element_id={element_id}, document_id={document_id}, metadata={metadata}", "warning")
            
            # Also search for images by looking for specific image references in the query
            image_keywords = ['figure', 'fig', 'image', 'chart', 'graph', 'diagram', 'photo', 'picture']
            query_lower = query_text.lower()
            
            if any(keyword in query_lower for keyword in image_keywords):
                # Try to find specific figure numbers or references
                additional_images = await self._find_images_by_reference(query_text)
                relevant_images.extend(additional_images)
            
            # Remove duplicates and sort by similarity
            seen_ids = set()
            unique_images = []
            for img in relevant_images:
                if img['element_id'] not in seen_ids:
                    unique_images.append(img)
                    seen_ids.add(img['element_id'])
            
            unique_images.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # If no images found through vector search, try to get ALL images from the database
            if not unique_images:
                self.log_activity("No images found through vector search, trying to get all images from database")
                unique_images = await self._get_all_images_from_database()
            
            self.log_activity(f"Found {len(unique_images)} relevant images")
            return unique_images[:top_k]
            
        except Exception as e:
            self.log_activity(f"Image search failed: {e}", "error")
            return []
    
    async def _find_images_by_reference(self, query_text: str) -> List[Dict[str, Any]]:
        try:
            # Extract potential figure references (e.g., "Figure 1", "Fig 2")
            import re
            
            figure_patterns = [
                r'figure\s*(\d+)',
                r'fig\.?\s*(\d+)',
                r'image\s*(\d+)',
                r'chart\s*(\d+)',
                r'graph\s*(\d+)'
            ]
            
            referenced_numbers = set()
            for pattern in figure_patterns:
                matches = re.findall(pattern, query_text.lower())
                referenced_numbers.update(matches)
            
            if not referenced_numbers:
                return []
            
            # Search for images with matching references in captions or metadata
            # This is a simplified approach - in practice, you'd have better figure numbering
            images_by_reference = []
            
            # Get all image elements and check their captions
            all_documents = await mongodb_client.list_documents()
            
            for doc in all_documents:
                elements = await mongodb_client.get_elements_by_document(doc.document_id)
                
                for element in elements:
                    if element.content_type == ContentType.IMAGE and element.image_content:
                        caption = element.image_content.caption or ""
                        description = element.image_content.description or ""
                        
                        # Check if any referenced number appears in caption or description
                        for num in referenced_numbers:
                            if (num in caption.lower()) or (num in description.lower()):
                                image_info = {
                                    'element_id': element.element_id,
                                    'document_id': element.document_id,
                                    'image_content': element.image_content,
                                    'metadata': element.metadata,
                                    'bounding_box': element.bounding_box,
                                    'similarity_score': 0.9  # High score for exact reference match
                                }
                                images_by_reference.append(image_info)
                                break
            
            return images_by_reference
            
        except Exception as e:
            self.log_activity(f"Reference-based image search failed: {e}", "error")
            return []
    
    async def _analyze_images(self, query_text: str, relevant_images: List[Dict[str, Any]]) -> str:
        try:
            # Prepare image information for analysis
            images_context = []
            
            for i, image_info in enumerate(relevant_images[:3]):  # Limit to 3 images
                image_content = image_info.get('image_content')
                if not image_content:
                    continue
                
                document_id = image_info['document_id']
                
                # Format image information
                image_text = f"Image {i+1} from document '{document_id}':\n"
                
                if image_content.image_id:
                    image_text += f"Image ID: {image_content.image_id}\n"
                
                if image_content.caption:
                    image_text += f"Caption: {image_content.caption}\n"
                
                if image_content.description:
                    image_text += f"Description: {image_content.description}\n"
                
                if image_content.alt_text:
                    image_text += f"Alt text: {image_content.alt_text}\n"
                
                # Add metadata information
                metadata = image_info.get('metadata', {})
                if metadata:
                    image_text += f"Additional info: "
                    relevant_metadata = []
                    for key, value in metadata.items():
                        if key in ['page_number', 'image_format', 'image_size']:
                            relevant_metadata.append(f"{key}: {value}")
                    if relevant_metadata:
                        image_text += ", ".join(relevant_metadata) + "\n"
                
                images_context.append(image_text)
            
            if not images_context:
                # Try to provide a basic summary even without detailed content
                basic_summary = f"I found {len(relevant_images)} image(s) in the document"
                if relevant_images:
                    doc_ids = list(set([img['document_id'] for img in relevant_images]))
                    basic_summary += f" from document(s): {', '.join(doc_ids)}"
                    
                    # Try to extract any basic info that's available
                    image_types = []
                    for img in relevant_images:
                        if img.get('image_content'):
                            if img['image_content'].caption:
                                image_types.append("captioned image")
                            elif img['image_content'].description:
                                image_types.append("described image")
                            else:
                                image_types.append("image")
                    
                    if image_types:
                        basic_summary += f". Types found: {', '.join(set(image_types))}"
                else:
                    basic_summary += " but couldn't extract detailed information for analysis."
                
                return basic_summary
            
            # Create analysis prompt
            system_prompt = """You are an expert at analyzing images and figures in documents.

            Instructions:
            1. Answer the user's question based on the image information provided
            2. Use captions, descriptions, and metadata to understand what the images show
            3. Be specific about which image you're referencing
            4. If the question asks about visual elements not described in the available information, explain this limitation
            5. Provide context about where the images are located (document, page)
            6. If multiple images are relevant, compare or synthesize information appropriately
            """
            
            context = "\n".join(images_context)
            
            user_prompt = f"""Image information:
            {context}

            Question: {query_text}

            Please analyze the image information and provide a comprehensive answer:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            answer = response.generations[0][0].text.strip()
            
            # Note: In a full implementation, you might also integrate with vision models
            # to analyze the actual image content, not just metadata and captions
            
            return answer
            
        except Exception as e:
            self.log_activity(f"Image analysis failed: {e}", "error")
            return f"I encountered an error while analyzing the images: {str(e)}"
    
    async def _get_all_images_from_database(self) -> List[Dict[str, Any]]:
        """Get all images from the database as a fallback when vector search fails."""
        try:
            all_images = []
            
            # Get all documents
            documents = await mongodb_client.list_documents()
            
            for doc in documents:
                # Get image elements from each document
                elements = await mongodb_client.get_elements_by_document(doc.document_id)
                
                for element in elements:
                    if element.content_type == ContentType.IMAGE and element.image_content:
                        image_info = {
                            'element_id': element.element_id,
                            'document_id': element.document_id,
                            'image_content': element.image_content,
                            'metadata': element.metadata,
                            'bounding_box': element.bounding_box,
                            'similarity_score': 0.5  # Default score since no vector similarity
                        }
                        all_images.append(image_info)
            
            self.log_activity(f"Found {len(all_images)} total images in database")
            return all_images
            
        except Exception as e:
            self.log_activity(f"Failed to get all images from database: {e}", "error")
            return []
    
