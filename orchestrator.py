import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
from agents.parsing.layout_analyzer import LayoutAnalyzerAgent
from agents.parsing.text_extractor import TextExtractorAgent
from agents.parsing.table_extractor import TableExtractorAgent
from agents.parsing.image_processor import ImageProcessorAgent
from database.models import Document, DocumentMetadata, DocumentElement, VectorEmbedding, ContentType
from database.mongodb_client import mongodb_client
from database.vector_store import vector_store
from config.settings import get_settings, ensure_directories
from utils.logger import logger
from utils.chunking import intelligent_chunker
from utils.pdf_utils import safe_pdf_document, force_pdf_cleanup

class DocumentProcessingOrchestrator:
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize agents
        self.layout_agent = LayoutAnalyzerAgent()
        self.text_agent = TextExtractorAgent()
        self.table_agent = TableExtractorAgent()
        self.image_agent = ImageProcessorAgent()
        
        ensure_directories()
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            logger.info(f"Starting document processing: {file_path}")
            
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Step 1: Create document metadata
            document = await self._create_document_record(file_path)
            
            # Step 2: Layout Analysis
            logger.info("Step 1: Layout Analysis")
            layout_response = await self.layout_agent.process(file_path)
            
            if layout_response.response_type == "error":
                raise Exception(f"Layout analysis failed: {layout_response.content}")
            
            layout_elements = layout_response.content
            
            # Step 3: Content Extraction (parallel processing)
            logger.info("Step 2: Content Extraction")
            extraction_tasks = [
                self.text_agent.process({
                    'file_path': file_path,
                    'layout_elements': layout_elements,
                    'document_id': document.document_id
                }),
                self.table_agent.process({
                    'file_path': file_path,
                    'layout_elements': layout_elements,
                    'document_id': document.document_id
                }),
                self.image_agent.process({
                    'file_path': file_path,
                    'layout_elements': layout_elements,
                    'document_id': document.document_id
                })
            ]
            
            text_response, table_response, image_response = await asyncio.gather(*extraction_tasks)
            
            # Step 4: Combine all extracted elements
            logger.info("Step 3: Combining extracted elements")
            all_elements = []
            
            # Process text extraction results
            if text_response.response_type != "error" and isinstance(text_response.content, list):
                all_elements.extend(text_response.content)
            
            # Process table extraction results
            if table_response.response_type != "error" and isinstance(table_response.content, list):
                all_elements.extend(table_response.content)
            
            # Process image extraction results
            if image_response.response_type != "error" and isinstance(image_response.content, list):
                all_elements.extend(image_response.content)
            
            # Step 5: Store elements in database
            logger.info("Step 4: Storing elements in database")
            if all_elements:
                await mongodb_client.insert_elements(all_elements)
                # Update document with element IDs
                element_ids = [element.element_id for element in all_elements]
                await mongodb_client.update_document_elements(document.document_id, element_ids)
            
            # Step 6: Generate and store embeddings
            logger.info("Step 5: Generating embeddings")
            embeddings = await self._generate_embeddings(all_elements)
            
            if embeddings:
                # Store in MongoDB
                await mongodb_client.insert_embeddings(embeddings)
                
                # Store in vector database
                vector_store.add_embeddings(embeddings)
            
            # Step 7: Update document status
            await mongodb_client.update_document_status(document.document_id, "completed")
            
            processing_time = time.time() - start_time
            
            result = {
                "status": "success",
                "document_id": document.document_id,
                "processing_time": processing_time,
                "elements_extracted": len(all_elements),
                "embeddings_generated": len(embeddings),
                "breakdown": {
                    "text_elements": len([e for e in all_elements if e.content_type in [ContentType.TEXT, ContentType.HEADER, ContentType.LIST]]),
                    "table_elements": len([e for e in all_elements if e.content_type == ContentType.TABLE]),
                    "image_elements": len([e for e in all_elements if e.content_type == ContentType.IMAGE])
                }
            }
            
            logger.info(f"Document processing completed in {processing_time:.2f}s: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            
            # Update document status with error
            if 'document' in locals():
                await mongodb_client.update_document_status(
                    document.document_id, 
                    "failed", 
                    [str(e)]
                )
            
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        finally:
            # Force cleanup of PDF resources
            force_pdf_cleanup()
    
    async def _create_document_record(self, file_path: str) -> Document:
        try:
            file_path_obj = Path(file_path)
            
            # Extract basic file metadata
            basic_metadata = {
                "filename": file_path_obj.name,
                "file_path": str(file_path_obj.absolute()),
                "file_size": file_path_obj.stat().st_size,
                "file_type": file_path_obj.suffix.lower()
            }
            
            # Extract document-specific metadata (e.g., PDF metadata)
            document_metadata = await self._extract_document_metadata(file_path)
            
            # Combine basic and document-specific metadata
            all_metadata = {**basic_metadata, **document_metadata}
            metadata = DocumentMetadata(**all_metadata)
            
            document = Document(
                metadata=metadata,
                processing_status="processing"
            )
            
            # Insert into database
            await mongodb_client.insert_document(document)
            
            logger.info(f"Created document record: {document.document_id}")
            logger.info(f"Document metadata: {metadata.model_dump()}")
            return document
            
        except Exception as e:
            logger.error(f"Failed to create document record: {e}")
            raise
    
    async def _extract_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract document-specific metadata based on file type."""
        try:
            file_path_obj = Path(file_path)
            file_extension = file_path_obj.suffix.lower()
            
            if file_extension == '.pdf':
                return await self._extract_pdf_metadata(file_path)
            elif file_extension in ['.docx', '.doc']:
                return await self._extract_docx_metadata(file_path)
            else:
                # Return empty metadata for unsupported file types
                return {}
                
        except Exception as e:
            logger.error(f"Failed to extract document metadata from {file_path}: {e}")
            # Return empty metadata instead of raising to not break document processing
            return {}
    
    async def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF using PyMuPDF."""
        try:
            with safe_pdf_document(file_path) as pdf_document:
                # Extract basic PDF info
                page_count = len(pdf_document)
                
                # Extract PDF metadata
                pdf_metadata = pdf_document.metadata
                
                # Convert PDF dates to datetime objects
                creation_date = None
                modification_date = None
                
                if pdf_metadata.get('creationDate'):
                    try:
                        # PDF dates are typically in format "D:YYYYMMDDHHmmSSOHH'mm'"
                        creation_date_str = pdf_metadata['creationDate']
                        if creation_date_str.startswith('D:'):
                            creation_date_str = creation_date_str[2:]  # Remove 'D:' prefix
                        # Extract just the date part (first 14 characters: YYYYMMDDHHMMSS)
                        date_part = creation_date_str[:14]
                        if len(date_part) >= 8:  # At least YYYYMMDD
                            # Pad with zeros if needed
                            date_part = date_part.ljust(14, '0')
                            creation_date = datetime.strptime(date_part, '%Y%m%d%H%M%S')
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse PDF creation date: {pdf_metadata.get('creationDate')}, error: {e}")
                
                if pdf_metadata.get('modDate'):
                    try:
                        modification_date_str = pdf_metadata['modDate']
                        if modification_date_str.startswith('D:'):
                            modification_date_str = modification_date_str[2:]
                        date_part = modification_date_str[:14]
                        if len(date_part) >= 8:
                            date_part = date_part.ljust(14, '0')
                            modification_date = datetime.strptime(date_part, '%Y%m%d%H%M%S')
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse PDF modification date: {pdf_metadata.get('modDate')}, error: {e}")
                
                # Build metadata dictionary
                metadata = {
                    'page_count': page_count,
                    'author': pdf_metadata.get('author', '').strip() or None,
                    'title': pdf_metadata.get('title', '').strip() or None,
                    'creation_date': creation_date,
                    'modification_date': modification_date,
                    'language': pdf_metadata.get('language', '').strip() or None
                }
                
                # Log extracted metadata for debugging
                logger.info(f"Extracted PDF metadata: page_count={page_count}, "
                           f"author={metadata['author']}, title={metadata['title']}, "
                           f"creation_date={metadata['creation_date']}, "
                           f"modification_date={metadata['modification_date']}, "
                           f"language={metadata['language']}")
                
                return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract PDF metadata: {e}")
            return {}
    
    async def _extract_docx_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from DOCX files."""
        try:
            # For now, return basic metadata. Could be enhanced later with python-docx
            return {}
        except Exception as e:
            logger.error(f"Failed to extract DOCX metadata: {e}")
            return {}
    
    async def _generate_embeddings(self, elements: List[DocumentElement]) -> List[VectorEmbedding]:
        try:
            # Use intelligent chunking to create better contextual chunks
            logger.info(f"Creating intelligent chunks from {len(elements)} elements")
            text_chunks = intelligent_chunker.create_chunks(elements)
            
            if not text_chunks:
                logger.warning("Intelligent chunking failed, falling back to simple chunking")
                text_chunks = await self._create_simple_chunks(elements)
            
            # Prepare texts for embedding
            texts_to_embed = [chunk.content for chunk in text_chunks]
            
            # Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(text_chunks)} intelligent chunks")
            embedding_vectors = vector_store.generate_embeddings(texts_to_embed)
            
            # Create VectorEmbedding objects from chunks
            embeddings = []
            for chunk, embedding in zip(text_chunks, embedding_vectors):
                # Determine primary content type
                primary_content_type = self._get_primary_content_type(chunk.content_types)
                
                vector_embedding = VectorEmbedding(
                    element_id=chunk.chunk_id,  # Use chunk ID as element ID
                    document_id=chunk.document_id,
                    content_type=primary_content_type,
                    embedding=embedding,
                    text_content=chunk.content,
                    metadata={
                        "chunk_type": "intelligent_chunk",
                        "source_elements": chunk.source_elements,
                        "content_types": [ct.value for ct in chunk.content_types],
                        "page_numbers": chunk.page_numbers,
                        "chunk_length": len(chunk.content),
                        "element_count": len(chunk.source_elements),
                        **chunk.metadata
                    }
                )
                embeddings.append(vector_embedding)
            
            logger.info(f"Generated {len(embeddings)} embeddings from intelligent chunks")
            logger.info(f"Average chunk size: {sum(len(chunk.content) for chunk in text_chunks) / len(text_chunks):.0f} characters")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Last resort fallback
            try:
                logger.info("Attempting simple fallback chunking")
                return await self._create_simple_embeddings(elements)
            except Exception as fallback_error:
                logger.error(f"Fallback embedding generation also failed: {fallback_error}")
                return []
    
    def _get_primary_content_type(self, content_types: List[ContentType]) -> ContentType:
        """Determine the primary content type for a chunk."""
        # Priority order: HEADER > TABLE > IMAGE > LIST > TEXT
        priority_order = [ContentType.HEADER, ContentType.TABLE, ContentType.IMAGE, ContentType.LIST, ContentType.TEXT]
        
        for content_type in priority_order:
            if content_type in content_types:
                return content_type
        
        # Default to TEXT if no match
        return ContentType.TEXT
    
    async def _create_simple_chunks(self, elements: List[DocumentElement]):
        """Simple fallback chunking method."""
        from utils.chunking import TextChunk
        
        chunks = []
        current_content = []
        current_elements = []
        current_size = 0
        chunk_size = 1500  # Simpler, fixed chunk size
        
        for element in elements:
            try:
                # Extract text content
                text_content = ""
                if hasattr(element, 'text_content') and element.text_content and hasattr(element.text_content, 'content'):
                    text_content = str(element.text_content.content or "")
                elif hasattr(element, 'table_content') and element.table_content:
                    if hasattr(element.table_content, 'headers') and element.table_content.headers:
                        text_content += "Headers: " + " | ".join(str(h) for h in element.table_content.headers) + "\n"
                    if hasattr(element.table_content, 'rows') and element.table_content.rows:
                        for i, row in enumerate(element.table_content.rows[:5]):  # Limit rows
                            text_content += " | ".join(str(cell) for cell in row) + "\n"
                elif hasattr(element, 'image_content') and element.image_content:
                    if hasattr(element.image_content, 'caption') and element.image_content.caption:
                        text_content = f"Image: {element.image_content.caption}"
                
                if text_content.strip():
                    text_size = len(text_content)
                    
                    if current_size + text_size > chunk_size and current_content:
                        # Create chunk
                        chunk = TextChunk(
                            content="\n".join(current_content),
                            chunk_id=f"simple_chunk_{len(chunks)}",
                            document_id=getattr(element, 'document_id', 'unknown'),
                            source_elements=current_elements,
                            content_types=[ContentType.TEXT],
                            page_numbers=[0],
                            metadata={"chunk_type": "simple", "element_count": len(current_elements)}
                        )
                        chunks.append(chunk)
                        
                        # Reset
                        current_content = [text_content]
                        current_elements = [getattr(element, 'element_id', 'unknown')]
                        current_size = text_size
                    else:
                        current_content.append(text_content)
                        current_elements.append(getattr(element, 'element_id', 'unknown'))
                        current_size += text_size
                        
            except Exception as e:
                logger.warning(f"Error processing element in simple chunking: {e}")
                continue
        
        # Add remaining content
        if current_content:
            chunk = TextChunk(
                content="\n".join(current_content),
                chunk_id=f"simple_chunk_{len(chunks)}",
                document_id=getattr(elements[0], 'document_id', 'unknown') if elements else 'unknown',
                source_elements=current_elements,
                content_types=[ContentType.TEXT],
                page_numbers=[0],
                metadata={"chunk_type": "simple", "element_count": len(current_elements)}
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} simple chunks")
        return chunks
    
    async def _create_simple_embeddings(self, elements: List[DocumentElement]) -> List[VectorEmbedding]:
        """Simple fallback embedding generation."""
        embeddings = []
        
        try:
            for element in elements:
                text_content = ""
                
                # Simple text extraction
                if hasattr(element, 'text_content') and element.text_content and hasattr(element.text_content, 'content'):
                    text_content = str(element.text_content.content or "")
                
                if text_content.strip() and len(text_content) > 50:  # Only substantial content
                    embedding_vector = vector_store.generate_embeddings([text_content])[0]
                    
                    vector_embedding = VectorEmbedding(
                        element_id=getattr(element, 'element_id', 'unknown'),
                        document_id=getattr(element, 'document_id', 'unknown'),
                        content_type=getattr(element, 'content_type', ContentType.TEXT),
                        embedding=embedding_vector,
                        text_content=text_content,
                        metadata={
                            "chunk_type": "simple_fallback",
                            "content_length": len(text_content)
                        }
                    )
                    embeddings.append(vector_embedding)
            
            logger.info(f"Generated {len(embeddings)} simple embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Simple embedding generation failed: {e}")
            return []
    
    async def process_multiple_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        start_time = time.time()
        results = []
        
        logger.info(f"Processing {len(file_paths)} documents")
        
        for file_path in file_paths:
            try:
                result = await self.process_document(file_path)
                results.append({
                    "file_path": file_path,
                    **result
                })
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({
                    "file_path": file_path,
                    "status": "error",
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        
        summary = {
            "total_documents": len(file_paths),
            "successful": len([r for r in results if r.get("status") == "success"]),
            "failed": len([r for r in results if r.get("status") == "error"]),
            "total_processing_time": total_time,
            "results": results
        }
        
        logger.info(f"Batch processing completed: {summary['successful']}/{summary['total_documents']} successful")
        return summary
    
    async def get_processing_status(self, document_id: str) -> Dict[str, Any]:
        try:
            document = await mongodb_client.get_document(document_id)
            
            if not document:
                return {"status": "not_found"}
            
            # Get elements count
            elements = await mongodb_client.get_elements_by_document(document_id)
            
            # Get embeddings count
            embeddings = await mongodb_client.get_embeddings_by_document(document_id)
            
            return {
                "status": document.processing_status,
                "document_id": document_id,
                "filename": document.metadata.filename,
                "elements_count": len(elements),
                "embeddings_count": len(embeddings),
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "errors": document.processing_errors
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            return {"status": "error", "error": str(e)}

# Global orchestrator instance
document_orchestrator = DocumentProcessingOrchestrator()