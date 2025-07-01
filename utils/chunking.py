import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from database.models import DocumentElement, ContentType, BoundingBox
from config.settings import get_settings
from utils.logger import logger
from utils.text_processing import smart_text_split, calculate_semantic_similarity, extract_keywords

@dataclass
class TextChunk:
    """Represents a text chunk with metadata for embeddings."""
    content: str
    chunk_id: str
    document_id: str
    source_elements: List[str]  # List of element IDs that contributed to this chunk
    content_types: List[ContentType]
    page_numbers: List[int]
    metadata: Dict[str, Any]

class IntelligentChunker:
    """Advanced chunking system that creates larger, more contextual chunks."""
    
    def __init__(self):
        self.settings = get_settings()
        self.chunk_size = self.settings.chunk_size
        self.min_chunk_size = self.settings.min_chunk_size
        self.max_chunk_size = self.settings.max_chunk_size
        self.overlap = self.settings.chunk_overlap
    
    def create_chunks(self, elements: List[DocumentElement]) -> List[TextChunk]:
        """Create intelligent chunks from document elements."""
        try:
            logger.info(f"Creating intelligent chunks from {len(elements)} elements")
            
            # Group elements by document and page for better context
            grouped_elements = self._group_elements_by_context(elements)
            
            chunks = []
            for group_key, group_elements in grouped_elements.items():
                try:
                    if isinstance(group_key, tuple) and len(group_key) >= 2:
                        document_id, page_num = group_key[0], group_key[1]
                    else:
                        logger.warning(f"Invalid group_key format: {group_key}, using defaults")
                        document_id = str(group_key) if group_key else "unknown"
                        page_num = 0
                    
                    page_chunks = self._create_page_chunks(group_elements, document_id, page_num)
                    chunks.extend(page_chunks)
                except Exception as e:
                    logger.error(f"Error processing group {group_key}: {e}")
                    continue
            
            # Post-process chunks to ensure quality
            optimized_chunks = self._optimize_chunks(chunks)
            
            logger.info(f"Created {len(optimized_chunks)} optimized chunks")
            return optimized_chunks
            
        except Exception as e:
            logger.error(f"Failed to create chunks: {e}")
            raise
    
    def _group_elements_by_context(self, elements: List[DocumentElement]) -> Dict[Tuple[str, int], List[DocumentElement]]:
        """Group elements by document and page for contextual chunking."""
        grouped = {}
        
        for element in elements:
            try:
                # Determine page number
                page_num = 0
                if element.bounding_box and hasattr(element.bounding_box, 'page_number') and element.bounding_box.page_number is not None:
                    page_num = element.bounding_box.page_number
                
                # Ensure we have a valid document_id
                document_id = getattr(element, 'document_id', 'unknown')
                if not document_id:
                    document_id = 'unknown'
                
                key = (document_id, page_num)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(element)
                
            except Exception as e:
                logger.warning(f"Error grouping element {getattr(element, 'element_id', 'unknown')}: {e}")
                # Add to default group
                default_key = ('unknown', 0)
                if default_key not in grouped:
                    grouped[default_key] = []
                grouped[default_key].append(element)
        
        # Sort elements within each group by reading order
        for key in grouped:
            try:
                grouped[key] = self._sort_elements_by_reading_order(grouped[key])
            except Exception as e:
                logger.warning(f"Error sorting elements for group {key}: {e}")
        
        logger.info(f"Grouped {len(elements)} elements into {len(grouped)} groups")
        return grouped
    
    def _sort_elements_by_reading_order(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """Sort elements by their reading order (top to bottom, left to right)."""
        def get_sort_key(element):
            try:
                if hasattr(element, 'bounding_box') and element.bounding_box:
                    # Sort by y1 (top), then x1 (left)
                    y1 = getattr(element.bounding_box, 'y1', float('inf'))
                    x1 = getattr(element.bounding_box, 'x1', float('inf'))
                    return (y1, x1)
                elif hasattr(element, 'text_content') and element.text_content and hasattr(element.text_content, 'reading_order') and element.text_content.reading_order is not None:
                    return (element.text_content.reading_order, 0)
                else:
                    return (float('inf'), float('inf'))
            except Exception:
                return (float('inf'), float('inf'))
        
        try:
            return sorted(elements, key=get_sort_key)
        except Exception as e:
            logger.warning(f"Error sorting elements: {e}, returning original order")
            return elements
    
    def _create_page_chunks(self, elements: List[DocumentElement], document_id: str, page_num: int) -> List[TextChunk]:
        """Create chunks from elements on a single page."""
        chunks = []
        current_chunk_content = []
        current_chunk_elements = []
        current_chunk_types = []
        current_chunk_size = 0
        
        for element in elements:
            element_text = self._extract_element_text(element)
            if not element_text.strip():
                continue
            
            element_size = len(element_text)
            
            # Check if adding this element would exceed max chunk size
            if (current_chunk_size + element_size > self.max_chunk_size and 
                current_chunk_content and 
                current_chunk_size >= self.min_chunk_size):
                
                # Create chunk from current content
                chunk = self._create_chunk_from_content(
                    current_chunk_content, current_chunk_elements, 
                    current_chunk_types, document_id, page_num
                )
                if chunk:
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_content, overlap_elements, overlap_types = self._create_overlap(
                    current_chunk_content, current_chunk_elements, current_chunk_types
                )
                current_chunk_content = overlap_content + [element_text]
                current_chunk_elements = overlap_elements + [element.element_id]
                current_chunk_types = list(set(overlap_types + [element.content_type]))
                current_chunk_size = sum(len(text) for text in current_chunk_content)
            else:
                # Add to current chunk
                current_chunk_content.append(element_text)
                current_chunk_elements.append(element.element_id)
                current_chunk_types.append(element.content_type)
                current_chunk_size += element_size
        
        # Create final chunk if there's remaining content
        if current_chunk_content and current_chunk_size >= self.min_chunk_size:
            chunk = self._create_chunk_from_content(
                current_chunk_content, current_chunk_elements, 
                current_chunk_types, document_id, page_num
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _extract_element_text(self, element: DocumentElement) -> str:
        """Extract text content from an element."""
        try:
            if hasattr(element, 'text_content') and element.text_content and hasattr(element.text_content, 'content') and element.text_content.content:
                return str(element.text_content.content)
            elif hasattr(element, 'table_content') and element.table_content:
                # Create rich table representation
                table_parts = []
                
                # Add table caption if available
                if hasattr(element.table_content, 'caption') and element.table_content.caption:
                    table_parts.append(f"Table: {element.table_content.caption}")
                
                # Add headers
                if hasattr(element.table_content, 'headers') and element.table_content.headers:
                    table_parts.append("Headers: " + " | ".join(str(h) for h in element.table_content.headers))
                
                # Add all rows (not limited like before)
                if hasattr(element.table_content, 'rows') and element.table_content.rows:
                    for i, row in enumerate(element.table_content.rows):
                        try:
                            row_text = " | ".join(str(cell) for cell in row)
                            table_parts.append(f"Row {i+1}: {row_text}")
                        except Exception as e:
                            logger.warning(f"Error processing table row {i}: {e}")
                            continue
                
                return "\n".join(table_parts)
            elif hasattr(element, 'image_content') and element.image_content:
                # Create descriptive image text
                image_parts = []
                if hasattr(element.image_content, 'caption') and element.image_content.caption:
                    image_parts.append(f"Image Caption: {element.image_content.caption}")
                if hasattr(element.image_content, 'description') and element.image_content.description:
                    image_parts.append(f"Image Description: {element.image_content.description}")
                if hasattr(element.image_content, 'alt_text') and element.image_content.alt_text:
                    image_parts.append(f"Image Alt Text: {element.image_content.alt_text}")
                
                return " ".join(image_parts)
            
            return ""
            
        except Exception as e:
            logger.warning(f"Error extracting text from element {getattr(element, 'element_id', 'unknown')}: {e}")
            return ""
    
    def _create_overlap(self, content: List[str], elements: List[str], types: List[ContentType]) -> Tuple[List[str], List[str], List[ContentType]]:
        """Create overlap content for the next chunk."""
        try:
            if not content or not elements or not types:
                return [], [], []
            
            # Ensure all lists have the same length
            min_length = min(len(content), len(elements), len(types))
            if min_length == 0:
                return [], [], []
            
            content = content[:min_length]
            elements = elements[:min_length]
            types = types[:min_length]
            
            # Calculate overlap size
            total_chars = sum(len(str(text)) for text in content)
            target_overlap = min(self.overlap, total_chars // 2)
            
            # Take content from the end that fits within overlap size
            overlap_content = []
            overlap_elements = []
            overlap_types = []
            current_size = 0
            
            for i in range(len(content) - 1, -1, -1):
                try:
                    text = str(content[i]) if content[i] else ""
                    text_len = len(text)
                    
                    if current_size + text_len <= target_overlap:
                        overlap_content.insert(0, text)
                        overlap_elements.insert(0, elements[i])
                        overlap_types.insert(0, types[i])
                        current_size += text_len
                    else:
                        # Take partial content if it fits
                        remaining_space = target_overlap - current_size
                        if remaining_space > 100:  # Only if significant space remains
                            partial_text = text[-remaining_space:] if text_len > remaining_space else text
                            overlap_content.insert(0, partial_text)
                            overlap_elements.insert(0, elements[i])
                            overlap_types.insert(0, types[i])
                        break
                except Exception as e:
                    logger.warning(f"Error processing overlap item {i}: {e}")
                    continue
            
            return overlap_content, overlap_elements, list(set(overlap_types))
            
        except Exception as e:
            logger.warning(f"Error creating overlap: {e}")
            return [], [], []
    
    def _create_chunk_from_content(
        self, 
        content: List[str], 
        elements: List[str], 
        types: List[ContentType], 
        document_id: str, 
        page_num: int
    ) -> Optional[TextChunk]:
        """Create a TextChunk from accumulated content."""
        if not content:
            return None
        
        # Combine content intelligently
        combined_content = self._combine_content_intelligently(content, types)
        
        # Create chunk ID
        chunk_id = f"{document_id}_page{page_num}_chunk_{len(elements)}_{hash(combined_content) % 10000}"
        
        
        # Create metadata
        metadata = {
            "page_number": page_num,
            "element_count": len(elements),
            "content_types": [ct.value for ct in set(types)],
            "chunk_length": len(combined_content),
            "has_tables": ContentType.TABLE in types,
            "has_images": ContentType.IMAGE in types,
            "has_headers": ContentType.HEADER in types
        }
        
        return TextChunk(
            content=combined_content,
            chunk_id=chunk_id,
            document_id=document_id,
            source_elements=elements,
            content_types=list(set(types)),
            page_numbers=[page_num],
            metadata=metadata
        )
    
    def _combine_content_intelligently(self, content: List[str], types: List[ContentType]) -> str:
        """Intelligently combine content with appropriate separators."""
        if not content:
            return ""
        
        combined_parts = []
        
        for i, (text, content_type) in enumerate(zip(content, types)):
            # Add content type prefix for clarity
            if content_type == ContentType.HEADER:
                combined_parts.append(f"## {text}")
            elif content_type == ContentType.TABLE:
                combined_parts.append(f"[TABLE]\n{text}\n[/TABLE]")
            elif content_type == ContentType.IMAGE:
                combined_parts.append(f"[IMAGE] {text}")
            elif content_type == ContentType.LIST:
                combined_parts.append(f"• {text}")
            else:
                combined_parts.append(text)
        
        # Join with appropriate separators
        return "\n\n".join(combined_parts)
    
    
    def _optimize_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Optimize chunks by merging small ones and splitting large ones."""
        optimized = []
        
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If chunk is too small, try to merge with next chunk
            if (len(current_chunk.content) < self.min_chunk_size and 
                i + 1 < len(chunks) and
                chunks[i + 1].document_id == current_chunk.document_id):
                
                next_chunk = chunks[i + 1]
                merged_chunk = self._merge_chunks(current_chunk, next_chunk)
                
                if merged_chunk and len(merged_chunk.content) <= self.max_chunk_size:
                    optimized.append(merged_chunk)
                    i += 2  # Skip next chunk as it's been merged
                    continue
            
            # If chunk is too large, split it
            if len(current_chunk.content) > self.max_chunk_size:
                split_chunks = self._split_chunk(current_chunk)
                optimized.extend(split_chunks)
            else:
                optimized.append(current_chunk)
            
            i += 1
        
        return optimized
    
    def _merge_chunks(self, chunk1: TextChunk, chunk2: TextChunk) -> Optional[TextChunk]:
        """Merge two adjacent chunks."""
        try:
            merged_content = chunk1.content + "\n\n" + chunk2.content
            merged_elements = chunk1.source_elements + chunk2.source_elements
            merged_types = list(set(chunk1.content_types + chunk2.content_types))
            merged_pages = list(set(chunk1.page_numbers + chunk2.page_numbers))
            
            # Create new metadata
            merged_metadata = {**chunk1.metadata, **chunk2.metadata}
            merged_metadata.update({
                "element_count": len(merged_elements),
                "chunk_length": len(merged_content),
                "content_types": [ct.value for ct in merged_types],
                "is_merged": True
            })
            
            # Create new chunk ID
            chunk_id = f"{chunk1.document_id}_merged_{hash(merged_content) % 10000}"
            
            
            return TextChunk(
                content=merged_content,
                chunk_id=chunk_id,
                document_id=chunk1.document_id,
                source_elements=merged_elements,
                content_types=merged_types,
                page_numbers=merged_pages,
                metadata=merged_metadata
            )
        except Exception as e:
            logger.warning(f"Failed to merge chunks: {e}")
            return None
    
    def _split_chunk(self, chunk: TextChunk) -> List[TextChunk]:
        """Split a large chunk into smaller ones using smart text splitting."""
        try:
            content = chunk.content
            
            # Use smart text splitting that respects boundaries
            split_texts = smart_text_split(
                content, 
                target_size=self.chunk_size,
                max_size=self.max_chunk_size,
                overlap=self.overlap
            )
            
            split_chunks = []
            for i, split_content in enumerate(split_texts):
                if len(split_content.strip()) >= 100:  # Only keep substantial chunks
                    split_chunk = self._create_split_chunk(chunk, split_content, i)
                    split_chunks.append(split_chunk)
            
            return split_chunks if split_chunks else [chunk]
            
        except Exception as e:
            logger.warning(f"Failed to split chunk: {e}")
            return [chunk]
    
    def _create_split_chunk(self, original_chunk: TextChunk, content: str, split_index: int) -> TextChunk:
        """Create a split chunk from original chunk."""
        chunk_id = f"{original_chunk.chunk_id}_split_{split_index}"
        
        metadata = original_chunk.metadata.copy()
        metadata.update({
            "chunk_length": len(content),
            "is_split": True,
            "split_index": split_index,
            "original_chunk_id": original_chunk.chunk_id
        })
        
        return TextChunk(
            content=content,
            chunk_id=chunk_id,
            document_id=original_chunk.document_id,
            source_elements=original_chunk.source_elements,
            content_types=original_chunk.content_types,
            page_numbers=original_chunk.page_numbers,
            metadata=metadata
        )

# Global chunker instance
intelligent_chunker = IntelligentChunker()