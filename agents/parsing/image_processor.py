import time
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
from agents.base_agent import BaseAgent
from database.models import AgentResponse, ImageContent, DocumentElement, ContentType, BoundingBox
from utils.logger import logger

class ImageProcessorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ImageProcessor",
            description="Processes and extracts images from documents with metadata and captions"
        )
    
    async def process(self, input_data: Any, **kwargs) -> AgentResponse:
        start_time = time.time()
        
        try:
            if isinstance(input_data, str):
                file_path = input_data
                layout_elements = kwargs.get('layout_elements', [])
            elif isinstance(input_data, dict) and 'file_path' in input_data:
                file_path = input_data['file_path']
                layout_elements = input_data.get('layout_elements', [])
                document_id = input_data.get('document_id', Path(file_path).stem)
            else:
                raise ValueError("Invalid input data format")
            
            elements = await self._extract_images_from_file(file_path, layout_elements, document_id)
            
            processing_time = time.time() - start_time
            
            response = self.create_response(
                content=elements,
                response_type="image_extraction",
                sources=[file_path],
                metadata={"total_images": len(elements)},
                processing_time=processing_time
            )
            
            self.log_activity(f"Processed {len(elements)} images")
            return response
            
        except Exception as e:
            self.log_activity(f"Image processing failed: {e}", "error")
            processing_time = time.time() - start_time
            
            return self.create_response(
                content=f"Image processing failed: {str(e)}",
                response_type="error",
                processing_time=processing_time
            )
    
    async def _extract_images_from_file(self, file_path: str, layout_elements: List[Dict], document_id: str = None) -> List[DocumentElement]:
        self.log_activity(f"Starting image extraction from: {file_path}")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not document_id:
            document_id = Path(file_path).stem
            
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return await self._extract_from_pdf(file_path, layout_elements, document_id)
        elif file_extension in ['.docx', '.doc']:
            return await self._extract_from_docx(file_path, document_id)
        else:
            raise ValueError(f"Unsupported file format for image extraction: {file_extension}")
    
    async def _extract_from_pdf(self, file_path: str, layout_elements: List[Dict], document_id: str) -> List[DocumentElement]:
        elements = []
        
        # Filter layout elements for images
        image_layout_elements = [
            elem for elem in layout_elements 
            if elem.get('content_type') == ContentType.IMAGE.value
        ]
        
        pdf_document = None
        try:
            pdf_document = fitz.open(file_path)
            
            # Create output directory for images
            output_dir = Path(self.settings.processed_dir) / "images" / document_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            extracted_images = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Get all images on the page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image data
                        image_data = await self._extract_image_from_pdf(
                            pdf_document, img, page_num, img_index, output_dir
                        )
                        
                        if image_data:
                            # Try to find corresponding layout element
                            layout_elem = None
                            if img_index < len(image_layout_elements):
                                layout_elem = image_layout_elements[img_index]
                            
                            # Create bounding box if available
                            bounding_box = None
                            if layout_elem:
                                bbox_info = layout_elem.get('bounding_box', {})
                                if bbox_info:
                                    bounding_box = BoundingBox(
                                        x1=bbox_info['x1'],
                                        y1=bbox_info['y1'],
                                        x2=bbox_info['x2'],
                                        y2=bbox_info['y2'],
                                        page_number=page_num
                                    )
                            
                            # Try to find caption nearby
                            caption = await self._find_image_caption(page, bounding_box, page_num)
                            
                            image_content = ImageContent(
                                image_path=image_data['path'],
                                image_base64=image_data.get('base64'),
                                caption=caption,
                                image_id=f"{document_id}_page{page_num}_img{img_index}",
                                description=image_data.get('description')
                            )
                            
                            element = DocumentElement(
                                document_id=document_id,
                                content_type=ContentType.IMAGE,
                                bounding_box=bounding_box,
                                image_content=image_content,
                                metadata={
                                    "extraction_method": "pdf_image_extraction",
                                    "page_number": page_num,
                                    "image_index": img_index,
                                    "image_format": image_data.get('format'),
                                    "image_size": image_data.get('size')
                                }
                            )
                            elements.append(element)
                            extracted_images.append(image_data)
                            
                    except Exception as e:
                        self.log_activity(f"Failed to extract image {img_index} from page {page_num}: {e}", "warning")
                        continue
            
            self.log_activity(f"Extracted {len(extracted_images)} images from PDF")
            
        except Exception as e:
            self.log_activity(f"Error extracting images from PDF: {e}", "error")
            raise
        finally:
            # Ensure PDF document is always closed
            if pdf_document is not None:
                try:
                    pdf_document.close()
                except:
                    pass
        
        return elements
    
    async def _extract_image_from_pdf(
        self, 
        pdf_document, 
        img_info, 
        page_num: int, 
        img_index: int, 
        output_dir: Path
    ) -> Optional[Dict[str, Any]]:
        pix = None
        converted_pix = None
        try:
            # Get image object reference
            xref = img_info[0]
            pix = fitz.Pixmap(pdf_document, xref)
            
            # Skip if image is too small (likely decorative)
            from config.settings import get_settings
            settings = get_settings()
            min_size = settings.thresholds.get("min_image_size", 50.0)
            if pix.width < min_size or pix.height < min_size:
                return None
            
            # Convert to PIL Image
            if pix.n - pix.alpha < 4:  # GRAY or RGB
                img_data = pix.tobytes("ppm")
                pil_image = Image.open(io.BytesIO(img_data))
            else:  # CMYK
                converted_pix = fitz.Pixmap(fitz.csRGB, pix)
                img_data = converted_pix.tobytes("ppm")
                pil_image = Image.open(io.BytesIO(img_data))
            
            # Save image to file
            image_filename = f"page{page_num:03d}_img{img_index:03d}.png"
            image_path = output_dir / image_filename
            pil_image.save(image_path, "PNG")
            
            # Convert to base64 for storage (optional)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Get image info
            image_data = {
                'path': str(image_path),
                'base64': img_base64,
                'format': 'PNG',
                'size': (pil_image.width, pil_image.height),
                'description': f"Image extracted from page {page_num + 1}"
            }
            
            return image_data
            
        except Exception as e:
            self.log_activity(f"Error extracting individual image: {e}", "warning")
            return None
        finally:
            # Ensure Pixmap objects are properly cleaned up
            if pix is not None:
                try:
                    pix = None
                except:
                    pass
            if converted_pix is not None:
                try:
                    converted_pix = None
                except:
                    pass
    
    async def _find_image_caption(self, page, bounding_box: Optional[BoundingBox], page_num: int) -> Optional[str]:
        try:
            if not bounding_box:
                return None
            
            # Look for text near the image (below or above)
            from config.settings import get_settings
            settings = get_settings()
            search_margin = settings.thresholds.get("caption_search_margin", 50.0)
            
            # Search below the image
            below_rect = fitz.Rect(
                bounding_box.x1,
                bounding_box.y2,
                bounding_box.x2,
                bounding_box.y2 + search_margin
            )
            
            below_text = page.get_text("text", clip=below_rect).strip()
            
            # Search above the image
            above_rect = fitz.Rect(
                bounding_box.x1,
                max(0, bounding_box.y1 - search_margin),
                bounding_box.x2,
                bounding_box.y1
            )
            
            above_text = page.get_text("text", clip=above_rect).strip()
            
            # Look for caption patterns
            caption_patterns = ['Figure', 'Fig.', 'Image', 'Photo', 'Chart', 'Graph', 'Diagram']
            
            for text in [below_text, above_text]:
                if text and any(pattern.lower() in text.lower() for pattern in caption_patterns):
                    # Clean up the caption
                    lines = text.split('\n')
                    for line in lines:
                        if any(pattern.lower() in line.lower() for pattern in caption_patterns):
                            return line.strip()
            
            # If no pattern match, return the first non-empty text found
            return below_text or above_text or None
            
        except Exception as e:
            self.log_activity(f"Error finding image caption: {e}", "warning")
            return None
    
    async def _extract_from_docx(self, file_path: str, document_id: str) -> List[DocumentElement]:
        elements = []
        
        try:
            from docx import Document
            from docx.document import Document as DocumentType
            
            doc = Document(file_path)
            
            # Create output directory for images
            output_dir = Path(self.settings.processed_dir) / "images" / document_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract images from document relationships
            image_parts = []
            if hasattr(doc, 'part') and hasattr(doc.part, 'related_parts'):
                for rel_id, related_part in doc.part.related_parts.items():
                    if "image" in related_part.content_type:
                        image_parts.append((rel_id, related_part))
            
            for i, (rel_id, image_part) in enumerate(image_parts):
                try:
                    # Get image data
                    image_data = image_part.blob
                    
                    # Determine image format
                    image_format = image_part.content_type.split('/')[-1]
                    if image_format == 'jpeg':
                        image_format = 'jpg'
                    
                    # Save image
                    image_filename = f"docx_img{i:03d}.{image_format}"
                    image_path = output_dir / image_filename
                    
                    with open(image_path, 'wb') as f:
                        f.write(image_data)
                    
                    # Convert to base64
                    img_base64 = base64.b64encode(image_data).decode()
                    
                    # Try to get image dimensions
                    try:
                        pil_image = Image.open(io.BytesIO(image_data))
                        image_size = (pil_image.width, pil_image.height)
                    except:
                        image_size = (0, 0)
                    
                    image_content = ImageContent(
                        image_path=str(image_path),
                        image_base64=img_base64,
                        image_id=f"{document_id}_docx_img{i}",
                        description=f"Image extracted from DOCX document"
                    )
                    
                    element = DocumentElement(
                        document_id=document_id,
                        content_type=ContentType.IMAGE,
                        image_content=image_content,
                        metadata={
                            "extraction_method": "docx_image_extraction",
                            "image_index": i,
                            "image_format": image_format,
                            "image_size": image_size,
                            "content_type": image_part.content_type
                        }
                    )
                    elements.append(element)
                    
                except Exception as e:
                    self.log_activity(f"Failed to extract DOCX image {i}: {e}", "warning")
                    continue
            
            self.log_activity(f"Extracted {len(elements)} images from DOCX")
            
        except Exception as e:
            self.log_activity(f"Error extracting images from DOCX: {e}", "error")
            raise
        
        return elements