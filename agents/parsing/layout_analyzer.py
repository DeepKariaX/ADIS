import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
from agents.base_agent import BaseAgent
from database.models import AgentResponse, BoundingBox, ContentType
from utils.logger import logger
from utils.pdf_utils import safe_pdf_document, safe_pixmap

# Import PyMuPDF with fallback
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

# Optional imports for advanced layout analysis
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
except ImportError:
    LAYOUTPARSER_AVAILABLE = False
    lp = None

class LayoutAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="LayoutAnalyzer",
            description="Analyzes document layout to identify content blocks and their types"
        )
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        import os
        
        # Skip model initialization in test mode to avoid hanging
        if os.environ.get('PYTEST_CURRENT_TEST') or os.environ.get('TESTING'):
            self.log_activity("Test mode detected, skipping model initialization")
            self.model = None
            return
            
        try:
            if not LAYOUTPARSER_AVAILABLE:
                self.log_activity("LayoutParser not available, using basic layout analysis", "warning")
                self.model = None
                return
                
            # Initialize layout analysis model (using LayoutParser with PubLayNet)
            # This can be slow on first run as it downloads the model (~330MB)
            self.log_activity("Initializing layout model (this may take a while on first run - downloading ~330MB model)")
            
            try:
                # Use a progress callback to show download progress and extend timeout during active downloads
                import time
                start_time = time.time()
                
                self.model = lp.Detectron2LayoutModel(
                    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.settings.thresholds.get("layout_detection_threshold", 0.8)],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                )
                
                elapsed = time.time() - start_time
                self.log_activity(f"Layout analysis model initialized successfully in {elapsed:.1f}s")
                
            except Exception as download_error:
                # If model download/initialization fails, fall back gracefully
                error_msg = str(download_error)
                if "download" in error_msg.lower() or "connection" in error_msg.lower():
                    self.log_activity("Model download failed (network issue), falling back to basic analysis", "warning")
                else:
                    self.log_activity(f"Model initialization failed: {error_msg}, falling back to basic analysis", "warning")
                self.model = None
                
        except Exception as e:
            self.log_activity(f"Failed to initialize layout model: {e}", "warning")
            self.log_activity("Falling back to basic layout analysis", "info")
            # Fallback to basic layout analysis
            self.model = None
    
    async def process(self, file_path: str, **kwargs) -> AgentResponse:
        start_time = time.time()
        
        try:
            self.log_activity(f"Starting layout analysis for: {file_path}")
            
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Process PDF document
            layout_elements = await self._analyze_pdf_layout(file_path)
            
            processing_time = time.time() - start_time
            
            response = self.create_response(
                content=layout_elements,
                response_type="layout_analysis",
                sources=[file_path],
                metadata={"file_path": file_path, "total_elements": len(layout_elements)},
                processing_time=processing_time
            )
            
            self.log_activity(f"Completed layout analysis: found {len(layout_elements)} elements")
            return response
            
        except Exception as e:
            self.log_activity(f"Layout analysis failed: {e}", "error")
            processing_time = time.time() - start_time
            
            return self.create_response(
                content=f"Layout analysis failed: {str(e)}",
                response_type="error",
                sources=[file_path],
                processing_time=processing_time
            )
    
    async def _analyze_pdf_layout(self, file_path: str) -> List[Dict[str, Any]]:
        layout_elements = []
        
        try:
            with safe_pdf_document(file_path) as pdf_document:
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    
                    if self.model:
                        # Use LayoutParser model
                        elements = await self._analyze_page_with_model(page, page_num)
                    else:
                        # Fallback to basic text block analysis
                        elements = await self._analyze_page_basic(page, page_num)
                    
                    layout_elements.extend(elements)
            
        except Exception as e:
            self.log_activity(f"Error analyzing PDF layout: {e}", "error")
            raise
        
        return layout_elements
    
    async def _analyze_page_with_model(self, page, page_num: int) -> List[Dict[str, Any]]:
        elements = []
        
        try:
            with safe_pixmap(page.get_pixmap()) as pix:
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Run layout detection
                layout_result = self.model.detect(img)
                
                for block in layout_result:
                    bbox = block.coordinates
                    
                    element = {
                        "content_type": self._map_content_type(block.type),
                        "bounding_box": {
                            "x1": bbox[0],
                            "y1": bbox[1],
                            "x2": bbox[2],
                            "y2": bbox[3],
                            "page_number": page_num
                        },
                        "confidence": block.score,
                        "metadata": {
                            "detection_method": "layoutparser",
                            "model_type": block.type
                        }
                    }
                    elements.append(element)
                
        except Exception as e:
            self.log_activity(f"Error in model-based analysis: {e}", "warning")
            # Fallback to basic analysis
            elements = await self._analyze_page_basic(page, page_num)
        
        return elements
    
    async def _analyze_page_basic(self, page, page_num: int) -> List[Dict[str, Any]]:
        elements = []
        
        try:
            # Get text blocks from PDF
            text_blocks = page.get_text("dict")["blocks"]
            
            for block in text_blocks:
                if "lines" in block:  # Text block
                    bbox = block["bbox"]
                    element = {
                        "content_type": self._classify_text_block(block),
                        "bounding_box": {
                            "x1": bbox[0],
                            "y1": bbox[1],
                            "x2": bbox[2],
                            "y2": bbox[3],
                            "page_number": page_num
                        },
                        "confidence": 0.7,
                        "metadata": {
                            "detection_method": "basic_pdf_analysis",
                            "block_type": "text"
                        }
                    }
                    elements.append(element)
                elif "image" in block:  # Image block
                    bbox = block["bbox"]
                    element = {
                        "content_type": ContentType.IMAGE.value,
                        "bounding_box": {
                            "x1": bbox[0],
                            "y1": bbox[1],
                            "x2": bbox[2],
                            "y2": bbox[3],
                            "page_number": page_num
                        },
                        "confidence": 0.8,
                        "metadata": {
                            "detection_method": "basic_pdf_analysis",
                            "block_type": "image"
                        }
                    }
                    elements.append(element)
            
            # Try to detect tables using heuristics
            table_elements = await self._detect_tables_basic(page, page_num)
            elements.extend(table_elements)
            
        except Exception as e:
            self.log_activity(f"Error in basic analysis: {e}", "error")
            raise
        
        return elements
    
    def _map_content_type(self, model_type: str) -> str:
        mapping = {
            "Text": ContentType.TEXT.value,
            "Title": ContentType.HEADER.value,
            "List": ContentType.LIST.value,
            "Table": ContentType.TABLE.value,
            "Figure": ContentType.IMAGE.value
        }
        return mapping.get(model_type, ContentType.TEXT.value)
    
    def _classify_text_block(self, block: Dict) -> str:
        # Simple heuristics to classify text blocks
        try:
            if not block.get("lines"):
                return ContentType.TEXT.value
            
            first_line = block["lines"][0]
            if not first_line.get("spans"):
                return ContentType.TEXT.value
            
            first_span = first_line["spans"][0]
            font_size = first_span.get("size", 12)
            font_flags = first_span.get("flags", 0)
            
            # Check if it's a header (large font or bold)
            from config.settings import get_settings
            settings = get_settings()
            large_font_threshold = settings.thresholds.get("large_font_size", 16.0)
            if font_size > large_font_threshold or (font_flags & 2**4):  # Bold flag
                return ContentType.HEADER.value
            
            # Check if it's a list (starts with bullet points or numbers)
            text = first_span.get("text", "").strip()
            if text and (text[0] in "•-*" or (text[0].isdigit() and "." in text[:5])):
                return ContentType.LIST.value
            
            return ContentType.TEXT.value
            
        except Exception:
            return ContentType.TEXT.value
    
    async def _detect_tables_basic(self, page, page_num: int) -> List[Dict[str, Any]]:
        elements = []
        
        try:
            # Try to find tables using text alignment analysis
            tables = page.find_tables()
            
            for table in tables:
                bbox = table.bbox
                element = {
                    "content_type": ContentType.TABLE.value,
                    "bounding_box": {
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                        "page_number": page_num
                    },
                    "confidence": 0.75,
                    "metadata": {
                        "detection_method": "pdf_table_detection",
                        "table_cells": len(table.cells) if hasattr(table, 'cells') else 0
                    }
                }
                elements.append(element)
                
        except Exception as e:
            self.log_activity(f"Table detection failed: {e}", "warning")
        
        return elements