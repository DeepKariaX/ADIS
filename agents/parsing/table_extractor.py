import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
import camelot
import tabula
from agents.base_agent import BaseAgent
from database.models import AgentResponse, TableContent, TableCell, DocumentElement, ContentType, BoundingBox
from utils.logger import logger

class TableExtractorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="TableExtractor",
            description="Extracts tabular data from documents and converts to structured format"
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
            
            elements = await self._extract_tables_from_file(file_path, layout_elements, document_id)
            
            processing_time = time.time() - start_time
            
            response = self.create_response(
                content=elements,
                response_type="table_extraction",
                sources=[file_path],
                metadata={"total_tables": len(elements)},
                processing_time=processing_time
            )
            
            self.log_activity(f"Extracted {len(elements)} tables")
            return response
            
        except Exception as e:
            self.log_activity(f"Table extraction failed: {e}", "error")
            processing_time = time.time() - start_time
            
            return self.create_response(
                content=f"Table extraction failed: {str(e)}",
                response_type="error",
                processing_time=processing_time
            )
    
    async def _extract_tables_from_file(self, file_path: str, layout_elements: List[Dict], document_id: str = None) -> List[DocumentElement]:
        self.log_activity(f"Starting table extraction from: {file_path}")
        
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
            raise ValueError(f"Unsupported file format for table extraction: {file_extension}")
    
    async def _extract_from_pdf(self, file_path: str, layout_elements: List[Dict], document_id: str) -> List[DocumentElement]:
        elements = []
        
        # Filter layout elements for tables
        table_layout_elements = [
            elem for elem in layout_elements 
            if elem.get('content_type') == ContentType.TABLE.value
        ]
        
        try:
            # Method 1: Use Camelot for table extraction
            camelot_tables = await self._extract_with_camelot(file_path)
            
            # Method 2: Use Tabula as fallback
            if not camelot_tables:
                tabula_tables = await self._extract_with_tabula(file_path)
                camelot_tables = tabula_tables
            
            # Method 3: Use layout-guided extraction if we have layout info
            if table_layout_elements:
                layout_tables = await self._extract_with_layout(file_path, table_layout_elements)
                # Merge with other methods, preferring layout-guided results
                camelot_tables = self._merge_table_results(camelot_tables, layout_tables)
            
            # Convert to DocumentElements
            for i, table_data in enumerate(camelot_tables):
                table_content = await self._create_table_content(table_data, f"table_{i}")
                
                # Find corresponding layout element if available
                bounding_box = None
                if i < len(table_layout_elements):
                    bbox_info = table_layout_elements[i].get('bounding_box', {})
                    if bbox_info:
                        bounding_box = BoundingBox(
                            x1=bbox_info['x1'],
                            y1=bbox_info['y1'],
                            x2=bbox_info['x2'],
                            y2=bbox_info['y2'],
                            page_number=bbox_info.get('page_number', 0)
                        )
                
                element = DocumentElement(
                    document_id=document_id,
                    content_type=ContentType.TABLE,
                    bounding_box=bounding_box,
                    table_content=table_content,
                    metadata={
                        "extraction_method": table_data.get('method', 'camelot'),
                        "table_index": i,
                        "confidence": table_data.get('confidence', 0.8)
                    }
                )
                elements.append(element)
                
        except Exception as e:
            self.log_activity(f"Error extracting tables from PDF: {e}", "error")
            raise
        
        return elements
    
    async def _extract_with_camelot(self, file_path: str) -> List[Dict[str, Any]]:
        tables = []
        
        try:
            # Extract tables using Camelot
            camelot_tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
            
            for table in camelot_tables:
                lattice_threshold = self.settings.thresholds.get("table_accuracy_lattice", 50.0)
                if table.accuracy > lattice_threshold:  # Only include tables with decent accuracy
                    df = table.df
                    table_data = {
                        'dataframe': df,
                        'method': 'camelot_lattice',
                        'confidence': table.accuracy / 100.0,
                        'page': table.page
                    }
                    tables.append(table_data)
            
            # Try stream flavor if lattice didn't find many tables
            if len(tables) < 2:
                camelot_stream = camelot.read_pdf(file_path, pages='all', flavor='stream')
                for table in camelot_stream:
                    stream_threshold = self.settings.thresholds.get("table_accuracy_stream", 40.0)
                    if table.accuracy > stream_threshold:
                        df = table.df
                        table_data = {
                            'dataframe': df,
                            'method': 'camelot_stream',
                            'confidence': table.accuracy / 100.0,
                            'page': table.page
                        }
                        tables.append(table_data)
            
        except Exception as e:
            self.log_activity(f"Camelot extraction failed: {e}", "warning")
        
        return tables
    
    async def _extract_with_tabula(self, file_path: str) -> List[Dict[str, Any]]:
        tables = []
        
        try:
            # Extract tables using Tabula
            tabula_tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            
            for i, df in enumerate(tabula_tables):
                if not df.empty:
                    table_data = {
                        'dataframe': df,
                        'method': 'tabula',
                        'confidence': self.settings.thresholds.get("table_confidence_default", 0.7),  # Default confidence for tabula
                        'page': i + 1  # Approximate page number
                    }
                    tables.append(table_data)
                    
        except Exception as e:
            self.log_activity(f"Tabula extraction failed: {e}", "warning")
        
        return tables
    
    async def _extract_with_layout(self, file_path: str, table_elements: List[Dict]) -> List[Dict[str, Any]]:
        tables = []
        pdf_document = None
        
        try:
            pdf_document = fitz.open(file_path)
            
            for table_elem in table_elements:
                bbox_info = table_elem.get('bounding_box', {})
                if not bbox_info:
                    continue
                
                page_num = bbox_info.get('page_number', 0)
                page = pdf_document.load_page(page_num)
                
                # Extract text from table region
                rect = fitz.Rect(bbox_info['x1'], bbox_info['y1'], bbox_info['x2'], bbox_info['y2'])
                
                # Try to extract table structure from the region
                table_text = page.get_text("text", clip=rect)
                
                # Simple table parsing (split by lines and common delimiters)
                df = await self._parse_text_table(table_text)
                
                if not df.empty:
                    table_data = {
                        'dataframe': df,
                        'method': 'layout_guided',
                        'confidence': table_elem.get('confidence', 0.8),
                        'page': page_num + 1,
                        'bbox': bbox_info
                    }
                    tables.append(table_data)
            
        except Exception as e:
            self.log_activity(f"Layout-guided extraction failed: {e}", "warning")
        finally:
            # Ensure PDF document is always closed
            if pdf_document is not None:
                try:
                    pdf_document.close()
                except:
                    pass
        
        return tables
    
    async def _parse_text_table(self, table_text: str) -> pd.DataFrame:
        try:
            lines = table_text.strip().split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            
            if not lines:
                return pd.DataFrame()
            
            # Try different delimiters
            delimiters = ['\t', '|', ',', ';']
            best_df = pd.DataFrame()
            max_columns = 0
            
            for delimiter in delimiters:
                try:
                    rows = []
                    for line in lines:
                        row = [cell.strip() for cell in line.split(delimiter)]
                        if len(row) > 1:  # Must have at least 2 columns
                            rows.append(row)
                    
                    if rows and len(rows) > 1:
                        # Make all rows the same length
                        max_len = max(len(row) for row in rows)
                        normalized_rows = [row + [''] * (max_len - len(row)) for row in rows]
                        
                        df = pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0])
                        if len(df.columns) > max_columns:
                            best_df = df
                            max_columns = len(df.columns)
                            
                except Exception:
                    continue
            
            return best_df
            
        except Exception as e:
            self.log_activity(f"Text table parsing failed: {e}", "warning")
            return pd.DataFrame()
    
    def _merge_table_results(self, method1_tables: List[Dict], method2_tables: List[Dict]) -> List[Dict]:
        # Simple merge strategy: prefer method2 (layout-guided) when available
        if not method1_tables:
            return method2_tables
        if not method2_tables:
            return method1_tables
        
        # For now, just combine both lists
        # In a more sophisticated implementation, we would match tables by position/content
        return method2_tables + method1_tables
    
    async def _create_table_content(self, table_data: Dict, table_id: str) -> TableContent:
        try:
            df = table_data['dataframe']
            
            # Extract headers and rows (ensure headers are strings)
            headers = [str(col) for col in df.columns.tolist()]
            rows = df.values.tolist()
            
            # Convert all values to strings
            rows = [[str(cell) if pd.notna(cell) else '' for cell in row] for row in rows]
            
            # Create table cells
            cells = []
            for row_idx, row in enumerate(rows):
                for col_idx, cell_content in enumerate(row):
                    cell = TableCell(
                        content=cell_content,
                        row=row_idx,
                        column=col_idx
                    )
                    cells.append(cell)
            
            return TableContent(
                headers=headers,
                rows=rows,
                cells=cells,
                table_id=table_id
            )
            
        except Exception as e:
            self.log_activity(f"Error creating table content: {e}", "error")
            return TableContent(headers=[], rows=[], cells=[])
    
    async def _extract_from_docx(self, file_path: str, document_id: str) -> List[DocumentElement]:
        elements = []
        
        try:
            from docx import Document
            
            doc = Document(file_path)
            
            for i, table in enumerate(doc.tables):
                # Extract table data
                rows = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    rows.append(row_data)
                
                if rows:
                    # Assume first row is headers (ensure strings)
                    headers = [str(cell) for cell in rows[0]] if rows else []
                    data_rows = rows[1:] if len(rows) > 1 else []
                    
                    # Create table cells
                    cells = []
                    for row_idx, row in enumerate(data_rows):
                        for col_idx, cell_content in enumerate(row):
                            cell = TableCell(
                                content=cell_content,
                                row=row_idx,
                                column=col_idx
                            )
                            cells.append(cell)
                    
                    table_content = TableContent(
                        headers=headers,
                        rows=data_rows,
                        cells=cells,
                        table_id=f"docx_table_{i}"
                    )
                    
                    element = DocumentElement(
                        document_id=document_id,
                        content_type=ContentType.TABLE,
                        table_content=table_content,
                        metadata={
                            "extraction_method": "docx",
                            "table_index": i,
                            "rows": len(data_rows),
                            "columns": len(headers)
                        }
                    )
                    elements.append(element)
        
        except Exception as e:
            self.log_activity(f"Error extracting tables from DOCX: {e}", "error")
            raise
        
        return elements