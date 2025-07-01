import fitz
import gc
import tempfile
import os
from typing import Optional, Any
from contextlib import contextmanager
from utils.logger import logger

class SafePDFManager:
    """Context manager for safe PDF document handling with proper resource cleanup."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.pdf_document: Optional[fitz.Document] = None
        
    def __enter__(self) -> fitz.Document:
        try:
            self.pdf_document = fitz.open(self.file_path)
            return self.pdf_document
        except Exception as e:
            logger.error(f"Failed to open PDF document {self.file_path}: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pdf_document is not None:
            try:
                self.pdf_document.close()
            except Exception as e:
                logger.warning(f"Error closing PDF document: {e}")
            finally:
                self.pdf_document = None
        
        # Force garbage collection to help with memory cleanup
        gc.collect()

class SafePixmapManager:
    """Context manager for safe Pixmap handling with proper cleanup."""
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pixmap: Optional[fitz.Pixmap] = None
        
    def __enter__(self) -> fitz.Pixmap:
        try:
            self.pixmap = fitz.Pixmap(*self.args, **self.kwargs)
            return self.pixmap
        except Exception as e:
            logger.error(f"Failed to create Pixmap: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pixmap is not None:
            try:
                self.pixmap = None
            except Exception as e:
                logger.warning(f"Error cleaning up Pixmap: {e}")

@contextmanager
def safe_pdf_document(file_path: str):
    """Context manager for safe PDF document handling."""
    pdf_document = None
    try:
        pdf_document = fitz.open(file_path)
        yield pdf_document
    except Exception as e:
        logger.error(f"Error with PDF document {file_path}: {e}")
        raise
    finally:
        if pdf_document is not None:
            try:
                pdf_document.close()
            except:
                pass
        # Force cleanup
        gc.collect()

@contextmanager
def safe_pixmap(*args, **kwargs):
    """Context manager for safe Pixmap handling."""
    pixmap = None
    try:
        pixmap = fitz.Pixmap(*args, **kwargs)
        yield pixmap
    except Exception as e:
        logger.error(f"Error with Pixmap: {e}")
        raise
    finally:
        if pixmap is not None:
            try:
                pixmap = None
            except:
                pass

def cleanup_temp_files():
    """Clean up any remaining temporary files."""
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith(('tmp', 'page-')) and filename.endswith('.pdf'):
                temp_file = os.path.join(temp_dir, filename)
                try:
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
                except (OSError, PermissionError):
                    # File might be in use, skip it
                    pass
    except Exception as e:
        logger.warning(f"Error during temp file cleanup: {e}")

def force_pdf_cleanup():
    """Force cleanup of PyMuPDF resources."""
    try:
        # Force garbage collection
        gc.collect()
        
        # Clean up temp files
        cleanup_temp_files()
        
    except Exception as e:
        logger.warning(f"Error during PDF cleanup: {e}")