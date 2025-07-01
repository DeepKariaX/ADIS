from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    page_number: int

class TextContent(BaseModel):
    content: str
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    reading_order: Optional[int] = None

class TableCell(BaseModel):
    content: str
    row: int
    column: int
    rowspan: int = 1
    colspan: int = 1

class TableContent(BaseModel):
    headers: List[str]
    rows: List[List[str]]
    cells: List[TableCell]
    caption: Optional[str] = None
    table_id: Optional[str] = None

class ImageContent(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    description: Optional[str] = None
    image_id: Optional[str] = None

class DocumentElement(BaseModel):
    element_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    document_id: str
    content_type: ContentType
    bounding_box: Optional[BoundingBox] = None
    text_content: Optional[TextContent] = None
    table_content: Optional[TableContent] = None
    image_content: Optional[ImageContent] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class DocumentMetadata(BaseModel):
    filename: str
    file_path: str
    file_size: int
    file_type: str
    page_count: Optional[int] = None
    author: Optional[str] = None
    title: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    language: Optional[str] = None

class Document(BaseModel):
    document_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    metadata: DocumentMetadata
    elements: List[str] = Field(default_factory=list)  # Store element IDs, not full objects
    processing_status: str = "pending"
    processing_errors: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class VectorEmbedding(BaseModel):
    element_id: str
    document_id: str
    content_type: ContentType
    embedding: List[float]
    text_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class QueryIntent(str, Enum):
    TEXT_SEARCH = "text_search"
    TABLE_QUERY = "table_query" 
    IMAGE_QUERY = "image_query"
    MULTI_MODAL = "multi_modal"
    UNKNOWN = "unknown"

class UserQuery(BaseModel):
    """Clean user query structure"""
    query_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    query_text: str
    intent: QueryIntent = QueryIntent.UNKNOWN
    
    # User context (optional)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Query metadata
    query_type: Optional[str] = None  # "chat", "api", "batch"
    expected_response_format: Optional[str] = None  # "text", "json", "markdown"
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

class AgentResponse(BaseModel):
    agent_name: str
    response_type: str
    content: Union[str, Dict[str, Any], List[Any]]
    sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Optional[float] = None

class SourceReference(BaseModel):
    """Document-level source reference"""
    document_id: str  # The actual document_id from documents collection
    document_name: str
    page_count: Optional[int] = None

class DetailedSourceReference(BaseModel):
    """Chunk/element-level source reference for detailed tracking"""
    element_id: str  # The chunk or element ID used
    document_id: str  # The actual document_id from documents collection
    content_type: Optional[ContentType] = None
    page_number: Optional[int] = None
    chunk_type: Optional[str] = None  # "intelligent_chunk", "simple_chunk", etc.

class QueryResponse(BaseModel):
    """Cleaner query response structure"""
    query_id: str
    query_text: str
    
    # Core response data
    answer: str
    intent: QueryIntent
    reasoning: Optional[str] = None
    
    # Processing details
    primary_agent: str  # Main agent that handled the query
    sub_agents_used: List[str] = Field(default_factory=list)  # Any sub-agents called
    
    # Source information
    sources: List[SourceReference] = Field(default_factory=list)  # Documents used
    detailed_sources: List[DetailedSourceReference] = Field(default_factory=list)  # Chunks used
    
    # Metadata
    processing_time: float
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)

# Keep the old model for backward compatibility during migration
class LegacyQueryResponse(BaseModel):
    query_id: str
    query_text: str
    final_answer: str
    agent_responses: List[AgentResponse] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    processing_time: float
    created_at: datetime = Field(default_factory=datetime.now)