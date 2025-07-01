import asyncio
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import IndexModel, ASCENDING, TEXT
from config.settings import get_settings
from database.models import Document, DocumentElement, VectorEmbedding, UserQuery, QueryResponse, LegacyQueryResponse, SourceReference, DetailedSourceReference
from utils.logger import logger

class MongoDBClient:
    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.documents: Optional[AsyncIOMotorCollection] = None
        self.elements: Optional[AsyncIOMotorCollection] = None
        self.embeddings: Optional[AsyncIOMotorCollection] = None
        self.queries: Optional[AsyncIOMotorCollection] = None
        self.responses: Optional[AsyncIOMotorCollection] = None

    async def connect(self):
        try:
            self.client = AsyncIOMotorClient(self.settings.mongodb_url)
            self.database = self.client[self.settings.database_name]
            
            # Collections
            self.documents = self.database.documents
            self.elements = self.database.elements
            self.embeddings = self.database.embeddings
            self.queries = self.database.queries
            self.responses = self.database.responses
            
            # Create indexes
            await self._create_indexes()
            
            logger.info("Connected to MongoDB successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self):
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def _create_indexes(self):
        # Document indexes
        await self.documents.create_index([("document_id", ASCENDING)], unique=True)
        await self.documents.create_index([("metadata.filename", ASCENDING)])
        await self.documents.create_index([("processing_status", ASCENDING)])
        
        # Element indexes
        await self.elements.create_index([("element_id", ASCENDING)], unique=True)
        await self.elements.create_index([("document_id", ASCENDING)])
        await self.elements.create_index([("content_type", ASCENDING)])
        
        # Embedding indexes
        await self.embeddings.create_index([("element_id", ASCENDING)], unique=True)
        await self.embeddings.create_index([("document_id", ASCENDING)])
        await self.embeddings.create_index([("content_type", ASCENDING)])
        
        # Query/Response indexes
        await self.queries.create_index([("query_id", ASCENDING)], unique=True)
        await self.responses.create_index([("query_id", ASCENDING)])

    # Document operations
    async def insert_document(self, document: Document) -> str:
        try:
            result = await self.documents.insert_one(document.model_dump())
            logger.info(f"Inserted document: {document.document_id}")
            return document.document_id
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            raise

    async def get_document(self, document_id: str) -> Optional[Document]:
        try:
            doc_data = await self.documents.find_one({"document_id": document_id})
            if doc_data:
                return Document(**doc_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            raise

    async def update_document_status(self, document_id: str, status: str, errors: List[str] = None):
        try:
            update_data = {"processing_status": status}
            if errors:
                update_data["processing_errors"] = errors
            
            await self.documents.update_one(
                {"document_id": document_id},
                {"$set": update_data}
            )
            logger.info(f"Updated document {document_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
            raise

    async def update_document_metadata(self, document_id: str, metadata_updates: Dict[str, Any]):
        try:
            # Build update data with metadata prefix
            update_data = {}
            for key, value in metadata_updates.items():
                update_data[f"metadata.{key}"] = value
            
            await self.documents.update_one(
                {"document_id": document_id},
                {"$set": update_data}
            )
            logger.info(f"Updated document {document_id} metadata: {list(metadata_updates.keys())}")
        except Exception as e:
            logger.error(f"Failed to update document metadata: {e}")
            raise

    async def update_document_elements(self, document_id: str, element_ids: List[str]):
        try:
            await self.documents.update_one(
                {"document_id": document_id},
                {"$set": {"elements": element_ids}}
            )
            logger.info(f"Updated document {document_id} with {len(element_ids)} elements")
        except Exception as e:
            logger.error(f"Failed to update document elements: {e}")
            raise

    async def list_documents(self, limit: int = 100) -> List[Document]:
        try:
            cursor = self.documents.find().limit(limit)
            documents = []
            async for doc_data in cursor:
                documents.append(Document(**doc_data))
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise

    # Element operations
    async def insert_elements(self, elements: List[DocumentElement]) -> List[str]:
        try:
            if not elements:
                return []
            
            element_dicts = [element.model_dump() for element in elements]
            result = await self.elements.insert_many(element_dicts)
            
            element_ids = [element.element_id for element in elements]
            logger.info(f"Inserted {len(element_ids)} elements")
            return element_ids
        except Exception as e:
            logger.error(f"Failed to insert elements: {e}")
            raise

    async def get_elements_by_document(self, document_id: str) -> List[DocumentElement]:
        try:
            cursor = self.elements.find({"document_id": document_id})
            elements = []
            async for element_data in cursor:
                elements.append(DocumentElement(**element_data))
            return elements
        except Exception as e:
            logger.error(f"Failed to get elements for document {document_id}: {e}")
            raise

    async def get_elements_by_type(self, document_id: str, content_type: str) -> List[DocumentElement]:
        try:
            cursor = self.elements.find({
                "document_id": document_id,
                "content_type": content_type
            })
            elements = []
            async for element_data in cursor:
                elements.append(DocumentElement(**element_data))
            return elements
        except Exception as e:
            logger.error(f"Failed to get elements by type: {e}")
            raise

    # Embedding operations
    async def insert_embeddings(self, embeddings: List[VectorEmbedding]) -> List[str]:
        try:
            if not embeddings:
                return []
            
            embedding_dicts = [embedding.model_dump() for embedding in embeddings]
            result = await self.embeddings.insert_many(embedding_dicts)
            
            element_ids = [embedding.element_id for embedding in embeddings]
            logger.info(f"Inserted {len(element_ids)} embeddings")
            return element_ids
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            raise

    async def get_embeddings_by_document(self, document_id: str) -> List[VectorEmbedding]:
        try:
            cursor = self.embeddings.find({"document_id": document_id})
            embeddings = []
            async for embedding_data in cursor:
                embeddings.append(VectorEmbedding(**embedding_data))
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings for document {document_id}: {e}")
            raise

    # Query operations
    async def insert_query(self, query: UserQuery) -> str:
        try:
            await self.queries.insert_one(query.model_dump())
            logger.info(f"Inserted query: {query.query_id}")
            return query.query_id
        except Exception as e:
            logger.error(f"Failed to insert query: {e}")
            raise

    async def insert_response(self, response: QueryResponse) -> str:
        try:
            await self.responses.insert_one(response.model_dump())
            logger.info(f"Inserted response for query: {response.query_id}")
            return response.query_id
        except Exception as e:
            logger.error(f"Failed to insert response: {e}")
            raise

    async def insert_legacy_response(self, response: LegacyQueryResponse) -> str:
        """Insert using the old response format for backward compatibility"""
        try:
            await self.responses.insert_one(response.model_dump())
            logger.info(f"Inserted legacy response for query: {response.query_id}")
            return response.query_id
        except Exception as e:
            logger.error(f"Failed to insert legacy response: {e}")
            raise

    async def create_clean_response(
        self, 
        query_id: str, 
        query_text: str, 
        answer: str, 
        intent: str,
        primary_agent: str,
        sources: List[str] = None,
        detailed_sources: List[Dict[str, Any]] = None,
        reasoning: str = None,
        sub_agents: List[str] = None,
        processing_time: float = 0.0,
        model_used: str = None
    ) -> QueryResponse:
        """Create a clean, well-structured response object"""
        from database.models import QueryIntent
        
        # Parse intent
        try:
            parsed_intent = QueryIntent(intent.lower())
        except ValueError:
            parsed_intent = QueryIntent.UNKNOWN
        
        # Parse document-level sources (unique documents used)
        clean_sources = []
        document_ids_used = set()
        
        if sources:
            for source in sources:
                # Extract document identifier (could be filename or timestamp)
                if "(Page" in source:
                    doc_identifier = source.split(" (Page ")[0]
                else:
                    doc_identifier = source
                
                # Get actual document_id and name from database
                doc_id, doc_name = await self._resolve_document_info(doc_identifier)
                
                if doc_id not in document_ids_used:
                    # Get page count from document metadata
                    page_count = None
                    try:
                        document = await self.documents.find_one({"document_id": doc_id})
                        if document and document.get("metadata", {}).get("page_count"):
                            page_count = document["metadata"]["page_count"]
                    except Exception as e:
                        logger.warning(f"Could not get page count for document {doc_id}: {e}")
                    
                    clean_sources.append(SourceReference(
                        document_id=doc_id,
                        document_name=doc_name,
                        page_count=page_count
                    ))
                    document_ids_used.add(doc_id)
        
        # Parse detailed chunk sources
        clean_detailed_sources = []
        if detailed_sources:
            for detailed_source in detailed_sources:
                # Get document_id for this chunk
                element_id = detailed_source.get('element_id')
                chunk_document_id = detailed_source.get('document_id')
                
                if element_id and chunk_document_id:
                    # Resolve to actual document_id
                    actual_doc_id, _ = await self._resolve_document_info(chunk_document_id)
                    
                    clean_detailed_sources.append(DetailedSourceReference(
                        element_id=element_id,
                        document_id=actual_doc_id,
                        content_type=detailed_source.get('content_type'),
                        page_number=detailed_source.get('page_number'),
                        chunk_type=detailed_source.get('chunk_type')
                    ))
        
        return QueryResponse(
            query_id=query_id,
            query_text=query_text,
            answer=answer,
            intent=parsed_intent,
            reasoning=reasoning,
            primary_agent=primary_agent,
            sub_agents_used=sub_agents or [],
            sources=clean_sources,
            detailed_sources=clean_detailed_sources,
            processing_time=processing_time,
            model_used=model_used
        )
    
    async def _resolve_document_info(self, document_identifier: str) -> tuple[str, str]:
        """Get actual document_id and filename from any identifier"""
        try:
            # First try to find by document_id (timestamp)
            document = await self.documents.find_one({"document_id": document_identifier})
            if document and document.get("metadata", {}).get("filename"):
                return document["document_id"], document["metadata"]["filename"]
            
            # If not found, try to find by filename (direct match)
            document = await self.documents.find_one({"metadata.filename": document_identifier})
            if document and document.get("metadata", {}).get("filename"):
                return document["document_id"], document["metadata"]["filename"]
            
            # Try without file extension (handle cases like "1406.2661v1.pdf" -> "1406.2661v1")
            if "." in document_identifier:
                base_name = document_identifier.rsplit('.', 1)[0]
                document = await self.documents.find_one({"metadata.filename": {"$regex": f"^{base_name}\\.", "$options": "i"}})
                if document and document.get("metadata", {}).get("filename"):
                    return document["document_id"], document["metadata"]["filename"]
            
            # Try adding common extensions
            for ext in ['.pdf', '.txt', '.docx', '.doc']:
                document = await self.documents.find_one({"metadata.filename": f"{document_identifier}{ext}"})
                if document and document.get("metadata", {}).get("filename"):
                    return document["document_id"], document["metadata"]["filename"]
            
            # Fallback: return the identifier as both id and name
            logger.warning(f"Could not resolve document info for {document_identifier}, using fallback")
            return document_identifier, document_identifier
            
        except Exception as e:
            logger.warning(f"Could not resolve document info for {document_identifier}: {e}")
            return document_identifier, document_identifier

    async def _get_document_name(self, document_identifier: str) -> str:
        """Get actual document filename from document ID or filename (legacy method)"""
        _, doc_name = await self._resolve_document_info(document_identifier)
        return doc_name

# Global MongoDB client instance
mongodb_client = MongoDBClient()