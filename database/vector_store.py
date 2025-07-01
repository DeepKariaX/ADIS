import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from config.settings import get_settings
from database.models import VectorEmbedding, ContentType
from utils.logger import logger

class VectorStore:
    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()

    def _initialize(self):
        try:
            # Initialize ChromaDB
            persist_directory = self.settings.vector_db_path
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=self.settings.vector_collection_name,
                metadata={"hnsw:space": self.settings.vector_space_type}
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.settings.embedding_model)
            
            logger.info(f"Initialized vector store with {self.collection.count()} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to only include ChromaDB-compatible types."""
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = value
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples to strings
                sanitized[key] = str(value)
            elif isinstance(value, dict):
                # Convert dictionaries to JSON strings
                import json
                try:
                    sanitized[key] = json.dumps(value)
                except (TypeError, ValueError):
                    sanitized[key] = str(value)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized

    def add_embeddings(self, embeddings: List[VectorEmbedding]) -> bool:
        try:
            if not embeddings:
                return True
            
            # Prepare data for ChromaDB
            ids = [emb.element_id for emb in embeddings]
            vectors = [emb.embedding for emb in embeddings]
            documents = [emb.text_content for emb in embeddings]
            metadatas = []
            
            for emb in embeddings:
                metadata = {
                    "document_id": emb.document_id,
                    "content_type": emb.content_type.value,
                    "created_at": emb.created_at.isoformat(),
                    **emb.metadata
                }
                # Sanitize metadata to ensure ChromaDB compatibility
                sanitized_metadata = self._sanitize_metadata(metadata)
                metadatas.append(sanitized_metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(embeddings)} embeddings to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise

    def search_similar(
        self, 
        query_text: str, 
        top_k: int = 10,
        content_types: Optional[List[ContentType]] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query_text])[0]
            
            # Prepare filter conditions
            where_conditions = {}
            if content_types:
                where_conditions["content_type"] = {"$in": [ct.value for ct in content_types]}
            if document_ids:
                where_conditions["document_id"] = {"$in": document_ids}
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_conditions if where_conditions else None
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = {
                        "element_id": results['ids'][0][i],
                        "text_content": results['documents'][0][i],
                        "distance": results['distances'][0][i] if results['distances'] else None,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar embeddings")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            raise

    def search_by_content_type(
        self, 
        content_type: ContentType, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        try:
            results = self.collection.get(
                where={"content_type": content_type.value},
                limit=limit
            )
            
            formatted_results = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    result = {
                        "element_id": results['ids'][i],
                        "text_content": results['documents'][i] if results['documents'] else "",
                        "metadata": results['metadatas'][i] if results['metadatas'] else {}
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} embeddings of type {content_type.value}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search by content type: {e}")
            raise

    def delete_document_embeddings(self, document_id: str) -> bool:
        try:
            # Get all embeddings for this document
            results = self.collection.get(where={"document_id": document_id})
            
            if results['ids']:
                # Delete embeddings
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} embeddings for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings for document {document_id}: {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            
            # Get distribution by content type
            content_type_counts = {}
            for content_type in ContentType:
                results = self.collection.get(where={"content_type": content_type.value}, limit=1)
                if results['ids']:
                    type_count = len(self.collection.get(where={"content_type": content_type.value})['ids'])
                    content_type_counts[content_type.value] = type_count
            
            return {
                "total_embeddings": count,
                "content_type_distribution": content_type_counts
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            raise

# Global vector store instance
vector_store = VectorStore()