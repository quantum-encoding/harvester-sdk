"""
Google AI Studio (GenAI) Embeddings Provider
Text embeddings for semantic search, RAG, classification, and clustering

Models available:
- gemini-embedding-001 (Stable)
- gemini-embedding-exp-03-07 (Experimental)
"""

import logging
import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import os
from enum import Enum
from dataclasses import dataclass

try:
    from google import genai
    from google.genai import types
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    genai = None
    types = None
    cosine_similarity = None
    
from .base_provider import BaseProvider

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Supported embedding task types"""
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"

@dataclass
class EmbeddingResult:
    """Container for embedding results"""
    text: str
    embedding: np.ndarray
    dimension: int
    normalized: bool = False

class GenAIEmbeddingsProvider(BaseProvider):
    """
    Google AI Studio Embeddings Provider
    
    This provider uses Google's Gemini embedding models for generating
    high-quality text embeddings for various NLP tasks.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        if genai is None:
            raise ImportError(
                "Google GenAI not installed. Install with: pip install google-generativeai scikit-learn"
            )
        
        # API key from environment or config
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Google GenAI API key required. Set GOOGLE_GENAI_API_KEY environment variable")
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        
        # Model configurations
        self.model_configs = {
            "gemini-embedding-001": {
                "name": "Gemini Embedding 001 (Stable)",
                "max_input_tokens": 2048,
                "supported_dimensions": [128, 256, 512, 768, 1536, 2048, 3072],
                "recommended_dimensions": [768, 1536, 3072],
                "default_dimension": 3072,
                "version": "stable"
            },
            "gemini-embedding-exp-03-07": {
                "name": "Gemini Embedding Experimental",
                "max_input_tokens": 2048,
                "supported_dimensions": [128, 256, 512, 768, 1536, 2048, 3072],
                "recommended_dimensions": [768, 1536, 3072],
                "default_dimension": 3072,
                "version": "experimental"
            }
        }
        
        self.default_model = config.get("default_model", "gemini-embedding-001")
        
        # Task type descriptions for user guidance
        self.task_descriptions = {
            TaskType.SEMANTIC_SIMILARITY: "Assess text similarity for recommendations, duplicate detection",
            TaskType.CLASSIFICATION: "Classify texts for sentiment analysis, spam detection",
            TaskType.CLUSTERING: "Group texts for document organization, anomaly detection",
            TaskType.RETRIEVAL_DOCUMENT: "Index documents for search (use for documents)",
            TaskType.RETRIEVAL_QUERY: "General search queries (use for queries)",
            TaskType.CODE_RETRIEVAL_QUERY: "Natural language queries for code search",
            TaskType.QUESTION_ANSWERING: "Find documents that answer questions",
            TaskType.FACT_VERIFICATION: "Retrieve evidence for fact-checking"
        }
        
        logger.info(f"GenAI Embeddings provider initialized with {len(self.model_configs)} models")
    
    async def embed_text(
        self,
        text: Union[str, List[str]],
        model: str = None,
        task_type: Union[str, TaskType] = None,
        output_dimensionality: int = None,
        normalize: bool = True,
        **kwargs
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Generate embeddings for text
        
        Args:
            text: Single text or list of texts to embed
            model: Model to use (gemini-embedding-001 or experimental)
            task_type: Task type for optimization (see TaskType enum)
            output_dimensionality: Size of embedding vector (128-3072)
            normalize: Whether to normalize embeddings (recommended)
            **kwargs: Additional parameters
        
        Returns:
            EmbeddingResult or list of EmbeddingResults
        """
        model = model or self.default_model
        
        if model not in self.model_configs:
            logger.warning(f"Unknown model {model}, using default")
            model = self.default_model
        
        config_info = self.model_configs[model]
        
        # Handle single vs multiple texts
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Validate dimensionality
        if output_dimensionality:
            if output_dimensionality not in config_info["supported_dimensions"]:
                logger.warning(f"Dimension {output_dimensionality} not in supported list. Using default.")
                output_dimensionality = None
        
        output_dimensionality = output_dimensionality or config_info["default_dimension"]
        
        try:
            # Build config
            embed_config = {}
            
            if task_type:
                # Convert string to TaskType if needed
                if isinstance(task_type, str):
                    task_type = TaskType[task_type]
                embed_config["task_type"] = task_type.value
            
            if output_dimensionality != config_info["default_dimension"]:
                embed_config["output_dimensionality"] = output_dimensionality
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} text(s) with dimension {output_dimensionality}")
            
            if embed_config:
                config_obj = types.EmbedContentConfig(**embed_config)
                result = await asyncio.to_thread(
                    self.client.models.embed_content,
                    model=model,
                    contents=texts,
                    config=config_obj
                )
            else:
                result = await asyncio.to_thread(
                    self.client.models.embed_content,
                    model=model,
                    contents=texts
                )
            
            # Process results
            embedding_results = []
            
            for i, embedding_obj in enumerate(result.embeddings):
                # Convert to numpy array
                embedding_array = np.array(embedding_obj.values)
                
                # Normalize if requested and not default dimension
                if normalize and output_dimensionality != 3072:
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm
                
                embedding_results.append(
                    EmbeddingResult(
                        text=texts[i],
                        embedding=embedding_array,
                        dimension=len(embedding_array),
                        normalized=normalize
                    )
                )
            
            logger.info(f"Successfully generated {len(embedding_results)} embeddings")
            
            # Return single result if single input
            return embedding_results[0] if is_single else embedding_results
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    async def embed_for_rag(
        self,
        documents: List[str],
        queries: List[str] = None,
        model: str = None,
        output_dimensionality: int = 768,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate embeddings optimized for RAG systems
        
        Args:
            documents: List of documents to index
            queries: Optional list of queries to embed
            model: Model to use
            output_dimensionality: Embedding size (768 recommended for RAG)
            **kwargs: Additional parameters
        
        Returns:
            Dict with document and query embeddings
        """
        result = {"success": True}
        
        try:
            # Embed documents with RETRIEVAL_DOCUMENT task
            logger.info(f"Embedding {len(documents)} documents for RAG")
            
            doc_embeddings = await self.embed_text(
                text=documents,
                model=model,
                task_type=TaskType.RETRIEVAL_DOCUMENT,
                output_dimensionality=output_dimensionality,
                normalize=True
            )
            
            result["document_embeddings"] = doc_embeddings
            result["document_count"] = len(documents)
            
            # Embed queries if provided
            if queries:
                logger.info(f"Embedding {len(queries)} queries for RAG")
                
                query_embeddings = await self.embed_text(
                    text=queries,
                    model=model,
                    task_type=TaskType.RETRIEVAL_QUERY,
                    output_dimensionality=output_dimensionality,
                    normalize=True
                )
                
                result["query_embeddings"] = query_embeddings
                result["query_count"] = len(queries)
            
            result["dimension"] = output_dimensionality
            result["model"] = model or self.default_model
            
            return result
            
        except Exception as e:
            logger.error(f"RAG embedding failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def semantic_search(
        self,
        query: str,
        documents: List[str],
        model: str = None,
        top_k: int = 5,
        output_dimensionality: int = 768,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on documents
        
        Args:
            query: Search query
            documents: List of documents to search
            model: Model to use
            top_k: Number of top results to return
            output_dimensionality: Embedding dimension
            **kwargs: Additional parameters
        
        Returns:
            List of top matching documents with scores
        """
        try:
            # Generate query embedding
            query_embedding = await self.embed_text(
                text=query,
                model=model,
                task_type=TaskType.RETRIEVAL_QUERY,
                output_dimensionality=output_dimensionality,
                normalize=True
            )
            
            # Generate document embeddings
            doc_embeddings = await self.embed_text(
                text=documents,
                model=model,
                task_type=TaskType.RETRIEVAL_DOCUMENT,
                output_dimensionality=output_dimensionality,
                normalize=True
            )
            
            # Calculate similarities
            query_vec = query_embedding.embedding.reshape(1, -1)
            doc_matrix = np.array([d.embedding for d in doc_embeddings])
            
            if cosine_similarity:
                similarities = cosine_similarity(query_vec, doc_matrix)[0]
            else:
                # Manual cosine similarity if sklearn not available
                similarities = np.dot(doc_matrix, query_vec.T).flatten()
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    "document": documents[idx],
                    "score": float(similarities[idx]),
                    "rank": len(results) + 1
                })
            
            logger.info(f"Semantic search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def calculate_similarity(
        self,
        texts: List[str],
        model: str = None,
        output_dimensionality: int = 768,
        return_matrix: bool = False,
        **kwargs
    ) -> Union[Dict[str, float], np.ndarray]:
        """
        Calculate pairwise similarity between texts
        
        Args:
            texts: List of texts to compare
            model: Model to use
            output_dimensionality: Embedding dimension
            return_matrix: Return full similarity matrix if True
            **kwargs: Additional parameters
        
        Returns:
            Similarity scores as dict or matrix
        """
        try:
            # Generate embeddings
            embeddings = await self.embed_text(
                text=texts,
                model=model,
                task_type=TaskType.SEMANTIC_SIMILARITY,
                output_dimensionality=output_dimensionality,
                normalize=True
            )
            
            # Create embedding matrix
            embedding_matrix = np.array([e.embedding for e in embeddings])
            
            # Calculate similarity matrix
            if cosine_similarity:
                similarity_matrix = cosine_similarity(embedding_matrix)
            else:
                # Manual calculation
                norm_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
                similarity_matrix = np.dot(norm_matrix, norm_matrix.T)
            
            if return_matrix:
                return similarity_matrix
            
            # Convert to pairwise dict
            results = {}
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    key = f"{i}_{j}"
                    results[key] = {
                        "text1": texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i],
                        "text2": texts[j][:50] + "..." if len(texts[j]) > 50 else texts[j],
                        "similarity": float(similarity_matrix[i, j])
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return {} if not return_matrix else np.array([])
    
    async def cluster_texts(
        self,
        texts: List[str],
        n_clusters: int = 5,
        model: str = None,
        output_dimensionality: int = 768,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Cluster texts into groups
        
        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters
            model: Model to use
            output_dimensionality: Embedding dimension
            **kwargs: Additional parameters
        
        Returns:
            Dict with cluster assignments and centroids
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.error("sklearn required for clustering. Install with: pip install scikit-learn")
            return {"success": False, "error": "sklearn not installed"}
        
        try:
            # Generate embeddings
            embeddings = await self.embed_text(
                text=texts,
                model=model,
                task_type=TaskType.CLUSTERING,
                output_dimensionality=output_dimensionality,
                normalize=True
            )
            
            # Create embedding matrix
            embedding_matrix = np.array([e.embedding for e in embeddings])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42)
            cluster_labels = kmeans.fit_predict(embedding_matrix)
            
            # Organize results by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    "text": texts[i],
                    "index": i
                })
            
            # Calculate cluster centroids
            centroids = kmeans.cluster_centers_
            
            logger.info(f"Clustered {len(texts)} texts into {len(clusters)} clusters")
            
            return {
                "success": True,
                "clusters": clusters,
                "n_clusters": len(clusters),
                "centroids": centroids.tolist(),
                "labels": cluster_labels.tolist()
            }
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_info(self, model: str = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        model = model or self.default_model
        
        if model in self.model_configs:
            return self.model_configs[model]
        else:
            return {
                "error": f"Unknown model: {model}",
                "available_models": list(self.model_configs.keys())
            }
    
    def get_task_types(self) -> Dict[str, str]:
        """Get available task types with descriptions"""
        return {
            task.name: desc 
            for task, desc in self.task_descriptions.items()
        }
    
    async def complete(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Compatibility method for BaseProvider interface
        Returns embedding as string representation
        """
        result = await self.embed_text(prompt, model, **kwargs)
        
        if isinstance(result, EmbeddingResult):
            return f"Embedding generated: dimension={result.dimension}, normalized={result.normalized}"
        else:
            return f"Generated {len(result)} embeddings"


# Example usage
if __name__ == "__main__":
    async def test_embeddings():
        config = {
            "api_key": "YOUR_API_KEY"  # Or set GOOGLE_GENAI_API_KEY env var
        }
        
        provider = GenAIEmbeddingsProvider(config)
        
        # Test 1: Simple embedding
        result = await provider.embed_text(
            text="What is the meaning of life?",
            output_dimensionality=768,
            normalize=True
        )
        
        print(f"✅ Generated embedding: dimension={result.dimension}")
        print(f"   Normalized: {result.normalized}")
        print(f"   First 5 values: {result.embedding[:5]}")
        
        # Test 2: Semantic similarity
        texts = [
            "What is the meaning of life?",
            "What is the purpose of existence?",
            "How do I bake a cake?"
        ]
        
        similarities = await provider.calculate_similarity(
            texts=texts,
            output_dimensionality=768
        )
        
        print("\n✅ Semantic Similarities:")
        for key, data in similarities.items():
            print(f"   '{data['text1']}' vs '{data['text2']}': {data['similarity']:.4f}")
        
        # Test 3: Semantic search
        documents = [
            "The meaning of life is to find happiness.",
            "Chocolate cake recipe: mix flour, sugar, eggs, and cocoa.",
            "The purpose of existence is a philosophical question.",
            "Life's meaning varies for each individual.",
            "Baking requires precise measurements and timing."
        ]
        
        results = await provider.semantic_search(
            query="What is life about?",
            documents=documents,
            top_k=3
        )
        
        print("\n✅ Semantic Search Results:")
        for result in results:
            print(f"   Rank {result['rank']}: {result['document'][:50]}... (score: {result['score']:.4f})")
        
        # Test 4: RAG embeddings
        rag_result = await provider.embed_for_rag(
            documents=documents,
            queries=["What is life?", "How to bake?"],
            output_dimensionality=768
        )
        
        if rag_result["success"]:
            print(f"\n✅ RAG Embeddings:")
            print(f"   Documents embedded: {rag_result['document_count']}")
            print(f"   Queries embedded: {rag_result.get('query_count', 0)}")
            print(f"   Dimension: {rag_result['dimension']}")
    
    # Run test
    import asyncio
    asyncio.run(test_embeddings())