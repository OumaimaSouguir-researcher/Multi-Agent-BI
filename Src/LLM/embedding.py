import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import asyncio
import aiohttp

from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingGenerator:
    """
    Generate and manage text embeddings using Ollama
    Supports embedding generation, similarity search, and vector operations
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        self.base_url = base_url or settings.OLLAMA_HOST
        self.model = model or settings.OLLAMA_EMBEDDING_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_enabled = True
        
        # Statistics
        self.stats = {
            "embeddings_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def connect(self):
        """Create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"Embedding generator connected to {self.base_url}")
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Embedding generator disconnected")
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            model: Model to use (default: self.model)
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector as numpy array
        """
        if not text:
            logger.warning("Empty text provided for embedding")
            return None
        
        # Check cache
        if use_cache and self._cache_enabled and text in self._embedding_cache:
            self.stats["cache_hits"] += 1
            return self._embedding_cache[text]
        
        self.stats["cache_misses"] += 1
        
        if not self.session:
            await self.connect()
        
        payload = {
            "model": model or self.model,
            "prompt": text
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = np.array(result.get("embedding", []))
                    
                    if embedding.size > 0:
                        self.stats["embeddings_generated"] += 1
                        
                        # Cache the embedding
                        if use_cache and self._cache_enabled:
                            self._embedding_cache[text] = embedding
                        
                        return embedding
                    else:
                        logger.error("Empty embedding returned")
                        return None
                else:
                    error_text = await response.text()
                    logger.error(f"Embedding generation failed: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        max_concurrent: int = 5
    ) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts concurrently
        
        Args:
            texts: List of input texts
            model: Model to use
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of embedding vectors
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(text: str):
            async with semaphore:
                return await self.generate_embedding(text, model)
        
        tasks = [generate_with_semaphore(text) for text in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, Exception):
                logger.error(f"Failed to generate embedding for text {i}: {emb}")
                results.append(None)
            else:
                results.append(emb)
        
        return results
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score between -1 and 1
        """
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    @staticmethod
    def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Distance value
        """
        if vec1 is None or vec2 is None:
            return float('inf')
        
        return float(np.linalg.norm(vec1 - vec2))
    
    async def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find most similar texts to a query
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar texts with scores
        """
        # Generate query embedding
        query_emb = await self.generate_embedding(query)
        if query_emb is None:
            return []
        
        # Generate candidate embeddings
        candidate_embs = await self.generate_embeddings_batch(candidates)
        
        # Calculate similarities
        similarities = []
        for i, (text, emb) in enumerate(zip(candidates, candidate_embs)):
            if emb is not None:
                sim = self.cosine_similarity(query_emb, emb)
                if sim >= threshold:
                    similarities.append({
                        "text": text,
                        "index": i,
                        "similarity": sim
                    })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    async def semantic_search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search over documents
        
        Args:
            query: Search query
            documents: List of document dicts
            text_field: Field containing text to search
            top_k: Number of results
            
        Returns:
            Top matching documents with scores
        """
        # Extract texts from documents
        texts = [doc.get(text_field, "") for doc in documents]
        
        # Find similar texts
        similar = await self.find_similar(query, texts, top_k=top_k)
        
        # Map back to documents
        results = []
        for item in similar:
            doc = documents[item["index"]].copy()
            doc["similarity_score"] = item["similarity"]
            results.append(doc)
        
        return results
    
    def cluster_embeddings(
        self,
        embeddings: List[np.ndarray],
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """
        Cluster embeddings using K-means
        
        Args:
            embeddings: List of embedding vectors
            n_clusters: Number of clusters
            
        Returns:
            Clustering results
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.error("scikit-learn not installed, cannot perform clustering")
            return {"error": "scikit-learn required for clustering"}
        
        # Filter out None embeddings
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        
        if len(valid_embeddings) < n_clusters:
            return {
                "error": f"Not enough embeddings ({len(valid_embeddings)}) for {n_clusters} clusters"
            }
        
        # Stack embeddings into matrix
        X = np.vstack(valid_embeddings)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Calculate cluster statistics
        clusters = {}
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            clusters[i] = {
                "size": len(cluster_indices),
                "indices": cluster_indices.tolist(),
                "center": kmeans.cluster_centers_[i].tolist()
            }
        
        return {
            "n_clusters": n_clusters,
            "clusters": clusters,
            "inertia": float(kmeans.inertia_)
        }
    
    def reduce_dimensions(
        self,
        embeddings: List[np.ndarray],
        n_components: int = 2
    ) -> Optional[np.ndarray]:
        """
        Reduce embedding dimensionality for visualization
        
        Args:
            embeddings: List of embedding vectors
            n_components: Target dimensions (2 or 3)
            
        Returns:
            Reduced embeddings matrix
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            logger.error("scikit-learn not installed, cannot reduce dimensions")
            return None
        
        # Filter out None embeddings
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        
        if len(valid_embeddings) < n_components:
            logger.error("Not enough embeddings for dimensionality reduction")
            return None
        
        # Stack embeddings
        X = np.vstack(valid_embeddings)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(X)
        
        logger.info(f"Reduced dimensions from {X.shape[1]} to {n_components}")
        logger.info(f"Explained variance: {sum(pca.explained_variance_ratio_):.2%}")
        
        return reduced
    
    def clear_cache(self):
        """Clear embedding cache"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def enable_cache(self):
        """Enable embedding cache"""
        self._cache_enabled = True
    
    def disable_cache(self):
        """Disable embedding cache"""
        self._cache_enabled = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        cache_size = len(self._embedding_cache)
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = (self.stats["cache_hits"] / total_requests) * 100
        
        return {
            **self.stats,
            "cache_size": cache_size,
            "cache_hit_rate": f"{hit_rate:.2f}%",
            "cache_enabled": self._cache_enabled
        }