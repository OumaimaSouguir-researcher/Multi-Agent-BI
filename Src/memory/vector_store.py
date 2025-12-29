import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import pickle
from pathlib import Path
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Store vectoriel pour stocker et rechercher des embeddings
    Supporte la recherche de similarité, le clustering et la persistance
    """
    
    def __init__(
        self,
        dimension: int = 768,
        metric: str = "cosine",
        persist_path: Optional[str] = None
    ):
        """
        Args:
            dimension: Dimension des vecteurs
            metric: Métrique de distance (cosine, euclidean, dot)
            persist_path: Chemin pour la persistance
        """
        self.dimension = dimension
        self.metric = metric
        self.persist_path = persist_path
        
        # Stockage principal
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        
        # Index pour recherche rapide
        self._id_to_index: Dict[str, int] = {}
        
        # Statistiques
        self.stats = {
            "total_vectors": 0,
            "total_searches": 0,
            "last_updated": None
        }
        
        # Charger depuis le disque si le chemin existe
        if persist_path and Path(persist_path).exists():
            self.load()
    
    def add(
        self,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """
        Ajouter un vecteur au store
        
        Args:
            vector: Vecteur d'embedding
            metadata: Métadonnées associées
            id: ID unique (généré automatiquement si None)
            
        Returns:
            ID du vecteur ajouté
        """
        # Valider la dimension
        if len(vector) != self.dimension:
            raise ValueError(
                f"Dimension du vecteur ({len(vector)}) ne correspond pas "
                f"à la dimension du store ({self.dimension})"
            )
        
        # Générer un ID si nécessaire
        if id is None:
            id = f"vec_{len(self.vectors)}_{datetime.now().timestamp()}"
        
        # Vérifier si l'ID existe déjà
        if id in self._id_to_index:
            logger.warning(f"ID {id} existe déjà, mise à jour du vecteur")
            return self.update(id, vector, metadata)
        
        # Ajouter le vecteur
        index = len(self.vectors)
        self.vectors.append(vector)
        self.metadata.append(metadata or {})
        self.ids.append(id)
        self._id_to_index[id] = index
        
        # Mettre à jour les stats
        self.stats["total_vectors"] += 1
        self.stats["last_updated"] = datetime.now().isoformat()
        
        logger.debug(f"Vecteur ajouté avec ID: {id}")
        return id
    
    def add_batch(
        self,
        vectors: List[np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Ajouter plusieurs vecteurs en batch
        
        Args:
            vectors: Liste de vecteurs
            metadata_list: Liste de métadonnées
            ids: Liste d'IDs
            
        Returns:
            Liste des IDs ajoutés
        """
        if metadata_list is None:
            metadata_list = [None] * len(vectors)
        
        if ids is None:
            ids = [None] * len(vectors)
        
        added_ids = []
        for vector, metadata, id in zip(vectors, metadata_list, ids):
            added_id = self.add(vector, metadata, id)
            added_ids.append(added_id)
        
        logger.info(f"Batch de {len(vectors)} vecteurs ajouté")
        return added_ids
    
    def get(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Récupérer un vecteur par son ID
        
        Returns:
            Tuple (vector, metadata) ou None si non trouvé
        """
        if id not in self._id_to_index:
            return None
        
        index = self._id_to_index[id]
        return self.vectors[index], self.metadata[index]
    
    def update(
        self,
        id: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mettre à jour un vecteur existant
        
        Returns:
            True si mis à jour, False si non trouvé
        """
        if id not in self._id_to_index:
            logger.warning(f"ID {id} non trouvé pour mise à jour")
            return False
        
        index = self._id_to_index[id]
        
        if vector is not None:
            if len(vector) != self.dimension:
                raise ValueError("Dimension du vecteur incorrecte")
            self.vectors[index] = vector
        
        if metadata is not None:
            self.metadata[index].update(metadata)
        
        self.stats["last_updated"] = datetime.now().isoformat()
        logger.debug(f"Vecteur {id} mis à jour")
        return True
    
    def delete(self, id: str) -> bool:
        """
        Supprimer un vecteur
        
        Returns:
            True si supprimé, False si non trouvé
        """
        if id not in self._id_to_index:
            return False
        
        index = self._id_to_index[id]
        
        # Supprimer des listes
        del self.vectors[index]
        del self.metadata[index]
        del self.ids[index]
        
        # Reconstruire l'index
        self._rebuild_index()
        
        self.stats["total_vectors"] -= 1
        self.stats["last_updated"] = datetime.now().isoformat()
        
        logger.debug(f"Vecteur {id} supprimé")
        return True
    
    def _rebuild_index(self):
        """Reconstruire l'index ID -> position"""
        self._id_to_index = {id: i for i, id in enumerate(self.ids)}
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Rechercher les vecteurs les plus similaires
        
        Args:
            query_vector: Vecteur de requête
            top_k: Nombre de résultats à retourner
            filter_metadata: Filtrer par métadonnées
            threshold: Seuil de similarité minimum
            
        Returns:
            Liste de résultats avec scores et métadonnées
        """
        if len(self.vectors) == 0:
            return []
        
        self.stats["total_searches"] += 1
        
        # Calculer les similarités
        similarities = []
        for i, vector in enumerate(self.vectors):
            # Appliquer le filtre de métadonnées
            if filter_metadata:
                if not self._matches_filter(self.metadata[i], filter_metadata):
                    continue
            
            # Calculer la similarité
            score = self._compute_similarity(query_vector, vector)
            
            # Appliquer le seuil
            if threshold is not None and score < threshold:
                continue
            
            similarities.append({
                "id": self.ids[i],
                "score": float(score),
                "metadata": self.metadata[i],
                "vector": vector
            })
        
        # Trier par score décroissant
        similarities.sort(key=lambda x: x["score"], reverse=True)
        
        # Retourner top_k résultats
        return similarities[:top_k]
    
    def _compute_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Calculer la similarité entre deux vecteurs"""
        if self.metric == "cosine":
            # Similarité cosinus
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        
        elif self.metric == "euclidean":
            # Distance euclidienne (convertie en similarité)
            dist = np.linalg.norm(vec1 - vec2)
            return float(1.0 / (1.0 + dist))
        
        elif self.metric == "dot":
            # Produit scalaire
            return float(np.dot(vec1, vec2))
        
        else:
            raise ValueError(f"Métrique inconnue: {self.metric}")
    
    def _matches_filter(
        self,
        metadata: Dict[str, Any],
        filter_dict: Dict[str, Any]
    ) -> bool:
        """Vérifier si les métadonnées correspondent au filtre"""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    def get_all(
        self,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Récupérer tous les vecteurs (avec filtre optionnel)
        
        Returns:
            Liste de dicts avec id, vector, metadata
        """
        results = []
        for i, (id, vector, metadata) in enumerate(
            zip(self.ids, self.vectors, self.metadata)
        ):
            if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                continue
            
            results.append({
                "id": id,
                "vector": vector,
                "metadata": metadata
            })
        
        return results
    
    def cluster(
        self,
        n_clusters: int = 5,
        method: str = "kmeans"
    ) -> Dict[str, Any]:
        """
        Clustériser les vecteurs
        
        Args:
            n_clusters: Nombre de clusters
            method: Méthode de clustering (kmeans, hierarchical)
            
        Returns:
            Résultats de clustering
        """
        if len(self.vectors) < n_clusters:
            return {
                "error": f"Pas assez de vecteurs ({len(self.vectors)}) pour {n_clusters} clusters"
            }
        
        try:
            if method == "kmeans":
                from sklearn.cluster import KMeans
                
                X = np.vstack(self.vectors)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(X)
                
                # Organiser par clusters
                clusters = {}
                for i, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append({
                        "id": self.ids[i],
                        "metadata": self.metadata[i]
                    })
                
                return {
                    "method": method,
                    "n_clusters": n_clusters,
                    "clusters": clusters,
                    "centroids": kmeans.cluster_centers_.tolist(),
                    "inertia": float(kmeans.inertia_)
                }
            
            elif method == "hierarchical":
                from sklearn.cluster import AgglomerativeClustering
                
                X = np.vstack(self.vectors)
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering.fit_predict(X)
                
                clusters = {}
                for i, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append({
                        "id": self.ids[i],
                        "metadata": self.metadata[i]
                    })
                
                return {
                    "method": method,
                    "n_clusters": n_clusters,
                    "clusters": clusters
                }
            
            else:
                return {"error": f"Méthode inconnue: {method}"}
        
        except ImportError:
            logger.error("scikit-learn requis pour le clustering")
            return {"error": "scikit-learn non installé"}
    
    def save(self, path: Optional[str] = None):
        """
        Sauvegarder le store sur disque
        
        Args:
            path: Chemin de sauvegarde (utilise self.persist_path si None)
        """
        path = path or self.persist_path
        if not path:
            raise ValueError("Aucun chemin de persistance fourni")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "dimension": self.dimension,
            "metric": self.metric,
            "vectors": [v.tolist() for v in self.vectors],
            "metadata": self.metadata,
            "ids": self.ids,
            "stats": self.stats
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Vector store sauvegardé dans {path}")
    
    def load(self, path: Optional[str] = None):
        """
        Charger le store depuis le disque
        
        Args:
            path: Chemin de chargement (utilise self.persist_path si None)
        """
        path = path or self.persist_path
        if not path:
            raise ValueError("Aucun chemin de persistance fourni")
        
        path = Path(path)
        if not path.exists():
            logger.warning(f"Fichier {path} non trouvé")
            return
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data["dimension"]
        self.metric = data["metric"]
        self.vectors = [np.array(v) for v in data["vectors"]]
        self.metadata = data["metadata"]
        self.ids = data["ids"]
        self.stats = data.get("stats", self.stats)
        
        # Reconstruire l'index
        self._rebuild_index()
        
        logger.info(f"Vector store chargé depuis {path} ({len(self.vectors)} vecteurs)")
    
    def clear(self):
        """Effacer tous les vecteurs"""
        self.vectors.clear()
        self.metadata.clear()
        self.ids.clear()
        self._id_to_index.clear()
        self.stats["total_vectors"] = 0
        self.stats["last_updated"] = datetime.now().isoformat()
        logger.info("Vector store effacé")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtenir les statistiques du store"""
        return {
            **self.stats,
            "dimension": self.dimension,
            "metric": self.metric,
            "current_size": len(self.vectors),
            "memory_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimer l'utilisation mémoire en MB"""
        if not self.vectors:
            return 0.0
        
        # Taille d'un vecteur
        vector_size = self.vectors[0].nbytes
        total_vectors_size = vector_size * len(self.vectors)
        
        # Estimation grossière pour métadonnées
        metadata_size = len(json.dumps(self.metadata).encode('utf-8'))
        
        total_bytes = total_vectors_size + metadata_size
        return total_bytes / (1024 * 1024)  # Convertir en MB
    
    def __len__(self) -> int:
        """Nombre de vecteurs dans le store"""
        return len(self.vectors)
    
    def __contains__(self, id: str) -> bool:
        """Vérifier si un ID existe"""
        return id in self._id_to_index
    
    def __repr__(self) -> str:
        return (
            f"VectorStore(dimension={self.dimension}, "
            f"metric={self.metric}, "
            f"size={len(self.vectors)})"
        )