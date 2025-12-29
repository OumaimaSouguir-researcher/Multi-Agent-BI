"""
Long-Term Memory (LTM)
---------------------
Mémoire long terme pour agents LLM avec recherche sémantique,
graphe de connaissances et consolidation automatique.

Knowledge Types:
- Facts (factual knowledge with confidence)
- Procedures (skills with success rate)
- Concepts (abstract notions with properties)
- Relations (links between knowledge nodes)

Features:
- Recherche sémantique vectorielle (embedding mockable)
- Consolidation (renforcement / oubli)
- Graphe de connaissances
- Export JSON
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import uuid
import time
import json
import math


# --------------------------------------------------
# Base Knowledge Node
# --------------------------------------------------
@dataclass
class KnowledgeNode:
    id: str
    node_type: str
    content: str
    embedding: List[float]
    strength: float
    created_at: float
    last_accessed: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reinforce(self, amount: float = 0.05):
        self.strength = min(1.0, self.strength + amount)
        self.last_accessed = time.time()

    def decay(self, rate: float):
        self.strength = max(0.0, self.strength - rate)


# --------------------------------------------------
# Knowledge Graph Relation
# --------------------------------------------------
@dataclass
class KnowledgeRelation:
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 0.5


# --------------------------------------------------
# Long Term Memory
# --------------------------------------------------
class LongTermMemory:
    def __init__(
        self,
        decay_rate: float = 0.001,
        reinforce_threshold: float = 0.75,
    ):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.relations: List[KnowledgeRelation] = []

        self.decay_rate = decay_rate
        self.reinforce_threshold = reinforce_threshold

    # --------------------------------------------------
    # Add Knowledge
    # --------------------------------------------------
    def add_fact(
        self,
        content: str,
        confidence: float = 0.8,
    ) -> str:
        return self._add_node(
            node_type="fact",
            content=content,
            strength=confidence,
            metadata={"confidence": confidence},
        )

    def add_procedure(
        self,
        content: str,
        success_rate: float = 0.6,
    ) -> str:
        return self._add_node(
            node_type="procedure",
            content=content,
            strength=success_rate,
            metadata={"success_rate": success_rate},
        )

    def add_concept(
        self,
        content: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._add_node(
            node_type="concept",
            content=content,
            strength=0.5,
            metadata={"properties": properties or {}},
        )

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 0.5,
    ):
        if source_id in self.nodes and target_id in self.nodes:
            self.relations.append(
                KnowledgeRelation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    weight=weight,
                )
            )

    def _add_node(
        self,
        node_type: str,
        content: str,
        strength: float,
        metadata: Dict[str, Any],
    ) -> str:
        node_id = str(uuid.uuid4())
        embedding = self._embed(content)

        now = time.time()
        self.nodes[node_id] = KnowledgeNode(
            id=node_id,
            node_type=node_type,
            content=content,
            embedding=embedding,
            strength=max(0.0, min(1.0, strength)),
            created_at=now,
            last_accessed=now,
            metadata=metadata,
        )
        return node_id

    # --------------------------------------------------
    # Semantic Search
    # --------------------------------------------------
    def semantic_search(
        self,
        query: str,
        node_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[KnowledgeNode]:
        query_vec = self._embed(query)
        scored = []

        for node in self.nodes.values():
            if node_type and node.node_type != node_type:
                continue

            similarity = self._cosine_similarity(
                query_vec, node.embedding
            )

            score = similarity * node.strength
            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, node in scored[:top_k]:
            node.last_accessed = time.time()
            if score >= self.reinforce_threshold:
                node.reinforce()
            results.append(node)

        return results

    # --------------------------------------------------
    # Consolidation (Reinforcement & Forgetting)
    # --------------------------------------------------
    def consolidate(self):
        now = time.time()
        for node in self.nodes.values():
            time_delta = now - node.last_accessed
            decay_amount = time_delta * self.decay_rate
            node.decay(decay_amount)

    # --------------------------------------------------
    # Knowledge Graph Utilities
    # --------------------------------------------------
    def get_neighbors(
        self,
        node_id: str,
        relation_type: Optional[str] = None,
    ) -> List[KnowledgeNode]:
        neighbors = []
        for rel in self.relations:
            if rel.source_id == node_id:
                if relation_type and rel.relation_type != relation_type:
                    continue
                target = self.nodes.get(rel.target_id)
                if target:
                    neighbors.append(target)
        return neighbors

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    def export_json(self, path: str):
        data = {
            "nodes": {
                node_id: asdict(node)
                for node_id, node in self.nodes.items()
            },
            "relations": [
                asdict(rel) for rel in self.relations
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------
    # Embedding & Math (Mockable)
    # --------------------------------------------------
    @staticmethod
    def _embed(text: str) -> List[float]:
        """
        Placeholder embedding.
        Remplacer par OpenAI / SentenceTransformers / etc.
        """
        return [
            math.sin(hash(word) % 1000)
            for word in text.lower().split()[:32]
        ]

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "nodes": len(self.nodes),
            "relations": len(self.relations),
            "decay_rate": self.decay_rate,
        }
