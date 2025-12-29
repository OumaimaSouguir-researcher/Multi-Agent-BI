"""
Short-Term Memory (STM)
----------------------
Mémoire de travail à capacité limitée pour agents LLM.

Features:
- FIFO storage avec capacité configurable
- Expiration automatique (TTL)
- Compression des souvenirs anciens
- Recherche par type, agent, contenu
- Scoring d'importance
- Formatage du contexte pour LLM
- Nettoyage automatique
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque
import time
import uuid
import math
import re


@dataclass
class MemoryItem:
    id: str
    content: str
    memory_type: str
    agent_id: str
    importance: float
    created_at: float
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: float) -> bool:
        if self.ttl is None:
            return False
        return (now - self.created_at) > self.ttl


class ShortTermMemory:
    def __init__(
        self,
        capacity: int = 50,
        default_ttl: Optional[float] = 900.0,  # 15 minutes
        compression_threshold: int = 30,
        auto_cleanup: bool = True,
    ):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold
        self.auto_cleanup = auto_cleanup

        self._memory: deque[MemoryItem] = deque()

    # -----------------------------
    # Core API
    # -----------------------------
    def add(
        self,
        content: str,
        memory_type: str,
        agent_id: str,
        importance: float = 0.5,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        now = time.time()
        mem = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            agent_id=agent_id,
            importance=self._normalize_importance(importance),
            created_at=now,
            ttl=ttl if ttl is not None else self.default_ttl,
            metadata=metadata or {},
        )

        self._memory.append(mem)

        if len(self._memory) > self.capacity:
            self._memory.popleft()

        if self.auto_cleanup:
            self.cleanup()

        if len(self._memory) >= self.compression_threshold:
            self._compress_old_memories()

        return mem.id

    def search(
        self,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        min_importance: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        now = time.time()
        results = []

        for mem in self._memory:
            if mem.is_expired(now):
                continue

            if memory_type and mem.memory_type != memory_type:
                continue

            if agent_id and mem.agent_id != agent_id:
                continue

            if mem.importance < min_importance:
                continue

            if query and not self._match_query(query, mem.content):
                continue

            results.append(mem)

        results.sort(
            key=lambda m: (m.importance, m.created_at),
            reverse=True,
        )

        return results[:limit] if limit else results

    def cleanup(self) -> None:
        now = time.time()
        self._memory = deque(
            mem for mem in self._memory if not mem.is_expired(now)
        )

    def clear(self) -> None:
        self._memory.clear()

    # -----------------------------
    # LLM Context Formatting
    # -----------------------------
    def format_for_llm(
        self,
        agent_id: Optional[str] = None,
        max_tokens_hint: int = 1500,
    ) -> str:
        """
        Retourne un contexte STM prêt à être injecté dans un prompt LLM.
        """
        memories = self.search(agent_id=agent_id)

        context_blocks = []
        token_estimate = 0

        for mem in memories:
            block = (
                f"[{mem.memory_type.upper()}]"
                f"(importance={mem.importance:.2f})\n"
                f"{mem.content}\n"
            )

            block_tokens = self._estimate_tokens(block)
            if token_estimate + block_tokens > max_tokens_hint:
                break

            context_blocks.append(block)
            token_estimate += block_tokens

        return "\n".join(context_blocks)

    # -----------------------------
    # Compression
    # -----------------------------
    def _compress_old_memories(self) -> None:
        """
        Compresse les souvenirs les moins importants en une version résumée.
        """
        if len(self._memory) < self.compression_threshold:
            return

        sorted_memories = sorted(
            self._memory,
            key=lambda m: (m.importance, m.created_at),
        )

        to_compress = sorted_memories[: math.ceil(len(sorted_memories) * 0.3)]

        summary_content = " | ".join(
            mem.content for mem in to_compress
        )

        for mem in to_compress:
            self._memory.remove(mem)

        compressed = MemoryItem(
            id=str(uuid.uuid4()),
            content=f"[COMPRESSED MEMORY] {summary_content}",
            memory_type="compressed",
            agent_id="system",
            importance=0.3,
            created_at=time.time(),
            ttl=self.default_ttl,
            metadata={"compressed_count": len(to_compress)},
        )

        self._memory.appendleft(compressed)

    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def _normalize_importance(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Heuristique simple (~4 chars/token)
        return max(1, len(text) // 4)

    @staticmethod
    def _match_query(query: str, text: str) -> bool:
        pattern = re.escape(query.lower())
        return re.search(pattern, text.lower()) is not None

    # -----------------------------
    # Debug / Introspection
    # -----------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "current_size": len(self._memory),
            "compression_threshold": self.compression_threshold,
            "auto_cleanup": self.auto_cleanup,
        }
