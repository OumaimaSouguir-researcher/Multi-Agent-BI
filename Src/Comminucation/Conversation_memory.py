"""
Conversation Memory
===================
Manages short-term conversation context and memory for agents.
"""

from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
from typing import Dict, Any

class MemoryEntry:
    """
    Represents a single entry in conversation memory
    """
    
    def __init__(
        self,
        entry_id: str,
        agent_id: str,
        content: Any,
        entry_type: str = "message",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.entry_id = entry_id
        self.agent_id = agent_id
        self.content = content
        self.entry_type = entry_type
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.access_count = 0
        self.last_accessed = self.timestamp
        
    def access(self) -> None:
        """Mark this entry as accessed"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary"""
        return {
            'entry_id': self.entry_id,
            'agent_id': self.agent_id,
            'content': self.content,
            'entry_type': self.entry_type,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create entry from dictionary"""
        entry = cls(
            entry_id=data['entry_id'],
            agent_id=data['agent_id'],
            content=data['content'],
            entry_type=data.get('entry_type', 'message'),
            metadata=data.get('metadata', {})
        )
        entry.access_count = data.get('access_count', 0)
        return entry
    
    def __repr__(self) -> str:
        return f"MemoryEntry(id={self.entry_id[:8]}, agent={self.agent_id}, type={self.entry_type})"


class ConversationContext:
    """
    Represents the context of a conversation
    """
    
    def __init__(
        self,
        conversation_id: str,
        participants: Optional[List[str]] = None,
        topic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.conversation_id = conversation_id
        self.participants: Set[str] = set(participants or [])
        self.topic = topic
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        self.status = "active"
        self.summary: Optional[str] = None
        self.tags: Set[str] = set()
        
    def add_participant(self, agent_id: str) -> None:
        """Add a participant to the conversation"""
        self.participants.add(agent_id)
        self.last_updated = datetime.now()
    
    def remove_participant(self, agent_id: str) -> None:
        """Remove a participant from the conversation"""
        self.participants.discard(agent_id)
        self.last_updated = datetime.now()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the conversation"""
        self.tags.add(tag)
    
    def set_summary(self, summary: str) -> None:
        """Set conversation summary"""
        self.summary = summary
        self.last_updated = datetime.now()
    
    def close(self) -> None:
        """Close the conversation"""
        self.status = "closed"
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            'conversation_id': self.conversation_id,
            'participants': list(self.participants),
            'topic': self.topic,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'status': self.status,
            'summary': self.summary,
            'tags': list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create context from dictionary"""
        context = cls(
            conversation_id=data['conversation_id'],
            participants=data.get('participants', []),
            topic=data.get('topic'),
            metadata=data.get('metadata', {})
        )
        context.status = data.get('status', 'active')
        context.summary = data.get('summary')
        context.tags = set(data.get('tags', []))
        return context
    
    def __repr__(self) -> str:
        return f"ConversationContext(id={self.conversation_id[:8]}, participants={len(self.participants)}, status={self.status})"


class ConversationMemory:
    """
    Manages short-term conversation memory and context for agents
    """
    
    def __init__(
        self,
        max_entries: int = 1000,
        retention_hours: int = 24,
        enable_compression: bool = True
    ):
        self.max_entries = max_entries
        self.retention_hours = retention_hours
        self.enable_compression = enable_compression
        
        # Storage
        self.entries: deque = deque(maxlen=max_entries)
        self.entry_index: Dict[str, MemoryEntry] = {}
        self.contexts: Dict[str, ConversationContext] = {}
        
        # Indexing for fast retrieval
        self.agent_entries: Dict[str, List[str]] = defaultdict(list)
        self.conversation_entries: Dict[str, List[str]] = defaultdict(list)
        self.type_entries: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'total_contexts': 0,
            'total_accesses': 0,
            'compressions': 0
        }
    
    def add_entry(
        self,
        agent_id: str,
        content: Any,
        entry_type: str = "message",
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """
        Add an entry to memory
        
        Args:
            agent_id: Agent creating the entry
            content: Entry content
            entry_type: Type of entry
            conversation_id: Optional conversation ID
            metadata: Optional metadata
            
        Returns:
            Created memory entry
        """
        entry_id = f"entry_{len(self.entry_index)}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        entry = MemoryEntry(
            entry_id=entry_id,
            agent_id=agent_id,
            content=content,
            entry_type=entry_type,
            metadata=metadata or {}
        )
        
        # Add conversation ID to metadata if provided
        if conversation_id:
            entry.metadata['conversation_id'] = conversation_id
            self.conversation_entries[conversation_id].append(entry_id)
        
        # Store entry
        self.entries.append(entry)
        self.entry_index[entry_id] = entry
        
        # Update indices
        self.agent_entries[agent_id].append(entry_id)
        self.type_entries[entry_type].append(entry_id)
        
        # Update stats
        self.stats['total_entries'] += 1
        
        # Clean old entries if needed
        self._cleanup_old_entries()
        
        return entry
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific entry by ID"""
        entry = self.entry_index.get(entry_id)
        if entry:
            entry.access()
            self.stats['total_accesses'] += 1
        return entry
    
    def get_entries_by_agent(
        self,
        agent_id: str,
        limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Get entries by agent"""
        entry_ids = self.agent_entries.get(agent_id, [])
        if limit:
            entry_ids = entry_ids[-limit:]
        
        entries = []
        for entry_id in entry_ids:
            entry = self.entry_index.get(entry_id)
            if entry:
                entry.access()
                entries.append(entry)
        
        return entries
    
    def get_entries_by_conversation(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Get entries by conversation"""
        entry_ids = self.conversation_entries.get(conversation_id, [])
        if limit:
            entry_ids = entry_ids[-limit:]
        
        entries = []
        for entry_id in entry_ids:
            entry = self.entry_index.get(entry_id)
            if entry:
                entry.access()
                entries.append(entry)
        
        return entries
    
    def get_entries_by_type(
        self,
        entry_type: str,
        limit: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Get entries by type"""
        entry_ids = self.type_entries.get(entry_type, [])
        if limit:
            entry_ids = entry_ids[-limit:]
        
        entries = []
        for entry_id in entry_ids:
            entry = self.entry_index.get(entry_id)
            if entry:
                entry.access()
                entries.append(entry)
        
        return entries
    
    def get_recent_entries(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most recent entries"""
        recent = list(self.entries)[-limit:]
        for entry in recent:
            entry.access()
        return recent
    
    def search_entries(
        self,
        query: str,
        agent_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        entry_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """
        Search entries by content
        
        Args:
            query: Search query
            agent_id: Filter by agent
            conversation_id: Filter by conversation
            entry_type: Filter by type
            
        Returns:
            List of matching entries
        """
        results = []
        query_lower = query.lower()
        
        for entry in self.entries:
            # Apply filters
            if agent_id and entry.agent_id != agent_id:
                continue
            if conversation_id and entry.metadata.get('conversation_id') != conversation_id:
                continue
            if entry_type and entry.entry_type != entry_type:
                continue
            
            # Search in content
            content_str = json.dumps(entry.content).lower()
            if query_lower in content_str:
                entry.access()
                results.append(entry)
        
        return results
    
    def create_context(
        self,
        conversation_id: str,
        participants: Optional[List[str]] = None,
        topic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationContext:
        """Create a new conversation context"""
        context = ConversationContext(
            conversation_id=conversation_id,
            participants=participants,
            topic=topic,
            metadata=metadata
        )
        
        self.contexts[conversation_id] = context
        self.stats['total_contexts'] += 1
        
        return context
    
    def get_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context"""
        return self.contexts.get(conversation_id)
    
    def update_context(
        self,
        conversation_id: str,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[ConversationContext]:
        """Update conversation context"""
        context = self.contexts.get(conversation_id)
        if context:
            if summary:
                context.set_summary(summary)
            if tags:
                for tag in tags:
                    context.add_tag(tag)
        return context
    
    def close_context(self, conversation_id: str) -> bool:
        """Close a conversation context"""
        context = self.contexts.get(conversation_id)
        if context:
            context.close()
            return True
        return False
    
    def get_active_contexts(self) -> List[ConversationContext]:
        """Get all active conversation contexts"""
        return [ctx for ctx in self.contexts.values() if ctx.status == "active"]
    
    def _cleanup_old_entries(self) -> None:
        """Remove entries older than retention period"""
        if self.retention_hours <= 0:
            return
        
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        entries_to_remove = []
        
        for entry in self.entries:
            if entry.timestamp < cutoff_time:
                entries_to_remove.append(entry.entry_id)
        
        for entry_id in entries_to_remove:
            self._remove_entry(entry_id)
    
    def _remove_entry(self, entry_id: str) -> None:
        """Remove an entry from memory"""
        if entry_id in self.entry_index:
            entry = self.entry_index[entry_id]
            
            # Remove from indices
            if entry_id in self.agent_entries[entry.agent_id]:
                self.agent_entries[entry.agent_id].remove(entry_id)
            if entry_id in self.type_entries[entry.entry_type]:
                self.type_entries[entry.entry_type].remove(entry_id)
            
            conversation_id = entry.metadata.get('conversation_id')
            if conversation_id and entry_id in self.conversation_entries[conversation_id]:
                self.conversation_entries[conversation_id].remove(entry_id)
            
            # Remove from index
            del self.entry_index[entry_id]
    
    def clear_agent_memory(self, agent_id: str) -> int:
        """Clear all entries for a specific agent"""
        entry_ids = self.agent_entries.get(agent_id, []).copy()
        for entry_id in entry_ids:
            self._remove_entry(entry_id)
        
        if agent_id in self.agent_entries:
            del self.agent_entries[agent_id]
        
        return len(entry_ids)
    
    def clear_conversation_memory(self, conversation_id: str) -> int:
        """Clear all entries for a specific conversation"""
        entry_ids = self.conversation_entries.get(conversation_id, []).copy()
        for entry_id in entry_ids:
            self._remove_entry(entry_id)
        
        if conversation_id in self.conversation_entries:
            del self.conversation_entries[conversation_id]
        
        if conversation_id in self.contexts:
            del self.contexts[conversation_id]
        
        return len(entry_ids)
    
    def clear_all(self) -> None:
        """Clear all memory"""
        self.entries.clear()
        self.entry_index.clear()
        self.contexts.clear()
        self.agent_entries.clear()
        self.conversation_entries.clear()
        self.type_entries.clear()


def get_statistics(self) -> Dict[str, Any]:
    """Get memory statistics"""
    return {
        **self.stats,
        "current_entries": len(self.entry_index),
        "current_contexts": len(self.contexts),
    }
