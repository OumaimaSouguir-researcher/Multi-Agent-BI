"""
Communication Module
====================
Handles message queuing, routing, and inter-agent communication protocols.

This module provides:
- Message routing between agents
- Message queuing and prioritization
- Protocol definitions for agent communication
- Message type definitions
- Conversation memory management
"""

from .protocol import (
    CommunicationProtocol,
    MessageRouter,
    MessageQueue,
    MessageBroker
)

from .message_types import (
    Message,
    MessageType,
    MessagePriority,
    RequestMessage,
    ResponseMessage,
    NotificationMessage,
    ErrorMessage
)

from .conversation_memory import (
    ConversationMemory,
    ConversationContext,
    MemoryEntry
)

__version__ = "1.0.0"
__author__ = "Your Team"

__all__ = [
    # Protocol
    'CommunicationProtocol',
    'MessageRouter',
    'MessageQueue',
    'MessageBroker',
    
    # Message Types
    'Message',
    'MessageType',
    'MessagePriority',
    'RequestMessage',
    'ResponseMessage',
    'NotificationMessage',
    'ErrorMessage',
    
    # Memory
    'ConversationMemory',
    'ConversationContext',
    'MemoryEntry'
]


# Module-level configuration
DEFAULT_CONFIG = {
    'max_queue_size': 1000,
    'message_timeout': 30,  # seconds
    'enable_logging': True,
    'enable_metrics': True,
    'retry_attempts': 3,
    'retry_delay': 1.0  # seconds
}


def get_default_config():
    """
    Get default configuration for communication module
    
    Returns:
        dict: Default configuration settings
    """
    return DEFAULT_CONFIG.copy()


def initialize_communication_system(config=None):
    """
    Initialize the communication system with optional custom configuration
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        MessageBroker: Configured message broker instance
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    broker = MessageBroker(config)
    return broker


# Version check helper
def check_compatibility(required_version: str) -> bool:
    """
    Check if current version is compatible with required version
    
    Args:
        required_version: Required version string (e.g., "1.0.0")
        
    Returns:
        bool: True if compatible, False otherwise
    """
    current = tuple(map(int, __version__.split('.')))
    required = tuple(map(int, required_version.split('.')))
    
    # Check major version compatibility
    return current[0] == required[0] and current >= required


# Module initialization
print(f"Communication Module v{__version__} loaded")