"""
Message Types
=============
Defines various message types and structures for inter-agent communication.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid


class MessageType(Enum):
    """Standard message types for agent communication"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"
    COMMAND = "command"
    QUERY = "query"
    UPDATE = "update"
    ACK = "acknowledgment"
    HEARTBEAT = "heartbeat"
    BROADCAST = "broadcast"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10
    
    def __lt__(self, other):
        if isinstance(other, MessagePriority):
            return self.value < other.value
        return NotImplemented


class MessageStatus(Enum):
    """Message delivery and processing status"""
    CREATED = "created"
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Message:
    """
    Base message class for all inter-agent communications
    """
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        message_type: MessageType,
        content: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.message_id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.status = MessageStatus.CREATED
        self.retry_count = 0
        self.max_retries = 3
        self.parent_message_id: Optional[str] = None
        self.conversation_id: Optional[str] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation"""
        return {
            'message_id': self.message_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'message_type': self.message_type.value if isinstance(self.message_type, MessageType) else self.message_type,
            'content': self.content,
            'priority': self.priority.value if isinstance(self.priority, MessagePriority) else self.priority,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value if isinstance(self.status, MessageStatus) else self.status,
            'retry_count': self.retry_count,
            'parent_message_id': self.parent_message_id,
            'conversation_id': self.conversation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        message_type = MessageType(data['message_type']) if isinstance(data['message_type'], str) else data['message_type']
        priority = MessagePriority(data.get('priority', 5)) if isinstance(data.get('priority'), int) else data.get('priority', MessagePriority.NORMAL)
        
        msg = cls(
            sender=data['sender'],
            receiver=data['receiver'],
            message_type=message_type,
            content=data['content'],
            priority=priority,
            metadata=data.get('metadata', {})
        )
        
        msg.message_id = data.get('message_id', msg.message_id)
        msg.retry_count = data.get('retry_count', 0)
        msg.parent_message_id = data.get('parent_message_id')
        msg.conversation_id = data.get('conversation_id')
        
        if 'status' in data:
            msg.status = MessageStatus(data['status']) if isinstance(data['status'], str) else data['status']
        
        return msg
    
    def set_parent(self, parent_message_id: str) -> None:
        """Set parent message for threading"""
        self.parent_message_id = parent_message_id
    
    def set_conversation(self, conversation_id: str) -> None:
        """Set conversation ID for grouping messages"""
        self.conversation_id = conversation_id
    
    def mark_status(self, status: MessageStatus) -> None:
        """Update message status"""
        self.status = status
    
    def increment_retry(self) -> bool:
        """
        Increment retry counter
        
        Returns:
            bool: True if can retry, False if max retries reached
        """
        self.retry_count += 1
        return self.retry_count < self.max_retries
    
    def __repr__(self) -> str:
        return f"Message(id={self.message_id[:8]}, type={self.message_type.value}, {self.sender}->{self.receiver})"
    
    def __lt__(self, other):
        """Compare messages by priority for queue ordering"""
        if isinstance(other, Message):
            return self.priority.value > other.priority.value  # Higher priority first
        return NotImplemented


class RequestMessage(Message):
    """
    Request message - expects a response
    """
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        request_type: str,
        request_data: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        timeout: int = 30,
        metadata: Optional[Dict[str, Any]] = None
    ):
        content = {
            'request_type': request_type,
            'request_data': request_data,
            'timeout': timeout
        }
        super().__init__(sender, receiver, MessageType.REQUEST, content, priority, metadata)
        self.request_type = request_type
        self.request_data = request_data
        self.timeout = timeout
        self.response: Optional['ResponseMessage'] = None
    
    def create_response(
        self,
        response_data: Any,
        success: bool = True,
        error: Optional[str] = None
    ) -> 'ResponseMessage':
        """Create a response message for this request"""
        response = ResponseMessage(
            sender=self.receiver,
            receiver=self.sender,
            request_id=self.message_id,
            response_data=response_data,
            success=success,
            error=error,
            priority=self.priority
        )
        response.set_parent(self.message_id)
        if self.conversation_id:
            response.set_conversation(self.conversation_id)
        return response
    
    def __repr__(self) -> str:
        return f"RequestMessage(id={self.message_id[:8]}, type={self.request_type}, {self.sender}->{self.receiver})"


class ResponseMessage(Message):
    """
    Response message - reply to a request
    """
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        request_id: str,
        response_data: Any,
        success: bool = True,
        error: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        content = {
            'request_id': request_id,
            'response_data': response_data,
            'success': success,
            'error': error
        }
        super().__init__(sender, receiver, MessageType.RESPONSE, content, priority, metadata)
        self.request_id = request_id
        self.response_data = response_data
        self.success = success
        self.error = error
        self.set_parent(request_id)
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"ResponseMessage(id={self.message_id[:8]}, status={status}, {self.sender}->{self.receiver})"


class NotificationMessage(Message):
    """
    Notification message - one-way information broadcast
    """
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        notification_type: str,
        notification_data: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        content = {
            'notification_type': notification_type,
            'notification_data': notification_data
        }
        super().__init__(sender, receiver, MessageType.NOTIFICATION, content, priority, metadata)
        self.notification_type = notification_type
        self.notification_data = notification_data
    
    def __repr__(self) -> str:
        return f"NotificationMessage(id={self.message_id[:8]}, type={self.notification_type}, {self.sender}->{self.receiver})"


class ErrorMessage(Message):
    """
    Error message - indicates an error occurred
    """
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        priority: MessagePriority = MessagePriority.HIGH,
        metadata: Optional[Dict[str, Any]] = None
    ):
        content = {
            'error_code': error_code,
            'error_message': error_message,
            'error_details': error_details or {}
        }
        super().__init__(sender, receiver, MessageType.ERROR, content, priority, metadata)
        self.error_code = error_code
        self.error_message = error_message
        self.error_details = error_details or {}
    
    def __repr__(self) -> str:
        return f"ErrorMessage(id={self.message_id[:8]}, code={self.error_code}, {self.sender}->{self.receiver})"


class CommandMessage(Message):
    """
    Command message - instructs an agent to perform an action
    """
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        command: str,
        parameters: Optional[Dict[str, Any]] = None,
        priority: MessagePriority = MessagePriority.HIGH,
        metadata: Optional[Dict[str, Any]] = None
    ):
        content = {
            'command': command,
            'parameters': parameters or {}
        }
        super().__init__(sender, receiver, MessageType.COMMAND, content, priority, metadata)
        self.command = command
        self.parameters = parameters or {}
    
    def __repr__(self) -> str:
        return f"CommandMessage(id={self.message_id[:8]}, command={self.command}, {self.sender}->{self.receiver})"


class QueryMessage(Message):
    """
    Query message - requests specific information
    """
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        query_type: str,
        query_params: Optional[Dict[str, Any]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        content = {
            'query_type': query_type,
            'query_params': query_params or {}
        }
        super().__init__(sender, receiver, MessageType.QUERY, content, priority, metadata)
        self.query_type = query_type
        self.query_params = query_params or {}
    
    def __repr__(self) -> str:
        return f"QueryMessage(id={self.message_id[:8]}, query={self.query_type}, {self.sender}->{self.receiver})"


class UpdateMessage(Message):
    """
    Update message - provides status or data updates
    """
    
    def __init__(
        self,
        sender: str,
        receiver: str,
        update_type: str,
        update_data: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        content = {
            'update_type': update_type,
            'update_data': update_data
        }
        super().__init__(sender, receiver, MessageType.UPDATE, content, priority, metadata)
        self.update_type = update_type
        self.update_data = update_data
    
    def __repr__(self) -> str:
        return f"UpdateMessage(id={self.message_id[:8]}, type={self.update_type}, {self.sender}->{self.receiver})"


# Helper functions

def create_request(
    sender: str,
    receiver: str,
    request_type: str,
    request_data: Any,
    **kwargs
) -> RequestMessage:
    """Helper function to create a request message"""
    return RequestMessage(sender, receiver, request_type, request_data, **kwargs)


def create_response(
    sender: str,
    receiver: str,
    request_id: str,
    response_data: Any,
    success: bool = True,
    **kwargs
) -> ResponseMessage:
    """Helper function to create a response message"""
    return ResponseMessage(sender, receiver, request_id, response_data, success, **kwargs)


def create_notification(
    sender: str,
    receiver: str,
    notification_type: str,
    notification_data: Any,
    **kwargs
) -> NotificationMessage:
    """Helper function to create a notification message"""
    return NotificationMessage(sender, receiver, notification_type, notification_data, **kwargs)


def create_error(
    sender: str,
    receiver: str,
    error_code: str,
    error_message: str,
    **kwargs
) -> ErrorMessage:
    """Helper function to create an error message"""
    return ErrorMessage(sender, receiver, error_code, error_message, **kwargs)


# Example usage
if __name__ == "__main__":
    print("Message Types Module Test")
    print("="*50)
    
    # Create a request
    request = RequestMessage(
        sender="researcher_agent",
        receiver="analyst_agent",
        request_type="analyze_data",
        request_data={"data": [1, 2, 3, 4, 5]},
        priority=MessagePriority.HIGH
    )
    print(f"✓ Created: {request}")
    
    # Create a response
    response = request.create_response(
        response_data={"mean": 3.0, "median": 3},
        success=True
    )
    print(f"✓ Created: {response}")
    
    # Create a notification
    notification = NotificationMessage(
        sender="supervisor_agent",
        receiver="all_agents",
        notification_type="task_completed",
        notification_data={"task_id": "task_123"}
    )
    print(f"✓ Created: {notification}")
    
    # Create an error
    error = ErrorMessage(
        sender="validator_agent",
        receiver="analyst_agent",
        error_code="VALIDATION_FAILED",
        error_message="Data validation failed",
        error_details={"field": "data", "reason": "empty"}
    )
    print(f"✓ Created: {error}")
    
    # Test serialization
    print("\nSerialization Test:")
    request_dict = request.to_dict()
    print(f"✓ Serialized to dict: {len(request_dict)} fields")
    
    reconstructed = Message.from_dict(request_dict)
    print(f"✓ Reconstructed: {reconstructed}")