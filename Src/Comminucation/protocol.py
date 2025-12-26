"""
Communication Protocol for Multi-Agent System
Handles message passing, queuing, and routing between agents using Redis
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis
from dataclasses import dataclass, asdict

from agents.base_agent import AgentMessage, AgentRole, MessageType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class CommunicationProtocol:
    """
    Manages all inter-agent communication using Redis as message broker
    Supports priority queues, pub/sub, and request-response patterns
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
        # Message tracking
        self.message_history: List[AgentMessage] = []
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Subscriber callbacks
        self.subscribers: Dict[str, List[Callable]] = {}
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0
        }
        
    async def connect(self) -> bool:
        """Establish connection to Redis"""
        try:
            self.redis_client = await redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
                socket_connect_timeout=5
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Initialize pubsub
            self.pubsub = self.redis_client.pubsub()
            
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self):
        """Close Redis connection"""
        try:
            if self.pubsub:
                await self.pubsub.close()
            if self.redis_client:
                await self.redis_client.close()
            logger.info("Disconnected from Redis")
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
    
    def _get_queue_name(self, agent_role: AgentRole, priority: int = 2) -> str:
        """Generate queue name based on agent role and priority"""
        return f"queue:{agent_role.value}:priority_{priority}"
    
    def _get_broadcast_channel(self, topic: str = "all") -> str:
        """Generate broadcast channel name"""
        return f"broadcast:{topic}"
    
    async def send_message(
        self,
        message: AgentMessage,
        timeout: Optional[int] = None
    ) -> bool:
        """
        Send a message to an agent's queue
        
        Args:
            message: AgentMessage to send
            timeout: Optional TTL for the message in seconds
            
        Returns:
            bool: True if message was sent successfully
        """
        if not self.redis_client:
            logger.error("Redis client not connected")
            return False
        
        try:
            # Serialize message
            message_data = json.dumps(asdict(message), default=str)
            
            # Get appropriate queue based on priority
            queue_name = self._get_queue_name(
                message.recipient,
                message.priority
            )
            
            # Push to queue (left push for FIFO with right pop)
            await self.redis_client.lpush(queue_name, message_data)
            
            # Set TTL if specified
            if timeout:
                await self.redis_client.expire(queue_name, timeout)
            
            # Track message
            self.message_history.append(message)
            self.stats["messages_sent"] += 1
            
            # Store in message tracking hash
            message_key = f"message:{message.message_id}"
            await self.redis_client.hset(
                message_key,
                mapping={
                    "data": message_data,
                    "sent_at": datetime.now().isoformat(),
                    "status": "sent"
                }
            )
            await self.redis_client.expire(message_key, 3600)  # 1 hour TTL
            
            logger.info(
                f"Sent message {message.message_id} from {message.sender.value} "
                f"to {message.recipient.value} (priority: {message.priority})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.stats["messages_failed"] += 1
            return False
    
    async def receive_message(
        self,
        agent_role: AgentRole,
        timeout: int = 0,
        priority_order: bool = True
    ) -> Optional[AgentMessage]:
        """
        Receive a message from agent's queue
        
        Args:
            agent_role: Role of the receiving agent
            timeout: Timeout in seconds (0 = non-blocking, None = blocking)
            priority_order: If True, check high priority queues first
            
        Returns:
            AgentMessage or None
        """
        if not self.redis_client:
            logger.error("Redis client not connected")
            return None
        
        try:
            # Define priority order
            priorities = [4, 3, 2, 1] if priority_order else [2, 1, 3, 4]
            
            for priority in priorities:
                queue_name = self._get_queue_name(agent_role, priority)
                
                # Try to pop message (right pop for FIFO)
                if timeout == 0:
                    # Non-blocking
                    result = await self.redis_client.rpop(queue_name)
                else:
                    # Blocking with timeout
                    result = await self.redis_client.brpop(
                        queue_name,
                        timeout=timeout or 0
                    )
                    if result:
                        _, result = result
                
                if result:
                    # Deserialize message
                    message_dict = json.loads(result)
                    
                    # Convert string enums back to enum objects
                    message_dict['sender'] = AgentRole(message_dict['sender'])
                    message_dict['recipient'] = AgentRole(message_dict['recipient'])
                    message_dict['message_type'] = MessageType(message_dict['message_type'])
                    message_dict['timestamp'] = datetime.fromisoformat(message_dict['timestamp'])
                    
                    message = AgentMessage(**message_dict)
                    
                    # Update tracking
                    self.stats["messages_received"] += 1
                    
                    message_key = f"message:{message.message_id}"
                    await self.redis_client.hset(
                        message_key,
                        "status",
                        "received"
                    )
                    
                    logger.info(
                        f"Received message {message.message_id} "
                        f"for {agent_role.value}"
                    )
                    
                    return message
            
            # No messages in any queue
            return None
            
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
    
    async def send_request(
        self,
        message: AgentMessage,
        timeout: int = 30
    ) -> Optional[AgentMessage]:
        """
        Send a request and wait for response (request-response pattern)
        
        Args:
            message: Request message
            timeout: Maximum time to wait for response in seconds
            
        Returns:
            Response message or None if timeout
        """
        # Create future for response
        response_future = asyncio.Future()
        self.pending_responses[message.message_id] = response_future
        
        # Send request
        success = await self.send_message(message)
        if not success:
            del self.pending_responses[message.message_id]
            return None
        
        try:
            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Request {message.message_id} timed out")
            return None
        finally:
            # Cleanup
            if message.message_id in self.pending_responses:
                del self.pending_responses[message.message_id]
    
    async def send_response(
        self,
        original_message: AgentMessage,
        response_content: Dict[str, Any]
    ) -> bool:
        """
        Send a response to a request
        
        Args:
            original_message: The original request message
            response_content: Response data
            
        Returns:
            bool: Success status
        """
        response = AgentMessage(
            sender=original_message.recipient,
            recipient=original_message.sender,
            message_type=MessageType.RESPONSE,
            content=response_content,
            conversation_id=original_message.conversation_id,
            parent_message_id=original_message.message_id,
            priority=original_message.priority
        )
        
        success = await self.send_message(response)
        
        # Resolve pending response future if exists
        if original_message.message_id in self.pending_responses:
            future = self.pending_responses[original_message.message_id]
            if not future.done():
                future.set_result(response)
        
        return success
    
    async def broadcast_message(
        self,
        sender: AgentRole,
        content: Dict[str, Any],
        topic: str = "all"
    ) -> bool:
        """
        Broadcast a message to all subscribers on a channel
        
        Args:
            sender: Agent sending the broadcast
            content: Message content
            topic: Broadcast topic/channel
            
        Returns:
            bool: Success status
        """
        if not self.redis_client:
            logger.error("Redis client not connected")
            return False
        
        try:
            message = {
                "sender": sender.value,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }
            
            channel = self._get_broadcast_channel(topic)
            await self.redis_client.publish(
                channel,
                json.dumps(message)
            )
            
            logger.info(f"Broadcast message on {channel} from {sender.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return False
    
    async def subscribe(
        self,
        topic: str,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Subscribe to a broadcast channel
        
        Args:
            topic: Channel topic to subscribe to
            callback: Function to call when message received
        """
        if not self.pubsub:
            logger.error("PubSub not initialized")
            return
        
        try:
            channel = self._get_broadcast_channel(topic)
            
            # Subscribe to channel
            await self.pubsub.subscribe(channel)
            
            # Store callback
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(callback)
            
            logger.info(f"Subscribed to {channel}")
            
            # Start listening in background
            asyncio.create_task(self._listen_to_broadcasts(topic))
            
        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
    
    async def _listen_to_broadcasts(self, topic: str):
        """Background task to listen for broadcast messages"""
        if not self.pubsub:
            return
        
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    
                    # Call all registered callbacks
                    if topic in self.subscribers:
                        for callback in self.subscribers[topic]:
                            await callback(data)
                            
        except Exception as e:
            logger.error(f"Error listening to broadcasts: {e}")
    
    async def get_queue_length(self, agent_role: AgentRole) -> Dict[int, int]:
        """
        Get the number of pending messages for an agent by priority
        
        Returns:
            Dict mapping priority level to queue length
        """
        if not self.redis_client:
            return {}
        
        lengths = {}
        for priority in [1, 2, 3, 4]:
            queue_name = self._get_queue_name(agent_role, priority)
            length = await self.redis_client.llen(queue_name)
            lengths[priority] = length
        
        return lengths
    
    async def clear_queue(self, agent_role: AgentRole, priority: Optional[int] = None):
        """Clear an agent's message queue"""
        if not self.redis_client:
            return
        
        if priority:
            queue_name = self._get_queue_name(agent_role, priority)
            await self.redis_client.delete(queue_name)
        else:
            # Clear all priority queues
            for p in [1, 2, 3, 4]:
                queue_name = self._get_queue_name(agent_role, p)
                await self.redis_client.delete(queue_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            **self.stats,
            "message_history_size": len(self.message_history),
            "pending_responses": len(self.pending_responses)
        }
    
    async def health_check(self) -> bool:
        """Check if Redis connection is healthy"""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.ping()
            return True
        except:
            return False