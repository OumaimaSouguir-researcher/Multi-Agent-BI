"""
Base Agent class for all agents in the multi-agent system.

This module defines the abstract base class that all agents must inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    All specialized agents (Orchestrator, Researcher, Coder, etc.) 
    must inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, role: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a base agent.
        
        Args:
            name: Unique name identifier for the agent
            role: Role description of the agent
            config: Optional configuration dictionary
        """
        self.name = name
        self.role = role
        self.config = config or {}
        self.logger = self._setup_logger()
        self.state = "idle"  # idle, working, waiting, completed
        self.task_history: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the agent."""
        logger = logging.getLogger(f"Agent.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.
        
        Args:
            task: Dictionary containing task details
            
        Returns:
            Dictionary containing task results
        """
        pass
    
    @abstractmethod
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate if the agent can handle the given task.
        
        Args:
            task: Dictionary containing task details
            
        Returns:
            True if task is valid and can be handled, False otherwise
        """
        pass
    
    def update_state(self, new_state: str):
        """
        Update the agent's current state.
        
        Args:
            new_state: New state value
        """
        old_state = self.state
        self.state = new_state
        self.logger.info(f"State changed: {old_state} -> {new_state}")
    
    def log_task(self, task: Dict[str, Any], result: Dict[str, Any]):
        """
        Log a completed task to the agent's history.
        
        Args:
            task: The task that was executed
            result: The result of the task execution
        """
        self.task_history.append({
            "timestamp": datetime.now(),
            "task": task,
            "result": result
        })
        self.logger.info(f"Task completed: {task.get('id', 'unknown')}")
    
    def get_capabilities(self) -> List[str]:
        """
        Get the list of capabilities this agent has.
        
        Returns:
            List of capability strings
        """
        return self.config.get("capabilities", [])
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            "name": self.name,
            "role": self.role,
            "state": self.state,
            "tasks_completed": len(self.task_history),
            "uptime": (datetime.now() - self.created_at).total_seconds(),
            "capabilities": self.get_capabilities()
        }
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message from another agent or the orchestrator.
        
        Args:
            message: Dictionary containing message details
            
        Returns:
            Dictionary containing response
        """
        self.logger.info(f"Processing message from: {message.get('sender', 'unknown')}")
        
        msg_type = message.get("type")
        
        if msg_type == "task":
            return await self.execute_task(message.get("content"))
        elif msg_type == "status_request":
            return self.get_status()
        elif msg_type == "ping":
            return {"status": "alive", "agent": self.name}
        else:
            self.logger.warning(f"Unknown message type: {msg_type}")
            return {"error": "Unknown message type", "received_type": msg_type}
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', role='{self.role}', state='{self.state}')"