"""
Multi-Agent Orchestration System

A flexible system for orchestrating multiple AI agents to handle complex tasks
through task decomposition, agent selection, and response synthesis.
"""

from .task_decomposer import TaskDecomposer
from .agent_selector import AgentSelector
from .workflow_engine import WorkflowEngine
from .response_synthesizer import ResponseSynthesizer

__version__ = "1.0.0"
__all__ = [
    "TaskDecomposer", 
    "AgentSelector",
    "WorkflowEngine",
    "ResponseSynthesizer",
]


