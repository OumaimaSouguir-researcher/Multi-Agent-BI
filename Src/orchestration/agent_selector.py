"""
Agent Selector Module

Selects the most appropriate agent(s) for each subtask based on
capabilities, availability, and task requirements.
"""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import random


class AgentCapability(Enum):
    """Available agent capabilities"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    CODING = "coding"
    WRITING = "writing"
    CALCULATION = "calculation"
    DATA_PROCESSING = "data_processing"
    CREATIVE = "creative"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    GENERAL = "general"


class AgentStatus(Enum):
    """Agent availability status"""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class Agent:
    """Represents an AI agent with specific capabilities"""
    id: str
    name: str
    capabilities: Set[AgentCapability]
    specialization_score: Dict[AgentCapability, float]  # 0-1 score for each capability
    max_concurrent_tasks: int = 3
    current_load: int = 0
    status: AgentStatus = AgentStatus.AVAILABLE
    performance_history: List[float] = field(default_factory=list)  # Success rates
    
    @property
    def availability_score(self) -> float:
        """Calculate how available the agent is (0-1)"""
        if self.status != AgentStatus.AVAILABLE:
            return 0.0
        return 1.0 - (self.current_load / self.max_concurrent_tasks)
    
    @property
    def average_performance(self) -> float:
        """Calculate average performance score"""
        if not self.performance_history:
            return 0.7  # Default score for new agents
        return sum(self.performance_history[-10:]) / min(10, len(self.performance_history))


@dataclass
class AgentAssignment:
    """Represents an assignment of an agent to a task"""
    task_id: str
    agent_id: str
    confidence_score: float  # How confident we are in this assignment
    estimated_time: float  # Estimated time in seconds
    reasoning: str


class AgentSelector:
    """
    Selects appropriate agents for subtasks based on capabilities and availability.
    """
    
    def __init__(self):
        """Initialize the AgentSelector with a pool of agents"""
        self.agents: Dict[str, Agent] = {}
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize a default pool of specialized agents"""
        default_agents = [
            Agent(
                id="agent_research_1",
                name="Research Specialist",
                capabilities={
                    AgentCapability.RESEARCH,
                    AgentCapability.QUESTION_ANSWERING,
                    AgentCapability.GENERAL
                },
                specialization_score={
                    AgentCapability.RESEARCH: 0.95,
                    AgentCapability.QUESTION_ANSWERING: 0.85,
                    AgentCapability.GENERAL: 0.7
                }
            ),
            Agent(
                id="agent_analyst_1",
                name="Data Analyst",
                capabilities={
                    AgentCapability.ANALYSIS,
                    AgentCapability.DATA_PROCESSING,
                    AgentCapability.CALCULATION
                },
                specialization_score={
                    AgentCapability.ANALYSIS: 0.9,
                    AgentCapability.DATA_PROCESSING: 0.85,
                    AgentCapability.CALCULATION: 0.8
                }
            ),
            Agent(
                id="agent_coder_1",
                name="Code Generator",
                capabilities={
                    AgentCapability.CODING,
                    AgentCapability.DATA_PROCESSING,
                    AgentCapability.GENERAL
                },
                specialization_score={
                    AgentCapability.CODING: 0.95,
                    AgentCapability.DATA_PROCESSING: 0.75,
                    AgentCapability.GENERAL: 0.6
                }
            ),
            Agent(
                id="agent_writer_1",
                name="Content Writer",
                capabilities={
                    AgentCapability.WRITING,
                    AgentCapability.CREATIVE,
                    AgentCapability.SUMMARIZATION
                },
                specialization_score={
                    AgentCapability.WRITING: 0.9,
                    AgentCapability.CREATIVE: 0.85,
                    AgentCapability.SUMMARIZATION: 0.8
                }
            ),
            Agent(
                id="agent_translator_1",
                name="Language Translator",
                capabilities={
                    AgentCapability.TRANSLATION,
                    AgentCapability.WRITING,
                    AgentCapability.GENERAL
                },
                specialization_score={
                    AgentCapability.TRANSLATION: 0.95,
                    AgentCapability.WRITING: 0.75,
                    AgentCapability.GENERAL: 0.65
                }
            ),
            Agent(
                id="agent_generalist_1",
                name="Generalist Agent",
                capabilities={cap for cap in AgentCapability},
                specialization_score={cap: 0.6 for cap in AgentCapability},
                max_concurrent_tasks=5
            )
        ]
        
        for agent in default_agents:
            self.agents[agent.id] = agent
    
    def add_agent(self, agent: Agent):
        """Add a new agent to the pool"""
        self.agents[agent.id] = agent
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the pool"""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def select_agent(
        self,
        task_id: str,
        task_type: str,
        complexity: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[AgentAssignment]:
        """
        Select the best agent for a given task.
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task (should match AgentCapability)
            complexity: Task complexity (1-10)
            context: Optional context information
            
        Returns:
            AgentAssignment if a suitable agent is found, None otherwise
        """
        # Convert task_type string to AgentCapability
        try:
            required_capability = AgentCapability(task_type.lower())
        except ValueError:
            required_capability = AgentCapability.GENERAL
        
        # Find capable agents
        capable_agents = [
            agent for agent in self.agents.values()
            if required_capability in agent.capabilities
            and agent.status == AgentStatus.AVAILABLE
        ]
        
        if not capable_agents:
            # Fallback to any available agent
            capable_agents = [
                agent for agent in self.agents.values()
                if agent.status == AgentStatus.AVAILABLE
            ]
        
        if not capable_agents:
            return None
        
        # Score each capable agent
        agent_scores = []
        for agent in capable_agents:
            score = self._calculate_agent_score(
                agent,
                required_capability,
                complexity,
                context
            )
            agent_scores.append((agent, score))
        
        # Select best agent
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        best_agent, best_score = agent_scores[0]
        
        # Create assignment
        assignment = AgentAssignment(
            task_id=task_id,
            agent_id=best_agent.id,
            confidence_score=best_score,
            estimated_time=self._estimate_time(complexity, best_agent),
            reasoning=self._generate_reasoning(
                best_agent,
                required_capability,
                best_score
            )
        )
        
        # Update agent load
        best_agent.current_load += 1
        
        return assignment
    
    def _calculate_agent_score(
        self,
        agent: Agent,
        capability: AgentCapability,
        complexity: int,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate a score for how well an agent matches a task.
        
        Returns:
            Score between 0 and 1
        """
        # Base score from specialization
        specialization_score = agent.specialization_score.get(capability, 0.5)
        
        # Availability factor
        availability_factor = agent.availability_score
        
        # Performance history factor
        performance_factor = agent.average_performance
        
        # Complexity handling (prefer specialists for complex tasks)
        if complexity > 7:
            complexity_factor = specialization_score
        else:
            complexity_factor = 1.0
        
        # Weighted combination
        score = (
            specialization_score * 0.4 +
            availability_factor * 0.3 +
            performance_factor * 0.2 +
            complexity_factor * 0.1
        )
        
        return score
    
    def _estimate_time(self, complexity: int, agent: Agent) -> float:
        """
        Estimate time to complete task in seconds.
        
        Args:
            complexity: Task complexity (1-10)
            agent: Agent to perform the task
            
        Returns:
            Estimated time in seconds
        """
        # Base time per complexity level
        base_time = complexity * 10
        
        # Adjust based on agent performance
        efficiency_factor = agent.average_performance
        
        return base_time / efficiency_factor
    
    def _generate_reasoning(
        self,
        agent: Agent,
        capability: AgentCapability,
        score: float
    ) -> str:
        """Generate human-readable reasoning for the selection"""
        reasons = []
        
        spec_score = agent.specialization_score.get(capability, 0)
        if spec_score > 0.8:
            reasons.append(f"highly specialized in {capability.value}")
        elif spec_score > 0.6:
            reasons.append(f"competent in {capability.value}")
        else:
            reasons.append(f"capable of {capability.value}")
        
        if agent.availability_score > 0.8:
            reasons.append("readily available")
        elif agent.availability_score > 0.5:
            reasons.append("available with some load")
        
        if agent.average_performance > 0.8:
            reasons.append("strong track record")
        
        return f"Selected {agent.name}: {', '.join(reasons)} (confidence: {score:.2f})"
    
    def release_agent(self, agent_id: str, task_success: float):
        """
        Release an agent after task completion and update performance.
        
        Args:
            agent_id: ID of the agent to release
            task_success: Success score for the task (0-1)
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.current_load = max(0, agent.current_load - 1)
            agent.performance_history.append(task_success)
            
            # Keep only last 50 performance scores
            if len(agent.performance_history) > 50:
                agent.performance_history = agent.performance_history[-50:]
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of an agent"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            "id": agent.id,
            "name": agent.name,
            "status": agent.status.value,
            "current_load": agent.current_load,
            "max_concurrent_tasks": agent.max_concurrent_tasks,
            "availability_score": agent.availability_score,
            "average_performance": agent.average_performance,
            "capabilities": [cap.value for cap in agent.capabilities]
        }
    
    def get_all_agents_status(self) -> List[Dict[str, Any]]:
        """Get status of all agents"""
        return [
            self.get_agent_status(agent_id)
            for agent_id in self.agents.keys()
        ]
    
    def set_agent_status(self, agent_id: str, status: AgentStatus):
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id].status = status