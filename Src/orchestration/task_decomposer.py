"""
Task Decomposer Module

Breaks down complex user queries into manageable subtasks that can be
distributed to specialized agents.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re


class TaskType(Enum):
    """Types of tasks that can be identified"""
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


@dataclass
class SubTask:
    """Represents a decomposed subtask"""
    id: str
    description: str
    task_type: TaskType
    priority: int
    dependencies: List[str]
    context: Dict[str, Any]
    estimated_complexity: int  # 1-10 scale


class TaskDecomposer:
    """
    Decomposes complex tasks into smaller, manageable subtasks.
    """
    
    def __init__(self, max_subtasks: int = 10):
        """
        Initialize the TaskDecomposer.
        
        Args:
            max_subtasks: Maximum number of subtasks to generate
        """
        self.max_subtasks = max_subtasks
        self.task_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[TaskType, List[str]]:
        """Initialize regex patterns for task type detection"""
        return {
            TaskType.RESEARCH: [
                r'\b(research|investigate|find|search|look up|explore)\b',
                r'\b(what is|who is|where is|when did|how did)\b'
            ],
            TaskType.ANALYSIS: [
                r'\b(analyze|compare|evaluate|assess|examine)\b',
                r'\b(pros and cons|advantages|disadvantages)\b'
            ],
            TaskType.CODING: [
                r'\b(code|program|implement|develop|debug|script)\b',
                r'\b(function|class|algorithm|api|database)\b'
            ],
            TaskType.WRITING: [
                r'\b(write|draft|compose|create|generate)\b.*\b(document|article|essay|report|email)\b'
            ],
            TaskType.CALCULATION: [
                r'\b(calculate|compute|solve|determine)\b',
                r'\b(math|equation|formula|statistics)\b'
            ],
            TaskType.DATA_PROCESSING: [
                r'\b(process|transform|clean|parse|extract)\b.*\b(data|file|csv|json|xml)\b'
            ],
            TaskType.CREATIVE: [
                r'\b(design|create|generate|imagine|brainstorm)\b',
                r'\b(story|poem|art|music|creative)\b'
            ],
            TaskType.TRANSLATION: [
                r'\b(translate|convert)\b.*\b(language|french|spanish|german)\b'
            ],
            TaskType.SUMMARIZATION: [
                r'\b(summarize|condense|brief|overview)\b'
            ]
        }
    
    def decompose(self, query: str, context: Optional[Dict[str, Any]] = None) -> List[SubTask]:
        """
        Decompose a query into subtasks.
        
        Args:
            query: The user query to decompose
            context: Optional context information
            
        Returns:
            List of SubTask objects
        """
        context = context or {}
        
        # Detect task types present in the query
        detected_types = self._detect_task_types(query)
        
        # Check if task is simple or complex
        if self._is_simple_task(query, detected_types):
            return self._create_simple_task(query, detected_types, context)
        
        # Complex task decomposition
        return self._decompose_complex_task(query, detected_types, context)
    
    def _detect_task_types(self, query: str) -> List[TaskType]:
        """Detect which task types are present in the query"""
        detected = []
        query_lower = query.lower()
        
        for task_type, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected.append(task_type)
                    break
        
        # Default to question answering if no specific type detected
        if not detected:
            detected.append(TaskType.QUESTION_ANSWERING)
        
        return detected
    
    def _is_simple_task(self, query: str, detected_types: List[TaskType]) -> bool:
        """Determine if a task is simple enough to not need decomposition"""
        # Simple heuristics
        word_count = len(query.split())
        sentence_count = len(re.split(r'[.!?]+', query))
        
        return (
            len(detected_types) <= 1 and
            word_count < 20 and
            sentence_count <= 2 and
            'and' not in query.lower() and
            'then' not in query.lower()
        )
    
    def _create_simple_task(
        self,
        query: str,
        detected_types: List[TaskType],
        context: Dict[str, Any]
    ) -> List[SubTask]:
        """Create a single subtask for simple queries"""
        task_type = detected_types[0] if detected_types else TaskType.QUESTION_ANSWERING
        
        return [SubTask(
            id="task_1",
            description=query,
            task_type=task_type,
            priority=1,
            dependencies=[],
            context=context,
            estimated_complexity=3
        )]
    
    def _decompose_complex_task(
        self,
        query: str,
        detected_types: List[TaskType],
        context: Dict[str, Any]
    ) -> List[SubTask]:
        """Decompose a complex query into multiple subtasks"""
        subtasks = []
        
        # Split by common conjunctions
        segments = re.split(r'\b(and then|then|after that|next|finally|and)\b', query, flags=re.IGNORECASE)
        segments = [s.strip() for s in segments if s.strip() and s.lower() not in ['and', 'then', 'and then', 'after that', 'next', 'finally']]
        
        # If no clear segmentation, try to identify logical components
        if len(segments) <= 1:
            segments = self._identify_logical_components(query, detected_types)
        
        # Create subtasks from segments
        for idx, segment in enumerate(segments[:self.max_subtasks]):
            segment_types = self._detect_task_types(segment)
            task_type = segment_types[0] if segment_types else detected_types[0]
            
            # Determine dependencies (sequential by default)
            dependencies = [f"task_{idx}"] if idx > 0 else []
            
            subtasks.append(SubTask(
                id=f"task_{idx + 1}",
                description=segment,
                task_type=task_type,
                priority=idx + 1,
                dependencies=dependencies,
                context=context,
                estimated_complexity=self._estimate_complexity(segment, task_type)
            ))
        
        return subtasks
    
    def _identify_logical_components(self, query: str, detected_types: List[TaskType]) -> List[str]:
        """Identify logical components in a query without clear separators"""
        components = []
        
        # Look for questions
        questions = re.split(r'[?]', query)
        if len(questions) > 1:
            components.extend([q.strip() + '?' for q in questions[:-1] if q.strip()])
        
        # If multiple task types detected, try to split by type
        if len(detected_types) > 1 and not components:
            # This is a simplified approach - could be more sophisticated
            sentences = re.split(r'[.!]', query)
            components.extend([s.strip() for s in sentences if s.strip()])
        
        # Fallback: return the original query
        return components if components else [query]
    
    def _estimate_complexity(self, description: str, task_type: TaskType) -> int:
        """Estimate the complexity of a subtask (1-10)"""
        word_count = len(description.split())
        
        # Base complexity by task type
        type_complexity = {
            TaskType.QUESTION_ANSWERING: 2,
            TaskType.RESEARCH: 5,
            TaskType.ANALYSIS: 6,
            TaskType.CODING: 7,
            TaskType.WRITING: 6,
            TaskType.CALCULATION: 4,
            TaskType.DATA_PROCESSING: 7,
            TaskType.CREATIVE: 8,
            TaskType.TRANSLATION: 4,
            TaskType.SUMMARIZATION: 3,
        }
        
        base = type_complexity.get(task_type, 5)
        
        # Adjust based on description length
        if word_count < 5:
            complexity = base - 1
        elif word_count > 20:
            complexity = base + 2
        else:
            complexity = base
        
        return max(1, min(10, complexity))
    
    def visualize_decomposition(self, subtasks: List[SubTask]) -> str:
        """
        Create a text visualization of the task decomposition.
        
        Args:
            subtasks: List of subtasks to visualize
            
        Returns:
            String representation of the task structure
        """
        lines = ["Task Decomposition:", "=" * 50]
        
        for task in subtasks:
            lines.append(f"\n[{task.id}] {task.task_type.value.upper()}")
            lines.append(f"Description: {task.description}")
            lines.append(f"Priority: {task.priority}")
            lines.append(f"Complexity: {task.estimated_complexity}/10")
            if task.dependencies:
                lines.append(f"Dependencies: {', '.join(task.dependencies)}")
            lines.append("-" * 50)
        
        return "\n".join(lines)
    

