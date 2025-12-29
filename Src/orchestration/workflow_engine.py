"""
Workflow Engine Module

Orchestrates the execution of tasks across multiple agents,
managing dependencies, parallel execution, and error handling.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import time
from task_decomposer import SubTask, TaskType
from agent_selector import AgentSelector, AgentAssignment


class TaskStatus(Enum):
    """Status of a task in the workflow"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Overall workflow status"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution"""
    task_id: str
    status: TaskStatus
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Complete workflow execution result"""
    workflow_id: str
    status: WorkflowStatus
    task_results: List[TaskResult]
    total_execution_time: float
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """
    Orchestrates task execution across multiple agents with dependency management.
    """
    
    def __init__(
        self,
        agent_selector: AgentSelector,
        max_parallel_tasks: int = 5,
        timeout: float = 300.0
    ):
        """
        Initialize the WorkflowEngine.
        
        Args:
            agent_selector: AgentSelector instance for agent assignment
            max_parallel_tasks: Maximum number of tasks to run in parallel
            timeout: Maximum time in seconds for workflow execution
        """
        self.agent_selector = agent_selector
        self.max_parallel_tasks = max_parallel_tasks
        self.timeout = timeout
        self.workflows: Dict[str, WorkflowResult] = {}
        self._task_executors: Dict[str, Callable] = {}
        self._register_default_executors()
    
    def _register_default_executors(self):
        """Register default task executors for each task type"""
        self._task_executors[TaskType.RESEARCH.value] = self._execute_research_task
        self._task_executors[TaskType.ANALYSIS.value] = self._execute_analysis_task
        self._task_executors[TaskType.CODING.value] = self._execute_coding_task
        self._task_executors[TaskType.WRITING.value] = self._execute_writing_task
        self._task_executors[TaskType.CALCULATION.value] = self._execute_calculation_task
        self._task_executors[TaskType.DATA_PROCESSING.value] = self._execute_data_processing_task
        self._task_executors[TaskType.CREATIVE.value] = self._execute_creative_task
        self._task_executors[TaskType.TRANSLATION.value] = self._execute_translation_task
        self._task_executors[TaskType.SUMMARIZATION.value] = self._execute_summarization_task
        self._task_executors[TaskType.QUESTION_ANSWERING.value] = self._execute_qa_task
    
    def register_executor(self, task_type: str, executor: Callable):
        """Register a custom executor for a task type"""
        self._task_executors[task_type] = executor
    
    async def execute_workflow(
        self,
        workflow_id: str,
        tasks: List[SubTask],
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Execute a complete workflow of tasks.
        
        Args:
            workflow_id: Unique identifier for this workflow
            tasks: List of SubTask objects to execute
            context: Optional shared context for all tasks
            
        Returns:
            WorkflowResult containing all task results
        """
        start_time = datetime.now()
        context = context or {}
        
        # Initialize workflow
        workflow_result = WorkflowResult(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            task_results=[],
            total_execution_time=0.0,
            start_time=start_time,
            end_time=start_time,
            metadata={"total_tasks": len(tasks)}
        )
        
        self.workflows[workflow_id] = workflow_result
        
        # Create task status tracking
        task_statuses: Dict[str, TaskStatus] = {
            task.id: TaskStatus.PENDING for task in tasks
        }
        
        task_results: Dict[str, TaskResult] = {}
        
        try:
            # Execute tasks respecting dependencies
            await self._execute_tasks_with_dependencies(
                tasks,
                task_statuses,
                task_results,
                context
            )
            
            # Finalize workflow
            workflow_result.task_results = list(task_results.values())
            workflow_result.status = self._determine_workflow_status(task_results)
            
        except asyncio.TimeoutError:
            workflow_result.status = WorkflowStatus.FAILED
            workflow_result.metadata["error"] = "Workflow execution timeout"
            
        except Exception as e:
            workflow_result.status = WorkflowStatus.FAILED
            workflow_result.metadata["error"] = str(e)
        
        finally:
            workflow_result.end_time = datetime.now()
            workflow_result.total_execution_time = (
                workflow_result.end_time - workflow_result.start_time
            ).total_seconds()
        
        return workflow_result
    
    async def _execute_tasks_with_dependencies(
        self,
        tasks: List[SubTask],
        task_statuses: Dict[str, TaskStatus],
        task_results: Dict[str, TaskResult],
        context: Dict[str, Any]
    ):
        """Execute tasks respecting dependencies and parallelization limits"""
        tasks_by_id = {task.id: task for task in tasks}
        
        while any(status == TaskStatus.PENDING or status == TaskStatus.READY 
                  for status in task_statuses.values()):
            
            # Find ready tasks (dependencies met)
            ready_tasks = []
            for task_id, status in task_statuses.items():
                if status == TaskStatus.PENDING:
                    task = tasks_by_id[task_id]
                    if self._are_dependencies_met(task, task_statuses):
                        task_statuses[task_id] = TaskStatus.READY
                        ready_tasks.append(task)
            
            # Execute ready tasks in parallel (up to max_parallel_tasks)
            if ready_tasks:
                batch_size = min(len(ready_tasks), self.max_parallel_tasks)
                batch = ready_tasks[:batch_size]
                
                # Execute batch
                batch_results = await asyncio.gather(*[
                    self._execute_task(task, task_statuses, context, task_results)
                    for task in batch
                ], return_exceptions=True)
                
                # Process results
                for task, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        task_results[task.id] = TaskResult(
                            task_id=task.id,
                            status=TaskStatus.FAILED,
                            output=None,
                            error=str(result)
                        )
                        task_statuses[task.id] = TaskStatus.FAILED
                    else:
                        task_results[task.id] = result
                        task_statuses[task.id] = result.status
            
            else:
                # No ready tasks, check for deadlock
                if all(status in [TaskStatus.COMPLETED, TaskStatus.FAILED] 
                       for status in task_statuses.values()):
                    break
                else:
                    # Avoid busy waiting
                    await asyncio.sleep(0.1)
    
    def _are_dependencies_met(
        self,
        task: SubTask,
        task_statuses: Dict[str, TaskStatus]
    ) -> bool:
        """Check if all dependencies for a task are completed"""
        if not task.dependencies:
            return True
        
        return all(
            task_statuses.get(dep_id) == TaskStatus.COMPLETED
            for dep_id in task.dependencies
        )
    
    async def _execute_task(
        self,
        task: SubTask,
        task_statuses: Dict[str, TaskStatus],
        context: Dict[str, Any],
        completed_results: Dict[str, TaskResult]
    ) -> TaskResult:
        """Execute a single task"""
        task_statuses[task.id] = TaskStatus.RUNNING
        start_time = time.time()
        
        # Select agent for task
        assignment = self.agent_selector.select_agent(
            task_id=task.id,
            task_type=task.task_type.value,
            complexity=task.estimated_complexity,
            context=context
        )
        
        if not assignment:
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                output=None,
                error="No available agent for task",
                execution_time=0.0
            )
        
        try:
            # Get executor for task type
            executor = self._task_executors.get(
                task.task_type.value,
                self._execute_default_task
            )
            
            # Prepare task context with dependency results
            task_context = context.copy()
            task_context["dependencies"] = {
                dep_id: completed_results[dep_id].output
                for dep_id in task.dependencies
                if dep_id in completed_results
            }
            
            # Execute task
            output = await executor(task, assignment, task_context)
            
            execution_time = time.time() - start_time
            
            # Release agent with success score
            self.agent_selector.release_agent(assignment.agent_id, 1.0)
            
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                output=output,
                execution_time=execution_time,
                metadata={
                    "agent_id": assignment.agent_id,
                    "confidence": assignment.confidence_score
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Release agent with failure score
            if assignment:
                self.agent_selector.release_agent(assignment.agent_id, 0.0)
            
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                output=None,
                error=str(e),
                execution_time=execution_time
            )
    
    # Task-specific executors (these would call actual AI models in production)
    
    async def _execute_research_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a research task"""
        await asyncio.sleep(0.5)  # Simulate processing
        return {
            "type": "research",
            "query": task.description,
            "findings": f"Research results for: {task.description}",
            "sources": ["source1.com", "source2.com"],
            "confidence": 0.85
        }
    
    async def _execute_analysis_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an analysis task"""
        await asyncio.sleep(0.5)
        return {
            "type": "analysis",
            "subject": task.description,
            "findings": f"Analysis of: {task.description}",
            "insights": ["Insight 1", "Insight 2", "Insight 3"],
            "confidence": 0.82
        }
    
    async def _execute_coding_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a coding task"""
        await asyncio.sleep(0.7)
        return {
            "type": "coding",
            "description": task.description,
            "code": f"# Generated code for: {task.description}\ndef solution():\n    pass",
            "language": "python",
            "confidence": 0.9
        }
    
    async def _execute_writing_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a writing task"""
        await asyncio.sleep(0.6)
        return {
            "type": "writing",
            "topic": task.description,
            "content": f"Written content for: {task.description}",
            "word_count": 250,
            "confidence": 0.88
        }
    
    async def _execute_calculation_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a calculation task"""
        await asyncio.sleep(0.3)
        return {
            "type": "calculation",
            "problem": task.description,
            "result": 42.0,  # Placeholder result
            "steps": ["Step 1", "Step 2", "Step 3"],
            "confidence": 0.95
        }
    
    async def _execute_data_processing_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a data processing task"""
        await asyncio.sleep(0.8)
        return {
            "type": "data_processing",
            "operation": task.description,
            "processed_records": 1000,
            "output_format": "json",
            "confidence": 0.87
        }
    
    async def _execute_creative_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a creative task"""
        await asyncio.sleep(0.9)
        return {
            "type": "creative",
            "prompt": task.description,
            "creation": f"Creative output for: {task.description}",
            "style": "imaginative",
            "confidence": 0.75
        }
    
    async def _execute_translation_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a translation task"""
        await asyncio.sleep(0.4)
        return {
            "type": "translation",
            "source": task.description,
            "translated": f"Translated: {task.description}",
            "source_lang": "en",
            "target_lang": "es",
            "confidence": 0.92
        }
    
    async def _execute_summarization_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a summarization task"""
        await asyncio.sleep(0.4)
        return {
            "type": "summarization",
            "original": task.description,
            "summary": f"Summary of: {task.description}",
            "compression_ratio": 0.3,
            "confidence": 0.86
        }
    
    async def _execute_qa_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a question answering task"""
        await asyncio.sleep(0.3)
        return {
            "type": "question_answering",
            "question": task.description,
            "answer": f"Answer to: {task.description}",
            "sources": ["knowledge_base"],
            "confidence": 0.84
        }
    
    async def _execute_default_task(
        self,
        task: SubTask,
        assignment: AgentAssignment,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Default executor for unknown task types"""
        await asyncio.sleep(0.5)
        return {
            "type": "general",
            "task": task.description,
            "output": f"Processed: {task.description}",
            "confidence": 0.7
        }
    
    def _determine_workflow_status(
        self,
        task_results: Dict[str, TaskResult]
    ) -> WorkflowStatus:
        """Determine overall workflow status from task results"""
        if not task_results:
            return WorkflowStatus.FAILED
        
        statuses = [result.status for result in task_results.values()]
        
        if all(status == TaskStatus.COMPLETED for status in statuses):
            return WorkflowStatus.COMPLETED
        elif any(status == TaskStatus.FAILED for status in statuses):
            return WorkflowStatus.FAILED
        else:
            return WorkflowStatus.RUNNING
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        completed_tasks = sum(
            1 for result in workflow.task_results
            if result.status == TaskStatus.COMPLETED
        )
        
        return {
            "workflow_id": workflow.workflow_id,
            "status": workflow.status.value,
            "progress": f"{completed_tasks}/{workflow.metadata.get('total_tasks', 0)}",
            "execution_time": workflow.total_execution_time,
            "start_time": workflow.start_time.isoformat(),
            "completed_tasks": completed_tasks
        }
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.CANCELLED
                return True
        return False
    

