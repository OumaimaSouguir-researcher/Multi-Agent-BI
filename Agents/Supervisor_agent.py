"""
Supervisor Agent - Orchestrator
Responsible for coordinating all agents, managing workflows, and ensuring task completion
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import json
import asyncio
from collections import defaultdict


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority enumeration"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AgentType(Enum):
    """Types of agents in the system"""
    RESEARCHER = "researcher"
    DATA_ANALYST = "data_analyst"
    STRATEGIST = "strategist"
    VALIDATOR = "validator"


class Task:
    """Represents a task to be executed by an agent"""
    
    def __init__(self, task_id: str, task_type: str, agent_type: AgentType, 
                 data: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM):
        self.task_id = task_id
        self.task_type = task_type
        self.agent_type = agent_type
        self.data = data
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.dependencies = []
        self.retry_count = 0
        self.max_retries = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'agent_type': self.agent_type.value,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_count': self.retry_count
        }


class SupervisorAgent:
    """
    Supervisor Agent - Orchestrates all other agents and manages workflows
    Coordinates complex multi-agent tasks and ensures quality control
    """
    
    def __init__(self, agents: Optional[Dict[AgentType, Any]] = None):
        """
        Initialize the Supervisor Agent
        
        Args:
            agents: Dictionary mapping agent types to agent instances
        """
        self.agents = agents or {}
        self.task_queue: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.agent_performance: Dict[AgentType, Dict[str, Any]] = defaultdict(
            lambda: {'tasks_completed': 0, 'tasks_failed': 0, 'avg_duration': 0.0}
        )
        
    def register_agent(self, agent_type: AgentType, agent_instance: Any) -> None:
        """
        Register an agent with the supervisor
        
        Args:
            agent_type: Type of agent
            agent_instance: Instance of the agent
        """
        self.agents[agent_type] = agent_instance
        print(f"✓ Registered {agent_type.value} agent")
    
    def create_task(self, task_type: str, agent_type: AgentType, 
                   data: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM) -> Task:
        """
        Create a new task
        
        Args:
            task_type: Type of task
            agent_type: Agent to handle the task
            data: Task data
            priority: Task priority
            
        Returns:
            Created task
        """
        task_id = f"task_{len(self.task_queue) + len(self.completed_tasks) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        task = Task(task_id, task_type, agent_type, data, priority)
        self.task_queue.append(task)
        self._sort_task_queue()
        return task
    
    def _sort_task_queue(self) -> None:
        """Sort task queue by priority"""
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
    
    def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a single task
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        if task.agent_type not in self.agents:
            task.status = TaskStatus.FAILED
            task.error = f"Agent {task.agent_type.value} not registered"
            return {'success': False, 'error': task.error}
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        self.active_tasks[task.task_id] = task
        
        try:
            agent = self.agents[task.agent_type]
            
            # Execute based on agent type and task type
            if task.agent_type == AgentType.RESEARCHER:
                result = self._execute_research_task(agent, task)
            elif task.agent_type == AgentType.DATA_ANALYST:
                result = self._execute_analysis_task(agent, task)
            elif task.agent_type == AgentType.STRATEGIST:
                result = self._execute_strategy_task(agent, task)
            elif task.agent_type == AgentType.VALIDATOR:
                result = self._execute_validation_task(agent, task)
            else:
                raise ValueError(f"Unknown agent type: {task.agent_type}")
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Update performance metrics
            self._update_agent_performance(task)
            
            return {'success': True, 'result': result}
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.retry_count += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                print(f"⚠ Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                task.status = TaskStatus.PENDING
                self.task_queue.append(task)
                self._sort_task_queue()
            
            return {'success': False, 'error': str(e)}
        
        finally:
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            if task.status == TaskStatus.COMPLETED:
                self.completed_tasks[task.task_id] = task
    
    def _execute_research_task(self, agent: Any, task: Task) -> Dict[str, Any]:
        """Execute research task"""
        if task.task_type == 'gather_information':
            query = task.data.get('query', '')
            return agent.gather_information(query)
        elif task.task_type == 'search':
            query = task.data.get('query', '')
            num_results = task.data.get('num_results', 10)
            return agent.search(query, num_results)
        else:
            raise ValueError(f"Unknown research task type: {task.task_type}")
    
    def _execute_analysis_task(self, agent: Any, task: Task) -> Dict[str, Any]:
        """Execute data analysis task"""
        if task.task_type == 'analyze_data':
            data = task.data.get('data', [])
            return agent.analyze_data(data)
        elif task.task_type == 'statistical_analysis':
            data = task.data.get('data', [])
            return agent.perform_statistical_analysis(data)
        else:
            raise ValueError(f"Unknown analysis task type: {task.task_type}")
    
    def _execute_strategy_task(self, agent: Any, task: Task) -> Dict[str, Any]:
        """Execute strategy task"""
        if task.task_type == 'create_strategy':
            context = task.data.get('context', {})
            objectives = task.data.get('objectives', [])
            return agent.create_strategy(context, objectives)
        elif task.task_type == 'optimize_plan':
            plan = task.data.get('plan', {})
            return agent.optimize_plan(plan)
        else:
            raise ValueError(f"Unknown strategy task type: {task.task_type}")
    
    def _execute_validation_task(self, agent: Any, task: Task) -> Dict[str, Any]:
        """Execute validation task"""
        if task.task_type == 'validate_research':
            data = task.data.get('data', {})
            return agent.validate_research_output(data)
        elif task.task_type == 'validate_analysis':
            data = task.data.get('data', {})
            return agent.validate_analysis_output(data)
        elif task.task_type == 'validate_strategy':
            data = task.data.get('data', {})
            return agent.validate_strategy_output(data)
        elif task.task_type == 'cross_validate':
            outputs = task.data.get('outputs', {})
            return agent.cross_validate_outputs(outputs)
        else:
            raise ValueError(f"Unknown validation task type: {task.task_type}")
    
    def _update_agent_performance(self, task: Task) -> None:
        """Update agent performance metrics"""
        metrics = self.agent_performance[task.agent_type]
        
        if task.status == TaskStatus.COMPLETED:
            metrics['tasks_completed'] += 1
            
            # Calculate duration
            if task.started_at and task.completed_at:
                duration = (task.completed_at - task.started_at).total_seconds()
                total_tasks = metrics['tasks_completed']
                metrics['avg_duration'] = (
                    (metrics['avg_duration'] * (total_tasks - 1) + duration) / total_tasks
                )
        elif task.status == TaskStatus.FAILED:
            metrics['tasks_failed'] += 1
    
    def execute_workflow(self, workflow_name: str, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete workflow with multiple tasks
        
        Args:
            workflow_name: Name of the workflow
            workflow_config: Workflow configuration
            
        Returns:
            Workflow execution results
        """
        workflow_id = f"workflow_{len(self.workflows) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        workflow = {
            'workflow_id': workflow_id,
            'name': workflow_name,
            'config': workflow_config,
            'status': 'in_progress',
            'tasks': [],
            'results': {},
            'started_at': datetime.now(),
            'completed_at': None
        }
        
        self.workflows[workflow_id] = workflow
        
        try:
            # Execute workflow steps
            steps = workflow_config.get('steps', [])
            
            for step in steps:
                step_name = step.get('name', '')
                agent_type = AgentType(step.get('agent_type', ''))
                task_type = step.get('task_type', '')
                task_data = step.get('data', {})
                priority = TaskPriority(step.get('priority', TaskPriority.MEDIUM.value))
                
                # Create and execute task
                task = self.create_task(task_type, agent_type, task_data, priority)
                workflow['tasks'].append(task.task_id)
                
                result = self.execute_task(task)
                workflow['results'][step_name] = result
                
                # If task failed and is critical, stop workflow
                if not result['success'] and priority == TaskPriority.CRITICAL:
                    workflow['status'] = 'failed'
                    workflow['error'] = f"Critical task {step_name} failed: {result.get('error')}"
                    return workflow
                
                # Pass results to next step if configured
                if 'output_to_next' in step and result['success']:
                    next_step_index = steps.index(step) + 1
                    if next_step_index < len(steps):
                        steps[next_step_index]['data'].update({
                            'previous_result': result['result']
                        })
            
            workflow['status'] = 'completed'
            workflow['completed_at'] = datetime.now()
            
            # Log execution
            self._log_workflow_execution(workflow)
            
            return workflow
            
        except Exception as e:
            workflow['status'] = 'failed'
            workflow['error'] = str(e)
            workflow['completed_at'] = datetime.now()
            return workflow
    
    def create_research_analysis_workflow(self, query: str, objectives: List[str]) -> Dict[str, Any]:
        """
        Create a complete research -> analysis -> strategy -> validation workflow
        
        Args:
            query: Research query
            objectives: Strategic objectives
            
        Returns:
            Workflow execution results
        """
        workflow_config = {
            'steps': [
                {
                    'name': 'research',
                    'agent_type': AgentType.RESEARCHER.value,
                    'task_type': 'gather_information',
                    'data': {'query': query},
                    'priority': TaskPriority.HIGH.value,
                    'output_to_next': True
                },
                {
                    'name': 'validate_research',
                    'agent_type': AgentType.VALIDATOR.value,
                    'task_type': 'validate_research',
                    'data': {},
                    'priority': TaskPriority.MEDIUM.value
                },
                {
                    'name': 'analysis',
                    'agent_type': AgentType.DATA_ANALYST.value,
                    'task_type': 'analyze_data',
                    'data': {},
                    'priority': TaskPriority.HIGH.value,
                    'output_to_next': True
                },
                {
                    'name': 'validate_analysis',
                    'agent_type': AgentType.VALIDATOR.value,
                    'task_type': 'validate_analysis',
                    'data': {},
                    'priority': TaskPriority.MEDIUM.value
                },
                {
                    'name': 'strategy',
                    'agent_type': AgentType.STRATEGIST.value,
                    'task_type': 'create_strategy',
                    'data': {'objectives': objectives},
                    'priority': TaskPriority.HIGH.value,
                    'output_to_next': True
                },
                {
                    'name': 'validate_strategy',
                    'agent_type': AgentType.VALIDATOR.value,
                    'task_type': 'validate_strategy',
                    'data': {},
                    'priority': TaskPriority.MEDIUM.value
                },
                {
                    'name': 'cross_validation',
                    'agent_type': AgentType.VALIDATOR.value,
                    'task_type': 'cross_validate',
                    'data': {
                        'outputs': {
                            'research': None,
                            'analysis': None,
                            'strategy': None
                        }
                    },
                    'priority': TaskPriority.CRITICAL.value
                }
            ]
        }
        
        return self.execute_workflow('Research-Analysis-Strategy', workflow_config)
    
    def _log_workflow_execution(self, workflow: Dict[str, Any]) -> None:
        """Log workflow execution to history"""
        execution_log = {
            'workflow_id': workflow['workflow_id'],
            'name': workflow['name'],
            'status': workflow['status'],
            'duration': None,
            'tasks_count': len(workflow['tasks']),
            'timestamp': datetime.now().isoformat()
        }
        
        if workflow['started_at'] and workflow['completed_at']:
            duration = (workflow['completed_at'] - workflow['started_at']).total_seconds()
            execution_log['duration'] = duration
        
        self.execution_history.append(execution_log)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check active tasks
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].to_dict()
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id].to_dict()
        
        # Check task queue
        for task in self.task_queue:
            if task.task_id == task_id:
                return task.to_dict()
        
        return None
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow"""
        return self.workflows.get(workflow_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        for task in self.task_queue:
            if task.task_id == task_id:
                task.status = TaskStatus.CANCELLED
                self.task_queue.remove(task)
                self.completed_tasks[task_id] = task
                return True
        return False
    
    def get_agent_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all agents"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'agents': {}
        }
        
        for agent_type, metrics in self.agent_performance.items():
            total_tasks = metrics['tasks_completed'] + metrics['tasks_failed']
            success_rate = (metrics['tasks_completed'] / total_tasks * 100) if total_tasks > 0 else 0
            
            report['agents'][agent_type.value] = {
                'tasks_completed': metrics['tasks_completed'],
                'tasks_failed': metrics['tasks_failed'],
                'success_rate': f"{success_rate:.1f}%",
                'avg_duration_seconds': round(metrics['avg_duration'], 2)
            }
        
        return report
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'registered_agents': [agent_type.value for agent_type in self.agents.keys()],
            'tasks': {
                'pending': len(self.task_queue),
                'active': len(self.active_tasks),
                'completed': len(self.completed_tasks)
            },
            'workflows': {
                'total': len(self.workflows),
                'in_progress': sum(1 for w in self.workflows.values() if w['status'] == 'in_progress'),
                'completed': sum(1 for w in self.workflows.values() if w['status'] == 'completed'),
                'failed': sum(1 for w in self.workflows.values() if w['status'] == 'failed')
            }
        }
    
    def generate_execution_report(self) -> str:
        """Generate a comprehensive execution report"""
        status = self.get_system_status()
        performance = self.get_agent_performance_report()
        
        report = f"""
SUPERVISOR AGENT - EXECUTION REPORT
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM STATUS
-------------
Registered Agents: {', '.join(status['registered_agents'])}

Tasks:
- Pending: {status['tasks']['pending']}
- Active: {status['tasks']['active']}
- Completed: {status['tasks']['completed']}

Workflows:
- Total: {status['workflows']['total']}
- In Progress: {status['workflows']['in_progress']}
- Completed: {status['workflows']['completed']}
- Failed: {status['workflows']['failed']}

AGENT PERFORMANCE
-----------------
"""
        
        for agent_name, metrics in performance['agents'].items():
            report += f"\n{agent_name.upper()}:\n"
            report += f"  ✓ Completed: {metrics['tasks_completed']}\n"
            report += f"  ✗ Failed: {metrics['tasks_failed']}\n"
            report += f"  Success Rate: {metrics['success_rate']}\n"
            report += f"  Avg Duration: {metrics['avg_duration_seconds']}s\n"
        
        report += "\nRECENT WORKFLOWS\n----------------\n"
        recent_workflows = sorted(
            self.execution_history[-5:], 
            key=lambda x: x['timestamp'], 
            reverse=True
        )
        
        for workflow in recent_workflows:
            status_icon = "✓" if workflow['status'] == 'completed' else "✗"
            report += f"{status_icon} {workflow['name']} - {workflow['status']}"
            if workflow['duration']:
                report += f" ({workflow['duration']:.2f}s)"
            report += "\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize supervisor
    supervisor = SupervisorAgent()
    
    # Simulate registering agents (in real scenario, these would be actual agent instances)
    class MockResearcherAgent:
        def gather_information(self, query):
            return {'query': query, 'results': ['Result 1', 'Result 2']}
    
    class MockDataAnalystAgent:
        def analyze_data(self, data):
            return {'analysis': 'Data analyzed', 'insights': ['Insight 1']}
    
    class MockStrategistAgent:
        def create_strategy(self, context, objectives):
            return {'strategy': 'Strategic plan', 'objectives': objectives}
    
    class MockValidatorAgent:
        def validate_research_output(self, data):
            return {'is_valid': True, 'scores': {'completeness': 0.9}}
        
        def validate_analysis_output(self, data):
            return {'is_valid': True, 'scores': {'accuracy': 0.95}}
        
        def validate_strategy_output(self, data):
            return {'is_valid': True, 'scores': {'feasibility': 0.85}}
        
        def cross_validate_outputs(self, outputs):
            return {'is_consistent': True, 'consistency_score': 0.9}
    
    # Register agents
    supervisor.register_agent(AgentType.RESEARCHER, MockResearcherAgent())
    supervisor.register_agent(AgentType.DATA_ANALYST, MockDataAnalystAgent())
    supervisor.register_agent(AgentType.STRATEGIST, MockStrategistAgent())
    supervisor.register_agent(AgentType.VALIDATOR, MockValidatorAgent())
    
    # Execute a complete workflow
    print("\n" + "="*50)
    print("EXECUTING COMPLETE WORKFLOW")
    print("="*50)
    
    workflow_result = supervisor.create_research_analysis_workflow(
        query="Market analysis for AI products",
        objectives=["Increase market share", "Improve customer satisfaction"]
    )
    
    print(f"\nWorkflow Status: {workflow_result['status']}")
    print(f"Tasks Executed: {len(workflow_result['tasks'])}")
    
    # Generate reports
    print("\n" + supervisor.generate_execution_report())
    
    print("\n" + "="*50)
    print("SYSTEM STATUS")
    print("="*50)
    print(json.dumps(supervisor.get_system_status(), indent=2))