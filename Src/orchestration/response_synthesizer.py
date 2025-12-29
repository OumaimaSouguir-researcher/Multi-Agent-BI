"""
Response Synthesizer Module

Combines and synthesizes results from multiple agents into coherent,
user-friendly responses.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from workflow_engine import TaskResult, WorkflowResult, TaskStatus
from task_decomposer import SubTask, TaskType


class SynthesisStrategy(Enum):
    """Strategy for synthesizing multiple results"""
    SEQUENTIAL = "sequential"  # Present results in order
    HIERARCHICAL = "hierarchical"  # Organize by importance/hierarchy
    NARRATIVE = "narrative"  # Create flowing narrative
    SUMMARY = "summary"  # Condense into summary
    STRUCTURED = "structured"  # Present as structured data


@dataclass
class SynthesizedResponse:
    """Final synthesized response"""
    content: str
    structure: Dict[str, Any]
    metadata: Dict[str, Any]
    confidence: float
    sources: List[str]
    warnings: List[str]


class ResponseSynthesizer:
    """
    Synthesizes results from multiple agents into coherent responses.
    """
    
    def __init__(self, default_strategy: SynthesisStrategy = SynthesisStrategy.NARRATIVE):
        """
        Initialize the ResponseSynthesizer.
        
        Args:
            default_strategy: Default synthesis strategy to use
        """
        self.default_strategy = default_strategy
        self.strategies = {
            SynthesisStrategy.SEQUENTIAL: self._synthesize_sequential,
            SynthesisStrategy.HIERARCHICAL: self._synthesize_hierarchical,
            SynthesisStrategy.NARRATIVE: self._synthesize_narrative,
            SynthesisStrategy.SUMMARY: self._synthesize_summary,
            SynthesisStrategy.STRUCTURED: self._synthesize_structured
        }
    
    def synthesize(
        self,
        workflow_result: WorkflowResult,
        tasks: List[SubTask],
        strategy: Optional[SynthesisStrategy] = None,
        user_query: Optional[str] = None
    ) -> SynthesizedResponse:
        """
        Synthesize workflow results into a final response.
        
        Args:
            workflow_result: Results from workflow execution
            tasks: Original list of subtasks
            strategy: Synthesis strategy to use (uses default if None)
            user_query: Original user query for context
            
        Returns:
            SynthesizedResponse with synthesized content
        """
        strategy = strategy or self.default_strategy
        
        # Filter successful results
        successful_results = [
            result for result in workflow_result.task_results
            if result.status == TaskStatus.COMPLETED
        ]
        
        failed_results = [
            result for result in workflow_result.task_results
            if result.status == TaskStatus.FAILED
        ]
        
        # Check if we have any results
        if not successful_results:
            return self._create_error_response(workflow_result, failed_results)
        
        # Get synthesis function
        synthesis_func = self.strategies.get(
            strategy,
            self._synthesize_narrative
        )
        
        # Create task lookup
        task_lookup = {task.id: task for task in tasks}
        
        # Synthesize content
        content = synthesis_func(successful_results, task_lookup, user_query)
        
        # Extract metadata
        metadata = self._extract_metadata(workflow_result, successful_results)
        
        # Calculate confidence
        confidence = self._calculate_confidence(successful_results)
        
        # Extract sources
        sources = self._extract_sources(successful_results)
        
        # Generate warnings
        warnings = self._generate_warnings(workflow_result, failed_results)
        
        # Create structured data
        structure = self._create_structure(successful_results, task_lookup)
        
        return SynthesizedResponse(
            content=content,
            structure=structure,
            metadata=metadata,
            confidence=confidence,
            sources=sources,
            warnings=warnings
        )
    
    def _synthesize_sequential(
        self,
        results: List[TaskResult],
        task_lookup: Dict[str, SubTask],
        user_query: Optional[str]
    ) -> str:
        """Synthesize results in sequential order"""
        parts = []
        
        if user_query:
            parts.append(f"Response to: {user_query}\n")
        
        for idx, result in enumerate(results, 1):
            task = task_lookup.get(result.task_id)
            if task:
                parts.append(f"\n{idx}. {task.description}")
                parts.append(self._format_result_output(result))
        
        return "\n".join(parts)
    
    def _synthesize_hierarchical(
        self,
        results: List[TaskResult],
        task_lookup: Dict[str, SubTask],
        user_query: Optional[str]
    ) -> str:
        """Synthesize results in hierarchical structure"""
        parts = []
        
        if user_query:
            parts.append(f"# {user_query}\n")
        
        # Group by task type
        by_type: Dict[TaskType, List[TaskResult]] = {}
        for result in results:
            task = task_lookup.get(result.task_id)
            if task:
                if task.task_type not in by_type:
                    by_type[task.task_type] = []
                by_type[task.task_type].append(result)
        
        # Present each type
        for task_type, type_results in by_type.items():
            parts.append(f"\n## {task_type.value.title()}")
            for result in type_results:
                parts.append(self._format_result_output(result))
        
        return "\n".join(parts)
    
    def _synthesize_narrative(
        self,
        results: List[TaskResult],
        task_lookup: Dict[str, SubTask],
        user_query: Optional[str]
    ) -> str:
        """Synthesize results into a flowing narrative"""
        parts = []
        
        if user_query:
            parts.append(f"Regarding your query about {user_query}:\n")
        
        # Create narrative flow
        for idx, result in enumerate(results):
            task = task_lookup.get(result.task_id)
            if not task:
                continue
            
            # Add transition
            if idx == 0:
                transition = "To begin with, "
            elif idx == len(results) - 1:
                transition = "Finally, "
            else:
                transition = "Additionally, "
            
            # Format based on task type
            content = self._format_narrative_content(result, task)
            parts.append(f"{transition}{content}")
        
        return "\n\n".join(parts)
    
    def _synthesize_summary(
        self,
        results: List[TaskResult],
        task_lookup: Dict[str, SubTask],
        user_query: Optional[str]
    ) -> str:
        """Synthesize results into a condensed summary"""
        parts = []
        
        if user_query:
            parts.append(f"Summary for: {user_query}\n")
        
        # Extract key findings
        key_findings = []
        for result in results:
            finding = self._extract_key_finding(result)
            if finding:
                key_findings.append(finding)
        
        if key_findings:
            parts.append("Key Findings:")
            for finding in key_findings:
                parts.append(f"• {finding}")
        
        # Add conclusion
        conclusion = self._generate_conclusion(results, task_lookup)
        if conclusion:
            parts.append(f"\nConclusion: {conclusion}")
        
        return "\n".join(parts)
    
    def _synthesize_structured(
        self,
        results: List[TaskResult],
        task_lookup: Dict[str, SubTask],
        user_query: Optional[str]
    ) -> str:
        """Synthesize results into structured format"""
        parts = []
        
        if user_query:
            parts.append(f"Query: {user_query}\n")
        
        parts.append("Results:")
        parts.append("-" * 60)
        
        for result in results:
            task = task_lookup.get(result.task_id)
            if task:
                parts.append(f"\nTask: {task.description}")
                parts.append(f"Type: {task.task_type.value}")
                parts.append(f"Status: {result.status.value}")
                
                if isinstance(result.output, dict):
                    for key, value in result.output.items():
                        parts.append(f"  {key}: {value}")
                else:
                    parts.append(f"  Output: {result.output}")
                
                parts.append("-" * 60)
        
        return "\n".join(parts)
    
    def _format_result_output(self, result: TaskResult) -> str:
        """Format a result's output for display"""
        if isinstance(result.output, dict):
            # Extract main content from dict
            if "content" in result.output:
                return str(result.output["content"])
            elif "answer" in result.output:
                return str(result.output["answer"])
            elif "findings" in result.output:
                return str(result.output["findings"])
            else:
                # Format as key-value pairs
                items = [f"{k}: {v}" for k, v in result.output.items()
                        if k not in ["type", "confidence"]]
                return "\n".join(items)
        return str(result.output)
    
    def _format_narrative_content(self, result: TaskResult, task: SubTask) -> str:
        """Format content for narrative style"""
        output = result.output
        
        if isinstance(output, dict):
            task_type = task.task_type.value
            
            if task_type == "research":
                findings = output.get("findings", "")
                return f"research indicates that {findings}"
            
            elif task_type == "analysis":
                findings = output.get("findings", "")
                return f"analysis reveals that {findings}"
            
            elif task_type == "coding":
                return "the requested code has been generated successfully"
            
            elif task_type == "writing":
                return "the content has been written as requested"
            
            elif task_type == "calculation":
                result_val = output.get("result", "")
                return f"calculations show that the result is {result_val}"
            
            elif task_type == "question_answering":
                answer = output.get("answer", "")
                return f"the answer is: {answer}"
            
            else:
                return f"regarding '{task.description}', the result is: {self._format_result_output(result)}"
        
        return f"for '{task.description}': {output}"
    
    def _extract_key_finding(self, result: TaskResult) -> Optional[str]:
        """Extract key finding from a result"""
        if isinstance(result.output, dict):
            # Look for key information
            for key in ["findings", "answer", "summary", "result", "content"]:
                if key in result.output:
                    value = result.output[key]
                    if isinstance(value, str) and len(value) < 200:
                        return value
                    elif isinstance(value, str):
                        return value[:197] + "..."
        
        output_str = str(result.output)
        if len(output_str) < 200:
            return output_str
        return output_str[:197] + "..."
    
    def _generate_conclusion(
        self,
        results: List[TaskResult],
        task_lookup: Dict[str, SubTask]
    ) -> str:
        """Generate a conclusion from results"""
        if len(results) == 1:
            return "The task has been completed successfully."
        
        task_types = set()
        for result in results:
            task = task_lookup.get(result.task_id)
            if task:
                task_types.add(task.task_type.value)
        
        if len(task_types) > 1:
            return f"Multiple tasks involving {', '.join(task_types)} have been completed successfully."
        else:
            return f"All {list(task_types)[0]} tasks have been completed."
    
    def _extract_metadata(
        self,
        workflow_result: WorkflowResult,
        results: List[TaskResult]
    ) -> Dict[str, Any]:
        """Extract metadata from workflow and results"""
        return {
            "workflow_id": workflow_result.workflow_id,
            "total_tasks": len(workflow_result.task_results),
            "successful_tasks": len(results),
            "execution_time": workflow_result.total_execution_time,
            "start_time": workflow_result.start_time.isoformat(),
            "end_time": workflow_result.end_time.isoformat()
        }
    
    def _calculate_confidence(self, results: List[TaskResult]) -> float:
        """Calculate overall confidence score"""
        if not results:
            return 0.0
        
        confidences = []
        for result in results:
            if isinstance(result.output, dict) and "confidence" in result.output:
                confidences.append(result.output["confidence"])
            else:
                # Default confidence for results without explicit confidence
                confidences.append(0.75)
        
        return sum(confidences) / len(confidences)
    
    def _extract_sources(self, results: List[TaskResult]) -> List[str]:
        """Extract source information from results"""
        sources = []
        
        for result in results:
            if isinstance(result.output, dict):
                if "sources" in result.output:
                    result_sources = result.output["sources"]
                    if isinstance(result_sources, list):
                        sources.extend(result_sources)
                    else:
                        sources.append(str(result_sources))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sources = []
        for source in sources:
            if source not in seen:
                seen.add(source)
                unique_sources.append(source)
        
        return unique_sources
    
    def _generate_warnings(
        self,
        workflow_result: WorkflowResult,
        failed_results: List[TaskResult]
    ) -> List[str]:
        """Generate warnings based on workflow execution"""
        warnings = []
        
        # Check for failures
        if failed_results:
            warnings.append(
                f"{len(failed_results)} task(s) failed during execution"
            )
        
        # Check for long execution time
        if workflow_result.total_execution_time > 60:
            warnings.append(
                f"Workflow took longer than expected ({workflow_result.total_execution_time:.1f}s)"
            )
        
        # Check for low confidence results
        low_confidence_count = 0
        for result in workflow_result.task_results:
            if isinstance(result.output, dict) and "confidence" in result.output:
                if result.output["confidence"] < 0.5:
                    low_confidence_count += 1
        
        if low_confidence_count > 0:
            warnings.append(
                f"{low_confidence_count} result(s) have low confidence scores"
            )
        
        return warnings
    
    def _create_structure(
        self,
        results: List[TaskResult],
        task_lookup: Dict[str, SubTask]
    ) -> Dict[str, Any]:
        """Create structured data representation"""
        structure = {
            "results": [],
            "summary": {},
            "statistics": {}
        }
        
        # Add individual results
        for result in results:
            task = task_lookup.get(result.task_id)
            structure["results"].append({
                "task_id": result.task_id,
                "task_description": task.description if task else "Unknown",
                "task_type": task.task_type.value if task else "unknown",
                "output": result.output,
                "execution_time": result.execution_time,
                "status": result.status.value
            })
        
        # Add statistics
        structure["statistics"] = {
            "total_results": len(results),
            "total_execution_time": sum(r.execution_time for r in results),
            "average_execution_time": sum(r.execution_time for r in results) / len(results) if results else 0
        }
        
        return structure
    
    def _create_error_response(
        self,
        workflow_result: WorkflowResult,
        failed_results: List[TaskResult]
    ) -> SynthesizedResponse:
        """Create response for failed workflows"""
        error_messages = [
            result.error for result in failed_results
            if result.error
        ]
        
        content = "The workflow encountered errors during execution:\n"
        content += "\n".join(f"• {error}" for error in error_messages)
        
        return SynthesizedResponse(
            content=content,
            structure={"errors": error_messages},
            metadata={
                "workflow_id": workflow_result.workflow_id,
                "failed_tasks": len(failed_results)
            },
            confidence=0.0,
            sources=[],
            warnings=["Workflow failed to complete successfully"]
        )
    
    def format_for_display(
        self,
        response: SynthesizedResponse,
        include_metadata: bool = True,
        include_warnings: bool = True
    ) -> str:
        """
        Format synthesized response for user display.
        
        Args:
            response: SynthesizedResponse to format
            include_metadata: Whether to include metadata
            include_warnings: Whether to include warnings
            
        Returns:
            Formatted string for display
        """
        parts = [response.content]
        
        if include_metadata and response.metadata:
            parts.append("\n" + "=" * 60)
            parts.append("Metadata:")
            parts.append(f"Confidence: {response.confidence:.2%}")
            parts.append(f"Execution time: {response.metadata.get('execution_time', 0):.2f}s")
            parts.append(f"Tasks completed: {response.metadata.get('successful_tasks', 0)}/{response.metadata.get('total_tasks', 0)}")
        
        if include_warnings and response.warnings:
            parts.append("\nWarnings:")
            for warning in response.warnings:
                parts.append(f"⚠ {warning}")
        
        if response.sources:
            parts.append("\nSources:")
            for source in response.sources:
                parts.append(f"• {source}")
        
        return "\n".join(parts)