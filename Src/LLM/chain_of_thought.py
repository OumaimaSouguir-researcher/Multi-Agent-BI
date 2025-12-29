import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ReasoningStep(Enum):
    """Types of reasoning steps in chain of thought"""
    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    INFERENCE = "inference"
    CONCLUSION = "conclusion"
    VERIFICATION = "verification"


class ChainOfThought:
    """
    Implements chain-of-thought reasoning for LLM agents
    Breaks down complex problems into structured reasoning steps
    """
    
    def __init__(self):
        self.reasoning_chains: Dict[str, List[Dict[str, Any]]] = {}
        self.current_chain_id: Optional[str] = None
    
    def start_chain(self, chain_id: str, problem: str) -> Dict[str, Any]:
        """
        Start a new reasoning chain
        
        Args:
            chain_id: Unique identifier for this chain
            problem: Problem statement to reason about
            
        Returns:
            Initial chain state
        """
        self.current_chain_id = chain_id
        self.reasoning_chains[chain_id] = []
        
        initial_step = {
            "step_number": 0,
            "type": ReasoningStep.OBSERVATION.value,
            "content": f"Problem: {problem}",
            "timestamp": None
        }
        
        self.reasoning_chains[chain_id].append(initial_step)
        logger.info(f"Started reasoning chain: {chain_id}")
        
        return {
            "chain_id": chain_id,
            "problem": problem,
            "steps": [initial_step]
        }
    
    def add_step(
        self,
        chain_id: str,
        step_type: ReasoningStep,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a reasoning step to the chain
        
        Args:
            chain_id: Chain identifier
            step_type: Type of reasoning step
            content: Step content/reasoning
            metadata: Additional metadata
            
        Returns:
            Updated step information
        """
        if chain_id not in self.reasoning_chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        chain = self.reasoning_chains[chain_id]
        step_number = len(chain)
        
        step = {
            "step_number": step_number,
            "type": step_type.value,
            "content": content,
            "metadata": metadata or {}
        }
        
        chain.append(step)
        logger.debug(f"Added {step_type.value} step to chain {chain_id}")
        
        return step
    
    def get_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        """Get all steps in a reasoning chain"""
        return self.reasoning_chains.get(chain_id, [])
    
    def format_chain_as_prompt(self, chain_id: str) -> str:
        """
        Format reasoning chain as a prompt for LLM
        
        Args:
            chain_id: Chain identifier
            
        Returns:
            Formatted prompt string
        """
        chain = self.get_chain(chain_id)
        if not chain:
            return ""
        
        prompt = "Let's think through this step by step:\n\n"
        
        for step in chain:
            step_type = step["type"]
            content = step["content"]
            step_num = step["step_number"]
            
            prompt += f"Step {step_num} ({step_type}):\n{content}\n\n"
        
        prompt += "Based on this reasoning, what is the next logical step or conclusion?"
        
        return prompt
    
    def extract_reasoning_from_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract reasoning steps from LLM response
        Looks for structured reasoning patterns
        
        Args:
            response: LLM response text
            
        Returns:
            List of extracted reasoning steps
        """
        steps = []
        
        # Common reasoning markers
        reasoning_markers = [
            ("First,", ReasoningStep.OBSERVATION),
            ("Second,", ReasoningStep.ANALYSIS),
            ("Third,", ReasoningStep.INFERENCE),
            ("Therefore,", ReasoningStep.CONCLUSION),
            ("To verify,", ReasoningStep.VERIFICATION),
            ("Let's analyze", ReasoningStep.ANALYSIS),
            ("We can observe", ReasoningStep.OBSERVATION),
            ("This suggests", ReasoningStep.INFERENCE),
            ("In conclusion", ReasoningStep.CONCLUSION)
        ]
        
        lines = response.split('\n')
        current_step = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for reasoning markers
            found_marker = False
            for marker, step_type in reasoning_markers:
                if line.lower().startswith(marker.lower()):
                    # Save previous step if exists
                    if current_step and current_content:
                        steps.append({
                            "type": current_step.value,
                            "content": " ".join(current_content)
                        })
                    
                    current_step = step_type
                    current_content = [line]
                    found_marker = True
                    break
            
            if not found_marker and current_step:
                current_content.append(line)
        
        # Add final step
        if current_step and current_content:
            steps.append({
                "type": current_step.value,
                "content": " ".join(current_content)
            })
        
        return steps
    
    def build_cot_prompt(self, problem: str, examples: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Build a chain-of-thought prompt with optional examples
        
        Args:
            problem: Problem to solve
            examples: Optional few-shot examples
            
        Returns:
            Formatted CoT prompt
        """
        prompt = "Let's approach this problem using step-by-step reasoning:\n\n"
        
        # Add examples if provided
        if examples:
            prompt += "Here are some examples of step-by-step reasoning:\n\n"
            for i, example in enumerate(examples, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Problem: {example.get('problem', '')}\n"
                prompt += f"Reasoning:\n{example.get('reasoning', '')}\n"
                prompt += f"Answer: {example.get('answer', '')}\n\n"
        
        # Add current problem
        prompt += f"Now, let's solve this problem:\n{problem}\n\n"
        prompt += "Please provide your reasoning step by step:\n"
        prompt += "1. First, observe and understand the problem\n"
        prompt += "2. Analyze the key components\n"
        prompt += "3. Make logical inferences\n"
        prompt += "4. Draw a conclusion\n"
        prompt += "5. Verify your reasoning\n\n"
        prompt += "Your response:"
        
        return prompt
    
    def evaluate_reasoning_quality(self, chain_id: str) -> Dict[str, Any]:
        """
        Evaluate the quality of reasoning in a chain
        
        Args:
            chain_id: Chain identifier
            
        Returns:
            Quality metrics
        """
        chain = self.get_chain(chain_id)
        if not chain:
            return {"error": "Chain not found"}
        
        # Count step types
        step_types = {}
        for step in chain:
            step_type = step["type"]
            step_types[step_type] = step_types.get(step_type, 0) + 1
        
        # Check for logical progression
        expected_order = [
            ReasoningStep.OBSERVATION.value,
            ReasoningStep.ANALYSIS.value,
            ReasoningStep.INFERENCE.value,
            ReasoningStep.CONCLUSION.value
        ]
        
        actual_order = [step["type"] for step in chain if step["step_number"] > 0]
        has_logical_flow = self._check_logical_flow(actual_order, expected_order)
        
        return {
            "total_steps": len(chain),
            "step_distribution": step_types,
            "has_logical_flow": has_logical_flow,
            "completeness_score": self._calculate_completeness(step_types),
            "chain_summary": self._summarize_chain(chain)
        }
    
    def _check_logical_flow(self, actual: List[str], expected: List[str]) -> bool:
        """Check if reasoning follows expected logical flow"""
        if not actual:
            return False
        
        # Check if key reasoning types appear in reasonable order
        last_index = -1
        for exp_step in expected:
            if exp_step in actual:
                current_index = actual.index(exp_step)
                if current_index < last_index:
                    return False
                last_index = current_index
        
        return True
    
    def _calculate_completeness(self, step_types: Dict[str, int]) -> float:
        """Calculate completeness score based on reasoning coverage"""
        required_types = [
            ReasoningStep.OBSERVATION.value,
            ReasoningStep.ANALYSIS.value,
            ReasoningStep.CONCLUSION.value
        ]
        
        present = sum(1 for rt in required_types if rt in step_types)
        score = (present / len(required_types)) * 100
        
        return round(score, 2)
    
    def _summarize_chain(self, chain: List[Dict[str, Any]]) -> str:
        """Create a brief summary of the reasoning chain"""
        if len(chain) <= 1:
            return "Empty chain"
        
        first_step = chain[1] if len(chain) > 1 else chain[0]
        last_step = chain[-1]
        
        return f"Reasoning from {first_step['type']} to {last_step['type']} in {len(chain)} steps"
    
    def clear_chain(self, chain_id: str):
        """Remove a reasoning chain"""
        if chain_id in self.reasoning_chains:
            del self.reasoning_chains[chain_id]
            logger.info(f"Cleared reasoning chain: {chain_id}")
    
    def get_all_chains(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all reasoning chains"""
        return self.reasoning_chains.copy()