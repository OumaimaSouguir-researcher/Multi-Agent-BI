import logging
from typing import Dict, Any, Optional, List
import asyncio

from .ollama_client import OllamaClient
from .prompt_templates import PromptTemplates, AgentRole
from .response_parser import ResponseParser

logger = logging.getLogger(__name__)


class LLMManager:
    """
    High-level manager for LLM operations
    Coordinates between OllamaClient, prompts, and response parsing
    """
    
    def __init__(self):
        self.client = OllamaClient()
        self.parser = ResponseParser()
        self.templates = PromptTemplates()
        
        # Cache for frequently used prompts
        self._prompt_cache: Dict[str, str] = {}
    
    async def __aenter__(self):
        await self.client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()
    
    async def generate_agent_response(
        self,
        agent_role: AgentRole,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response for a specific agent
        
        Args:
            agent_role: Role of the agent
            task: Task description
            context: Additional context
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Parsed agent response
        """
        try:
            # Build prompts
            system_prompt, user_prompt = self.templates.build_agent_prompt(
                agent_role=agent_role,
                task=task,
                context=context
            )
            
            # Generate response
            result = await self.client.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if "error" in result:
                logger.error(f"LLM generation failed: {result['error']}")
                return {"error": result["error"], "agent_role": agent_role.value}
            
            # Parse response
            response_text = result.get("response", "")
            parsed = self.parser.parse_json_response(response_text)
            
            # Add metadata
            return {
                "agent_role": agent_role.value,
                "content": parsed if parsed else response_text,
                "raw_response": response_text,
                "metadata": {
                    "model": result.get("model"),
                    "tokens_generated": result.get("eval_count"),
                    "duration_ms": result.get("total_duration", 0) // 1_000_000
                }
            }
            
        except Exception as e:
            logger.error(f"Error in generate_agent_response: {e}")
            return {"error": str(e), "agent_role": agent_role.value}
    
    async def chat_with_agent(
        self,
        agent_role: AgentRole,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Multi-turn conversation with an agent
        
        Args:
            agent_role: Role of the agent
            messages: Conversation history
            temperature: Sampling temperature
            
        Returns:
            Agent response
        """
        try:
            # Add system message if not present
            if not messages or messages[0].get("role") != "system":
                system_prompt = self.templates.get_system_prompt(agent_role)
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            # Generate chat response
            result = await self.client.chat(
                messages=messages,
                temperature=temperature
            )
            
            if "error" in result:
                logger.error(f"Chat failed: {result['error']}")
                return {"error": result["error"]}
            
            message = result.get("message", {})
            content = message.get("content", "")
            
            # Try to parse as JSON
            parsed = self.parser.parse_json_response(content)
            
            return {
                "role": message.get("role"),
                "content": parsed if parsed else content,
                "raw_content": content,
                "metadata": {
                    "model": result.get("model"),
                    "tokens": result.get("eval_count")
                }
            }
            
        except Exception as e:
            logger.error(f"Error in chat_with_agent: {e}")
            return {"error": str(e)}
    
    async def batch_generate(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple LLM requests concurrently
        
        Args:
            requests: List of request dicts with 'agent_role', 'task', 'context'
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(request: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.generate_agent_response(
                    agent_role=request.get("agent_role"),
                    task=request.get("task"),
                    context=request.get("context"),
                    temperature=request.get("temperature"),
                    max_tokens=request.get("max_tokens")
                )
        
        tasks = [process_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} failed: {result}")
                responses.append({"error": str(result), "request_index": i})
            else:
                responses.append(result)
        
        return responses
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        return self.client.get_statistics()