import aiohttp
import logging
from typing import Optional, Dict, Any, List, AsyncIterator
import json
import asyncio

from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OllamaClient:
    """
    Async client for Ollama LLM API
    Handles text generation, chat, and embeddings
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        self.base_url = base_url or settings.OLLAMA_HOST
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens_generated": 0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def connect(self):
        """Create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"Connected to Ollama at {self.base_url}")
    
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Disconnected from Ollama")
    
    async def health_check(self) -> bool:
        """Check if Ollama server is available"""
        try:
            if not self.session:
                await self.connect()
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        try:
            if not self.session:
                await self.connect()
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    logger.error(f"Failed to list models: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text completion
        
        Args:
            prompt: Input prompt
            model: Model to use (default: self.model)
            system: System prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            stream: Stream response
            
        Returns:
            Generated response with metadata
        """
        if not self.session:
            await self.connect()
        
        self.stats["total_requests"] += 1
        
        # Build request payload
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {}
        }
        
        # Add optional parameters
        if system:
            payload["system"] = system
        
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        else:
            payload["options"]["temperature"] = settings.OLLAMA_TEMPERATURE
        
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        else:
            payload["options"]["num_predict"] = settings.OLLAMA_MAX_TOKENS
        
        if top_p is not None:
            payload["options"]["top_p"] = top_p
        else:
            payload["options"]["top_p"] = settings.OLLAMA_TOP_P
        
        if stop:
            payload["options"]["stop"] = stop
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    if stream:
                        return {"stream": self._stream_response(response)}
                    else:
                        result = await response.json()
                        self.stats["successful_requests"] += 1
                        
                        # Update token statistics
                        if "eval_count" in result:
                            self.stats["total_tokens_generated"] += result["eval_count"]
                        
                        return {
                            "response": result.get("response", ""),
                            "model": result.get("model"),
                            "created_at": result.get("created_at"),
                            "done": result.get("done"),
                            "total_duration": result.get("total_duration"),
                            "load_duration": result.get("load_duration"),
                            "prompt_eval_count": result.get("prompt_eval_count"),
                            "eval_count": result.get("eval_count"),
                            "eval_duration": result.get("eval_duration")
                        }
                else:
                    self.stats["failed_requests"] += 1
                    error_text = await response.text()
                    logger.error(f"Ollama generate error: {error_text}")
                    return {"error": error_text, "status": response.status}
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Error calling Ollama generate: {e}")
            return {"error": str(e)}
    
    async def _stream_response(self, response) -> AsyncIterator[str]:
        """Stream response chunks"""
        async for line in response.content:
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                except json.JSONDecodeError:
                    continue
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Chat completion with conversation history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{"role": "user", "content": "Hello"}]
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Stream response
            
        Returns:
            Chat response with metadata
        """
        if not self.session:
            await self.connect()
        
        self.stats["total_requests"] += 1
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "options": {}
        }
        
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        else:
            payload["options"]["temperature"] = settings.OLLAMA_TEMPERATURE
        
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status == 200:
                    if stream:
                        return {"stream": self._stream_chat_response(response)}
                    else:
                        result = await response.json()
                        self.stats["successful_requests"] += 1
                        
                        return {
                            "message": result.get("message", {}),
                            "model": result.get("model"),
                            "created_at": result.get("created_at"),
                            "done": result.get("done"),
                            "total_duration": result.get("total_duration"),
                            "eval_count": result.get("eval_count")
                        }
                else:
                    self.stats["failed_requests"] += 1
                    error_text = await response.text()
                    logger.error(f"Ollama chat error: {error_text}")
                    return {"error": error_text, "status": response.status}
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Error calling Ollama chat: {e}")
            return {"error": str(e)}
    
    async def _stream_chat_response(self, response) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat response chunks"""
        async for line in response.content:
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data:
                        yield data["message"]
                except json.JSONDecodeError:
                    continue
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        success_rate = 0
        if self.stats["total_requests"] > 0:
            success_rate = (
                self.stats["successful_requests"] / self.stats["total_requests"] * 100
            )
        
        return {
            **self.stats,
            "success_rate": f"{success_rate:.2f}%"
        }