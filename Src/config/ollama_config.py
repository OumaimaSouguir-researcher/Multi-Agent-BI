from typing import Optional, Dict, Any, List
import aiohttp
import logging

from .settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OllamaConfig:
    """Ollama LLM configuration and client"""
    
    def __init__(self):
        self.settings = settings
        self.base_url = self.settings.OLLAMA_HOST
        self.model = self.settings.OLLAMA_MODEL
        self.embedding_model = self.settings.OLLAMA_EMBEDDING_MODEL
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.settings.OLLAMA_TIMEOUT)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if Ollama server is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("models", [])
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
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text completion
        
        Args:
            prompt: Input prompt
            model: Model name (default: settings.OLLAMA_MODEL)
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Stream response
            
        Returns:
            Generated text response
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature or self.settings.OLLAMA_TEMPERATURE,
                "top_p": self.settings.OLLAMA_TOP_P,
                "num_predict": max_tokens or self.settings.OLLAMA_MAX_TOKENS,
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    if stream:
                        return response
                    else:
                        return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama generate error: {error_text}")
                    return {"error": error_text}
        except Exception as e:
            logger.error(f"Error calling Ollama generate: {e}")
            return {"error": str(e)}
    
    async def embed(
        self,
        text: str,
        model: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Generate embeddings for text
        
        Args:
            text: Input text
            model: Embedding model name
            
        Returns:
            List of embedding values
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "model": model or self.embedding_model,
            "prompt": text
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("embedding")
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama embed error: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error calling Ollama embed: {e}")
            return None
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Chat completion
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Sampling temperature
            stream: Stream response
            
        Returns:
            Chat response
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature or self.settings.OLLAMA_TEMPERATURE,
                "top_p": self.settings.OLLAMA_TOP_P,
            }
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                if response.status == 200:
                    if stream:
                        return response
                    else:
                        return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama chat error: {error_text}")
                    return {"error": error_text}
        except Exception as e:
            logger.error(f"Error calling Ollama chat: {e}")
            return {"error": str(e)}
