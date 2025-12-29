"""
Base Tool Module

Abstract base class for all agent tools, providing a common interface
and utility methods for tool development.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging


class ToolStatus(Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


class ToolCategory(Enum):
    """Categories of tools"""
    DATA_ANALYSIS = "data_analysis"
    FILE_OPERATIONS = "file_operations"
    DATABASE = "database"
    WEB = "web"
    VISUALIZATION = "visualization"
    STATISTICS = "statistics"
    GENERAL = "general"


@dataclass
class ToolResult:
    """Result of a tool execution"""
    status: ToolStatus
    data: Any
    message: str
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def is_success(self) -> bool:
        """Check if execution was successful"""
        return self.status == ToolStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "status": self.status.value,
            "data": self.data,
            "message": self.message,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "errors": self.errors,
            "warnings": self.warnings
        }


@dataclass
class ToolConfig:
    """Configuration for a tool"""
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_enabled: bool = True
    cache_ttl: int = 3600
    log_level: str = "INFO"
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    All tools must inherit from this class and implement the execute method.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        config: Optional[ToolConfig] = None
    ):
        """
        Initialize the base tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            category: Category of the tool
            config: Optional configuration for the tool
        """
        self.name = name
        self.description = description
        self.category = category
        self.config = config or ToolConfig()
        self.logger = self._setup_logger()
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_execution_time = 0.0
        self._cache: Dict[str, Any] = {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the tool"""
        logger = logging.getLogger(f"Tool.{self.name}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        This method must be implemented by all subclasses.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult object containing execution results
        """
        pass
    
    @abstractmethod
    def validate_params(self, **kwargs) -> bool:
        """
        Validate tool parameters before execution.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        pass
    
    def run(self, **kwargs) -> ToolResult:
        """
        Run the tool with validation, timing, and error handling.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult object
        """
        start_time = datetime.now()
        self._execution_count += 1
        
        try:
            # Validate parameters
            if not self.validate_params(**kwargs):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message="Invalid parameters",
                    execution_time=0.0,
                    errors=["Parameter validation failed"]
                )
            
            # Check cache
            if self.config.cache_enabled:
                cache_key = self._generate_cache_key(**kwargs)
                if cache_key in self._cache:
                    self.logger.info(f"Cache hit for {self.name}")
                    cached_result = self._cache[cache_key]
                    cached_result.metadata["from_cache"] = True
                    return cached_result
            
            # Execute tool
            self.logger.info(f"Executing {self.name}")
            result = self.execute(**kwargs)
            
            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            self._total_execution_time += execution_time
            
            if result.is_success():
                self._success_count += 1
            else:
                self._failure_count += 1
            
            # Cache result if enabled
            if self.config.cache_enabled and result.is_success():
                cache_key = self._generate_cache_key(**kwargs)
                self._cache[cache_key] = result
            
            self.logger.info(
                f"Completed {self.name} in {execution_time:.2f}s - Status: {result.status.value}"
            )
            
            return result
            
        except Exception as e:
            self._failure_count += 1
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error executing {self.name}: {str(e)}")
            
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Tool execution failed: {str(e)}",
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def _generate_cache_key(self, **kwargs) -> str:
        """Generate a cache key from parameters"""
        import hashlib
        import json
        
        # Sort kwargs for consistent hashing
        sorted_params = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(f"{self.name}:{sorted_params}".encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the tool's cache"""
        self._cache.clear()
        self.logger.info(f"Cache cleared for {self.name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tool execution statistics"""
        return {
            "name": self.name,
            "category": self.category.value,
            "executions": self._execution_count,
            "successes": self._success_count,
            "failures": self._failure_count,
            "success_rate": (
                self._success_count / self._execution_count
                if self._execution_count > 0 else 0.0
            ),
            "total_execution_time": self._total_execution_time,
            "average_execution_time": (
                self._total_execution_time / self._execution_count
                if self._execution_count > 0 else 0.0
            ),
            "cache_size": len(self._cache)
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "config": {
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "cache_enabled": self.config.cache_enabled
            }
        }
    
    def __str__(self) -> str:
        return f"{self.name} ({self.category.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


class ToolRegistry:
    """
    Registry for managing multiple tools.
    """
    
    def __init__(self):
        """Initialize the tool registry"""
        self._tools: Dict[str, BaseTool] = {}
        self._tools_by_category: Dict[ToolCategory, List[BaseTool]] = {
            category: [] for category in ToolCategory
        }
    
    def register(self, tool: BaseTool):
        """
        Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} is already registered")
        
        self._tools[tool.name] = tool
        self._tools_by_category[tool.category].append(tool)
    
    def unregister(self, tool_name: str):
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of the tool to unregister
        """
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            self._tools_by_category[tool.category].remove(tool)
            del self._tools[tool_name]
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """
        Get all tools in a category.
        
        Args:
            category: Tool category
            
        Returns:
            List of tools in the category
        """
        return self._tools_by_category.get(category, [])
    
    def list_tools(self) -> List[str]:
        """Get list of all registered tool names"""
        return list(self._tools.keys())
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tools"""
        return {
            name: tool.get_statistics()
            for name, tool in self._tools.items()
        }
    
    def clear_all_caches(self):
        """Clear caches for all tools"""
        for tool in self._tools.values():
            tool.clear_cache()


# Global tool registry instance
tool_registry = ToolRegistry()