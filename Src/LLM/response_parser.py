import json
import re
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Parse and validate LLM responses
    Handles JSON extraction, error recovery, and format validation
    """
    
    @staticmethod
    def parse_json_response(response: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from LLM response
        Handles various JSON formats and markdown code blocks
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        if not response:
            return None
        
        # Try direct JSON parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object without code blocks
        json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_object_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
        
        logger.warning("Could not parse JSON from response")
        return None
    
    @staticmethod
    def extract_key_value_pairs(response: str) -> Dict[str, str]:
        """
        Extract key-value pairs from structured text
        Handles formats like "Key: Value" or "Key = Value"
        """
        pairs = {}
        
        # Pattern for "Key: Value" or "Key = Value"
        pattern = r'([A-Za-z_][A-Za-z0-9_\s]*?):\s*(.+?)(?=\n[A-Za-z_]|$)'
        matches = re.finditer(pattern, response, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            key = match.group(1).strip()
            value = match.group(2).strip()
            pairs[key] = value
        
        return pairs
    
    @staticmethod
    def extract_list_items(response: str) -> List[str]:
        """
        Extract list items from text
        Handles numbered lists, bullet points, etc.
        """
        items = []
        
        # Pattern for numbered lists (1. Item, 1) Item, etc.)
        numbered_pattern = r'^\s*\d+[\.)]\s*(.+?)$'
        
        # Pattern for bullet points (-, *, •)
        bullet_pattern = r'^\s*[-*•]\s*(.+?)$'
        
        for line in response.split('\n'):
            # Try numbered list
            match = re.match(numbered_pattern, line)
            if match:
                items.append(match.group(1).strip())
                continue
            
            # Try bullet points
            match = re.match(bullet_pattern, line)
            if match:
                items.append(match.group(1).strip())
        
        return items
    
    @staticmethod
    def validate_data_analysis_response(response: Dict[str, Any]) -> bool:
        """
        Validate data analyst response structure
        
        Expected fields:
        - insights: List of key findings
        - statistics: Statistical measures
        - recommendations: Action items
        """
        required_fields = ["insights", "statistics", "recommendations"]
        return all(field in response for field in required_fields)
    
    @staticmethod
    def validate_research_response(response: Dict[str, Any]) -> bool:
        """
        Validate researcher response structure
        
        Expected fields:
        - findings: Research results
        - sources: List of sources
        - confidence: Confidence level
        """
        required_fields = ["findings", "sources"]
        return all(field in response for field in required_fields)
    
    @staticmethod
    def validate_strategy_response(response: Dict[str, Any]) -> bool:
        """
        Validate strategist response structure
        
        Expected fields:
        - options: Strategic options
        - recommendation: Recommended approach
        - risks: Risk assessment
        """
        required_fields = ["options", "recommendation"]
        return all(field in response for field in required_fields)
    
    @classmethod
    def clean_response(cls, response: str) -> str:
        """
        Clean up LLM response text
        Remove excessive whitespace, fix encoding issues, etc.
        """
        if not response:
            return ""
        
        # Remove excessive whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)
        
        # Remove common artifacts
        response = response.replace('```json\n', '').replace('\n```', '')
        response = response.strip()
        
        return response
