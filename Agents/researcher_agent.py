"""
Researcher Agent - Specialized in information gathering and research.

This agent handles web searches, document analysis, information synthesis,
and knowledge extraction from various sources.
"""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import json
import re
from collections import defaultdict

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from base_agent import BaseAgent


class ResearcherAgent(BaseAgent):
    """
    Agent specialized in research and information gathering.
    
    Capabilities:
    - Web search and scraping
    - Document analysis
    - Information synthesis
    - Source verification
    - Knowledge extraction
    - Research report generation
    - Citation management
    """
    
    def __init__(self, name: str = "Researcher", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Researcher Agent.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
            
        Raises:
            ImportError: If requests library is not installed
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library is required for ResearcherAgent. "
                "Install it with: pip install requests"
            )
        
        default_config = {
            "capabilities": [
                "web_search",
                "document_analysis",
                "information_synthesis",
                "source_verification",
                "knowledge_extraction",
                "report_generation",
                "citation_management"
            ],
            "max_sources": 10,
            "search_depth": "comprehensive",  # quick, standard, comprehensive
            "verify_sources": True,
            "include_citations": True,
            "languages": ["en", "fr"],
            "timeout": 30  # seconds
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name=name, role="Researcher", config=default_config)
        self.research_cache: Dict[str, Any] = {}
        self.sources: List[Dict[str, Any]] = []
        self.knowledge_base: Dict[str, List[str]] = defaultdict(list)
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate if the task is appropriate for research.
        
        Args:
            task: Task dictionary
            
        Returns:
            True if task can be handled
        """
        required_fields = ["task_type"]
        
        if not all(field in task for field in required_fields):
            self.logger.error(f"Missing required fields: {required_fields}")
            return False
        
        task_type = task.get("task_type")
        valid_types = [
            "search",
            "research_topic",
            "analyze_document",
            "verify_information",
            "synthesize_findings",
            "generate_report",
            "extract_knowledge",
            "compare_sources"
        ]
        
        if task_type not in valid_types:
            self.logger.error(f"Invalid task type: {task_type}")
            return False
        
        return True
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a research task.
        
        Args:
            task: Dictionary containing:
                - task_type: Type of research task
                - query: Search query or topic
                - parameters: Optional research parameters
                
        Returns:
            Dictionary with research results
        """
        self.update_state("working")
        
        try:
            if not self.validate_task(task):
                return {
                    "success": False,
                    "error": "Invalid task format",
                    "timestamp": datetime.now().isoformat()
                }
            
            task_type = task["task_type"]
            query = task.get("query", "")
            parameters = task.get("parameters", {})
            
            # Route to appropriate research method
            if task_type == "search":
                result = await self._perform_search(query, parameters)
            elif task_type == "research_topic":
                result = await self._research_topic(query, parameters)
            elif task_type == "analyze_document":
                result = await self._analyze_document(task.get("document"), parameters)
            elif task_type == "verify_information":
                result = await self._verify_information(query, parameters)
            elif task_type == "synthesize_findings":
                result = await self._synthesize_findings(parameters)
            elif task_type == "generate_report":
                result = await self._generate_report(query, parameters)
            elif task_type == "extract_knowledge":
                result = await self._extract_knowledge(task.get("content"), parameters)
            elif task_type == "compare_sources":
                result = await self._compare_sources(task.get("sources"), parameters)
            else:
                result = {"error": f"Unknown task type: {task_type}"}
            
            result["success"] = True
            result["timestamp"] = datetime.now().isoformat()
            
            self.log_task(task, result)
            self.update_state("idle")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing research task: {str(e)}")
            self.update_state("idle")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _perform_search(self, query: str, params: Dict) -> Dict[str, Any]:
        """
        Perform a search operation.
        
        Args:
            query: Search query
            params: Search parameters
            
        Returns:
            Search results
        """
        self.logger.info(f"Performing search for: {query}")
        
        # Check cache first
        cache_key = self._generate_cache_key(query, params)
        if cache_key in self.research_cache:
            self.logger.info("Returning cached results")
            return self.research_cache[cache_key]
        
        max_results = params.get("max_results", self.config["max_sources"])
        
        # Simulate search results (in production, integrate with actual search API)
        results = {
            "query": query,
            "results": [],
            "total_found": 0,
            "search_time": datetime.now().isoformat()
        }
        
        # Store in cache
        self.research_cache[cache_key] = results
        
        return results
    
    async def _research_topic(self, topic: str, params: Dict) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a topic.
        
        Args:
            topic: Topic to research
            params: Research parameters
            
        Returns:
            Research findings
        """
        self.logger.info(f"Researching topic: {topic}")
        
        depth = params.get("depth", self.config["search_depth"])
        
        research_report = {
            "topic": topic,
            "depth": depth,
            "summary": "",
            "key_findings": [],
            "sources": [],
            "related_topics": [],
            "confidence_score": 0.0
        }
        
        # Step 1: Initial search
        search_results = await self._perform_search(topic, params)
        
        # Step 2: Analyze top sources
        if search_results.get("results"):
            for source in search_results["results"][:5]:
                self.sources.append({
                    "title": source.get("title", ""),
                    "url": source.get("url", ""),
                    "relevance": source.get("relevance", 0),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Step 3: Extract key findings (simulated)
        research_report["key_findings"] = [
            f"Finding related to {topic}",
            f"Key insight about {topic}",
            f"Important consideration regarding {topic}"
        ]
        
        # Step 4: Generate summary
        research_report["summary"] = f"Research on '{topic}' has been conducted with {depth} depth."
        
        # Step 5: Identify related topics
        research_report["related_topics"] = self._extract_related_topics(topic)
        
        # Step 6: Calculate confidence score
        research_report["confidence_score"] = self._calculate_confidence(research_report)
        
        research_report["sources"] = self.sources[-5:]  # Last 5 sources
        
        return research_report
    
    async def _analyze_document(self, document: Any, params: Dict) -> Dict[str, Any]:
        """
        Analyze a document and extract information.
        
        Args:
            document: Document content (text, dict, or file path)
            params: Analysis parameters
            
        Returns:
            Document analysis results
        """
        self.logger.info("Analyzing document")
        
        if isinstance(document, dict):
            content = document.get("content", "")
            doc_type = document.get("type", "text")
        elif isinstance(document, str):
            content = document
            doc_type = "text"
        else:
            return {"error": "Invalid document format"}
        
        analysis = {
            "document_type": doc_type,
            "length": len(content),
            "word_count": len(content.split()),
            "summary": "",
            "key_points": [],
            "entities": [],
            "topics": [],
            "sentiment": "neutral"
        }
        
        # Extract key points (simple extraction)
        sentences = self._split_into_sentences(content)
        analysis["key_points"] = sentences[:5] if len(sentences) > 5 else sentences
        
        # Extract entities (simple pattern matching)
        analysis["entities"] = self._extract_entities(content)
        
        # Extract topics
        analysis["topics"] = self._extract_topics(content)
        
        # Generate summary
        analysis["summary"] = self._generate_summary(content, max_length=200)
        
        return analysis
    
    async def _verify_information(self, claim: str, params: Dict) -> Dict[str, Any]:
        """
        Verify information against multiple sources.
        
        Args:
            claim: Claim to verify
            params: Verification parameters
            
        Returns:
            Verification results
        """
        self.logger.info(f"Verifying claim: {claim}")
        
        verification = {
            "claim": claim,
            "verification_status": "unverified",  # verified, disputed, unverified
            "confidence": 0.0,
            "supporting_sources": [],
            "contradicting_sources": [],
            "neutral_sources": [],
            "recommendation": ""
        }
        
        # Search for related information
        search_results = await self._perform_search(claim, params)
        
        # Analyze sources (simulated)
        # In production, this would involve NLP and fact-checking
        
        verification["recommendation"] = "Further verification recommended"
        
        return verification
    
    async def _synthesize_findings(self, params: Dict) -> Dict[str, Any]:
        """
        Synthesize findings from multiple sources.
        
        Args:
            params: Synthesis parameters
            
        Returns:
            Synthesized information
        """
        self.logger.info("Synthesizing research findings")
        
        synthesis = {
            "total_sources": len(self.sources),
            "main_themes": [],
            "consensus_points": [],
            "disagreements": [],
            "gaps": [],
            "synthesis_text": ""
        }
        
        # Analyze collected sources
        if self.sources:
            # Extract common themes
            synthesis["main_themes"] = self._extract_common_themes()
            
            # Identify consensus
            synthesis["consensus_points"] = [
                "Point of agreement across sources",
                "Common finding in multiple sources"
            ]
            
            # Identify disagreements
            synthesis["disagreements"] = [
                "Conflicting information found",
                "Different interpretations exist"
            ]
            
            # Identify knowledge gaps
            synthesis["gaps"] = self._identify_knowledge_gaps()
            
            # Generate synthesis text
            synthesis["synthesis_text"] = self._create_synthesis_text(synthesis)
        
        return synthesis
    
    async def _generate_report(self, topic: str, params: Dict) -> Dict[str, Any]:
        """
        Generate a comprehensive research report.
        
        Args:
            topic: Report topic
            params: Report parameters
            
        Returns:
            Research report
        """
        self.logger.info(f"Generating report on: {topic}")
        
        report_format = params.get("format", "structured")  # structured, narrative, academic
        
        report = {
            "title": f"Research Report: {topic}",
            "generated_at": datetime.now().isoformat(),
            "format": report_format,
            "sections": {}
        }
        
        # Executive Summary
        report["sections"]["executive_summary"] = {
            "content": f"This report provides a comprehensive analysis of {topic}.",
            "key_takeaways": [
                "Key finding 1",
                "Key finding 2",
                "Key finding 3"
            ]
        }
        
        # Introduction
        report["sections"]["introduction"] = {
            "content": f"Introduction to {topic} and research methodology.",
            "objectives": [
                f"Understand {topic}",
                f"Analyze key aspects of {topic}",
                f"Provide actionable insights"
            ]
        }
        
        # Main Findings
        report["sections"]["findings"] = {
            "content": "Detailed analysis of research findings.",
            "data": []
        }
        
        # Conclusions
        report["sections"]["conclusions"] = {
            "content": "Summary of key conclusions and recommendations.",
            "recommendations": [
                "Recommendation 1",
                "Recommendation 2"
            ]
        }
        
        # References
        report["sections"]["references"] = {
            "sources": self.sources,
            "citation_style": params.get("citation_style", "APA")
        }
        
        return report
    
    async def _extract_knowledge(self, content: str, params: Dict) -> Dict[str, Any]:
        """
        Extract structured knowledge from content.
        
        Args:
            content: Content to analyze
            params: Extraction parameters
            
        Returns:
            Extracted knowledge
        """
        self.logger.info("Extracting knowledge from content")
        
        knowledge = {
            "facts": [],
            "concepts": [],
            "relationships": [],
            "definitions": {},
            "examples": []
        }
        
        # Extract facts (simple sentence extraction)
        sentences = self._split_into_sentences(content)
        knowledge["facts"] = [s for s in sentences if self._is_factual_sentence(s)][:10]
        
        # Extract concepts (nouns and key terms)
        knowledge["concepts"] = self._extract_concepts(content)
        
        # Store in knowledge base
        topic = params.get("topic", "general")
        self.knowledge_base[topic].extend(knowledge["facts"])
        
        return knowledge
    
    async def _compare_sources(self, sources: List[Dict], params: Dict) -> Dict[str, Any]:
        """
        Compare multiple sources for consistency and reliability.
        
        Args:
            sources: List of source documents
            params: Comparison parameters
            
        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing {len(sources)} sources")
        
        comparison = {
            "total_sources": len(sources),
            "agreement_score": 0.0,
            "common_information": [],
            "unique_information": {},
            "contradictions": [],
            "reliability_scores": {}
        }
        
        # Calculate agreement between sources
        comparison["agreement_score"] = self._calculate_source_agreement(sources)
        
        # Identify common information
        comparison["common_information"] = self._find_common_information(sources)
        
        # Identify unique information per source
        for i, source in enumerate(sources):
            source_id = source.get("id", f"source_{i}")
            comparison["unique_information"][source_id] = self._extract_unique_info(source, sources)
        
        # Assess reliability
        for i, source in enumerate(sources):
            source_id = source.get("id", f"source_{i}")
            comparison["reliability_scores"][source_id] = self._assess_reliability(source)
        
        return comparison
    
    # Helper methods
    
    def _generate_cache_key(self, query: str, params: Dict) -> str:
        """Generate a cache key for search results."""
        param_str = json.dumps(params, sort_keys=True)
        return f"{query}_{hash(param_str)}"
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text (simplified)."""
        # Simple capitalized word extraction
        words = text.split()
        entities = []
        
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                entities.append({
                    "text": word,
                    "type": "unknown"
                })
        
        return entities[:10]  # Limit to 10
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text."""
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = defaultdict(int)
        
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        for word in words:
            if word not in stopwords and len(word) > 3:
                word_freq[word] += 1
        
        # Get top 5 most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5]]
    
    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the text."""
        if len(text) <= max_length:
            return text
        
        sentences = self._split_into_sentences(text)
        summary = ""
        
        for sentence in sentences:
            if len(summary) + len(sentence) <= max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip()
    
    def _extract_related_topics(self, topic: str) -> List[str]:
        """Extract related topics."""
        # Simulated related topics
        return [
            f"Advanced {topic}",
            f"{topic} applications",
            f"{topic} theory",
            f"{topic} best practices"
        ]
    
    def _calculate_confidence(self, research_report: Dict) -> float:
        """Calculate confidence score for research."""
        score = 0.0
        
        if research_report.get("sources"):
            score += min(len(research_report["sources"]) * 0.1, 0.5)
        
        if research_report.get("key_findings"):
            score += min(len(research_report["key_findings"]) * 0.1, 0.3)
        
        score += 0.2  # Base score
        
        return min(score, 1.0)
    
    def _extract_common_themes(self) -> List[str]:
        """Extract common themes from collected sources."""
        # Simplified theme extraction
        return ["Theme 1", "Theme 2", "Theme 3"]
    
    def _identify_knowledge_gaps(self) -> List[str]:
        """Identify gaps in current knowledge."""
        return ["Area requiring more research", "Unclear aspect"]
    
    def _create_synthesis_text(self, synthesis: Dict) -> str:
        """Create narrative synthesis from findings."""
        text = f"Analysis of {synthesis['total_sources']} sources reveals "
        text += f"{len(synthesis['main_themes'])} main themes. "
        text += f"There are {len(synthesis['consensus_points'])} points of consensus "
        text += f"and {len(synthesis['disagreements'])} areas of disagreement."
        return text
    
    def _is_factual_sentence(self, sentence: str) -> bool:
        """Check if sentence appears to be factual."""
        # Simple heuristic: contains numbers, dates, or specific terms
        patterns = [r'\d+', r'is', r'are', r'was', r'were']
        return any(re.search(pattern, sentence.lower()) for pattern in patterns)
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        topics = self._extract_topics(text)
        return topics
    
    def _calculate_source_agreement(self, sources: List[Dict]) -> float:
        """Calculate agreement score between sources."""
        if len(sources) < 2:
            return 1.0
        return 0.75  # Simulated agreement score
    
    def _find_common_information(self, sources: List[Dict]) -> List[str]:
        """Find information common across sources."""
        return ["Common fact 1", "Shared information"]
    
    def _extract_unique_info(self, source: Dict, all_sources: List[Dict]) -> List[str]:
        """Extract information unique to a source."""
        return ["Unique insight from this source"]
    
    def _assess_reliability(self, source: Dict) -> float:
        """Assess the reliability of a source."""
        score = 0.5  # Base score
        
        if source.get("author"):
            score += 0.1
        if source.get("publication_date"):
            score += 0.1
        if source.get("citations"):
            score += 0.2
        if source.get("peer_reviewed"):
            score += 0.1
        
        return min(score, 1.0)
    
    def get_research_history(self) -> List[Dict[str, Any]]:
        """Get the history of research tasks."""
        return self.task_history
    
    def get_knowledge_base(self, topic: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get accumulated knowledge base.
        
        Args:
            topic: Specific topic to retrieve (optional)
            
        Returns:
            Knowledge base dictionary
        """
        if topic:
            return {topic: self.knowledge_base.get(topic, [])}
        return dict(self.knowledge_base)
    
    def clear_cache(self):
        """Clear the research cache."""
        self.research_cache.clear()
        self.logger.info("Research cache cleared")
    
    def clear_sources(self):
        """Clear collected sources."""
        self.sources.clear()
        self.logger.info("Sources cleared")