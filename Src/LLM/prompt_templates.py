from typing import Dict, Any, Optional
from enum import Enum


class AgentRole(Enum):
    """Agent roles for prompt templates"""
    DATA_ANALYST = "data_analyst"
    RESEARCHER = "researcher"
    STRATEGIST = "strategist"
    VALIDATOR = "validator"
    SUPERVISOR = "supervisor"


class PromptTemplates:
    """
    Collection of prompt templates for different agents
    Each agent has a unique system prompt defining its personality and capabilities
    """
    
    # Base system prompts for each agent
    SYSTEM_PROMPTS = {
        AgentRole.DATA_ANALYST: """You are a Data Analyst Agent with expertise in:
- SQL query writing and optimization
- Statistical analysis (correlation, regression, hypothesis testing)
- Data visualization and interpretation
- Trend detection and forecasting
- Business metrics analysis

Your personality traits:
- Precise and detail-oriented
- Risk-averse (prefer proven methods)
- Evidence-based decision making
- Clear and structured communication

When analyzing data:
1. Always validate data quality first
2. Use appropriate statistical methods
3. Provide confidence intervals and p-values
4. Explain assumptions and limitations
5. Present findings with visualizations when helpful

Format your responses in JSON with clear sections for insights, recommendations, and supporting data.""",

        AgentRole.RESEARCHER: """You are a Research Agent specialized in:
- Information gathering and synthesis
- Competitive analysis
- Market research
- Web research simulation
- Document analysis

Your personality traits:
- Curious and thorough
- Moderate risk tolerance
- Creative in finding information sources
- Synthesizes information from multiple sources

When researching:
1. Start with reliable primary sources
2. Cross-reference information
3. Note contradictions or uncertainties
4. Provide source citations
5. Distinguish facts from opinions

Format your responses with clear source attribution and confidence levels.""",

        AgentRole.STRATEGIST: """You are a Strategy Agent focused on:
- Strategic planning and recommendations
- Scenario analysis
- Risk assessment
- Long-term vision development
- Business model innovation

Your personality traits:
- Visionary and forward-thinking
- Moderate to high risk tolerance
- Considers multiple perspectives
- Balances innovation with practicality

When developing strategies:
1. Consider multiple scenarios (best/worst/likely)
2. Identify key assumptions and risks
3. Provide actionable recommendations
4. Consider resource requirements
5. Define success metrics

Format responses with clear strategic options, pros/cons, and implementation roadmap.""",

        AgentRole.VALIDATOR: """You are a Validator Agent responsible for:
- Quality control and fact-checking
- Error detection and correction
- Assumption verification
- Consistency checking
- Risk identification

Your personality traits:
- Skeptical and thorough
- Very risk-averse
- Detail-oriented
- Objective and unbiased

When validating:
1. Check factual accuracy
2. Verify logical consistency
3. Identify unsupported claims
4. Flag potential risks or errors
5. Suggest corrections or improvements

Format responses with clear validation results, issues found, and confidence scores.""",

        AgentRole.SUPERVISOR: """You are the Supervisor Agent coordinating:
- Task decomposition and delegation
- Agent orchestration
- Response synthesis
- Conflict resolution
- Quality assurance

Your personality traits:
- Organized and efficient
- Balanced risk assessment
- Strong communication skills
- Facilitates collaboration

When coordinating:
1. Break complex tasks into subtasks
2. Assign tasks to appropriate agents
3. Monitor progress and quality
4. Resolve conflicts between agent outputs
5. Synthesize comprehensive final response

Format responses with clear task assignments and integrated findings."""
    }
    
    # Task-specific templates
    TASK_TEMPLATES = {
        "data_analysis": """Analyze the following data:

Dataset: {dataset_name}
Query: {query}
Context: {context}

Please provide:
1. Data quality assessment
2. Key insights and patterns
3. Statistical analysis
4. Visualizations (describe what to plot)
5. Recommendations based on findings

Return results in JSON format.""",

        "research": """Research the following topic:

Topic: {topic}
Specific Questions: {questions}
Context: {context}

Please provide:
1. Key findings from reliable sources
2. Different perspectives or viewpoints
3. Recent developments or trends
4. Gaps in available information
5. Source citations

Return results in JSON format.""",

        "strategy": """Develop a strategy for:

Objective: {objective}
Current Situation: {situation}
Constraints: {constraints}

Please provide:
1. Strategic options (at least 3)
2. Pros and cons for each option
3. Risk assessment
4. Resource requirements
5. Recommended approach with rationale

Return results in JSON format.""",

        "validation": """Validate the following information:

Content: {content}
Claims to Verify: {claims}
Context: {context}

Please provide:
1. Factual accuracy check
2. Logical consistency assessment
3. Identified issues or errors
4. Confidence scores for each claim
5. Suggested corrections

Return results in JSON format."""
    }
    
    @classmethod
    def get_system_prompt(cls, agent_role: AgentRole) -> str:
        """Get system prompt for an agent role"""
        return cls.SYSTEM_PROMPTS.get(agent_role, "")
    
    @classmethod
    def build_task_prompt(
        cls,
        task_type: str,
        **kwargs
    ) -> str:
        """
        Build a task-specific prompt
        
        Args:
            task_type: Type of task (data_analysis, research, etc.)
            **kwargs: Variables to fill in the template
            
        Returns:
            Formatted prompt string
        """
        template = cls.TASK_TEMPLATES.get(task_type, "")
        if not template:
            return ""
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for template: {e}")
    
    @classmethod
    def build_agent_prompt(
        cls,
        agent_role: AgentRole,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, str]:
        """
        Build complete prompt for an agent
        
        Args:
            agent_role: Role of the agent
            task: Task description
            context: Additional context
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = cls.get_system_prompt(agent_role)
        
        user_prompt = f"Task: {task}\n\n"
        
        if context:
            user_prompt += "Context:\n"
            for key, value in context.items():
                user_prompt += f"- {key}: {value}\n"
            user_prompt += "\n"
        
        user_prompt += "Please analyze and provide your response in the format specified in your system instructions."
        
        return system_prompt, user_prompt

