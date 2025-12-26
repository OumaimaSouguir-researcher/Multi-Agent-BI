"""
Strategist Agent - Specialized in strategic planning and decision-making.

This agent handles strategic planning, goal decomposition, resource allocation,
risk assessment, and decision optimization.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import defaultdict
from base_agent import BaseAgent


class Priority(Enum):
    """Priority levels for tasks and goals."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


class RiskLevel(Enum):
    """Risk assessment levels."""
    VERY_HIGH = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    VERY_LOW = 1


class StrategistAgent(BaseAgent):
    """
    Agent specialized in strategic planning and decision-making.
    
    Capabilities:
    - Strategic planning
    - Goal decomposition
    - Resource allocation
    - Risk assessment
    - Decision analysis
    - Timeline planning
    - Dependency management
    - Performance optimization
    """
    
    def __init__(self, name: str = "Strategist", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Strategist Agent.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        default_config = {
            "capabilities": [
                "strategic_planning",
                "goal_decomposition",
                "resource_allocation",
                "risk_assessment",
                "decision_analysis",
                "timeline_planning",
                "dependency_management",
                "performance_optimization"
            ],
            "planning_horizon": 90,  # days
            "max_goal_depth": 5,
            "risk_tolerance": "medium",  # low, medium, high
            "optimization_focus": "balanced"  # time, cost, quality, balanced
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name=name, role="Strategist", config=default_config)
        self.strategic_plans: Dict[str, Dict[str, Any]] = {}
        self.goals: Dict[str, Dict[str, Any]] = {}
        self.resources: Dict[str, Any] = {}
        self.decisions: List[Dict[str, Any]] = []
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate if the task is appropriate for strategic planning.
        
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
            "create_plan",
            "decompose_goal",
            "allocate_resources",
            "assess_risk",
            "analyze_decision",
            "create_timeline",
            "optimize_strategy",
            "evaluate_alternatives",
            "prioritize_tasks"
        ]
        
        if task_type not in valid_types:
            self.logger.error(f"Invalid task type: {task_type}")
            return False
        
        return True
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a strategic planning task.
        
        Args:
            task: Dictionary containing:
                - task_type: Type of strategic task
                - objective: Main objective or goal
                - parameters: Optional planning parameters
                
        Returns:
            Dictionary with strategic analysis results
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
            objective = task.get("objective", "")
            parameters = task.get("parameters", {})
            
            # Route to appropriate strategy method
            if task_type == "create_plan":
                result = await self._create_strategic_plan(objective, parameters)
            elif task_type == "decompose_goal":
                result = await self._decompose_goal(objective, parameters)
            elif task_type == "allocate_resources":
                result = await self._allocate_resources(task.get("plan_id"), parameters)
            elif task_type == "assess_risk":
                result = await self._assess_risk(objective, parameters)
            elif task_type == "analyze_decision":
                result = await self._analyze_decision(task.get("decision"), parameters)
            elif task_type == "create_timeline":
                result = await self._create_timeline(task.get("plan_id"), parameters)
            elif task_type == "optimize_strategy":
                result = await self._optimize_strategy(task.get("plan_id"), parameters)
            elif task_type == "evaluate_alternatives":
                result = await self._evaluate_alternatives(task.get("alternatives"), parameters)
            elif task_type == "prioritize_tasks":
                result = await self._prioritize_tasks(task.get("tasks"), parameters)
            else:
                result = {"error": f"Unknown task type: {task_type}"}
            
            result["success"] = True
            result["timestamp"] = datetime.now().isoformat()
            
            self.log_task(task, result)
            self.update_state("idle")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing strategic task: {str(e)}")
            self.update_state("idle")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _create_strategic_plan(self, objective: str, params: Dict) -> Dict[str, Any]:
        """
        Create a comprehensive strategic plan.
        
        Args:
            objective: Main objective to plan for
            params: Planning parameters
            
        Returns:
            Strategic plan
        """
        self.logger.info(f"Creating strategic plan for: {objective}")
        
        plan_id = f"plan_{len(self.strategic_plans) + 1}_{int(datetime.now().timestamp())}"
        horizon = params.get("horizon", self.config["planning_horizon"])
        
        plan = {
            "id": plan_id,
            "objective": objective,
            "created_at": datetime.now().isoformat(),
            "horizon_days": horizon,
            "status": "draft",
            "phases": [],
            "milestones": [],
            "success_criteria": [],
            "constraints": [],
            "estimated_completion": (datetime.now() + timedelta(days=horizon)).isoformat()
        }
        
        # Phase 1: Analysis & Research
        plan["phases"].append({
            "name": "Analysis & Research",
            "duration_days": horizon * 0.2,
            "activities": [
                "Market research",
                "Competitive analysis",
                "Resource assessment",
                "Risk identification"
            ],
            "deliverables": ["Research report", "SWOT analysis"]
        })
        
        # Phase 2: Strategy Development
        plan["phases"].append({
            "name": "Strategy Development",
            "duration_days": horizon * 0.3,
            "activities": [
                "Define strategic objectives",
                "Develop action plans",
                "Set KPIs",
                "Create roadmap"
            ],
            "deliverables": ["Strategic document", "Implementation roadmap"]
        })
        
        # Phase 3: Implementation
        plan["phases"].append({
            "name": "Implementation",
            "duration_days": horizon * 0.4,
            "activities": [
                "Execute action plans",
                "Monitor progress",
                "Adjust strategies",
                "Manage resources"
            ],
            "deliverables": ["Progress reports", "Adjustment plans"]
        })
        
        # Phase 4: Evaluation
        plan["phases"].append({
            "name": "Evaluation",
            "duration_days": horizon * 0.1,
            "activities": [
                "Measure outcomes",
                "Analyze results",
                "Document lessons learned",
                "Plan next steps"
            ],
            "deliverables": ["Final report", "Recommendations"]
        })
        
        # Define milestones
        plan["milestones"] = [
            {
                "name": "Research completed",
                "target_date": (datetime.now() + timedelta(days=horizon * 0.2)).isoformat(),
                "criteria": "All research activities completed"
            },
            {
                "name": "Strategy approved",
                "target_date": (datetime.now() + timedelta(days=horizon * 0.5)).isoformat(),
                "criteria": "Strategic plan validated and approved"
            },
            {
                "name": "Implementation at 50%",
                "target_date": (datetime.now() + timedelta(days=horizon * 0.7)).isoformat(),
                "criteria": "Half of action plans executed"
            },
            {
                "name": "Objective achieved",
                "target_date": (datetime.now() + timedelta(days=horizon)).isoformat(),
                "criteria": "All success criteria met"
            }
        ]
        
        # Define success criteria
        plan["success_criteria"] = [
            f"Achieve {objective}",
            "Stay within budget",
            "Meet all deadlines",
            "Maintain quality standards"
        ]
        
        # Store the plan
        self.strategic_plans[plan_id] = plan
        
        return plan
    
    async def _decompose_goal(self, goal: str, params: Dict) -> Dict[str, Any]:
        """
        Decompose a high-level goal into actionable sub-goals.
        
        Args:
            goal: High-level goal to decompose
            params: Decomposition parameters
            
        Returns:
            Goal decomposition tree
        """
        self.logger.info(f"Decomposing goal: {goal}")
        
        max_depth = params.get("max_depth", self.config["max_goal_depth"])
        
        decomposition = {
            "main_goal": goal,
            "goal_tree": self._build_goal_tree(goal, depth=0, max_depth=max_depth),
            "total_subgoals": 0,
            "actionable_tasks": []
        }
        
        # Extract actionable tasks (leaf nodes)
        decomposition["actionable_tasks"] = self._extract_leaf_goals(decomposition["goal_tree"])
        decomposition["total_subgoals"] = len(decomposition["actionable_tasks"])
        
        return decomposition
    
    def _build_goal_tree(self, goal: str, depth: int, max_depth: int) -> Dict[str, Any]:
        """
        Recursively build a goal decomposition tree.
        
        Args:
            goal: Goal to decompose
            depth: Current depth in tree
            max_depth: Maximum depth to traverse
            
        Returns:
            Goal tree node
        """
        node = {
            "goal": goal,
            "depth": depth,
            "subgoals": []
        }
        
        if depth >= max_depth:
            node["actionable"] = True
            return node
        
        # Generate subgoals (simplified logic)
        if depth == 0:
            subgoals = [
                f"Plan {goal}",
                f"Execute {goal}",
                f"Monitor {goal}",
                f"Optimize {goal}"
            ]
        elif depth == 1:
            subgoals = [
                f"Research for {goal}",
                f"Design approach for {goal}",
                f"Implement {goal}"
            ]
        else:
            subgoals = [
                f"Prepare {goal}",
                f"Validate {goal}"
            ]
        
        for subgoal in subgoals:
            node["subgoals"].append(self._build_goal_tree(subgoal, depth + 1, max_depth))
        
        node["actionable"] = len(node["subgoals"]) == 0
        
        return node
    
    def _extract_leaf_goals(self, tree: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract actionable leaf goals from tree."""
        leaves = []
        
        if tree.get("actionable", False):
            leaves.append({
                "goal": tree["goal"],
                "depth": tree["depth"]
            })
        
        for subgoal in tree.get("subgoals", []):
            leaves.extend(self._extract_leaf_goals(subgoal))
        
        return leaves
    
    async def _allocate_resources(self, plan_id: str, params: Dict) -> Dict[str, Any]:
        """
        Allocate resources to a strategic plan.
        
        Args:
            plan_id: ID of the plan
            params: Allocation parameters
            
        Returns:
            Resource allocation plan
        """
        self.logger.info(f"Allocating resources for plan: {plan_id}")
        
        available_resources = params.get("available_resources", {
            "budget": 100000,
            "team_members": 5,
            "time_weeks": 12
        })
        
        allocation = {
            "plan_id": plan_id,
            "available_resources": available_resources,
            "allocation": {},
            "utilization": {},
            "recommendations": []
        }
        
        if plan_id in self.strategic_plans:
            plan = self.strategic_plans[plan_id]
            phases = plan.get("phases", [])
            
            # Allocate budget across phases
            total_phases = len(phases)
            budget_per_phase = available_resources["budget"] / total_phases if total_phases > 0 else 0
            
            for i, phase in enumerate(phases):
                phase_name = phase["name"]
                allocation["allocation"][phase_name] = {
                    "budget": budget_per_phase,
                    "team_members": available_resources["team_members"],
                    "duration_weeks": phase["duration_days"] / 7
                }
            
            # Calculate utilization
            allocation["utilization"] = {
                "budget": 100.0,
                "team": 100.0,
                "time": 100.0
            }
            
            # Provide recommendations
            allocation["recommendations"] = [
                "Consider hiring additional specialists for technical phases",
                "Allocate contingency budget (10-15%)",
                "Plan buffer time for risk mitigation"
            ]
        else:
            allocation["error"] = f"Plan {plan_id} not found"
        
        return allocation
    
    async def _assess_risk(self, objective: str, params: Dict) -> Dict[str, Any]:
        """
        Assess risks associated with an objective.
        
        Args:
            objective: Objective to assess
            params: Assessment parameters
            
        Returns:
            Risk assessment report
        """
        self.logger.info(f"Assessing risks for: {objective}")
        
        assessment = {
            "objective": objective,
            "assessed_at": datetime.now().isoformat(),
            "overall_risk_level": RiskLevel.MEDIUM.name,
            "risks": [],
            "mitigation_strategies": {},
            "contingency_plans": []
        }
        
        # Identify potential risks
        risks = [
            {
                "id": "risk_1",
                "category": "Technical",
                "description": "Technical complexity may exceed team capabilities",
                "probability": 0.3,
                "impact": RiskLevel.HIGH.name,
                "risk_score": 0.3 * RiskLevel.HIGH.value
            },
            {
                "id": "risk_2",
                "category": "Resources",
                "description": "Insufficient budget allocation",
                "probability": 0.4,
                "impact": RiskLevel.MEDIUM.name,
                "risk_score": 0.4 * RiskLevel.MEDIUM.value
            },
            {
                "id": "risk_3",
                "category": "Timeline",
                "description": "Delays in critical dependencies",
                "probability": 0.5,
                "impact": RiskLevel.MEDIUM.name,
                "risk_score": 0.5 * RiskLevel.MEDIUM.value
            },
            {
                "id": "risk_4",
                "category": "Market",
                "description": "Changing market conditions",
                "probability": 0.2,
                "impact": RiskLevel.HIGH.name,
                "risk_score": 0.2 * RiskLevel.HIGH.value
            }
        ]
        
        assessment["risks"] = sorted(risks, key=lambda x: x["risk_score"], reverse=True)
        
        # Develop mitigation strategies
        for risk in assessment["risks"]:
            assessment["mitigation_strategies"][risk["id"]] = [
                f"Monitor {risk['category'].lower()} indicators closely",
                f"Develop alternative approaches for {risk['category'].lower()} challenges",
                "Establish early warning systems"
            ]
        
        # Contingency plans
        assessment["contingency_plans"] = [
            "Maintain 15% budget reserve",
            "Identify backup suppliers/partners",
            "Create flexible timeline with buffer periods",
            "Cross-train team members for critical roles"
        ]
        
        # Calculate overall risk level
        avg_risk_score = sum(r["risk_score"] for r in assessment["risks"]) / len(assessment["risks"])
        if avg_risk_score >= 2.0:
            assessment["overall_risk_level"] = RiskLevel.HIGH.name
        elif avg_risk_score >= 1.0:
            assessment["overall_risk_level"] = RiskLevel.MEDIUM.name
        else:
            assessment["overall_risk_level"] = RiskLevel.LOW.name
        
        return assessment
    
    async def _analyze_decision(self, decision: Dict[str, Any], params: Dict) -> Dict[str, Any]:
        """
        Analyze a decision and its implications.
        
        Args:
            decision: Decision to analyze
            params: Analysis parameters
            
        Returns:
            Decision analysis
        """
        self.logger.info("Analyzing decision")
        
        analysis = {
            "decision": decision,
            "analysis_date": datetime.now().isoformat(),
            "pros": [],
            "cons": [],
            "alternatives": [],
            "recommendation": "",
            "confidence": 0.0
        }
        
        # Analyze pros and cons
        analysis["pros"] = [
            "Aligns with strategic objectives",
            "Reasonable resource requirements",
            "Manageable risk profile"
        ]
        
        analysis["cons"] = [
            "May require additional training",
            "Timeline is ambitious",
            "Dependency on external factors"
        ]
        
        # Generate alternatives
        analysis["alternatives"] = [
            {
                "name": "Alternative A: Phased approach",
                "description": "Break into smaller phases",
                "pros": ["Lower risk", "More manageable"],
                "cons": ["Longer timeline", "Higher coordination overhead"]
            },
            {
                "name": "Alternative B: Outsource components",
                "description": "Leverage external expertise",
                "pros": ["Faster execution", "Access to expertise"],
                "cons": ["Higher cost", "Less control"]
            }
        ]
        
        # Provide recommendation
        analysis["recommendation"] = "Proceed with original decision with recommended risk mitigation measures"
        analysis["confidence"] = 0.75
        
        # Store decision
        self.decisions.append({
            "decision": decision,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
        
        return analysis
    
    async def _create_timeline(self, plan_id: str, params: Dict) -> Dict[str, Any]:
        """
        Create a detailed timeline for a plan.
        
        Args:
            plan_id: ID of the plan
            params: Timeline parameters
            
        Returns:
            Detailed timeline
        """
        self.logger.info(f"Creating timeline for plan: {plan_id}")
        
        timeline = {
            "plan_id": plan_id,
            "created_at": datetime.now().isoformat(),
            "start_date": datetime.now().isoformat(),
            "events": [],
            "dependencies": [],
            "critical_path": []
        }
        
        if plan_id in self.strategic_plans:
            plan = self.strategic_plans[plan_id]
            current_date = datetime.now()
            
            for phase in plan.get("phases", []):
                duration_days = phase["duration_days"]
                
                timeline["events"].append({
                    "name": f"{phase['name']} - Start",
                    "date": current_date.isoformat(),
                    "type": "phase_start",
                    "phase": phase["name"]
                })
                
                # Add activities
                activity_duration = duration_days / len(phase.get("activities", []))
                for activity in phase.get("activities", []):
                    timeline["events"].append({
                        "name": activity,
                        "date": current_date.isoformat(),
                        "type": "activity",
                        "phase": phase["name"],
                        "duration_days": activity_duration
                    })
                    current_date += timedelta(days=activity_duration)
                
                timeline["events"].append({
                    "name": f"{phase['name']} - Complete",
                    "date": current_date.isoformat(),
                    "type": "phase_end",
                    "phase": phase["name"]
                })
            
            timeline["end_date"] = current_date.isoformat()
            
            # Identify critical path (simplified)
            timeline["critical_path"] = [
                event["name"] for event in timeline["events"] 
                if event["type"] in ["phase_start", "phase_end"]
            ]
        
        return timeline
    
    async def _optimize_strategy(self, plan_id: str, params: Dict) -> Dict[str, Any]:
        """
        Optimize a strategic plan based on criteria.
        
        Args:
            plan_id: ID of the plan
            params: Optimization parameters
            
        Returns:
            Optimized plan recommendations
        """
        self.logger.info(f"Optimizing strategy for plan: {plan_id}")
        
        optimization = {
            "plan_id": plan_id,
            "optimization_focus": params.get("focus", self.config["optimization_focus"]),
            "recommendations": [],
            "potential_improvements": {},
            "estimated_impact": {}
        }
        
        # Generate optimization recommendations
        optimization["recommendations"] = [
            {
                "area": "Resource allocation",
                "recommendation": "Reallocate 15% of budget from phase 1 to phase 3",
                "expected_benefit": "Faster implementation",
                "risk": "Low"
            },
            {
                "area": "Timeline",
                "recommendation": "Parallelize activities in phase 2 and 3",
                "expected_benefit": "20% time reduction",
                "risk": "Medium"
            },
            {
                "area": "Team structure",
                "recommendation": "Add specialist for critical technical tasks",
                "expected_benefit": "Higher quality output",
                "risk": "Low"
            }
        ]
        
        optimization["potential_improvements"] = {
            "time_savings": "15-20%",
            "cost_reduction": "5-10%",
            "quality_improvement": "Moderate",
            "risk_reduction": "Significant"
        }
        
        return optimization
    
    async def _evaluate_alternatives(self, alternatives: List[Dict], params: Dict) -> Dict[str, Any]:
        """
        Evaluate multiple strategic alternatives.
        
        Args:
            alternatives: List of alternatives to evaluate
            params: Evaluation parameters
            
        Returns:
            Evaluation results with ranking
        """
        self.logger.info(f"Evaluating {len(alternatives)} alternatives")
        
        criteria = params.get("criteria", [
            {"name": "cost", "weight": 0.3},
            {"name": "time", "weight": 0.2},
            {"name": "quality", "weight": 0.3},
            {"name": "risk", "weight": 0.2}
        ])
        
        evaluation = {
            "alternatives_count": len(alternatives),
            "criteria": criteria,
            "scores": [],
            "ranking": [],
            "recommendation": ""
        }
        
        # Score each alternative
        for i, alt in enumerate(alternatives):
            alt_id = alt.get("id", f"alt_{i}")
            
            # Calculate weighted score (simplified)
            score = 0
            for criterion in criteria:
                criterion_score = alt.get(criterion["name"], 5) / 10  # Normalize to 0-1
                score += criterion_score * criterion["weight"]
            
            evaluation["scores"].append({
                "alternative_id": alt_id,
                "name": alt.get("name", f"Alternative {i+1}"),
                "score": round(score, 2),
                "details": alt
            })
        
        # Rank alternatives
        evaluation["scores"].sort(key=lambda x: x["score"], reverse=True)
        evaluation["ranking"] = [s["alternative_id"] for s in evaluation["scores"]]
        
        # Provide recommendation
        if evaluation["scores"]:
            best = evaluation["scores"][0]
            evaluation["recommendation"] = f"Recommend {best['name']} with score {best['score']}"
        
        return evaluation
    
    async def _prioritize_tasks(self, tasks: List[Dict], params: Dict) -> Dict[str, Any]:
        """
        Prioritize a list of tasks based on multiple factors.
        
        Args:
            tasks: List of tasks to prioritize
            params: Prioritization parameters
            
        Returns:
            Prioritized task list
        """
        self.logger.info(f"Prioritizing {len(tasks)} tasks")
        
        method = params.get("method", "eisenhower")  # eisenhower, impact, deadline
        
        prioritization = {
            "method": method,
            "total_tasks": len(tasks),
            "prioritized_tasks": []
        }
        
        for task in tasks:
            priority_score = self._calculate_priority_score(task, method)
            
            prioritization["prioritized_tasks"].append({
                "task": task,
                "priority_score": priority_score,
                "priority_level": self._get_priority_level(priority_score),
                "recommended_action": self._get_action_recommendation(priority_score)
            })
        
        # Sort by priority score
        prioritization["prioritized_tasks"].sort(
            key=lambda x: x["priority_score"], 
            reverse=True
        )
        
        return prioritization
    
    def _calculate_priority_score(self, task: Dict, method: str) -> float:
        """Calculate priority score for a task."""
        if method == "eisenhower":
            urgency = task.get("urgency", 5) / 10
            importance = task.get("importance", 5) / 10
            return (urgency * 0.4 + importance * 0.6)
        elif method == "impact":
            impact = task.get("impact", 5) / 10
            effort = task.get("effort", 5) / 10
            return impact / (effort + 0.1)  # Avoid division by zero
        else:  # deadline
            deadline = task.get("deadline_days", 30)
            return 1.0 / (deadline + 1)  # Sooner deadline = higher priority
    
    def _get_priority_level(self, score: float) -> str:
        """Convert score to priority level."""
        if score >= 0.8:
            return Priority.CRITICAL.name
        elif score >= 0.6:
            return Priority.HIGH.name
        elif score >= 0.4:
            return Priority.MEDIUM.name
        elif score >= 0.2:
            return Priority.LOW.name
        else:
            return Priority.MINIMAL.name
    
    def _get_action_recommendation(self, score: float) -> str:
        """Get action recommendation based on priority."""
        if score >= 0.8:
            return "Do immediately"
        elif score >= 0.6:
            return "Schedule soon"
        elif score >= 0.4:
            return "Plan for later"
        else:
            return "Consider delegating or deferring"
    
    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a strategic plan by ID."""
        return self.strategic_plans.get(plan_id)
    
    def list_plans(self) -> List[Dict[str, Any]]:
        """List all strategic plans."""
        return list(self.strategic_plans.values())
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get history of analyzed decisions."""
        return self.decisions
    
    def update_plan_status(self, plan_id: str, new_status: str) -> bool:
        """
        Update the status of a plan.
        
        Args:
            plan_id: ID of the plan
            new_status: New status (draft, active, completed, cancelled)
            
        Returns:
            True if updated successfully
        """
        if plan_id in self.strategic_plans:
            self.strategic_plans[plan_id]["status"] = new_status
            self.strategic_plans[plan_id]["updated_at"] = datetime.now().isoformat()
            self.logger.info(f"Plan {plan_id} status updated to {new_status}")
            return True
        return False