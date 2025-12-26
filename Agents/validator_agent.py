"""
Validator Agent - Quality Control Specialist
Responsible for validating outputs, checking accuracy, and ensuring quality standards
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import re


class ValidatorAgent:
    """
    Agent specialized in quality control and validation of outputs.
    Ensures accuracy, completeness, and adherence to standards.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Validator Agent
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.validation_history = []
        self.quality_thresholds = {
            'completeness': 0.8,
            'accuracy': 0.9,
            'consistency': 0.85,
            'relevance': 0.75
        }
        
    def validate_research_output(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate research output from researcher agent
        
        Args:
            research_data: Research output to validate
            
        Returns:
            Validation results with scores and recommendations
        """
        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'input_type': 'research',
            'is_valid': True,
            'scores': {},
            'issues': [],
            'recommendations': []
        }
        
        # Check completeness
        completeness_score = self._check_completeness(research_data)
        validation_result['scores']['completeness'] = completeness_score
        
        if completeness_score < self.quality_thresholds['completeness']:
            validation_result['is_valid'] = False
            validation_result['issues'].append('Research data is incomplete')
            validation_result['recommendations'].append('Gather more comprehensive information')
        
        # Check for required fields
        required_fields = ['sources', 'findings', 'summary']
        missing_fields = [field for field in required_fields if field not in research_data]
        
        if missing_fields:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f'Missing required fields: {", ".join(missing_fields)}')
            validation_result['recommendations'].append(f'Add missing fields: {", ".join(missing_fields)}')
        
        # Validate sources
        if 'sources' in research_data:
            source_validation = self._validate_sources(research_data['sources'])
            validation_result['scores']['source_quality'] = source_validation['score']
            validation_result['issues'].extend(source_validation['issues'])
            validation_result['recommendations'].extend(source_validation['recommendations'])
        
        self._log_validation(validation_result)
        return validation_result
    
    def validate_analysis_output(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis output from data analyst agent
        
        Args:
            analysis_data: Analysis output to validate
            
        Returns:
            Validation results with scores and recommendations
        """
        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'input_type': 'analysis',
            'is_valid': True,
            'scores': {},
            'issues': [],
            'recommendations': []
        }
        
        # Check for data integrity
        if 'data' in analysis_data:
            integrity_score = self._check_data_integrity(analysis_data['data'])
            validation_result['scores']['data_integrity'] = integrity_score
            
            if integrity_score < self.quality_thresholds['accuracy']:
                validation_result['is_valid'] = False
                validation_result['issues'].append('Data integrity concerns detected')
                validation_result['recommendations'].append('Review and clean data before analysis')
        
        # Validate statistical methods
        if 'methods' in analysis_data:
            method_validation = self._validate_methods(analysis_data['methods'])
            validation_result['scores']['method_validity'] = method_validation['score']
            validation_result['issues'].extend(method_validation['issues'])
            validation_result['recommendations'].extend(method_validation['recommendations'])
        
        # Check for conclusions
        if 'conclusions' not in analysis_data or not analysis_data['conclusions']:
            validation_result['is_valid'] = False
            validation_result['issues'].append('Missing or empty conclusions')
            validation_result['recommendations'].append('Provide clear conclusions based on analysis')
        
        self._log_validation(validation_result)
        return validation_result
    
    def validate_strategy_output(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate strategy output from strategist agent
        
        Args:
            strategy_data: Strategy output to validate
            
        Returns:
            Validation results with scores and recommendations
        """
        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'input_type': 'strategy',
            'is_valid': True,
            'scores': {},
            'issues': [],
            'recommendations': []
        }
        
        # Check for strategic components
        required_components = ['objectives', 'actions', 'timeline', 'metrics']
        missing_components = [comp for comp in required_components if comp not in strategy_data]
        
        if missing_components:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f'Missing strategic components: {", ".join(missing_components)}')
            validation_result['recommendations'].append(f'Include: {", ".join(missing_components)}')
        
        # Validate objectives are SMART
        if 'objectives' in strategy_data:
            smart_validation = self._validate_smart_objectives(strategy_data['objectives'])
            validation_result['scores']['objectives_quality'] = smart_validation['score']
            validation_result['issues'].extend(smart_validation['issues'])
            validation_result['recommendations'].extend(smart_validation['recommendations'])
        
        # Check feasibility
        if 'actions' in strategy_data:
            feasibility_score = self._check_feasibility(strategy_data['actions'])
            validation_result['scores']['feasibility'] = feasibility_score
            
            if feasibility_score < self.quality_thresholds['relevance']:
                validation_result['issues'].append('Some actions may not be feasible')
                validation_result['recommendations'].append('Review action items for practicality')
        
        self._log_validation(validation_result)
        return validation_result
    
    def cross_validate_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-validate outputs from multiple agents for consistency
        
        Args:
            outputs: Dictionary containing outputs from different agents
            
        Returns:
            Cross-validation results
        """
        validation_result = {
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'cross_validation',
            'is_consistent': True,
            'consistency_score': 0.0,
            'conflicts': [],
            'recommendations': []
        }
        
        # Check consistency between research and analysis
        if 'research' in outputs and 'analysis' in outputs:
            consistency = self._check_consistency(outputs['research'], outputs['analysis'])
            validation_result['consistency_score'] = consistency['score']
            validation_result['conflicts'].extend(consistency['conflicts'])
            
            if consistency['score'] < self.quality_thresholds['consistency']:
                validation_result['is_consistent'] = False
                validation_result['recommendations'].append('Resolve conflicts between research and analysis')
        
        # Check alignment between analysis and strategy
        if 'analysis' in outputs and 'strategy' in outputs:
            alignment = self._check_alignment(outputs['analysis'], outputs['strategy'])
            
            if not alignment['is_aligned']:
                validation_result['is_consistent'] = False
                validation_result['conflicts'].extend(alignment['misalignments'])
                validation_result['recommendations'].append('Ensure strategy aligns with analysis findings')
        
        self._log_validation(validation_result)
        return validation_result
    
    def _check_completeness(self, data: Dict[str, Any]) -> float:
        """Check completeness of data"""
        if not data:
            return 0.0
        
        total_fields = len(data)
        non_empty_fields = sum(1 for v in data.values() if v)
        
        return non_empty_fields / total_fields if total_fields > 0 else 0.0
    
    def _validate_sources(self, sources: List[Any]) -> Dict[str, Any]:
        """Validate quality and reliability of sources"""
        result = {
            'score': 1.0,
            'issues': [],
            'recommendations': []
        }
        
        if not sources:
            result['score'] = 0.0
            result['issues'].append('No sources provided')
            result['recommendations'].append('Include credible sources')
            return result
        
        # Check for source diversity
        if len(sources) < 3:
            result['score'] -= 0.2
            result['issues'].append('Limited source diversity')
            result['recommendations'].append('Include more diverse sources')
        
        # Check for source metadata
        sources_with_metadata = sum(1 for s in sources if isinstance(s, dict) and 'url' in s)
        if sources_with_metadata < len(sources) * 0.8:
            result['score'] -= 0.15
            result['issues'].append('Some sources lack metadata')
            result['recommendations'].append('Provide complete source information')
        
        return result
    
    def _check_data_integrity(self, data: Any) -> float:
        """Check integrity of data"""
        if not data:
            return 0.0
        
        score = 1.0
        
        # Check for null/empty values
        if isinstance(data, (list, dict)):
            total_items = len(data)
            if total_items == 0:
                return 0.0
            
            if isinstance(data, list):
                non_null_items = sum(1 for item in data if item is not None)
            else:
                non_null_items = sum(1 for v in data.values() if v is not None)
            
            score = non_null_items / total_items
        
        return score
    
    def _validate_methods(self, methods: Any) -> Dict[str, Any]:
        """Validate analytical methods used"""
        result = {
            'score': 1.0,
            'issues': [],
            'recommendations': []
        }
        
        if not methods:
            result['score'] = 0.0
            result['issues'].append('No methods specified')
            result['recommendations'].append('Document analytical methods used')
        
        return result
    
    def _validate_smart_objectives(self, objectives: List[Any]) -> Dict[str, Any]:
        """Validate if objectives are SMART (Specific, Measurable, Achievable, Relevant, Time-bound)"""
        result = {
            'score': 1.0,
            'issues': [],
            'recommendations': []
        }
        
        if not objectives:
            result['score'] = 0.0
            result['issues'].append('No objectives defined')
            result['recommendations'].append('Define SMART objectives')
            return result
        
        # Check for measurability indicators
        measurable_count = 0
        for obj in objectives:
            obj_str = str(obj).lower()
            if any(indicator in obj_str for indicator in ['%', 'number', 'increase', 'decrease', 'by', 'target']):
                measurable_count += 1
        
        if measurable_count < len(objectives) * 0.7:
            result['score'] -= 0.2
            result['issues'].append('Some objectives lack measurable criteria')
            result['recommendations'].append('Make objectives more measurable')
        
        return result
    
    def _check_feasibility(self, actions: List[Any]) -> float:
        """Check feasibility of proposed actions"""
        if not actions:
            return 0.0
        
        # Simple heuristic: more detailed actions are likely more feasible
        detailed_actions = sum(1 for action in actions if len(str(action)) > 50)
        return detailed_actions / len(actions) if len(actions) > 0 else 0.0
    
    def _check_consistency(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency between two data sets"""
        result = {
            'score': 1.0,
            'conflicts': []
        }
        
        # Check for conflicting conclusions
        if 'conclusions' in data1 and 'conclusions' in data2:
            # Simple check - in real implementation, would use NLP
            pass
        
        return result
    
    def _check_alignment(self, analysis: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Check alignment between analysis and strategy"""
        result = {
            'is_aligned': True,
            'misalignments': []
        }
        
        # Check if strategy addresses analysis findings
        if 'findings' in analysis and 'actions' in strategy:
            if not strategy['actions']:
                result['is_aligned'] = False
                result['misalignments'].append('Strategy lacks actions to address analysis findings')
        
        return result
    
    def _log_validation(self, validation_result: Dict[str, Any]) -> None:
        """Log validation result to history"""
        self.validation_history.append(validation_result)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed"""
        if not self.validation_history:
            return {
                'total_validations': 0,
                'success_rate': 0.0,
                'common_issues': []
            }
        
        total = len(self.validation_history)
        successful = sum(1 for v in self.validation_history if v.get('is_valid', False))
        
        # Collect all issues
        all_issues = []
        for validation in self.validation_history:
            all_issues.extend(validation.get('issues', []))
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_validations': total,
            'success_rate': successful / total if total > 0 else 0.0,
            'common_issues': [{'issue': issue, 'count': count} for issue, count in common_issues],
            'last_validation': self.validation_history[-1]['timestamp']
        }
    
    def generate_quality_report(self) -> str:
        """Generate a comprehensive quality report"""
        summary = self.get_validation_summary()
        
        report = f"""
QUALITY VALIDATION REPORT
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Statistics:
- Total Validations: {summary['total_validations']}
- Success Rate: {summary['success_rate']:.1%}

Common Issues Identified:
"""
        
        if summary['common_issues']:
            for i, issue_data in enumerate(summary['common_issues'], 1):
                report += f"{i}. {issue_data['issue']} (occurred {issue_data['count']} times)\n"
        else:
            report += "No issues identified\n"
        
        report += "\nRecommendations:\n"
        report += "- Continue monitoring quality metrics\n"
        report += "- Address common issues systematically\n"
        report += "- Maintain validation standards\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Initialize validator
    validator = ValidatorAgent()
    
    # Example: Validate research output
    research_output = {
        'sources': [
            {'url': 'https://example.com', 'title': 'Research Paper'},
            {'url': 'https://example2.com', 'title': 'Study'}
        ],
        'findings': ['Finding 1', 'Finding 2'],
        'summary': 'Research summary here'
    }
    
    research_validation = validator.validate_research_output(research_output)
    print("Research Validation:")
    print(json.dumps(research_validation, indent=2))
    
    # Example: Validate analysis output
    analysis_output = {
        'data': [1, 2, 3, 4, 5],
        'methods': ['statistical analysis', 'regression'],
        'conclusions': ['Conclusion 1', 'Conclusion 2']
    }
    
    analysis_validation = validator.validate_analysis_output(analysis_output)
    print("\nAnalysis Validation:")
    print(json.dumps(analysis_validation, indent=2))
    
    # Generate quality report
    print("\n" + validator.generate_quality_report())