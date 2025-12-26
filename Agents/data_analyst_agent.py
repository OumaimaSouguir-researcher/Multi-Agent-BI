"""
Data Analyst Agent - Specialized in data analysis and statistical operations.

This agent handles data processing, statistical analysis, visualization,
and data-driven insights generation.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
from base_agent import BaseAgent


class DataAnalystAgent(BaseAgent):
    """
    Agent specialized in data analysis tasks.
    
    Capabilities:
    - Data cleaning and preprocessing
    - Statistical analysis
    - Data visualization recommendations
    - Pattern recognition
    - Trend analysis
    - Data quality assessment
    """
    
    def __init__(self, name: str = "DataAnalyst", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Data Analyst Agent.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary
        """
        default_config = {
            "capabilities": [
                "data_cleaning",
                "statistical_analysis",
                "visualization",
                "trend_analysis",
                "data_quality_check",
                "correlation_analysis",
                "anomaly_detection"
            ],
            "max_dataset_size": 1000000,  # Maximum rows to process
            "supported_formats": ["csv", "json", "excel", "parquet"],
            "analysis_depth": "comprehensive"  # quick, standard, comprehensive
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name=name, role="Data Analyst", config=default_config)
        self.current_datasets: Dict[str, pd.DataFrame] = {}
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate if the task is appropriate for data analysis.
        
        Args:
            task: Task dictionary
            
        Returns:
            True if task can be handled
        """
        required_fields = ["task_type", "data"]
        
        if not all(field in task for field in required_fields):
            self.logger.error(f"Missing required fields: {required_fields}")
            return False
        
        task_type = task.get("task_type")
        valid_types = [
            "analyze_data",
            "clean_data",
            "visualize_data",
            "statistical_summary",
            "correlation_analysis",
            "trend_analysis",
            "anomaly_detection",
            "data_quality_check"
        ]
        
        if task_type not in valid_types:
            self.logger.error(f"Invalid task type: {task_type}")
            return False
        
        return True
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a data analysis task.
        
        Args:
            task: Dictionary containing:
                - task_type: Type of analysis
                - data: Data to analyze (DataFrame, dict, or dataset_id)
                - parameters: Optional analysis parameters
                
        Returns:
            Dictionary with analysis results
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
            data = task["data"]
            parameters = task.get("parameters", {})
            
            # Load data if needed
            df = self._load_data(data)
            
            # Route to appropriate analysis method
            if task_type == "analyze_data":
                result = await self._comprehensive_analysis(df, parameters)
            elif task_type == "clean_data":
                result = await self._clean_data(df, parameters)
            elif task_type == "visualize_data":
                result = await self._generate_visualization_spec(df, parameters)
            elif task_type == "statistical_summary":
                result = await self._statistical_summary(df, parameters)
            elif task_type == "correlation_analysis":
                result = await self._correlation_analysis(df, parameters)
            elif task_type == "trend_analysis":
                result = await self._trend_analysis(df, parameters)
            elif task_type == "anomaly_detection":
                result = await self._detect_anomalies(df, parameters)
            elif task_type == "data_quality_check":
                result = await self._data_quality_check(df, parameters)
            else:
                result = {"error": f"Unknown task type: {task_type}"}
            
            result["success"] = True
            result["timestamp"] = datetime.now().isoformat()
            
            self.log_task(task, result)
            self.update_state("idle")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task: {str(e)}")
            self.update_state("idle")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _load_data(self, data: Any) -> pd.DataFrame:
        """Load data into a DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, str):
            # Assume it's a dataset_id
            if data in self.current_datasets:
                return self.current_datasets[data]
            else:
                raise ValueError(f"Dataset not found: {data}")
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    async def _comprehensive_analysis(self, df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Perform comprehensive data analysis."""
        self.logger.info("Performing comprehensive analysis")
        
        analysis = {
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
                "column_types": df.dtypes.astype(str).to_dict()
            },
            "summary_statistics": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "unique_values": {col: df[col].nunique() for col in df.columns},
            "data_quality_score": self._calculate_quality_score(df)
        }
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis["column_classification"] = {
            "numeric": numeric_cols,
            "categorical": categorical_cols
        }
        
        # Additional insights
        if numeric_cols:
            analysis["numeric_insights"] = self._analyze_numeric_columns(df[numeric_cols])
        
        if categorical_cols:
            analysis["categorical_insights"] = self._analyze_categorical_columns(df[categorical_cols])
        
        return analysis
    
    async def _clean_data(self, df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Clean the dataset."""
        self.logger.info("Cleaning data")
        
        original_shape = df.shape
        cleaning_report = {
            "original_shape": original_shape,
            "operations_performed": []
        }
        
        # Remove duplicates
        if params.get("remove_duplicates", True):
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            if before != after:
                cleaning_report["operations_performed"].append({
                    "operation": "remove_duplicates",
                    "rows_removed": before - after
                })
        
        # Handle missing values
        missing_strategy = params.get("missing_strategy", "drop")
        if missing_strategy == "drop":
            before = len(df)
            df = df.dropna()
            after = len(df)
            if before != after:
                cleaning_report["operations_performed"].append({
                    "operation": "drop_missing",
                    "rows_removed": before - after
                })
        elif missing_strategy == "fill":
            fill_value = params.get("fill_value", 0)
            df = df.fillna(fill_value)
            cleaning_report["operations_performed"].append({
                "operation": "fill_missing",
                "fill_value": fill_value
            })
        
        # Remove outliers if requested
        if params.get("remove_outliers", False):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            before = len(df)
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
            after = len(df)
            if before != after:
                cleaning_report["operations_performed"].append({
                    "operation": "remove_outliers",
                    "rows_removed": before - after
                })
        
        cleaning_report["final_shape"] = df.shape
        cleaning_report["cleaned_data_preview"] = df.head(10).to_dict()
        
        return cleaning_report
    
    async def _statistical_summary(self, df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Generate statistical summary."""
        self.logger.info("Generating statistical summary")
        
        summary = {
            "basic_stats": df.describe().to_dict(),
            "additional_stats": {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            summary["additional_stats"][col] = {
                "median": float(df[col].median()),
                "mode": float(df[col].mode()[0]) if not df[col].mode().empty else None,
                "skewness": float(df[col].skew()),
                "kurtosis": float(df[col].kurtosis()),
                "variance": float(df[col].var())
            }
        
        return summary
    
    async def _correlation_analysis(self, df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Analyze correlations between variables."""
        self.logger.info("Performing correlation analysis")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No numeric columns found for correlation analysis"}
        
        correlation_matrix = numeric_df.corr().to_dict()
        
        # Find strong correlations
        strong_correlations = []
        threshold = params.get("threshold", 0.7)
        
        for col1 in numeric_df.columns:
            for col2 in numeric_df.columns:
                if col1 < col2:  # Avoid duplicates
                    corr_value = numeric_df[col1].corr(numeric_df[col2])
                    if abs(corr_value) >= threshold:
                        strong_correlations.append({
                            "variable1": col1,
                            "variable2": col2,
                            "correlation": float(corr_value)
                        })
        
        return {
            "correlation_matrix": correlation_matrix,
            "strong_correlations": strong_correlations,
            "threshold": threshold
        }
    
    async def _trend_analysis(self, df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Analyze trends in time series or sequential data."""
        self.logger.info("Performing trend analysis")
        
        target_column = params.get("target_column")
        
        if not target_column or target_column not in df.columns:
            return {"error": "Target column not specified or not found"}
        
        data = df[target_column].dropna()
        
        # Calculate basic trend metrics
        trend_info = {
            "overall_trend": "increasing" if data.iloc[-1] > data.iloc[0] else "decreasing",
            "percent_change": float((data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100),
            "volatility": float(data.std()),
            "data_points": len(data)
        }
        
        return trend_info
    
    async def _detect_anomalies(self, df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Detect anomalies in the dataset."""
        self.logger.info("Detecting anomalies")
        
        numeric_df = df.select_dtypes(include=[np.number])
        anomalies = {}
        
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomaly_indices = numeric_df[
                (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
            ].index.tolist()
            
            if anomaly_indices:
                anomalies[col] = {
                    "count": len(anomaly_indices),
                    "percentage": len(anomaly_indices) / len(df) * 100,
                    "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)}
                }
        
        return {"anomalies_by_column": anomalies}
    
    async def _data_quality_check(self, df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Perform comprehensive data quality assessment."""
        self.logger.info("Checking data quality")
        
        quality_report = {
            "completeness": {},
            "consistency": {},
            "validity": {},
            "overall_score": 0
        }
        
        # Completeness check
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            quality_report["completeness"][col] = {
                "missing_percentage": float(missing_pct),
                "score": float(100 - missing_pct)
            }
        
        # Consistency check (duplicates)
        duplicate_count = df.duplicated().sum()
        quality_report["consistency"]["duplicates"] = {
            "count": int(duplicate_count),
            "percentage": float(duplicate_count / len(df) * 100)
        }
        
        # Calculate overall quality score
        completeness_score = np.mean([v["score"] for v in quality_report["completeness"].values()])
        consistency_score = 100 - quality_report["consistency"]["duplicates"]["percentage"]
        
        quality_report["overall_score"] = float((completeness_score + consistency_score) / 2)
        
        return quality_report
    
    async def _generate_visualization_spec(self, df: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Generate visualization specifications."""
        self.logger.info("Generating visualization specifications")
        
        viz_type = params.get("viz_type", "auto")
        
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Recommend visualizations based on data types
        if len(numeric_cols) >= 2:
            recommendations.append({
                "type": "scatter_plot",
                "x_axis": numeric_cols[0],
                "y_axis": numeric_cols[1],
                "purpose": "Explore relationship between two numeric variables"
            })
        
        if numeric_cols:
            recommendations.append({
                "type": "histogram",
                "variable": numeric_cols[0],
                "purpose": "Show distribution of numeric data"
            })
        
        if categorical_cols and numeric_cols:
            recommendations.append({
                "type": "bar_chart",
                "category": categorical_cols[0],
                "value": numeric_cols[0],
                "purpose": "Compare numeric values across categories"
            })
        
        return {
            "recommendations": recommendations,
            "data_summary": {
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols
            }
        }
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
        
        return float((completeness + uniqueness) / 2)
    
    def _analyze_numeric_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numeric columns."""
        insights = {}
        
        for col in df.columns:
            insights[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "range": float(df[col].max() - df[col].min())
            }
        
        return insights
    
    def _analyze_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical columns."""
        insights = {}
        
        for col in df.columns:
            value_counts = df[col].value_counts()
            insights[col] = {
                "unique_values": int(df[col].nunique()),
                "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "top_5_values": value_counts.head(5).to_dict()
            }
        
        return insights
    
    def store_dataset(self, dataset_id: str, df: pd.DataFrame):
        """Store a dataset for later use."""
        self.current_datasets[dataset_id] = df
        self.logger.info(f"Stored dataset: {dataset_id}")
    
    def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Retrieve a stored dataset."""
        return self.current_datasets.get(dataset_id)
    
    def list_datasets(self) -> List[str]:
        """List all stored datasets."""
        return list(self.current_datasets.keys())