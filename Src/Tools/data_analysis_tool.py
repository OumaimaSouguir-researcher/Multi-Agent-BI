"""
Data Analysis Tool Module

Provides comprehensive data analysis capabilities including descriptive statistics,
data profiling, cleaning, transformation, and exploratory data analysis.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json

from base_tool import BaseTool, ToolResult, ToolStatus, ToolCategory, ToolConfig


class DataAnalysisTool(BaseTool):
    """
    Tool for data analysis operations on structured data.
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        """Initialize the Data Analysis Tool"""
        super().__init__(
            name="DataAnalysisTool",
            description="Perform data analysis: profiling, statistics, cleaning, transformation",
            category=ToolCategory.DATA_ANALYSIS,
            config=config
        )
        self._cached_dataframes: Dict[str, pd.DataFrame] = {}
    
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """
        Execute a data analysis operation.
        
        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            ToolResult with analysis results
        """
        operations = {
            "load": self._load_data,
            "profile": self._profile_data,
            "describe": self._describe_data,
            "clean": self._clean_data,
            "filter": self._filter_data,
            "aggregate": self._aggregate_data,
            "merge": self._merge_data,
            "pivot": self._pivot_data,
            "group": self._group_data,
            "transform": self._transform_data,
            "missing": self._analyze_missing,
            "outliers": self._detect_outliers,
            "correlations": self._calculate_correlations,
            "unique": self._get_unique_values,
            "sample": self._sample_data,
            "export": self._export_data
        }
        
        if operation not in operations:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Unknown operation: {operation}",
                execution_time=0.0,
                errors=[f"Valid operations: {', '.join(operations.keys())}"]
            )
        
        return operations[operation](**kwargs)
    
    def validate_params(self, operation: str, **kwargs) -> bool:
        """Validate parameters for data analysis operations"""
        required_params = {
            "load": ["source"],
            "profile": ["data_id"],
            "describe": ["data_id"],
            "clean": ["data_id"],
            "filter": ["data_id", "condition"],
            "aggregate": ["data_id", "columns"],
            "merge": ["data_id1", "data_id2"],
            "pivot": ["data_id"],
            "group": ["data_id", "by"],
            "transform": ["data_id"],
            "missing": ["data_id"],
            "outliers": ["data_id"],
            "correlations": ["data_id"],
            "unique": ["data_id", "column"],
            "sample": ["data_id"],
            "export": ["data_id", "path"]
        }
        
        if operation not in required_params:
            return False
        
        for param in required_params[operation]:
            if param not in kwargs:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        
        return True
    
    def _load_data(
        self,
        source: Union[str, pd.DataFrame, Dict, List],
        data_id: Optional[str] = None,
        file_type: Optional[str] = None,
        **read_kwargs
    ) -> ToolResult:
        """Load data from various sources"""
        try:
            # Generate data_id if not provided
            if not data_id:
                data_id = f"data_{len(self._cached_dataframes)}"
            
            # Load based on source type
            if isinstance(source, pd.DataFrame):
                df = source
            elif isinstance(source, (dict, list)):
                df = pd.DataFrame(source)
            elif isinstance(source, str):
                source_path = Path(source)
                if not source_path.exists():
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        data=None,
                        message=f"File not found: {source}",
                        execution_time=0.0,
                        errors=["Source file does not exist"]
                    )
                
                # Determine file type
                if not file_type:
                    file_type = source_path.suffix.lower()
                
                # Load based on file type
                if file_type in ['.csv', '.txt']:
                    df = pd.read_csv(source, **read_kwargs)
                elif file_type in ['.xlsx', '.xls']:
                    df = pd.read_excel(source, **read_kwargs)
                elif file_type == '.json':
                    df = pd.read_json(source, **read_kwargs)
                elif file_type == '.parquet':
                    df = pd.read_parquet(source, **read_kwargs)
                elif file_type in ['.h5', '.hdf5']:
                    df = pd.read_hdf(source, **read_kwargs)
                else:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        data=None,
                        message=f"Unsupported file type: {file_type}",
                        execution_time=0.0,
                        errors=["Supported types: csv, xlsx, json, parquet, hdf5"]
                    )
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message="Invalid source type",
                    execution_time=0.0,
                    errors=["Source must be filepath, DataFrame, dict, or list"]
                )
            
            # Cache the dataframe
            self._cached_dataframes[data_id] = df
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "data_id": data_id,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "memory_usage": df.memory_usage(deep=True).sum()
                },
                message=f"Loaded data with shape {df.shape}",
                execution_time=0.0,
                metadata={"source": str(source)}
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to load data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _profile_data(self, data_id: str) -> ToolResult:
        """Generate comprehensive data profile"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            profile = {
                "shape": {
                    "rows": len(df),
                    "columns": len(df.columns)
                },
                "columns": {},
                "missing_data": {
                    "total": df.isnull().sum().sum(),
                    "percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                },
                "duplicates": {
                    "count": df.duplicated().sum(),
                    "percentage": (df.duplicated().sum() / len(df)) * 100
                },
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
            # Profile each column
            for col in df.columns:
                col_data = df[col]
                col_profile = {
                    "dtype": str(col_data.dtype),
                    "missing": col_data.isnull().sum(),
                    "missing_pct": (col_data.isnull().sum() / len(col_data)) * 100,
                    "unique": col_data.nunique(),
                    "unique_pct": (col_data.nunique() / len(col_data)) * 100
                }
                
                # Numeric columns
                if pd.api.types.is_numeric_dtype(col_data):
                    col_profile.update({
                        "mean": float(col_data.mean()) if not col_data.empty else None,
                        "std": float(col_data.std()) if not col_data.empty else None,
                        "min": float(col_data.min()) if not col_data.empty else None,
                        "max": float(col_data.max()) if not col_data.empty else None,
                        "median": float(col_data.median()) if not col_data.empty else None,
                        "zeros": (col_data == 0).sum()
                    })
                
                # Categorical/Object columns
                elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                    value_counts = col_data.value_counts().head(10)
                    col_profile.update({
                        "top_values": value_counts.to_dict(),
                        "most_common": str(col_data.mode()[0]) if not col_data.mode().empty else None
                    })
                
                profile["columns"][col] = col_profile
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=profile,
                message=f"Generated profile for dataset with {df.shape[0]} rows and {df.shape[1]} columns",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to profile data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _describe_data(
        self,
        data_id: str,
        percentiles: Optional[List[float]] = None
    ) -> ToolResult:
        """Generate descriptive statistics"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            percentiles = percentiles or [0.25, 0.5, 0.75]
            description = df.describe(percentiles=percentiles)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "statistics": description.to_dict(),
                    "shape": df.shape
                },
                message="Generated descriptive statistics",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to describe data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _clean_data(
        self,
        data_id: str,
        drop_duplicates: bool = True,
        drop_missing: bool = False,
        fill_missing: Optional[Union[str, Dict]] = None,
        output_id: Optional[str] = None
    ) -> ToolResult:
        """Clean data by handling duplicates and missing values"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            df_cleaned = df.copy()
            operations = []
            
            # Drop duplicates
            if drop_duplicates:
                before = len(df_cleaned)
                df_cleaned = df_cleaned.drop_duplicates()
                after = len(df_cleaned)
                operations.append(f"Removed {before - after} duplicate rows")
            
            # Handle missing values
            if drop_missing:
                before = len(df_cleaned)
                df_cleaned = df_cleaned.dropna()
                after = len(df_cleaned)
                operations.append(f"Removed {before - after} rows with missing values")
            
            elif fill_missing is not None:
                if isinstance(fill_missing, dict):
                    df_cleaned = df_cleaned.fillna(fill_missing)
                else:
                    df_cleaned = df_cleaned.fillna(fill_missing)
                operations.append(f"Filled missing values")
            
            # Store cleaned data
            output_id = output_id or f"{data_id}_cleaned"
            self._cached_dataframes[output_id] = df_cleaned
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "output_id": output_id,
                    "original_shape": df.shape,
                    "cleaned_shape": df_cleaned.shape,
                    "operations": operations
                },
                message=f"Cleaned data: {'; '.join(operations)}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to clean data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _filter_data(
        self,
        data_id: str,
        condition: str,
        output_id: Optional[str] = None
    ) -> ToolResult:
        """Filter data based on condition"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            # Evaluate condition
            df_filtered = df.query(condition)
            
            output_id = output_id or f"{data_id}_filtered"
            self._cached_dataframes[output_id] = df_filtered
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "output_id": output_id,
                    "original_rows": len(df),
                    "filtered_rows": len(df_filtered),
                    "removed_rows": len(df) - len(df_filtered)
                },
                message=f"Filtered from {len(df)} to {len(df_filtered)} rows",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to filter data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _aggregate_data(
        self,
        data_id: str,
        columns: List[str],
        operations: Union[str, List[str], Dict] = "sum",
        output_id: Optional[str] = None
    ) -> ToolResult:
        """Aggregate data"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            df_agg = df[columns].agg(operations)
            
            result_data = df_agg.to_dict() if isinstance(df_agg, pd.Series) else df_agg.to_dict()
            
            if output_id:
                self._cached_dataframes[output_id] = pd.DataFrame(df_agg)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "aggregations": result_data,
                    "output_id": output_id
                },
                message="Aggregated data successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to aggregate data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _merge_data(
        self,
        data_id1: str,
        data_id2: str,
        how: str = "inner",
        on: Optional[Union[str, List[str]]] = None,
        output_id: Optional[str] = None
    ) -> ToolResult:
        """Merge two datasets"""
        try:
            df1 = self._get_dataframe(data_id1)
            df2 = self._get_dataframe(data_id2)
            
            if df1 is None:
                return self._data_not_found_error(data_id1)
            if df2 is None:
                return self._data_not_found_error(data_id2)
            
            df_merged = pd.merge(df1, df2, how=how, on=on)
            
            output_id = output_id or f"{data_id1}_{data_id2}_merged"
            self._cached_dataframes[output_id] = df_merged
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "output_id": output_id,
                    "shape": df_merged.shape,
                    "merge_type": how
                },
                message=f"Merged datasets: result shape {df_merged.shape}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to merge data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _pivot_data(
        self,
        data_id: str,
        index: str,
        columns: str,
        values: str,
        aggfunc: str = "mean",
        output_id: Optional[str] = None
    ) -> ToolResult:
        """Create pivot table"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            df_pivot = pd.pivot_table(
                df,
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )
            
            output_id = output_id or f"{data_id}_pivot"
            self._cached_dataframes[output_id] = df_pivot
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "output_id": output_id,
                    "shape": df_pivot.shape,
                    "preview": df_pivot.head(10).to_dict()
                },
                message="Created pivot table",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create pivot table: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _group_data(
        self,
        data_id: str,
        by: Union[str, List[str]],
        agg_dict: Optional[Dict] = None,
        output_id: Optional[str] = None
    ) -> ToolResult:
        """Group data and aggregate"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            grouped = df.groupby(by)
            
            if agg_dict:
                df_grouped = grouped.agg(agg_dict)
            else:
                df_grouped = grouped.size().to_frame('count')
            
            df_grouped = df_grouped.reset_index()
            
            output_id = output_id or f"{data_id}_grouped"
            self._cached_dataframes[output_id] = df_grouped
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "output_id": output_id,
                    "groups": len(df_grouped),
                    "preview": df_grouped.head(10).to_dict()
                },
                message=f"Grouped data into {len(df_grouped)} groups",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to group data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _transform_data(
        self,
        data_id: str,
        transformations: Dict[str, str],
        output_id: Optional[str] = None
    ) -> ToolResult:
        """Apply transformations to columns"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            df_transformed = df.copy()
            
            for col, transformation in transformations.items():
                if col in df_transformed.columns:
                    df_transformed[col] = df_transformed[col].apply(eval(transformation))
            
            output_id = output_id or f"{data_id}_transformed"
            self._cached_dataframes[output_id] = df_transformed
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "output_id": output_id,
                    "transformed_columns": list(transformations.keys())
                },
                message="Applied transformations",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to transform data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _analyze_missing(self, data_id: str) -> ToolResult:
        """Analyze missing data patterns"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            missing_data = {
                "total_missing": df.isnull().sum().sum(),
                "by_column": df.isnull().sum().to_dict(),
                "by_column_pct": (df.isnull().sum() / len(df) * 100).to_dict(),
                "rows_with_missing": df.isnull().any(axis=1).sum(),
                "complete_rows": (~df.isnull().any(axis=1)).sum()
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=missing_data,
                message=f"Found {missing_data['total_missing']} missing values",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to analyze missing data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _detect_outliers(
        self,
        data_id: str,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> ToolResult:
        """Detect outliers in numeric columns"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            outliers = {}
            
            for col in columns:
                if col not in df.columns:
                    continue
                
                col_data = df[col].dropna()
                
                if method == "iqr":
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - threshold * IQR
                    upper = Q3 + threshold * IQR
                    outlier_mask = (col_data < lower) | (col_data > upper)
                
                elif method == "zscore":
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    outlier_mask = z_scores > threshold
                
                else:
                    continue
                
                outliers[col] = {
                    "count": outlier_mask.sum(),
                    "percentage": (outlier_mask.sum() / len(col_data)) * 100,
                    "values": col_data[outlier_mask].tolist()[:10]  # First 10 outliers
                }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={"outliers": outliers, "method": method},
                message=f"Detected outliers using {method} method",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to detect outliers: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _calculate_correlations(
        self,
        data_id: str,
        method: str = "pearson",
        columns: Optional[List[str]] = None
    ) -> ToolResult:
        """Calculate correlation matrix"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            if columns:
                df_corr = df[columns]
            else:
                df_corr = df.select_dtypes(include=[np.number])
            
            corr_matrix = df_corr.corr(method=method)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "correlations": corr_matrix.to_dict(),
                    "method": method,
                    "shape": corr_matrix.shape
                },
                message=f"Calculated {method} correlations",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to calculate correlations: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _get_unique_values(
        self,
        data_id: str,
        column: str,
        limit: int = 100
    ) -> ToolResult:
        """Get unique values from a column"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            if column not in df.columns:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Column not found: {column}",
                    execution_time=0.0,
                    errors=[f"Available columns: {', '.join(df.columns)}"]
                )
            
            unique_values = df[column].unique()[:limit].tolist()
            value_counts = df[column].value_counts().head(limit).to_dict()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "column": column,
                    "unique_count": df[column].nunique(),
                    "unique_values": unique_values,
                    "value_counts": value_counts
                },
                message=f"Found {df[column].nunique()} unique values",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to get unique values: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _sample_data(
        self,
        data_id: str,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        output_id: Optional[str] = None
    ) -> ToolResult:
        """Sample data"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            df_sample = df.sample(n=n, frac=frac, random_state=random_state)
            
            if output_id:
                self._cached_dataframes[output_id] = df_sample
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "output_id": output_id,
                    "sample_size": len(df_sample),
                    "preview": df_sample.head(10).to_dict()
                },
                message=f"Sampled {len(df_sample)} rows",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to sample data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _export_data(
        self,
        data_id: str,
        path: str,
        file_type: Optional[str] = None,
        **write_kwargs
    ) -> ToolResult:
        """Export data to file"""
        try:
            df = self._get_dataframe(data_id)
            if df is None:
                return self._data_not_found_error(data_id)
            
            output_path = Path(path)
            
            if not file_type:
                file_type = output_path.suffix.lower()
            
            # Export based on file type
            if file_type in ['.csv', '.txt']:
                df.to_csv(path, **write_kwargs)
            elif file_type in ['.xlsx', '.xls']:
                df.to_excel(path, **write_kwargs)
            elif file_type == '.json':
                df.to_json(path, **write_kwargs)
            elif file_type == '.parquet':
                df.to_parquet(path, **write_kwargs)
            elif file_type in ['.h5', '.hdf5']:
                df.to_hdf(path, key='data', **write_kwargs)
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Unsupported file type: {file_type}",
                    execution_time=0.0,
                    errors=["Supported types: csv, xlsx, json, parquet, hdf5"]
                )
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "path": str(output_path),
                    "file_type": file_type,
                    "size": output_path.stat().st_size
                },
                message=f"Exported data to {path}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to export data: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _get_dataframe(self, data_id: str) -> Optional[pd.DataFrame]:
        """Retrieve cached dataframe"""
        return self._cached_dataframes.get(data_id)
    
    def _data_not_found_error(self, data_id: str) -> ToolResult:
        """Return data not found error"""
        return ToolResult(
            status=ToolStatus.ERROR,
            data=None,
            message=f"Data not found: {data_id}",
            execution_time=0.0,
            errors=[f"Available datasets: {', '.join(self._cached_dataframes.keys())}"]
        )
    
    def list_cached_data(self) -> List[str]:
        """List all cached datasets"""
        return list(self._cached_dataframes.keys())
    
    def clear_cached_data(self, data_id: Optional[str] = None):
        """Clear cached datasets"""
        if data_id:
            if data_id in self._cached_dataframes:
                del self._cached_dataframes[data_id]
        else:
            self._cached_dataframes.clear()