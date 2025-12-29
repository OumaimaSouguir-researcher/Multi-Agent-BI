"""
Visualization Tool Module

Provides data visualization capabilities including various plot types,
charts, and advanced visualization techniques.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import base64
from io import BytesIO

from base_tool import BaseTool, ToolResult, ToolStatus, ToolCategory, ToolConfig


class VisualizationTool(BaseTool):
    """
    Tool for creating data visualizations.
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        """Initialize the Visualization Tool"""
        super().__init__(
            name="VisualizationTool",
            description="Create data visualizations: plots, charts, graphs",
            category=ToolCategory.VISUALIZATION,
            config=config
        )
        # Set default style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """
        Execute a visualization operation.
        
        Args:
            operation: Type of visualization to create
            **kwargs: Visualization-specific parameters
            
        Returns:
            ToolResult with visualization data
        """
        operations = {
            "line": self._line_plot,
            "scatter": self._scatter_plot,
            "bar": self._bar_plot,
            "histogram": self._histogram,
            "box": self._box_plot,
            "violin": self._violin_plot,
            "heatmap": self._heatmap,
            "pie": self._pie_chart,
            "area": self._area_plot,
            "density": self._density_plot,
            "pair": self._pair_plot,
            "correlation": self._correlation_matrix,
            "distribution": self._distribution_plot,
            "time_series": self._time_series_plot,
            "3d_scatter": self._scatter_3d
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
        """Validate parameters for visualization operations"""
        required_params = {
            "line": ["x", "y"],
            "scatter": ["x", "y"],
            "bar": ["categories", "values"],
            "histogram": ["data"],
            "box": ["data"],
            "violin": ["data"],
            "heatmap": ["data"],
            "pie": ["values"],
            "area": ["x", "y"],
            "density": ["data"],
            "pair": ["data"],
            "correlation": ["data"],
            "distribution": ["data"],
            "time_series": ["data"],
            "3d_scatter": ["x", "y", "z"]
        }
        
        if operation not in required_params:
            return False
        
        for param in required_params[operation]:
            if param not in kwargs:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        
        return True
    
    def _save_figure(
        self,
        fig: plt.Figure,
        output_path: Optional[str] = None,
        return_base64: bool = True
    ) -> Dict[str, Any]:
        """Save figure to file and/or return as base64"""
        result = {}
        
        # Save to file if path provided
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            result["file_path"] = output_path
        
        # Return as base64
        if return_base64:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode()
            result["base64"] = img_base64
            buffer.close()
        
        plt.close(fig)
        return result
    
    def _line_plot(
        self,
        x: Union[List, np.ndarray],
        y: Union[List, np.ndarray, List[List]],
        labels: Optional[List[str]] = None,
        title: str = "Line Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create line plot"""
        try:
            fig, ax = plt.subplots()
            
            # Handle multiple y series
            if isinstance(y[0], (list, np.ndarray)):
                for idx, y_series in enumerate(y):
                    label = labels[idx] if labels and idx < len(labels) else f"Series {idx+1}"
                    ax.plot(x, y_series, label=label, **kwargs)
                ax.legend()
            else:
                ax.plot(x, y, **kwargs)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Line plot created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create line plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _scatter_plot(
        self,
        x: Union[List, np.ndarray],
        y: Union[List, np.ndarray],
        colors: Optional[Union[List, np.ndarray]] = None,
        sizes: Optional[Union[List, np.ndarray]] = None,
        title: str = "Scatter Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create scatter plot"""
        try:
            fig, ax = plt.subplots()
            
            scatter_kwargs = kwargs.copy()
            if colors is not None:
                scatter_kwargs['c'] = colors
            if sizes is not None:
                scatter_kwargs['s'] = sizes
            
            scatter = ax.scatter(x, y, alpha=0.6, **scatter_kwargs)
            
            if colors is not None:
                plt.colorbar(scatter, ax=ax)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Scatter plot created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create scatter plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _bar_plot(
        self,
        categories: Union[List, np.ndarray],
        values: Union[List, np.ndarray, List[List]],
        labels: Optional[List[str]] = None,
        title: str = "Bar Plot",
        xlabel: str = "Categories",
        ylabel: str = "Values",
        horizontal: bool = False,
        stacked: bool = False,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create bar plot"""
        try:
            fig, ax = plt.subplots()
            
            # Handle multiple value series
            if isinstance(values[0], (list, np.ndarray)):
                x_pos = np.arange(len(categories))
                width = 0.8 / len(values) if not stacked else 0.8
                
                for idx, val_series in enumerate(values):
                    label = labels[idx] if labels and idx < len(labels) else f"Series {idx+1}"
                    offset = (idx - len(values)/2 + 0.5) * width if not stacked else 0
                    bottom = np.sum(values[:idx], axis=0) if stacked and idx > 0 else None
                    
                    if horizontal:
                        ax.barh(x_pos + offset, val_series, width, label=label, 
                               left=bottom, **kwargs)
                    else:
                        ax.bar(x_pos + offset, val_series, width, label=label, 
                              bottom=bottom, **kwargs)
                
                ax.legend()
                if horizontal:
                    ax.set_yticks(x_pos)
                    ax.set_yticklabels(categories)
                else:
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(categories, rotation=45, ha='right')
            else:
                if horizontal:
                    ax.barh(categories, values, **kwargs)
                else:
                    ax.bar(categories, values, **kwargs)
                    ax.tick_params(axis='x', rotation=45)
            
            ax.set_title(title)
            if horizontal:
                ax.set_ylabel(xlabel)
                ax.set_xlabel(ylabel)
            else:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis='y' if not horizontal else 'x')
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Bar plot created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create bar plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _histogram(
        self,
        data: Union[List, np.ndarray, List[List]],
        bins: Union[int, str] = 'auto',
        labels: Optional[List[str]] = None,
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create histogram"""
        try:
            fig, ax = plt.subplots()
            
            # Handle multiple datasets
            if isinstance(data[0], (list, np.ndarray)):
                for idx, dataset in enumerate(data):
                    label = labels[idx] if labels and idx < len(labels) else f"Dataset {idx+1}"
                    ax.hist(dataset, bins=bins, alpha=0.6, label=label, **kwargs)
                ax.legend()
            else:
                ax.hist(data, bins=bins, **kwargs)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis='y')
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Histogram created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create histogram: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _box_plot(
        self,
        data: Union[List, np.ndarray, List[List]],
        labels: Optional[List[str]] = None,
        title: str = "Box Plot",
        ylabel: str = "Values",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create box plot"""
        try:
            fig, ax = plt.subplots()
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True, **kwargs)
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis='y')
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Box plot created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create box plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _violin_plot(
        self,
        data: Union[pd.DataFrame, List[List]],
        x: Optional[str] = None,
        y: Optional[str] = None,
        title: str = "Violin Plot",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create violin plot"""
        try:
            fig, ax = plt.subplots()
            
            if isinstance(data, pd.DataFrame):
                sns.violinplot(data=data, x=x, y=y, ax=ax, **kwargs)
            else:
                sns.violinplot(data=data, ax=ax, **kwargs)
            
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='y')
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Violin plot created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create violin plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _heatmap(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        annot: bool = True,
        fmt: str = ".2f",
        cmap: str = "coolwarm",
        title: str = "Heatmap",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create heatmap"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, ax=ax, **kwargs)
            
            ax.set_title(title)
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Heatmap created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create heatmap: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _pie_chart(
        self,
        values: Union[List, np.ndarray],
        labels: Optional[List[str]] = None,
        title: str = "Pie Chart",
        explode: Optional[List[float]] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create pie chart"""
        try:
            fig, ax = plt.subplots()
            
            ax.pie(values, labels=labels, explode=explode, autopct='%1.1f%%',
                   startangle=90, **kwargs)
            ax.axis('equal')
            ax.set_title(title)
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Pie chart created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create pie chart: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _area_plot(
        self,
        x: Union[List, np.ndarray],
        y: Union[List, np.ndarray, List[List]],
        labels: Optional[List[str]] = None,
        title: str = "Area Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        stacked: bool = True,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create area plot"""
        try:
            fig, ax = plt.subplots()
            
            if isinstance(y[0], (list, np.ndarray)):
                if stacked:
                    ax.stackplot(x, *y, labels=labels, alpha=0.7, **kwargs)
                else:
                    for idx, y_series in enumerate(y):
                        label = labels[idx] if labels and idx < len(labels) else f"Series {idx+1}"
                        ax.fill_between(x, y_series, alpha=0.5, label=label, **kwargs)
                if labels:
                    ax.legend()
            else:
                ax.fill_between(x, y, alpha=0.7, **kwargs)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Area plot created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create area plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _density_plot(
        self,
        data: Union[List, np.ndarray, List[List]],
        labels: Optional[List[str]] = None,
        title: str = "Density Plot",
        xlabel: str = "Value",
        ylabel: str = "Density",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create density plot (KDE)"""
        try:
            fig, ax = plt.subplots()
            
            if isinstance(data[0], (list, np.ndarray)):
                for idx, dataset in enumerate(data):
                    label = labels[idx] if labels and idx < len(labels) else f"Dataset {idx+1}"
                    sns.kdeplot(data=dataset, ax=ax, label=label, **kwargs)
                ax.legend()
            else:
                sns.kdeplot(data=data, ax=ax, **kwargs)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Density plot created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create density plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _pair_plot(
        self,
        data: pd.DataFrame,
        hue: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create pair plot"""
        try:
            pair_grid = sns.pairplot(data, hue=hue, **kwargs)
            
            output_data = self._save_figure(pair_grid.fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Pair plot created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create pair plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _correlation_matrix(
        self,
        data: pd.DataFrame,
        method: str = "pearson",
        title: str = "Correlation Matrix",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create correlation matrix visualization"""
        try:
            # Calculate correlation
            corr = data.corr(method=method)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                       vmin=-1, vmax=1, center=0, ax=ax, **kwargs)
            
            ax.set_title(title)
            
            output_data = self._save_figure(fig, output_path)
            output_data['correlation_matrix'] = corr.to_dict()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Correlation matrix created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create correlation matrix: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _distribution_plot(
        self,
        data: Union[List, np.ndarray],
        dist_type: str = "norm",
        title: str = "Distribution Plot",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Plot data distribution with fitted curve"""
        try:
            from scipy import stats as sp_stats
            
            fig, ax = plt.subplots()
            
            # Plot histogram
            n, bins, patches = ax.hist(data, bins='auto', density=True, 
                                       alpha=0.7, label='Data', **kwargs)
            
            # Fit distribution
            dist = getattr(sp_stats, dist_type)
            params = dist.fit(data)
            
            # Plot fitted distribution
            x = np.linspace(min(data), max(data), 100)
            ax.plot(x, dist.pdf(x, *params), 'r-', lw=2, 
                   label=f'Fitted {dist_type}')
            
            ax.set_title(title)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            output_data = self._save_figure(fig, output_path)
            output_data['distribution_params'] = list(params)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message=f"Distribution plot with {dist_type} fit created",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create distribution plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _time_series_plot(
        self,
        data: Union[pd.DataFrame, pd.Series],
        y: Optional[str] = None,
        title: str = "Time Series Plot",
        ylabel: str = "Value",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create time series plot"""
        try:
            fig, ax = plt.subplots()
            
            if isinstance(data, pd.DataFrame):
                if y:
                    data[y].plot(ax=ax, **kwargs)
                else:
                    data.plot(ax=ax, **kwargs)
            else:
                data.plot(ax=ax, **kwargs)
            
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="Time series plot created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create time series plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _scatter_3d(
        self,
        x: Union[List, np.ndarray],
        y: Union[List, np.ndarray],
        z: Union[List, np.ndarray],
        colors: Optional[Union[List, np.ndarray]] = None,
        title: str = "3D Scatter Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        zlabel: str = "Z",
        output_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Create 3D scatter plot"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter_kwargs = kwargs.copy()
            if colors is not None:
                scatter_kwargs['c'] = colors
            
            scatter = ax.scatter(x, y, z, **scatter_kwargs)
            
            if colors is not None:
                plt.colorbar(scatter, ax=ax)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)
            
            output_data = self._save_figure(fig, output_path)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=output_data,
                message="3D scatter plot created successfully",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to create 3D scatter plot: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )