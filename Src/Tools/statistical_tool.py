"""
Statistical Tool Module

Provides statistical analysis capabilities including hypothesis testing,
distributions, regression, and advanced statistical methods.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr

from base_tool import BaseTool, ToolResult, ToolStatus, ToolCategory, ToolConfig


class StatisticalTool(BaseTool):
    """
    Tool for statistical analysis and hypothesis testing.
    """
    
    def __init__(self, config: Optional[ToolConfig] = None):
        """Initialize the Statistical Tool"""
        super().__init__(
            name="StatisticalTool",
            description="Perform statistical analysis: hypothesis testing, distributions, regression",
            category=ToolCategory.STATISTICS,
            config=config
        )
    
    def execute(self, operation: str, **kwargs) -> ToolResult:
        """
        Execute a statistical operation.
        
        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters
            
        Returns:
            ToolResult with analysis results
        """
        operations = {
            "ttest": self._t_test,
            "anova": self._anova,
            "chi_square": self._chi_square_test,
            "correlation": self._correlation_test,
            "normality": self._normality_test,
            "regression": self._linear_regression,
            "distribution_fit": self._fit_distribution,
            "confidence_interval": self._confidence_interval,
            "bootstrap": self._bootstrap_analysis,
            "mann_whitney": self._mann_whitney_test,
            "kruskal": self._kruskal_wallis_test,
            "wilcoxon": self._wilcoxon_test,
            "binomial": self._binomial_test,
            "proportion": self._proportion_test,
            "variance": self._variance_test
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
        """Validate parameters for statistical operations"""
        required_params = {
            "ttest": ["data"],
            "anova": ["groups"],
            "chi_square": ["observed"],
            "correlation": ["x", "y"],
            "normality": ["data"],
            "regression": ["x", "y"],
            "distribution_fit": ["data"],
            "confidence_interval": ["data"],
            "bootstrap": ["data"],
            "mann_whitney": ["group1", "group2"],
            "kruskal": ["groups"],
            "wilcoxon": ["data"],
            "binomial": ["successes", "trials"],
            "proportion": ["successes", "trials"],
            "variance": ["data"]
        }
        
        if operation not in required_params:
            return False
        
        for param in required_params[operation]:
            if param not in kwargs:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        
        return True
    
    def _t_test(
        self,
        data: Union[List, np.ndarray],
        group2: Optional[Union[List, np.ndarray]] = None,
        mu: float = 0,
        alternative: str = "two-sided",
        paired: bool = False
    ) -> ToolResult:
        """Perform t-test"""
        try:
            data = np.array(data)
            
            if group2 is not None:
                # Two-sample t-test
                group2 = np.array(group2)
                if paired:
                    statistic, pvalue = stats.ttest_rel(data, group2, alternative=alternative)
                    test_type = "Paired t-test"
                else:
                    statistic, pvalue = stats.ttest_ind(data, group2, alternative=alternative)
                    test_type = "Independent t-test"
            else:
                # One-sample t-test
                statistic, pvalue = stats.ttest_1samp(data, mu, alternative=alternative)
                test_type = "One-sample t-test"
            
            result = {
                "test_type": test_type,
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "significant": pvalue < 0.05,
                "alternative": alternative,
                "sample1_mean": float(np.mean(data)),
                "sample1_std": float(np.std(data, ddof=1)),
                "sample1_size": len(data)
            }
            
            if group2 is not None:
                result.update({
                    "sample2_mean": float(np.mean(group2)),
                    "sample2_std": float(np.std(group2, ddof=1)),
                    "sample2_size": len(group2),
                    "mean_difference": float(np.mean(data) - np.mean(group2))
                })
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"{test_type}: p-value = {pvalue:.6f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to perform t-test: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _anova(
        self,
        groups: List[Union[List, np.ndarray]],
        post_hoc: bool = False
    ) -> ToolResult:
        """Perform one-way ANOVA"""
        try:
            # Convert groups to arrays
            groups = [np.array(g) for g in groups]
            
            # Perform ANOVA
            statistic, pvalue = stats.f_oneway(*groups)
            
            result = {
                "test_type": "One-way ANOVA",
                "f_statistic": float(statistic),
                "p_value": float(pvalue),
                "significant": pvalue < 0.05,
                "num_groups": len(groups),
                "group_means": [float(np.mean(g)) for g in groups],
                "group_stds": [float(np.std(g, ddof=1)) for g in groups],
                "group_sizes": [len(g) for g in groups]
            }
            
            # Post-hoc analysis (Tukey HSD approximation)
            if post_hoc and len(groups) > 2:
                pairwise = []
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        stat, p = stats.ttest_ind(groups[i], groups[j])
                        pairwise.append({
                            "comparison": f"Group {i+1} vs Group {j+1}",
                            "p_value": float(p),
                            "significant": p < 0.05
                        })
                result["pairwise_comparisons"] = pairwise
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"ANOVA: F={statistic:.4f}, p-value={pvalue:.6f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to perform ANOVA: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _chi_square_test(
        self,
        observed: Union[List, np.ndarray, List[List]],
        expected: Optional[Union[List, np.ndarray]] = None
    ) -> ToolResult:
        """Perform chi-square test"""
        try:
            observed = np.array(observed)
            
            if observed.ndim == 1:
                # Goodness of fit test
                if expected is None:
                    expected = np.ones_like(observed) * np.mean(observed)
                else:
                    expected = np.array(expected)
                
                statistic, pvalue = stats.chisquare(observed, expected)
                test_type = "Chi-square goodness of fit"
            else:
                # Test of independence
                statistic, pvalue, dof, expected = chi2_contingency(observed)
                test_type = "Chi-square test of independence"
            
            result = {
                "test_type": test_type,
                "chi_square_statistic": float(statistic),
                "p_value": float(pvalue),
                "significant": pvalue < 0.05,
                "observed": observed.tolist(),
                "expected": expected.tolist() if isinstance(expected, np.ndarray) else expected
            }
            
            if observed.ndim > 1:
                result["degrees_of_freedom"] = int(dof)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"{test_type}: χ²={statistic:.4f}, p-value={pvalue:.6f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to perform chi-square test: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _correlation_test(
        self,
        x: Union[List, np.ndarray],
        y: Union[List, np.ndarray],
        method: str = "pearson"
    ) -> ToolResult:
        """Test correlation between two variables"""
        try:
            x = np.array(x)
            y = np.array(y)
            
            if method == "pearson":
                correlation, pvalue = pearsonr(x, y)
                test_type = "Pearson correlation"
            elif method == "spearman":
                correlation, pvalue = spearmanr(x, y)
                test_type = "Spearman correlation"
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Unknown correlation method: {method}",
                    execution_time=0.0,
                    errors=["Valid methods: pearson, spearman"]
                )
            
            result = {
                "test_type": test_type,
                "correlation": float(correlation),
                "p_value": float(pvalue),
                "significant": pvalue < 0.05,
                "sample_size": len(x),
                "interpretation": self._interpret_correlation(correlation)
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"{test_type}: r={correlation:.4f}, p-value={pvalue:.6f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to test correlation: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _normality_test(
        self,
        data: Union[List, np.ndarray],
        method: str = "shapiro"
    ) -> ToolResult:
        """Test for normality"""
        try:
            data = np.array(data)
            
            if method == "shapiro":
                statistic, pvalue = stats.shapiro(data)
                test_type = "Shapiro-Wilk test"
            elif method == "kstest":
                statistic, pvalue = stats.kstest(data, 'norm')
                test_type = "Kolmogorov-Smirnov test"
            elif method == "anderson":
                result_anderson = stats.anderson(data)
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "test_type": "Anderson-Darling test",
                        "statistic": float(result_anderson.statistic),
                        "critical_values": result_anderson.critical_values.tolist(),
                        "significance_levels": result_anderson.significance_level.tolist()
                    },
                    message="Anderson-Darling test completed",
                    execution_time=0.0
                )
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    message=f"Unknown method: {method}",
                    execution_time=0.0,
                    errors=["Valid methods: shapiro, kstest, anderson"]
                )
            
            result = {
                "test_type": test_type,
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "is_normal": pvalue > 0.05,
                "sample_size": len(data),
                "mean": float(np.mean(data)),
                "std": float(np.std(data, ddof=1))
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"{test_type}: p-value={pvalue:.6f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to test normality: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _linear_regression(
        self,
        x: Union[List, np.ndarray],
        y: Union[List, np.ndarray],
        alpha: float = 0.05
    ) -> ToolResult:
        """Perform linear regression"""
        try:
            x = np.array(x).reshape(-1, 1) if np.array(x).ndim == 1 else np.array(x)
            y = np.array(y)
            
            # Add intercept column
            x_with_intercept = np.column_stack([np.ones(len(x)), x])
            
            # Calculate coefficients
            coefficients = np.linalg.lstsq(x_with_intercept, y, rcond=None)[0]
            
            # Predictions
            y_pred = x_with_intercept @ coefficients
            
            # Calculate statistics
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            n = len(y)
            k = x.shape[1]
            adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))
            
            # Standard error
            mse = ss_res / (n - k - 1)
            std_error = np.sqrt(mse)
            
            # F-statistic
            f_statistic = (ss_tot - ss_res) / k / (ss_res / (n - k - 1))
            f_pvalue = 1 - stats.f.cdf(f_statistic, k, n - k - 1)
            
            result = {
                "coefficients": {
                    "intercept": float(coefficients[0]),
                    "slopes": coefficients[1:].tolist()
                },
                "r_squared": float(r_squared),
                "adjusted_r_squared": float(adjusted_r_squared),
                "f_statistic": float(f_statistic),
                "f_pvalue": float(f_pvalue),
                "std_error": float(std_error),
                "residuals_mean": float(np.mean(residuals)),
                "residuals_std": float(np.std(residuals)),
                "sample_size": n,
                "significant": f_pvalue < alpha
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"Linear regression: R²={r_squared:.4f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to perform regression: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _fit_distribution(
        self,
        data: Union[List, np.ndarray],
        distribution: str = "norm"
    ) -> ToolResult:
        """Fit a distribution to data"""
        try:
            data = np.array(data)
            
            # Get distribution
            dist = getattr(stats, distribution)
            
            # Fit distribution
            params = dist.fit(data)
            
            # Goodness of fit test
            ks_statistic, ks_pvalue = stats.kstest(data, distribution, args=params)
            
            result = {
                "distribution": distribution,
                "parameters": {
                    "values": list(params),
                    "names": list(dist.shapes.split(', ')) + ['loc', 'scale'] if dist.shapes else ['loc', 'scale']
                },
                "ks_statistic": float(ks_statistic),
                "ks_pvalue": float(ks_pvalue),
                "good_fit": ks_pvalue > 0.05
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"Fitted {distribution} distribution: p-value={ks_pvalue:.6f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to fit distribution: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _confidence_interval(
        self,
        data: Union[List, np.ndarray],
        confidence: float = 0.95
    ) -> ToolResult:
        """Calculate confidence interval for mean"""
        try:
            data = np.array(data)
            
            mean = np.mean(data)
            sem = stats.sem(data)
            ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
            
            result = {
                "mean": float(mean),
                "confidence_level": confidence,
                "confidence_interval": {
                    "lower": float(ci[0]),
                    "upper": float(ci[1])
                },
                "margin_of_error": float(ci[1] - mean),
                "sample_size": len(data),
                "std_error": float(sem)
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"{confidence*100}% CI: [{ci[0]:.4f}, {ci[1]:.4f}]",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to calculate confidence interval: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _bootstrap_analysis(
        self,
        data: Union[List, np.ndarray],
        statistic: str = "mean",
        n_iterations: int = 10000,
        confidence: float = 0.95
    ) -> ToolResult:
        """Perform bootstrap analysis"""
        try:
            data = np.array(data)
            n = len(data)
            
            # Define statistic function
            stat_func = {
                "mean": np.mean,
                "median": np.median,
                "std": np.std
            }.get(statistic, np.mean)
            
            # Bootstrap
            bootstrap_stats = []
            for _ in range(n_iterations):
                sample = np.random.choice(data, size=n, replace=True)
                bootstrap_stats.append(stat_func(sample))
            
            bootstrap_stats = np.array(bootstrap_stats)
            
            # Calculate confidence interval
            alpha = 1 - confidence
            lower = np.percentile(bootstrap_stats, alpha/2 * 100)
            upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
            
            result = {
                "statistic": statistic,
                "original_value": float(stat_func(data)),
                "bootstrap_mean": float(np.mean(bootstrap_stats)),
                "bootstrap_std": float(np.std(bootstrap_stats)),
                "confidence_level": confidence,
                "confidence_interval": {
                    "lower": float(lower),
                    "upper": float(upper)
                },
                "n_iterations": n_iterations
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"Bootstrap {statistic}: {confidence*100}% CI [{lower:.4f}, {upper:.4f}]",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to perform bootstrap: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _mann_whitney_test(
        self,
        group1: Union[List, np.ndarray],
        group2: Union[List, np.ndarray],
        alternative: str = "two-sided"
    ) -> ToolResult:
        """Perform Mann-Whitney U test (non-parametric)"""
        try:
            group1 = np.array(group1)
            group2 = np.array(group2)
            
            statistic, pvalue = stats.mannwhitneyu(
                group1, group2, alternative=alternative
            )
            
            result = {
                "test_type": "Mann-Whitney U test",
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "significant": pvalue < 0.05,
                "alternative": alternative,
                "group1_median": float(np.median(group1)),
                "group2_median": float(np.median(group2)),
                "group1_size": len(group1),
                "group2_size": len(group2)
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"Mann-Whitney U test: p-value={pvalue:.6f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to perform Mann-Whitney test: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _kruskal_wallis_test(
        self,
        groups: List[Union[List, np.ndarray]]
    ) -> ToolResult:
        """Perform Kruskal-Wallis test (non-parametric ANOVA)"""
        try:
            groups = [np.array(g) for g in groups]
            
            statistic, pvalue = stats.kruskal(*groups)
            
            result = {
                "test_type": "Kruskal-Wallis H test",
                "h_statistic": float(statistic),
                "p_value": float(pvalue),
                "significant": pvalue < 0.05,
                "num_groups": len(groups),
                "group_medians": [float(np.median(g)) for g in groups],
                "group_sizes": [len(g) for g in groups]
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"Kruskal-Wallis test: H={statistic:.4f}, p-value={pvalue:.6f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to perform Kruskal-Wallis test: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _wilcoxon_test(
        self,
        data: Union[List, np.ndarray],
        y: Optional[Union[List, np.ndarray]] = None,
        alternative: str = "two-sided"
    ) -> ToolResult:
        """Perform Wilcoxon signed-rank test"""
        try:
            data = np.array(data)
            
            if y is not None:
                y = np.array(y)
                statistic, pvalue = stats.wilcoxon(data, y, alternative=alternative)
            else:
                statistic, pvalue = stats.wilcoxon(data, alternative=alternative)
            
            result = {
                "test_type": "Wilcoxon signed-rank test",
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "significant": pvalue < 0.05,
                "alternative": alternative,
                "sample_size": len(data)
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"Wilcoxon test: p-value={pvalue:.6f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to perform Wilcoxon test: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _binomial_test(
        self,
        successes: int,
        trials: int,
        probability: float = 0.5,
        alternative: str = "two-sided"
    ) -> ToolResult:
        """Perform binomial test"""
        try:
            pvalue = stats.binomtest(
                successes, trials, probability, alternative=alternative
            ).pvalue
            
            result = {
                "test_type": "Binomial test",
                "successes": successes,
                "trials": trials,
                "probability": probability,
                "p_value": float(pvalue),
                "significant": pvalue < 0.05,
                "alternative": alternative,
                "observed_proportion": successes / trials
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"Binomial test: p-value={pvalue:.6f}",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to perform binomial test: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _proportion_test(
        self,
        successes: Union[int, List[int]],
        trials: Union[int, List[int]],
        alternative: str = "two-sided"
    ) -> ToolResult:
        """Test proportions"""
        try:
            from statsmodels.stats.proportion import proportions_ztest
            
            if isinstance(successes, list) and isinstance(trials, list):
                # Two-sample proportion test
                stat, pvalue = proportions_ztest(successes, trials, alternative=alternative)
                test_type = "Two-sample proportion test"
                props = [s/t for s, t in zip(successes, trials)]
            else:
                # One-sample proportion test
                stat, pvalue = proportions_ztest(successes, trials, alternative=alternative)
                test_type = "One-sample proportion test"
                props = [successes / trials]
            
            result = {
                "test_type": test_type,
                "z_statistic": float(stat),
                "p_value": float(pvalue),
                "significant": pvalue < 0.05,
                "proportions": props,
                "alternative": alternative
            }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message=f"{test_type}: p-value={pvalue:.6f}",
                execution_time=0.0
            )
            
        except ImportError:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message="statsmodels package required for proportion tests",
                execution_time=0.0,
                errors=["Install with: pip install statsmodels"]
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to test proportions: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    def _variance_test(
        self,
        data: Union[List, np.ndarray],
        group2: Optional[Union[List, np.ndarray]] = None
    ) -> ToolResult:
        """Test variance (Levene's test or F-test)"""
        try:
            data = np.array(data)
            
            if group2 is not None:
                # Levene's test for equality of variances
                group2 = np.array(group2)
                statistic, pvalue = stats.levene(data, group2)
                test_type = "Levene's test"
                
                result = {
                    "test_type": test_type,
                    "statistic": float(statistic),
                    "p_value": float(pvalue),
                    "equal_variances": pvalue > 0.05,
                    "group1_var": float(np.var(data, ddof=1)),
                    "group2_var": float(np.var(group2, ddof=1))
                }
            else:
                # Single sample variance
                result = {
                    "variance": float(np.var(data, ddof=1)),
                    "std": float(np.std(data, ddof=1)),
                    "sample_size": len(data)
                }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=result,
                message="Variance test completed",
                execution_time=0.0
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                data=None,
                message=f"Failed to test variance: {str(e)}",
                execution_time=0.0,
                errors=[str(e)]
            )
    
    @staticmethod
    def _interpret_correlation(r: float) -> str:
        """Interpret correlation coefficient"""
        abs_r = abs(r)
        if abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        direction = "positive" if r > 0 else "negative"
        return f"{strength} {direction} correlation"