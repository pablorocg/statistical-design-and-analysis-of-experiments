"""
T-Distribution and T-Test Functions
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.figure import Figure
from typing_extensions import Literal


def calculate_t_critical(
    df: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    t_stat: Optional[float] = None,
    mean_diff: Optional[float] = None,
    std_error: Optional[float] = None,
) -> Dict:
    """
    Calculate t-distribution critical values and related statistics.

    Parameters:
    -----------
    df : int
        Degrees of freedom
    alpha : float, optional
        Significance level (default is 0.05)
    alternative : str, optional
        Type of test: 'two-sided', 'less', or 'greater' (default is 'two-sided')
    t_stat : float, optional
        t-statistic value for p-value calculation
    mean_diff : float, optional
        Observed difference between means or from hypothesized value
    std_error : float, optional
        Standard error of the mean difference

    Returns:
    --------
    dict
        Dictionary containing:
        - 'critical_values': Critical value(s) for the t-distribution
        - 'p_value': P-value if t_stat is provided
        - 'confidence_interval': Tuple of (lower, upper) bounds if mean_diff and std_error are provided
        - 'parameters': Dictionary of input parameters

    Examples:
    --------
    >>> # Left-tailed test with α = 0.01 and df = 16
    >>> results = calculate_t_critical(df=16, alpha=0.01, alternative='less')
    >>> print(results['critical_values']['lower'])

    >>> # Two-sided test with t-statistic
    >>> results = calculate_t_critical(df=16, alpha=0.05,
    ...                               alternative='two-sided', t_stat=-2.5)
    >>> print(results['p_value'])
    """
    # Input validation
    if not isinstance(df, int) or df <= 0:
        raise ValueError("Degrees of freedom must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError("Alternative must be 'two-sided', 'less', or 'greater'")

    # Calculate critical values based on alternative hypothesis
    critical_values = {}

    if alternative == "two-sided":
        critical_values["upper"] = stats.t.ppf(1 - alpha / 2, df)
        critical_values["lower"] = -critical_values["upper"]
    elif alternative == "less":
        critical_values["lower"] = stats.t.ppf(alpha, df)
    else:  # greater
        critical_values["upper"] = stats.t.ppf(1 - alpha, df)

    # Prepare results dictionary
    results = {
        "critical_values": critical_values,
        "parameters": {
            "df": df,
            "alpha": alpha,
            "alternative": alternative,
            "confidence_level": (1 - alpha) * 100,
        },
    }

    # Calculate p-value if t-statistic is provided
    if t_stat is not None:
        results["t_stat"] = t_stat
        
        if alternative == "two-sided":
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        elif alternative == "less":
            p_value = stats.t.cdf(t_stat, df)
        else:  # greater
            p_value = 1 - stats.t.cdf(t_stat, df)

        results["p_value"] = p_value
        
        # Add test result
        results["reject_null"] = p_value < alpha

    # Calculate confidence interval if mean difference and standard error are provided
    if mean_diff is not None and std_error is not None:
        results["mean_diff"] = mean_diff
        results["std_error"] = std_error
        
        if alternative == "two-sided":
            margin = abs(critical_values["lower"]) * std_error
            ci_lower = mean_diff - margin
            ci_upper = mean_diff + margin
        elif alternative == "less":
            ci_lower = float("-inf")
            ci_upper = mean_diff + abs(critical_values["lower"]) * std_error
        else:  # greater
            ci_lower = mean_diff - critical_values["upper"] * std_error
            ci_upper = float("inf")
            
        results["confidence_interval"] = (ci_lower, ci_upper)

    return results


def format_t_results(results: Dict, decimals: int = 4) -> str:
    """
    Format the results from calculate_t_critical into a readable string.

    Parameters:
    -----------
    results : dict
        Output dictionary from calculate_t_critical
    decimals : int, optional
        Number of decimal places to round to (default is 4)

    Returns:
    --------
    str
        Formatted string with the results
    """
    output = []
    params = results["parameters"]

    output.append("T-Distribution Analysis Results:")
    output.append("-------------------------------")
    output.append(f"Test Type: {params['alternative']}")
    output.append(f"Degrees of Freedom: {params['df']}")
    output.append(f"Alpha: {params['alpha']}")
    output.append(f"Confidence Level: {params['confidence_level']}%")

    output.append("\nCritical Values:")
    if "upper" in results["critical_values"]:
        output.append(f"  Upper: {results['critical_values']['upper']:.{decimals}f}")
    if "lower" in results["critical_values"]:
        output.append(f"  Lower: {results['critical_values']['lower']:.{decimals}f}")

    if "t_stat" in results:
        output.append(f"\nt-statistic: {results['t_stat']:.{decimals}f}")
        output.append(f"P-value: {results['p_value']:.{decimals}f}")
        
        # Add interpretation
        output.append("\nTest Interpretation:")
        if results["p_value"] < params["alpha"]:
            output.append(f"  Reject the null hypothesis (p={results['p_value']:.{decimals}f} < α={params['alpha']})")
        else:
            output.append(f"  Fail to reject the null hypothesis (p={results['p_value']:.{decimals}f} ≥ α={params['alpha']})")

    if "mean_diff" in results:
        output.append(f"\nMean Difference: {results['mean_diff']:.{decimals}f}")
        output.append(f"Standard Error: {results['std_error']:.{decimals}f}")

    if "confidence_interval" in results:
        ci = results["confidence_interval"]
        output.append(f"\n{params['confidence_level']}% Confidence Interval:")
        if ci[0] == float("-inf"):
            lower = "-∞"
        else:
            lower = f"{ci[0]:.{decimals}f}"
        if ci[1] == float("inf"):
            upper = "∞"
        else:
            upper = f"{ci[1]:.{decimals}f}"
        output.append(f"  ({lower}, {upper})")

    return "\n".join(output)


def visualize_t_distribution(
    results: Dict, show_plot: bool = True, figure_size: Tuple[int, int] = (10, 6)
) -> Optional[Figure]:
    """
    Visualize the t-distribution analysis results.

    Parameters:
    -----------
    results : dict
        Output dictionary from calculate_t_critical
    show_plot : bool, optional
        Whether to display the plot (default is True)
    figure_size : tuple, optional
        Size of the figure (width, height) in inches (default is (10, 6))

    Returns:
    --------
    matplotlib.figure.Figure or None
        Figure object if show_plot is False, None otherwise
    """
    # Extract parameters from results
    params = results["parameters"]
    df = params["df"]
    alpha = params["alpha"]
    alternative = params["alternative"]
    critical_values = results["critical_values"]
    t_stat = results.get("t_stat")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figure_size)

    # Calculate appropriate x-range for the t-distribution
    if "upper" in critical_values and "lower" in critical_values:
        x_min = min(critical_values["lower"] * 1.5, -4)
        x_max = max(critical_values["upper"] * 1.5, 4)
    elif "upper" in critical_values:
        x_min = -4
        x_max = max(critical_values["upper"] * 1.5, 4)
    else:  # "lower" in critical_values
        x_min = min(critical_values["lower"] * 1.5, -4)
        x_max = 4

    # Adjust range if t-statistic is provided
    if t_stat is not None:
        x_min = min(x_min, t_stat * 1.5 if t_stat < 0 else -4)
        x_max = max(x_max, t_stat * 1.5 if t_stat > 0 else 4)

    x = np.linspace(x_min, x_max, 1000)
    
    # Plot the t-distribution PDF
    y = stats.t.pdf(x, df)
    ax.plot(x, y, 'b-', lw=2, label=f't({df})')
    
    # Shade rejection regions based on alternative hypothesis
    if alternative == "two-sided":
        # Lower tail
        x_lower = np.linspace(x_min, critical_values["lower"], 100)
        y_lower = stats.t.pdf(x_lower, df)
        ax.fill_between(
            x_lower, 
            y_lower, 
            alpha=0.3, 
            color='r',
            label=f"Rejection region (α/2={alpha/2:.3f})"
        )
        
        # Upper tail
        x_upper = np.linspace(critical_values["upper"], x_max, 100)
        y_upper = stats.t.pdf(x_upper, df)
        ax.fill_between(x_upper, y_upper, alpha=0.3, color='r')
        
        # Add vertical lines for critical values
        ax.axvline(
            critical_values["lower"],
            color='r',
            linestyle='--',
            label=f"Critical values: {critical_values['lower']:.4f}, {critical_values['upper']:.4f}"
        )
        ax.axvline(critical_values["upper"], color='r', linestyle='--')
        
    elif alternative == "greater":
        # Upper tail only
        x_upper = np.linspace(critical_values["upper"], x_max, 100)
        y_upper = stats.t.pdf(x_upper, df)
        ax.fill_between(
            x_upper,
            y_upper,
            alpha=0.3,
            color='r',
            label=f"Rejection region (α={alpha:.3f})"
        )
        
        # Add vertical line for critical value
        ax.axvline(
            critical_values["upper"],
            color='r',
            linestyle='--',
            label=f"Critical value: {critical_values['upper']:.4f}"
        )
        
    else:  # less
        # Lower tail only
        x_lower = np.linspace(x_min, critical_values["lower"], 100)
        y_lower = stats.t.pdf(x_lower, df)
        ax.fill_between(
            x_lower,
            y_lower,
            alpha=0.3,
            color='r',
            label=f"Rejection region (α={alpha:.3f})"
        )
        
        # Add vertical line for critical value
        ax.axvline(
            critical_values["lower"],
            color='r',
            linestyle='--',
            label=f"Critical value: {critical_values['lower']:.4f}"
        )
    
    # Add t-statistic if provided
    if t_stat is not None:
        ax.axvline(
            t_stat,
            color='g',
            linestyle='-',
            linewidth=1.5,
            label=f"t-statistic: {t_stat:.4f} (p={results['p_value']:.4f})"
        )
        
        # Add confidence interval if provided
        if "confidence_interval" in results:
            ci = results["confidence_interval"]
            if ci[0] > float("-inf") and ci[1] < float("inf"):
                # For mean difference diagram, show in a separate subplot
                mean_diff = results.get("mean_diff")
                std_error = results.get("std_error")
                if mean_diff is not None and std_error is not None:
                    # Create a small inset axis for the mean difference
                    ax_inset = fig.add_axes([0.15, 0.55, 0.3, 0.2])
                    
                    # Plot the mean difference and CI
                    mean_x = [mean_diff]
                    mean_y = [0]
                    ax_inset.errorbar(
                        mean_x, mean_y, 
                        xerr=[[mean_diff - ci[0]], [ci[1] - mean_diff]],
                        fmt='o', color='g', capsize=5, capthick=2,
                        markersize=8
                    )
                    
                    ax_inset.axvline(x=0, linestyle='--', color='r', alpha=0.5)
                    ax_inset.set_yticks([])
                    ax_inset.set_title(f"{params['confidence_level']}% CI", fontsize=10)
                    ax_inset.set_xlabel('Mean Difference', fontsize=8)
                    
                    # Set x-limits with some padding
                    ci_width = ci[1] - ci[0]
                    ax_inset.set_xlim([ci[0] - 0.1 * ci_width, ci[1] + 0.1 * ci_width])
    
    # Add title and labels
    title = f"t-Distribution (df={df})"
    if alternative == "two-sided":
        title += f" - Two-sided Test (α={alpha:.3f})"
    elif alternative == "greater":
        title += f" - Right-tailed Test (α={alpha:.3f})"
    else:  # less
        title += f" - Left-tailed Test (α={alpha:.3f})"
        
    ax.set_title(title)
    ax.set_xlabel('t Value')
    ax.set_ylabel('Probability Density')
    
    # Add legend
    ax.legend(loc='best')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        return None
    else:
        return fig


def analyze_t_distribution(
    df: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    t_stat: Optional[float] = None,
    mean_diff: Optional[float] = None,
    std_error: Optional[float] = None,
    visualize: bool = True,
    figure_size: Tuple[int, int] = (10, 6),
) -> Tuple[Dict, Optional[Figure]]:
    """
    Analyze t-distribution with calculations and optional visualization.
    
    Parameters:
    -----------
    df : int
        Degrees of freedom
    alpha : float, optional
        Significance level (default is 0.05)
    alternative : str, optional
        Type of test: 'two-sided', 'less', or 'greater' (default is 'two-sided')
    t_stat : float, optional
        t-statistic value for p-value calculation
    mean_diff : float, optional
        Observed difference between means or from hypothesized value
    std_error : float, optional
        Standard error of the mean difference
    visualize : bool, optional
        Whether to create a visualization (default is True)
    figure_size : tuple, optional
        Size of the figure (width, height) in inches (default is (10, 6))
        
    Returns:
    --------
    tuple
        (results_dict, figure) where:
        - results_dict: Output dictionary from calculate_t_critical
        - figure: matplotlib Figure object or None if visualize is False
        
    Examples:
    --------
    >>> # One-sample t-test
    >>> results, fig = analyze_t_distribution(df=24, alpha=0.05, t_stat=-2.5, 
    ...                                      mean_diff=-1.2, std_error=0.48)
    >>> print(format_t_results(results))
    >>> plt.figure(fig.number)
    >>> plt.show()
    """
    # Calculate t-distribution results
    results = calculate_t_critical(df, alpha, alternative, t_stat, mean_diff, std_error)
    
    # Create visualization if requested
    fig = None
    if visualize:
        fig = visualize_t_distribution(results, show_plot=False, figure_size=figure_size)
    
    return results, fig