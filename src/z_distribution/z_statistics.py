"""
Z-Distribution Analysis Functions
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import stats
from typing_extensions import Literal


def calculate_z_critical(
    mu_0: float,
    x_bar: float,
    sigma: float,
    n: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> Dict:
    """
    Calculate z-distribution statistics for hypothesis testing with known variance.

    Parameters:
    -----------
    mu_0 : float
        Hypothesized population mean (null hypothesis value)
    x_bar : float
        Sample mean
    sigma : float
        Known population standard deviation
    n : int
        Sample size
    alpha : float, optional
        Significance level (default is 0.05)
    alternative : str, optional
        Type of test: 'two-sided', 'greater', or 'less' (default is 'two-sided')

    Returns:
    --------
    dict
        Dictionary containing:
        - 'z_statistic': Calculated z-statistic
        - 'critical_values': Critical value(s) for the z-distribution
        - 'p_value': P-value for the test
        - 'confidence_interval': Confidence interval for the mean
        - 'parameters': Dictionary of input parameters
    """
    # Input validation
    if n <= 0:
        raise ValueError("Sample size must be positive")
    if sigma <= 0:
        raise ValueError("Standard deviation must be positive")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")

    # Calculate z-statistic
    z_stat = (x_bar - mu_0) / (sigma / np.sqrt(n))

    # Calculate critical values based on alternative hypothesis
    critical_values = {}

    if alternative == "two-sided":
        critical_values["upper"] = stats.norm.ppf(1 - alpha / 2)
        critical_values["lower"] = -critical_values["upper"]
    elif alternative == "greater":
        critical_values["upper"] = stats.norm.ppf(1 - alpha)
    else:  # less
        critical_values["lower"] = stats.norm.ppf(alpha)

    # Calculate p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:  # less
        p_value = stats.norm.cdf(z_stat)

    # Calculate confidence interval for the mean
    margin = 0
    if alternative == "two-sided":
        margin = abs(critical_values["lower"]) * (sigma / np.sqrt(n))
        ci_lower = x_bar - margin
        ci_upper = x_bar + margin
    elif alternative == "greater":
        margin = critical_values["upper"] * (sigma / np.sqrt(n))
        ci_lower = x_bar - margin
        ci_upper = float("inf")
    else:  # less
        margin = abs(critical_values["lower"]) * (sigma / np.sqrt(n))
        ci_lower = float("-inf")
        ci_upper = x_bar + margin

    # Prepare results
    results = {
        "z_statistic": z_stat,
        "critical_values": critical_values,
        "p_value": p_value,
        "confidence_interval": (ci_lower, ci_upper),
        "parameters": {
            "mu_0": mu_0,
            "x_bar": x_bar,
            "sigma": sigma,
            "std_error": sigma / np.sqrt(n),
            "n": n,
            "alpha": alpha,
            "alternative": alternative,
            "confidence_level": (1 - alpha) * 100,
        },
        "reject_null": p_value < alpha
    }

    return results


def format_z_results(results: Dict, decimals: int = 4) -> str:
    """
    Format the results from calculate_z_critical into a readable string.

    Parameters:
    -----------
    results : dict
        Output dictionary from calculate_z_critical
    decimals : int, optional
        Number of decimal places to round to (default is 4)

    Returns:
    --------
    str
        Formatted string with the results
    """
    output = []
    params = results["parameters"]

    output.append("Z-Distribution Analysis Results:")
    output.append("-------------------------------")
    output.append(f"Test Type: {params['alternative']}")

    output.append("\nTest Parameters:")
    output.append(f"  Sample Mean (x̄): {params['x_bar']:.{decimals}f}")
    output.append(f"  Null Hypothesis (μ₀): {params['mu_0']:.{decimals}f}")
    output.append(f"  Population SD (σ): {params['sigma']:.{decimals}f}")
    output.append(f"  Standard Error: {params['std_error']:.{decimals}f}")
    output.append(f"  Sample Size (n): {params['n']}")
    output.append(f"  Alpha (α): {params['alpha']}")
    output.append(f"  Confidence Level: {params['confidence_level']}%")

    output.append("\nTest Statistics:")
    output.append(f"  Z-statistic: {results['z_statistic']:.{decimals}f}")

    output.append("\nCritical Values:")
    if "upper" in results["critical_values"]:
        output.append(f"  Upper: {results['critical_values']['upper']:.{decimals}f}")
    if "lower" in results["critical_values"]:
        output.append(f"  Lower: {results['critical_values']['lower']:.{decimals}f}")

    output.append(f"\nP-value: {results['p_value']:.{decimals}f}")

    # Add confidence interval
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

    # Add interpretation
    output.append("\nTest Interpretation:")
    if results["reject_null"]:
        output.append(f"  Reject the null hypothesis (p={results['p_value']:.{decimals}f} < α={params['alpha']})")
        
        # Add specific interpretation based on alternative
        if params["alternative"] == "two-sided":
            output.append(f"  Conclusion: There is evidence that μ ≠ {params['mu_0']}")
        elif params["alternative"] == "greater":
            output.append(f"  Conclusion: There is evidence that μ > {params['mu_0']}")
        else:  # less
            output.append(f"  Conclusion: There is evidence that μ < {params['mu_0']}")
    else:
        output.append(f"  Fail to reject the null hypothesis (p={results['p_value']:.{decimals}f} ≥ α={params['alpha']})")
        output.append(f"  Conclusion: Insufficient evidence to conclude that μ ≠ {params['mu_0']}")

    return "\n".join(output)


def visualize_z_distribution(
    results: Dict, show_plot: bool = True, figure_size: Tuple[int, int] = (12, 6)
) -> Optional[Figure]:
    """
    Visualize the z-distribution analysis results.

    Parameters:
    -----------
    results : dict
        Output dictionary from calculate_z_critical
    show_plot : bool, optional
        Whether to display the plot (default is True)
    figure_size : tuple, optional
        Size of the figure (width, height) in inches (default is (12, 6))

    Returns:
    --------
    matplotlib.figure.Figure or None
        Figure object if show_plot is False, None otherwise
    """
    # Extract parameters from results
    params = results["parameters"]
    mu_0 = params["mu_0"]
    x_bar = params["x_bar"]
    sigma = params["sigma"]
    std_error = params["std_error"]
    n = params["n"]
    alpha = params["alpha"]
    alternative = params["alternative"]
    critical_values = results["critical_values"]
    z_stat = results["z_statistic"]
    ci = results["confidence_interval"]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    
    # --- First subplot: Z-distribution ---
    # Calculate appropriate x-range for the z-distribution
    x_min = min(-4, z_stat - 1 if z_stat < 0 else -4)
    x_max = max(4, z_stat + 1 if z_stat > 0 else 4)
    
    x = np.linspace(x_min, x_max, 1000)
    y = stats.norm.pdf(x)
    
    # Plot the standard normal distribution
    ax1.plot(x, y, 'b-', lw=2, label='N(0, 1)')
    
    # Shade rejection regions based on alternative hypothesis
    if alternative == "two-sided":
        # Lower tail
        x_lower = np.linspace(x_min, critical_values["lower"], 100)
        y_lower = stats.norm.pdf(x_lower)
        ax1.fill_between(
            x_lower, 
            y_lower, 
            alpha=0.3, 
            color='r',
            label=f"Rejection region (α/2={alpha/2:.3f})"
        )
        
        # Upper tail
        x_upper = np.linspace(critical_values["upper"], x_max, 100)
        y_upper = stats.norm.pdf(x_upper)
        ax1.fill_between(x_upper, y_upper, alpha=0.3, color='r')
        
        # Add vertical lines for critical values
        ax1.axvline(
            critical_values["lower"],
            color='r',
            linestyle='--',
            label=f"Critical values: {critical_values['lower']:.4f}, {critical_values['upper']:.4f}"
        )
        ax1.axvline(critical_values["upper"], color='r', linestyle='--')
        
    elif alternative == "greater":
        # Upper tail only
        x_upper = np.linspace(critical_values["upper"], x_max, 100)
        y_upper = stats.norm.pdf(x_upper)
        ax1.fill_between(
            x_upper,
            y_upper,
            alpha=0.3,
            color='r',
            label=f"Rejection region (α={alpha:.3f})"
        )
        
        # Add vertical line for critical value
        ax1.axvline(
            critical_values["upper"],
            color='r',
            linestyle='--',
            label=f"Critical value: {critical_values['upper']:.4f}"
        )
        
    else:  # less
        # Lower tail only
        x_lower = np.linspace(x_min, critical_values["lower"], 100)
        y_lower = stats.norm.pdf(x_lower)
        ax1.fill_between(
            x_lower,
            y_lower,
            alpha=0.3,
            color='r',
            label=f"Rejection region (α={alpha:.3f})"
        )
        
        # Add vertical line for critical value
        ax1.axvline(
            critical_values["lower"],
            color='r',
            linestyle='--',
            label=f"Critical value: {critical_values['lower']:.4f}"
        )
    
    # Add z-statistic
    ax1.axvline(
        z_stat,
        color='g',
        linestyle='-',
        linewidth=1.5,
        label=f"Z-statistic: {z_stat:.4f} (p={results['p_value']:.4f})"
    )
    
    # Add title and labels
    title = "Standard Normal Distribution"
    if alternative == "two-sided":
        title += f" - Two-sided Test (α={alpha:.3f})"
    elif alternative == "greater":
        title += f" - Right-tailed Test (α={alpha:.3f})"
    else:  # less
        title += f" - Left-tailed Test (α={alpha:.3f})"
        
    ax1.set_title(title)
    ax1.set_xlabel('Z Value')
    ax1.set_ylabel('Probability Density')
    
    # Add legend
    ax1.legend(loc='best')
    
    # --- Second subplot: Mean with confidence interval ---
    
    # Create a horizontal error bar for the sample mean and CI
    ax2.errorbar(
        [x_bar], [1], 
        xerr=[[x_bar - ci[0]] if ci[0] != float("-inf") else [0], 
              [ci[1] - x_bar] if ci[1] != float("inf") else [0]],
        fmt='o', 
        color='g',
        capsize=5,
        capthick=2,
        markersize=8,
        label=f"Sample Mean: {x_bar:.4f}"
    )
    
    # Add null hypothesis value
    ax2.axvline(
        mu_0,
        color='r',
        linestyle='--',
        linewidth=1.5,
        label=f"Null Hypothesis: μ₀ = {mu_0:.4f}"
    )
    
    # Plot the sampling distribution around the null and sample mean
    if ci[0] != float("-inf") and ci[1] != float("inf"):
        x_range = np.linspace(
            min(mu_0, ci[0]) - 3*std_error, 
            max(mu_0, ci[1]) + 3*std_error, 
            1000
        )
    else:
        # If CI is one-sided, create a reasonable range
        x_range = np.linspace(
            mu_0 - 4*std_error, 
            mu_0 + 4*std_error, 
            1000
        )
    
    # Sampling distribution under null hypothesis
    y_null = stats.norm.pdf(x_range, loc=mu_0, scale=std_error)
    ax2.plot(
        x_range, 
        y_null/5 + 2,  # Scaled and shifted for visualization
        'r-', 
        lw=1.5, 
        alpha=0.7, 
        label="Sampling Dist. under H₀"
    )
    
    # Sampling distribution based on sample mean
    y_sample = stats.norm.pdf(x_range, loc=x_bar, scale=std_error)
    ax2.plot(
        x_range, 
        y_sample/5 + 3,  # Scaled and shifted for visualization
        'g-', 
        lw=1.5, 
        alpha=0.7, 
        label="Sampling Dist. based on x̄"
    )
    
    # Confidence interval text annotation
    if ci[0] == float("-inf"):
        ci_text = f"{params['confidence_level']}% CI: (-∞, {ci[1]:.4f})"
    elif ci[1] == float("inf"):
        ci_text = f"{params['confidence_level']}% CI: ({ci[0]:.4f}, ∞)"
    else:
        ci_text = f"{params['confidence_level']}% CI: ({ci[0]:.4f}, {ci[1]:.4f})"
    
    # Add text annotation for CI
    ax2.text(
        0.5, 0.05, 
        ci_text, 
        horizontalalignment='center',
        verticalalignment='center', 
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )
    
    # Add title and labels
    ax2.set_title(f"Sample Mean and {params['confidence_level']}% Confidence Interval\n(n={n}, σ={sigma:.4f})")
    ax2.set_xlabel('Mean Value')
    ax2.set_yticks([])  # Hide y-ticks since they're meaningless here
    
    # Add legend
    ax2.legend(loc='upper center')
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        return None
    else:
        return fig


def analyze_z_distribution(
    mu_0: float,
    x_bar: float,
    sigma: float,
    n: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    visualize: bool = True,
    figure_size: Tuple[int, int] = (12, 6),
) -> Tuple[Dict, Optional[Figure]]:
    """
    Analyze z-distribution with calculations and optional visualization.
    
    Parameters:
    -----------
    mu_0 : float
        Hypothesized population mean (null hypothesis value)
    x_bar : float
        Sample mean
    sigma : float
        Known population standard deviation
    n : int
        Sample size
    alpha : float, optional
        Significance level (default is 0.05)
    alternative : str, optional
        Type of test: 'two-sided', 'greater', or 'less' (default is 'two-sided')
    visualize : bool, optional
        Whether to create a visualization (default is True)
    figure_size : tuple, optional
        Size of the figure (width, height) in inches (default is (12, 6))
        
    Returns:
    --------
    tuple
        (results_dict, figure) where:
        - results_dict: Output dictionary from calculate_z_critical
        - figure: matplotlib Figure object or None if visualize is False
        
    Examples:
    --------
    >>> # One-sample z-test
    >>> results, fig = analyze_z_distribution(
    ...     mu_0=3.0, x_bar=3.1, sigma=0.2, n=20, 
    ...     alpha=0.05, alternative="greater"
    ... )
    >>> print(format_z_results(results))
    >>> plt.figure(fig.number)
    >>> plt.show()
    """
    # Calculate z-distribution results
    results = calculate_z_critical(mu_0, x_bar, sigma, n, alpha, alternative)
    
    # Create visualization if requested
    fig = None
    if visualize:
        fig = visualize_z_distribution(results, show_plot=False, figure_size=figure_size)
    
    return results, fig