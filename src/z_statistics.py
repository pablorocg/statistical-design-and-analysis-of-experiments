"""
Z-Distribution Analysis Functions - Optimized Version
"""
from typing import Dict, Optional, Tuple, Literal, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

# Set theme once at module level
sns.set_theme(style="whitegrid")

def calculate_z_critical(
    mu_0: float,
    x_bar: float,
    sigma: float,
    n: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> Dict[str, Any]:
    """Calculate z-distribution statistics for hypothesis testing with known variance."""
    # Input validation
    if n <= 0 or sigma <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid input parameters")
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")

    # Calculate standard error and z-statistic
    std_error = sigma / np.sqrt(n)
    z_stat = (x_bar - mu_0) / std_error

    # Calculate critical values based on alternative hypothesis
    critical_values = {}
    if alternative == "two-sided":
        critical_values["upper"] = stats.norm.ppf(1 - alpha / 2)
        critical_values["lower"] = -critical_values["upper"]
    elif alternative == "greater":
        critical_values["upper"] = stats.norm.ppf(1 - alpha)
    else:  # less
        critical_values["lower"] = stats.norm.ppf(alpha)

    # Calculate p-value based on test type
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z_stat)
    else:  # less
        p_value = stats.norm.cdf(z_stat)

    # Calculate confidence interval for the mean
    if alternative == "two-sided":
        margin = abs(critical_values["lower"]) * std_error
        ci = (x_bar - margin, x_bar + margin)
    elif alternative == "greater":
        margin = critical_values["upper"] * std_error
        ci = (x_bar - margin, float("inf"))
    else:  # less
        margin = abs(critical_values["lower"]) * std_error
        ci = (float("-inf"), x_bar + margin)

    # Prepare results dictionary
    results = {
        "z_statistic": z_stat,
        "critical_values": critical_values,
        "p_value": p_value,
        "confidence_interval": ci,
        "parameters": {
            "mu_0": mu_0, 
            "x_bar": x_bar, 
            "sigma": sigma,
            "std_error": std_error, 
            "n": n, 
            "alpha": alpha,
            "alternative": alternative, 
            "confidence_level": (1 - alpha) * 100,
        },
        "reject_null": p_value < alpha
    }

    return results

def format_z_results(results: Dict[str, Any], decimals: int = 4) -> str:
    """Format the results into a readable string."""
    params = results["parameters"]
    crit_vals = results["critical_values"]
    ci = results["confidence_interval"]
    
    # Start building the formatted output
    lines = [
        "Z-Distribution Analysis Results:",
        "-------------------------------",
        f"Test Type: {params['alternative']}",
        "\nTest Parameters:",
        f"  Sample Mean (x̄): {params['x_bar']:.{decimals}f}",
        f"  Null Hypothesis (μ₀): {params['mu_0']:.{decimals}f}",
        f"  Population SD (σ): {params['sigma']:.{decimals}f}",
        f"  Standard Error: {params['std_error']:.{decimals}f}",
        f"  Sample Size (n): {params['n']}",
        f"  Alpha (α): {params['alpha']}",
        f"  Confidence Level: {params['confidence_level']}%",
        "\nTest Statistics:",
        f"  Z-statistic: {results['z_statistic']:.{decimals}f}",
        "\nCritical Values:"
    ]
    
    # Add critical values
    for key, label in [("upper", "Upper"), ("lower", "Lower")]:
        if key in crit_vals:
            lines.append(f"  {label}: {crit_vals[key]:.{decimals}f}")
    
    lines.append(f"\nP-value: {results['p_value']:.{decimals}f}")
    
    # Format confidence interval
    lower = "-∞" if ci[0] == float("-inf") else f"{ci[0]:.{decimals}f}"
    upper = "∞" if ci[1] == float("inf") else f"{ci[1]:.{decimals}f}"
    lines.extend([
        f"\n{params['confidence_level']}% Confidence Interval:",
        f"  ({lower}, {upper})",
        "\nTest Interpretation:"
    ])
    
    # Add test interpretation based on results
    reject = results["reject_null"]
    p_val = results["p_value"]
    mu_0 = params["mu_0"]
    
    if reject:
        conclusion = {
            "two-sided": f"There is evidence that μ ≠ {mu_0}",
            "greater": f"There is evidence that μ > {mu_0}",
            "less": f"There is evidence that μ < {mu_0}"
        }
        lines.extend([
            f"  Reject the null hypothesis (p={p_val:.{decimals}f} < α={params['alpha']})",
            f"  Conclusion: {conclusion[params['alternative']]}"
        ])
    else:
        lines.extend([
            f"  Fail to reject the null hypothesis (p={p_val:.{decimals}f} ≥ α={params['alpha']})",
            f"  Conclusion: Insufficient evidence to conclude that μ ≠ {mu_0}"
        ])
    
    return "\n".join(lines)

def visualize_z_distribution(
    results: Dict[str, Any], 
    show_plot: bool = True, 
    figure_size: Tuple[int, int] = (12, 6)
) -> Optional[Figure]:
    """Visualize the z-distribution analysis results with enhanced clarity."""
    # Extract parameters
    params = results["parameters"]
    mu_0, x_bar = params["mu_0"], params["x_bar"]
    sigma, std_error = params["sigma"], params["std_error"]
    n, alpha = params["n"], params["alpha"]
    alternative = params["alternative"]
    critical_values = results["critical_values"]
    z_stat = results["z_statistic"]
    ci = results["confidence_interval"]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    
    # Get colors from seaborn palette
    colors = sns.color_palette("muted")
    main_color = colors[0]    # Blue
    reject_color = colors[3]  # Red/orange
    stat_color = colors[2]    # Green
    
    # --- First subplot: Z-distribution ---
    # Calculate appropriate x range
    x_min = min(-4, z_stat - 1 if z_stat < 0 else -4)
    x_max = max(4, z_stat + 1 if z_stat > 0 else 4)
    x = np.linspace(x_min, x_max, 1000)
    y = stats.norm.pdf(x)
    
    # Plot standard normal distribution
    ax1.plot(x, y, color=main_color, linewidth=2, label='N(0, 1)')
    
    # Handle visualization based on test type
    if alternative == "two-sided":
        lower, upper = critical_values["lower"], critical_values["upper"]
        
        # Shade rejection regions
        ax1.fill_between(
            np.linspace(x_min, lower, 100),
            stats.norm.pdf(np.linspace(x_min, lower, 100)),
            alpha=0.3, color=reject_color,
            label=f"Rejection region (α/2={alpha/2:.3f})"
        )
        ax1.fill_between(
            np.linspace(upper, x_max, 100),
            stats.norm.pdf(np.linspace(upper, x_max, 100)),
            alpha=0.3, color=reject_color
        )
        
        # Add critical lines
        ax1.axvline(x=lower, color=reject_color, linestyle='--', linewidth=2,
                   label=f"Critical values: {lower:.4f}, {upper:.4f}")
        ax1.axvline(x=upper, color=reject_color, linestyle='--', linewidth=2)
        
    elif alternative == "greater":
        upper = critical_values["upper"]
        
        # Shade upper rejection region
        ax1.fill_between(
            np.linspace(upper, x_max, 100),
            stats.norm.pdf(np.linspace(upper, x_max, 100)),
            alpha=0.3, color=reject_color,
            label=f"Rejection region (α={alpha:.3f})"
        )
        
        # Add critical line
        ax1.axvline(x=upper, color=reject_color, linestyle='--', linewidth=2,
                   label=f"Critical value: {upper:.4f}")
        
    else:  # less
        lower = critical_values["lower"]
        
        # Shade lower rejection region
        ax1.fill_between(
            np.linspace(x_min, lower, 100),
            stats.norm.pdf(np.linspace(x_min, lower, 100)),
            alpha=0.3, color=reject_color,
            label=f"Rejection region (α={alpha:.3f})"
        )
        
        # Add critical line
        ax1.axvline(x=lower, color=reject_color, linestyle='--', linewidth=2,
                   label=f"Critical value: {lower:.4f}")
    
    # Add z-statistic line
    ax1.axvline(x=z_stat, color=stat_color, linestyle='-', linewidth=2.5,
               label=f"Z-statistic: {z_stat:.4f} (p={results['p_value']:.4f})")
    
    # Format first subplot
    title_suffix = {
        "two-sided": f" - Two-sided Test (α={alpha:.3f})",
        "greater": f" - Right-tailed Test (α={alpha:.3f})",
        "less": f" - Left-tailed Test (α={alpha:.3f})"
    }
    ax1.set_title(f"Standard Normal Distribution{title_suffix[alternative]}", fontsize=12)
    ax1.set_xlabel('Z Value', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.legend(loc='best', frameon=True, framealpha=0.7)
    
    # --- Second subplot: Mean with confidence interval ---
    # Create error bar for the sample mean and CI
    xerr_left = x_bar - ci[0] if ci[0] != float("-inf") else 0
    xerr_right = ci[1] - x_bar if ci[1] != float("inf") else 0
    
    ax2.errorbar(
        [x_bar], [1], 
        xerr=[[xerr_left], [xerr_right]],
        fmt='o', color=stat_color, capsize=5, markersize=8,
        label=f"Sample Mean: {x_bar:.4f}"
    )
    
    # Add null hypothesis reference line
    ax2.axvline(x=mu_0, color=reject_color, linestyle='--', linewidth=1.5,
               label=f"Null Hypothesis: μ₀ = {mu_0:.4f}")
    
    # Determine x range for sampling distributions
    if ci[0] != float("-inf") and ci[1] != float("inf"):
        x_range = np.linspace(min(mu_0, ci[0]) - 3*std_error, max(mu_0, ci[1]) + 3*std_error, 1000)
    else:
        x_range = np.linspace(mu_0 - 4*std_error, mu_0 + 4*std_error, 1000)
    
    # Plot sampling distributions
    # Null hypothesis distribution
    y_null = stats.norm.pdf(x_range, loc=mu_0, scale=std_error)
    ax2.plot(x_range, y_null/5 + 2, color=reject_color, linewidth=1.5, 
            alpha=0.7, label="Sampling Dist. under H₀")
    
    # Sample mean distribution
    y_sample = stats.norm.pdf(x_range, loc=x_bar, scale=std_error)
    ax2.plot(x_range, y_sample/5 + 3, color=stat_color, linewidth=1.5,
            alpha=0.7, label="Sampling Dist. based on x̄")
    
    # Format confidence interval text
    if ci[0] == float("-inf"):
        ci_text = f"{params['confidence_level']}% CI: (-∞, {ci[1]:.4f})"
    elif ci[1] == float("inf"):
        ci_text = f"{params['confidence_level']}% CI: ({ci[0]:.4f}, ∞)"
    else:
        ci_text = f"{params['confidence_level']}% CI: ({ci[0]:.4f}, {ci[1]:.4f})"
    
    # Add CI text box
    ax2.text(0.5, 0.05, ci_text, horizontalalignment='center',
            transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
    
    # Format second subplot
    ax2.set_title(f"Sample Mean and {params['confidence_level']}% Confidence Interval\n(n={n}, σ={sigma:.4f})", 
                 fontsize=12)
    ax2.set_xlabel('Mean Value', fontsize=11)
    ax2.set_yticks([])
    ax2.legend(loc='upper center', frameon=True, framealpha=0.7)
    
    # Final formatting
    sns.despine(ax=ax1, left=False, bottom=False)
    sns.despine(ax=ax2, left=True, bottom=False)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        return None
    return fig

def analyze_z_distribution(
    mu_0: float,
    x_bar: float,
    sigma: float,
    n: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    visualize: bool = True,
    show_plot: bool = True,
    figure_size: Tuple[int, int] = (12, 6),
) -> Tuple[Dict[str, Any], Optional[Figure]]:
    """Analyze z-distribution with calculations and optional visualization."""
    results = calculate_z_critical(mu_0, x_bar, sigma, n, alpha, alternative)
    fig = None
    if visualize:
        fig = visualize_z_distribution(results, show_plot=show_plot, figure_size=figure_size)
    return results, fig