"""
Z-Distribution Analysis Functions
"""

from typing import Dict, Optional, Tuple, Literal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

# Set seaborn style
sns.set_theme(style="whitegrid")


def calculate_z_critical(
    mu_0: float,
    x_bar: float,
    sigma: float,
    n: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> Dict:
    """Calculate z-distribution statistics for hypothesis testing with known variance."""
    # Input validation
    if n <= 0 or sigma <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid input parameters")
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")

    # Calculate z-statistic
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

    # Calculate p-value
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

    # Prepare results
    results = {
        "z_statistic": z_stat,
        "critical_values": critical_values,
        "p_value": p_value,
        "confidence_interval": ci,
        "parameters": {
            "mu_0": mu_0, "x_bar": x_bar, "sigma": sigma,
            "std_error": std_error, "n": n, "alpha": alpha,
            "alternative": alternative, "confidence_level": (1 - alpha) * 100,
        },
        "reject_null": p_value < alpha
    }

    return results


def format_z_results(results: Dict, decimals: int = 4) -> str:
    """Format the results into a readable string."""
    params = results["parameters"]
    crit_vals = results["critical_values"]
    ci = results["confidence_interval"]
    
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
    if "upper" in crit_vals:
        lines.append(f"  Upper: {crit_vals['upper']:.{decimals}f}")
    if "lower" in crit_vals:
        lines.append(f"  Lower: {crit_vals['lower']:.{decimals}f}")
    
    lines.append(f"\nP-value: {results['p_value']:.{decimals}f}")
    
    # Add confidence interval
    lower = "-∞" if ci[0] == float("-inf") else f"{ci[0]:.{decimals}f}"
    upper = "∞" if ci[1] == float("inf") else f"{ci[1]:.{decimals}f}"
    lines.extend([
        f"\n{params['confidence_level']}% Confidence Interval:",
        f"  ({lower}, {upper})",
        "\nTest Interpretation:"
    ])
    
    # Add test interpretation
    if results["reject_null"]:
        conclusion = {
            "two-sided": f"There is evidence that μ ≠ {params['mu_0']}",
            "greater": f"There is evidence that μ > {params['mu_0']}",
            "less": f"There is evidence that μ < {params['mu_0']}"
        }
        lines.extend([
            f"  Reject the null hypothesis (p={results['p_value']:.{decimals}f} < α={params['alpha']})",
            f"  Conclusion: {conclusion[params['alternative']]}"
        ])
    else:
        lines.extend([
            f"  Fail to reject the null hypothesis (p={results['p_value']:.{decimals}f} ≥ α={params['alpha']})",
            f"  Conclusion: Insufficient evidence to conclude that μ ≠ {params['mu_0']}"
        ])
    
    return "\n".join(lines)


def visualize_z_distribution(
    results: Dict, show_plot: bool = True, figure_size: Tuple[int, int] = (12, 6)
) -> Optional[Figure]:
    """Visualize the z-distribution analysis results using Seaborn."""
    # Extract parameters
    params = results["parameters"]
    mu_0, x_bar = params["mu_0"], params["x_bar"]
    sigma, std_error = params["sigma"], params["std_error"]
    n, alpha = params["n"], params["alpha"]
    alternative = params["alternative"]
    critical_values = results["critical_values"]
    z_stat = results["z_statistic"]
    ci = results["confidence_interval"]
    
    # Create figure with Seaborn styling
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    
    # Seaborn color palette
    colors = sns.color_palette("muted")
    main_color = colors[0]    # Blue
    reject_color = colors[3]  # Red/orange
    stat_color = colors[2]    # Green
    
    # --- First subplot: Z-distribution ---
    x_min = min(-4, z_stat - 1 if z_stat < 0 else -4)
    x_max = max(4, z_stat + 1 if z_stat > 0 else 4)
    x = np.linspace(x_min, x_max, 1000)
    y = stats.norm.pdf(x)
    
    # Plot standard normal distribution
    sns.lineplot(x=x, y=y, color=main_color, ax=ax1, linewidth=2, label='N(0, 1)')
    
    # Shade rejection regions and add critical lines
    if alternative == "two-sided":
        lower, upper = critical_values["lower"], critical_values["upper"]
        
        # Lower tail
        x_lower = np.linspace(x_min, lower, 100)
        y_lower = stats.norm.pdf(x_lower)
        ax1.fill_between(x_lower, y_lower, alpha=0.3, color=reject_color, 
                        label=f"Rejection region (α/2={alpha/2:.3f})")
        
        # Upper tail
        x_upper = np.linspace(upper, x_max, 100)
        y_upper = stats.norm.pdf(x_upper)
        ax1.fill_between(x_upper, y_upper, alpha=0.3, color=reject_color)
        
        # Critical lines
        sns.lineplot(x=[lower, lower], y=[0, max(y)*1.1], color=reject_color, linestyle="--", 
                    ax=ax1, label=f"Critical values: {lower:.4f}, {upper:.4f}")
        sns.lineplot(x=[upper, upper], y=[0, max(y)*1.1], color=reject_color, linestyle="--", ax=ax1)
        
    elif alternative == "greater":
        upper = critical_values["upper"]
        
        # Upper tail
        x_upper = np.linspace(upper, x_max, 100)
        y_upper = stats.norm.pdf(x_upper)
        ax1.fill_between(x_upper, y_upper, alpha=0.3, color=reject_color,
                        label=f"Rejection region (α={alpha:.3f})")
        
        # Critical line
        sns.lineplot(x=[upper, upper], y=[0, max(y)*1.1], color=reject_color, linestyle="--", 
                    ax=ax1, label=f"Critical value: {upper:.4f}")
        
    else:  # less
        lower = critical_values["lower"]
        
        # Lower tail
        x_lower = np.linspace(x_min, lower, 100)
        y_lower = stats.norm.pdf(x_lower)
        ax1.fill_between(x_lower, y_lower, alpha=0.3, color=reject_color,
                        label=f"Rejection region (α={alpha:.3f})")
        
        # Critical line
        sns.lineplot(x=[lower, lower], y=[0, max(y)*1.1], color=reject_color, linestyle="--", 
                    ax=ax1, label=f"Critical value: {lower:.4f}")
    
    # Add z-statistic
    sns.lineplot(x=[z_stat, z_stat], y=[0, max(y)*1.1], color=stat_color, linestyle="-", 
                linewidth=1.5, ax=ax1, label=f"Z-statistic: {z_stat:.4f} (p={results['p_value']:.4f})")
    
    # Set title and labels with Seaborn styling
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
    # Create a horizontal error bar for the sample mean and CI
    ax2.errorbar(
        [x_bar], [1], 
        xerr=[[x_bar - ci[0]] if ci[0] != float("-inf") else [0], 
              [ci[1] - x_bar] if ci[1] != float("inf") else [0]],
        fmt='o', color=stat_color, capsize=5, markersize=8,
        label=f"Sample Mean: {x_bar:.4f}"
    )
    
    # Add null hypothesis value
    sns.lineplot(x=[mu_0, mu_0], y=[0, 2], color=reject_color, linestyle="--",
                linewidth=1.5, ax=ax2, label=f"Null Hypothesis: μ₀ = {mu_0:.4f}")
    
    # Plot sampling distributions
    if ci[0] != float("-inf") and ci[1] != float("inf"):
        x_range = np.linspace(min(mu_0, ci[0]) - 3*std_error, max(mu_0, ci[1]) + 3*std_error, 1000)
    else:
        x_range = np.linspace(mu_0 - 4*std_error, mu_0 + 4*std_error, 1000)
    
    # Sampling distribution under null hypothesis
    y_null = stats.norm.pdf(x_range, loc=mu_0, scale=std_error)
    sns.lineplot(x=x_range, y=y_null/5 + 2, color=reject_color, linewidth=1.5, 
                alpha=0.7, ax=ax2, label="Sampling Dist. under H₀")
    
    # Sampling distribution based on sample mean
    y_sample = stats.norm.pdf(x_range, loc=x_bar, scale=std_error)
    sns.lineplot(x=x_range, y=y_sample/5 + 3, color=stat_color, linewidth=1.5, 
                alpha=0.7, ax=ax2, label="Sampling Dist. based on x̄")
    
    # Confidence interval text annotation
    if ci[0] == float("-inf"):
        ci_text = f"{params['confidence_level']}% CI: (-∞, {ci[1]:.4f})"
    elif ci[1] == float("inf"):
        ci_text = f"{params['confidence_level']}% CI: ({ci[0]:.4f}, ∞)"
    else:
        ci_text = f"{params['confidence_level']}% CI: ({ci[0]:.4f}, {ci[1]:.4f})"
    
    # Add text box with CI
    ax2.text(0.5, 0.05, ci_text, horizontalalignment='center',
            transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
    
    # Set title and styling
    ax2.set_title(f"Sample Mean and {params['confidence_level']}% Confidence Interval\n(n={n}, σ={sigma:.4f})", 
                 fontsize=12)
    ax2.set_xlabel('Mean Value', fontsize=11)
    ax2.set_yticks([])
    ax2.legend(loc='upper center', frameon=True, framealpha=0.7)
    
    # Add despine for cleaner look
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
    figure_size: Tuple[int, int] = (12, 6),
) -> Tuple[Dict, Optional[Figure]]:
    """Analyze z-distribution with calculations and optional visualization."""
    results = calculate_z_critical(mu_0, x_bar, sigma, n, alpha, alternative)
    fig = visualize_z_distribution(results, show_plot=visualize, figure_size=figure_size) if visualize else None
    return results, fig