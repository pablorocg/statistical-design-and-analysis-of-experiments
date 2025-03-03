"""
T-Distribution and T-Test Functions - Optimized Version
"""
from typing import Dict, Optional, Tuple, Literal, Any, Union
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# Set theme once at module level
sns.set_theme(style="whitegrid")

def calculate_t_critical(
    df: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    t_stat: Optional[float] = None,
    mean_diff: Optional[float] = None,
    std_error: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate t-distribution critical values and related statistics."""
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
        
        # Calculate p-value based on test type
        if alternative == "two-sided":
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        elif alternative == "less":
            p_value = stats.t.cdf(t_stat, df)
        else:  # greater
            p_value = 1 - stats.t.cdf(t_stat, df)

        results["p_value"] = p_value
        results["reject_null"] = p_value < alpha

    # Calculate confidence interval if mean difference and standard error are provided
    if mean_diff is not None and std_error is not None:
        results["mean_diff"] = mean_diff
        results["std_error"] = std_error
        
        # Calculate CI based on test type
        if alternative == "two-sided":
            margin = abs(critical_values["lower"]) * std_error
            ci = (mean_diff - margin, mean_diff + margin)
        elif alternative == "less":
            ci = (float("-inf"), mean_diff + abs(critical_values["lower"]) * std_error)
        else:  # greater
            ci = (mean_diff - critical_values["upper"] * std_error, float("inf"))
            
        results["confidence_interval"] = ci

    return results

def format_t_results(results: Dict[str, Any], decimals: int = 4) -> str:
    """Format the results into a readable string."""
    params = results["parameters"]
    crit_vals = results["critical_values"]
    
    # Create formatted string with f-strings for readability
    lines = [
        "T-Distribution Analysis Results:",
        "-------------------------------",
        f"Test Type: {params['alternative']}",
        f"Degrees of Freedom: {params['df']}",
        f"Alpha: {params['alpha']}",
        f"Confidence Level: {params['confidence_level']}%",
        "\nCritical Values:"
    ]
    
    # Add critical values
    for key, label in [("upper", "Upper"), ("lower", "Lower")]:
        if key in crit_vals:
            lines.append(f"  {label}: {crit_vals[key]:.{decimals}f}")

    # Add test results if available
    if "t_stat" in results:
        p_value = results["p_value"]
        reject = p_value < params["alpha"]
        compare = "<" if reject else "≥"
        
        lines.extend([
            f"\nt-statistic: {results['t_stat']:.{decimals}f}",
            f"P-value: {p_value:.{decimals}f}",
            "\nTest Interpretation:",
            f"  {'Reject' if reject else 'Fail to reject'} the null hypothesis (p={p_value:.{decimals}f} {compare} α={params['alpha']})"
        ])

    # Add mean difference and standard error if provided
    if "mean_diff" in results:
        lines.extend([
            f"\nMean Difference: {results['mean_diff']:.{decimals}f}",
            f"Standard Error: {results['std_error']:.{decimals}f}"
        ])

    # Add confidence interval if provided
    if "confidence_interval" in results:
        ci = results["confidence_interval"]
        lower = "-∞" if ci[0] == float("-inf") else f"{ci[0]:.{decimals}f}"
        upper = "∞" if ci[1] == float("inf") else f"{ci[1]:.{decimals}f}"
        lines.extend([
            f"\n{params['confidence_level']}% Confidence Interval:",
            f"  ({lower}, {upper})"
        ])

    return "\n".join(lines)

def visualize_t_distribution(
    results: Dict[str, Any], 
    show_plot: bool = True, 
    figure_size: Tuple[int, int] = (10, 6)
) -> Optional[Figure]:
    """Visualize the t-distribution analysis results with enhanced clarity."""
    # Extract parameters
    params = results["parameters"]
    df, alpha = params["df"], params["alpha"]
    alternative = params["alternative"]
    critical_values = results["critical_values"]
    t_stat = results.get("t_stat")

    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Calculate appropriate x-range
    x_min = -4
    x_max = 4
    
    # Adjust range based on critical values and t-statistic
    if "lower" in critical_values:
        x_min = min(x_min, critical_values["lower"] * 1.5)
    if "upper" in critical_values:
        x_max = max(x_max, critical_values["upper"] * 1.5)
    if t_stat is not None:
        x_min = min(x_min, t_stat * 1.5 if t_stat < 0 else x_min)
        x_max = max(x_max, t_stat * 1.5 if t_stat > 0 else x_max)

    # Create x and y values for the distribution curve
    x = np.linspace(x_min, x_max, 1000)
    y = stats.t.pdf(x, df)
    
    # Plot the main distribution curve
    ax.plot(x, y, linewidth=2, label=f't({df})')
    
    # Get colors for visualization
    colors = sns.color_palette("muted")
    reject_color = colors[3]  # Red/orange for rejection regions
    stat_color = colors[2]    # Green for statistic
    
    # Visualization based on test type
    if alternative == "two-sided":
        lower, upper = critical_values["lower"], critical_values["upper"]
        
        # Shade rejection regions
        ax.fill_between(
            np.linspace(x_min, lower, 100), 
            stats.t.pdf(np.linspace(x_min, lower, 100), df),
            alpha=0.3, color=reject_color, 
            label=f"Rejection region (α/2={alpha/2:.3f})"
        )
        ax.fill_between(
            np.linspace(upper, x_max, 100),
            stats.t.pdf(np.linspace(upper, x_max, 100), df),
            alpha=0.3, color=reject_color
        )
        
        # Add critical lines
        ax.axvline(x=lower, color=reject_color, linestyle='--', linewidth=2, 
                  label=f"Critical values: {lower:.4f}, {upper:.4f}")
        ax.axvline(x=upper, color=reject_color, linestyle='--', linewidth=2)
        
    elif alternative == "greater":
        upper = critical_values["upper"]
        
        # Shade upper rejection region
        ax.fill_between(
            np.linspace(upper, x_max, 100),
            stats.t.pdf(np.linspace(upper, x_max, 100), df),
            alpha=0.3, color=reject_color,
            label=f"Rejection region (α={alpha:.3f})"
        )
        
        # Add critical line
        ax.axvline(x=upper, color=reject_color, linestyle='--', linewidth=2,
                  label=f"Critical value: {upper:.4f}")
        
    else:  # less
        lower = critical_values["lower"]
        
        # Shade lower rejection region
        ax.fill_between(
            np.linspace(x_min, lower, 100),
            stats.t.pdf(np.linspace(x_min, lower, 100), df),
            alpha=0.3, color=reject_color,
            label=f"Rejection region (α={alpha:.3f})"
        )
        
        # Add critical line
        ax.axvline(x=lower, color=reject_color, linestyle='--', linewidth=2,
                  label=f"Critical value: {lower:.4f}")
    
    # Add t-statistic if provided
    if t_stat is not None:
        ax.axvline(x=t_stat, color=stat_color, linestyle='-', linewidth=2,
                  label=f"t-statistic: {t_stat:.4f} (p={results['p_value']:.4f})")
        
        # Add confidence interval visualization if available
        if "confidence_interval" in results and "mean_diff" in results:
            ci = results["confidence_interval"]
            mean_diff = results["mean_diff"]
            
            # Create inset for confidence interval visualization
            if ci[0] > float("-inf") and ci[1] < float("inf"):
                # Create inset axes for CI visualization
                ax_inset = fig.add_axes([0.15, 0.55, 0.3, 0.2])
                
                # Calculate appropriate x range for inset
                ci_width = ci[1] - ci[0]
                x_padding = 0.1 * ci_width
                
                # Add reference line at zero
                ax_inset.axvline(x=0, color=reject_color, linestyle='--', alpha=0.5)
                
                # Plot mean with error bars
                ax_inset.errorbar(
                    [mean_diff], [0.5], 
                    xerr=[[mean_diff - ci[0]], [ci[1] - mean_diff]],
                    fmt='o', color=stat_color, capsize=5, markersize=8
                )
                
                # Format inset axes
                ax_inset.set_yticks([])
                ax_inset.set_title(f"{params['confidence_level']}% CI", fontsize=10)
                ax_inset.set_xlabel('Mean Difference', fontsize=8)
                ax_inset.set_xlim(ci[0] - x_padding, ci[1] + x_padding)
                sns.despine(ax=ax_inset, left=True)
    
    # Add title and labels
    title_suffix = {
        "two-sided": f" - Two-sided Test (α={alpha:.3f})",
        "greater": f" - Right-tailed Test (α={alpha:.3f})",
        "less": f" - Left-tailed Test (α={alpha:.3f})"
    }
    
    ax.set_title(f"t-Distribution (df={df}){title_suffix[alternative]}", fontsize=13)
    ax.set_xlabel('t Value', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.legend(loc='best', frameon=True, framealpha=0.7)
    ax.set_ylim(bottom=0)
    
    # Final formatting
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        return None
    return fig

def analyze_t_distribution(
    df: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    t_stat: Optional[float] = None,
    mean_diff: Optional[float] = None,
    std_error: Optional[float] = None,
    visualize: bool = True,
    show_plot: bool = True,
    figure_size: Tuple[int, int] = (10, 6),
) -> Tuple[Dict[str, Any], Optional[Figure]]:
    """Analyze t-distribution with calculations and optional visualization."""
    results = calculate_t_critical(df, alpha, alternative, t_stat, mean_diff, std_error)
    fig = None
    if visualize:
        fig = visualize_t_distribution(results, show_plot=show_plot, figure_size=figure_size)
    return results, fig