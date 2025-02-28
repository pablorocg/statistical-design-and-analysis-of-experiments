"""
T-Distribution and T-Test Functions
"""

from typing import Dict, Optional, Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from matplotlib.figure import Figure

# Set seaborn style
sns.set_theme(style="whitegrid")


def calculate_t_critical(
    df: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    t_stat: Optional[float] = None,
    mean_diff: Optional[float] = None,
    std_error: Optional[float] = None,
) -> Dict:
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


def format_t_results(results: Dict, decimals: int = 4) -> str:
    """Format the results into a readable string."""
    params = results["parameters"]
    crit_vals = results["critical_values"]
    
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
    if "upper" in crit_vals:
        lines.append(f"  Upper: {crit_vals['upper']:.{decimals}f}")
    if "lower" in crit_vals:
        lines.append(f"  Lower: {crit_vals['lower']:.{decimals}f}")

    # Add t-statistic and p-value if provided
    if "t_stat" in results:
        lines.extend([
            f"\nt-statistic: {results['t_stat']:.{decimals}f}",
            f"P-value: {results['p_value']:.{decimals}f}",
            "\nTest Interpretation:",
            f"  {'Reject' if results['p_value'] < params['alpha'] else 'Fail to reject'} the null hypothesis " +
            f"(p={results['p_value']:.{decimals}f} {('<' if results['p_value'] < params['alpha'] else '≥')} α={params['alpha']})"
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
    results: Dict, show_plot: bool = True, figure_size: Tuple[int, int] = (10, 6)
) -> Optional[Figure]:
    """Visualize the t-distribution analysis results using Seaborn."""
    # Extract parameters
    params = results["parameters"]
    df, alpha = params["df"], params["alpha"]
    alternative = params["alternative"]
    critical_values = results["critical_values"]
    t_stat = results.get("t_stat")

    # Create figure with Seaborn styling
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Calculate appropriate x-range
    x_min = -4
    x_max = 4
    
    if "lower" in critical_values:
        x_min = min(x_min, critical_values["lower"] * 1.5)
    if "upper" in critical_values:
        x_max = max(x_max, critical_values["upper"] * 1.5)
        
    # Adjust range if t-statistic is provided
    if t_stat is not None:
        x_min = min(x_min, t_stat * 1.5 if t_stat < 0 else x_min)
        x_max = max(x_max, t_stat * 1.5 if t_stat > 0 else x_max)

    x = np.linspace(x_min, x_max, 1000)
    
    # Plot PDF with Seaborn
    y = stats.t.pdf(x, df)
    sns.lineplot(x=x, y=y, color=sns.color_palette()[0], linewidth=2, label=f't({df})')
    
    # Use Seaborn color palette
    colors = sns.color_palette("muted")
    reject_color = colors[3]  # Usually red/orange in muted palette
    stat_color = colors[2]    # Usually green in muted palette
    
    # Shade regions and add critical lines
    if alternative == "two-sided":
        lower, upper = critical_values["lower"], critical_values["upper"]
        
        # Lower tail
        x_lower = np.linspace(x_min, lower, 100)
        y_lower = stats.t.pdf(x_lower, df)
        ax.fill_between(x_lower, y_lower, alpha=0.3, color=reject_color,
                      label=f"Rejection region (α/2={alpha/2:.3f})")
        
        # Upper tail
        x_upper = np.linspace(upper, x_max, 100)
        y_upper = stats.t.pdf(x_upper, df)
        ax.fill_between(x_upper, y_upper, alpha=0.3, color=reject_color)
        
        # Critical lines
        sns.lineplot(x=[lower, lower], y=[0, max(y)*1.1], color=reject_color, linestyle="--",
                  label=f"Critical values: {lower:.4f}, {upper:.4f}")
        sns.lineplot(x=[upper, upper], y=[0, max(y)*1.1], color=reject_color, linestyle="--")
        
    elif alternative == "greater":
        upper = critical_values["upper"]
        
        # Upper tail
        x_upper = np.linspace(upper, x_max, 100)
        y_upper = stats.t.pdf(x_upper, df)
        ax.fill_between(x_upper, y_upper, alpha=0.3, color=reject_color,
                      label=f"Rejection region (α={alpha:.3f})")
        
        # Critical line
        sns.lineplot(x=[upper, upper], y=[0, max(y)*1.1], color=reject_color, linestyle="--",
                  label=f"Critical value: {upper:.4f}")
        
    else:  # less
        lower = critical_values["lower"]
        
        # Lower tail
        x_lower = np.linspace(x_min, lower, 100)
        y_lower = stats.t.pdf(x_lower, df)
        ax.fill_between(x_lower, y_lower, alpha=0.3, color=reject_color,
                      label=f"Rejection region (α={alpha:.3f})")
        
        # Critical line
        sns.lineplot(x=[lower, lower], y=[0, max(y)*1.1], color=reject_color, linestyle="--",
                  label=f"Critical value: {lower:.4f}")
    
    # Add t-statistic if provided
    if t_stat is not None:
        sns.lineplot(x=[t_stat, t_stat], y=[0, max(y)*1.1], color=stat_color, linestyle="-",
                  linewidth=1.5, label=f"t-statistic: {t_stat:.4f} (p={results['p_value']:.4f})")
        
        # Add confidence interval visualization if provided
        if "confidence_interval" in results and "mean_diff" in results and "std_error" in results:
            ci = results["confidence_interval"]
            mean_diff = results["mean_diff"]
            
            # Create inset for CI visualization
            if ci[0] > float("-inf") and ci[1] < float("inf"):
                ax_inset = fig.add_axes([0.15, 0.55, 0.3, 0.2])
                
                ci_width = ci[1] - ci[0]
                x_ci = np.linspace(ci[0] - 0.1 * ci_width, ci[1] + 0.1 * ci_width, 2)
                
                # Add reference line at zero
                sns.lineplot(x=[0, 0], y=[0, 1], color=reject_color, linestyle="--", alpha=0.5, ax=ax_inset)
                
                # Plot mean with error bars
                ax_inset.errorbar(
                    [mean_diff], [0.5], 
                    xerr=[[mean_diff - ci[0]], [ci[1] - mean_diff]],
                    fmt='o', color=stat_color, capsize=5, markersize=8
                )
                
                ax_inset.set_yticks([])
                ax_inset.set_title(f"{params['confidence_level']}% CI", fontsize=10)
                ax_inset.set_xlabel('Mean Difference', fontsize=8)
                sns.despine(ax=ax_inset, left=True)
    
    # Add title and labels with Seaborn styling
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
    
    # Add despine for cleaner look
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
    figure_size: Tuple[int, int] = (10, 6),
) -> Tuple[Dict, Optional[Figure]]:
    """Analyze t-distribution with calculations and optional visualization."""
    results = calculate_t_critical(df, alpha, alternative, t_stat, mean_diff, std_error)
    fig = visualize_t_distribution(results, show_plot=visualize, figure_size=figure_size) if visualize else None
    return results, fig