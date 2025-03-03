"""
Chi-Squared Distribution Functions - Optimized Version
"""
from typing import Dict, Optional, Tuple, Literal, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

# Set theme once at module level
sns.set_theme(style="whitegrid")

def calculate_chi_squared(
    df: int,
    chi_stat: Optional[float] = None,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> Dict[str, Any]:
    """Calculate chi-squared distribution critical values and related statistics."""
    # Input validation
    if not isinstance(df, int) or df <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid input parameters")
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")

    # Calculate critical values based on alternative hypothesis
    critical_values = {}
    if alternative == "two-sided":
        critical_values = {
            "upper": stats.chi2.ppf(1 - alpha / 2, df),
            "lower": stats.chi2.ppf(alpha / 2, df),
        }
    elif alternative == "greater":
        critical_values = {"upper": stats.chi2.ppf(1 - alpha, df)}
    else:  # less
        critical_values = {"lower": stats.chi2.ppf(alpha, df)}

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

    # Add chi-squared statistic if provided or calculated
    if chi_stat is not None:
        results["test_statistic"] = chi_stat

        # Calculate p-value
        if alternative == "two-sided":
            p_value = 2 * min(
                stats.chi2.cdf(chi_stat, df), 1 - stats.chi2.cdf(chi_stat, df)
            )
        elif alternative == "greater":
            p_value = 1 - stats.chi2.cdf(chi_stat, df)
        else:  # less
            p_value = stats.chi2.cdf(chi_stat, df)

        results["p_value"] = p_value
        results["reject_null"] = p_value < alpha

    return results

def format_chi_squared_results(results: Dict[str, Any], decimals: int = 4) -> str:
    """Format the chi-squared results into a readable string."""
    params = results["parameters"]
    crit_vals = results["critical_values"]

    lines = [
        "Chi-Squared Distribution Analysis Results:",
        "----------------------------------------",
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
    if "test_statistic" in results:
        chi_stat = results["test_statistic"]
        p_value = results["p_value"]
        reject = p_value < params["alpha"]
        compare = "<" if reject else "≥"
        
        lines.extend([
            f"\nChi-squared statistic: {chi_stat:.{decimals}f}",
            f"P-value: {p_value:.{decimals}f}",
            "\nTest Interpretation:",
            f"  {'Reject' if reject else 'Fail to reject'} the null hypothesis "
            f"(p={p_value:.{decimals}f} {compare} α={params['alpha']})"
        ])

    return "\n".join(lines)

def visualize_chi_squared(
    results: Dict[str, Any], 
    show_plot: bool = True, 
    figure_size: Tuple[int, int] = (10, 6)
) -> Optional[Figure]:
    """Visualize the chi-squared distribution with enhanced clarity."""
    # Extract parameters
    params = results["parameters"]
    df, alpha = params["df"], params["alpha"]
    alternative = params["alternative"]
    critical_values = results["critical_values"]
    chi_stat = results.get("test_statistic")

    # Create figure
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Get colors from seaborn palette
    colors = sns.color_palette("muted")
    main_color = colors[0]    # Blue
    reject_color = colors[3]  # Red/orange
    stat_color = colors[2]    # Green
    
    # Calculate appropriate x-range
    x_max = max(df * 2, 10)  # Base range on degrees of freedom
    
    if "upper" in critical_values:
        x_max = max(x_max, critical_values["upper"] * 1.5)
    if chi_stat is not None:
        x_max = max(x_max, chi_stat * 1.2)

    # Create distribution curve
    x = np.linspace(0.001, x_max, 1000)
    y = stats.chi2.pdf(x, df)
    
    # Plot main distribution
    ax.plot(x, y, color=main_color, linewidth=2, label=f"χ²({df})")
    
    # Handle visualization based on test type
    if alternative == "two-sided":
        # Two-sided test with both tails
        if "lower" in critical_values and "upper" in critical_values:
            lower, upper = critical_values["lower"], critical_values["upper"]
            
            # Lower tail
            ax.fill_between(
                np.linspace(0.001, lower, 100),
                stats.chi2.pdf(np.linspace(0.001, lower, 100), df),
                alpha=0.3, color=reject_color,
                label=f"Rejection region (α/2={alpha/2:.3f})"
            )
            ax.axvline(x=lower, color=reject_color, linestyle='--', linewidth=2)
            
            # Upper tail
            ax.fill_between(
                np.linspace(upper, x_max, 100),
                stats.chi2.pdf(np.linspace(upper, x_max, 100), df),
                alpha=0.3, color=reject_color
            )
            ax.axvline(
                x=upper, color=reject_color, linestyle='--', linewidth=2,
                label=f"Critical values: {lower:.4f}, {upper:.4f}"
            )
            
    elif alternative == "greater":
        # Upper tail only
        upper = critical_values["upper"]
        ax.fill_between(
            np.linspace(upper, x_max, 100),
            stats.chi2.pdf(np.linspace(upper, x_max, 100), df),
            alpha=0.3, color=reject_color,
            label=f"Rejection region (α={alpha:.3f})"
        )
        ax.axvline(
            x=upper, color=reject_color, linestyle='--', linewidth=2,
            label=f"Critical value: {upper:.4f}"
        )
        
    else:  # less
        # Lower tail only
        lower = critical_values["lower"]
        ax.fill_between(
            np.linspace(0.001, lower, 100),
            stats.chi2.pdf(np.linspace(0.001, lower, 100), df),
            alpha=0.3, color=reject_color,
            label=f"Rejection region (α={alpha:.3f})"
        )
        ax.axvline(
            x=lower, color=reject_color, linestyle='--', linewidth=2,
            label=f"Critical value: {lower:.4f}"
        )
    
    # Add chi-squared statistic if provided
    if chi_stat is not None:
        ax.axvline(
            x=chi_stat, color=stat_color, linestyle='-', linewidth=2.5,
            label=f"χ² statistic: {chi_stat:.4f} (p={results['p_value']:.4f})"
        )
    
    # Format plot
    title_suffix = {
        "two-sided": f" - Two-sided Test (α={alpha:.3f})",
        "greater": f" - Right-tailed Test (α={alpha:.3f})",
        "less": f" - Left-tailed Test (α={alpha:.3f})"
    }
    
    ax.set_title(f"Chi-Squared Distribution (df={df}){title_suffix[alternative]}", fontsize=12)
    ax.set_xlabel("χ² Value", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(loc="best", frameon=True, framealpha=0.7)
    ax.set_ylim(bottom=0)
    
    # Final formatting
    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        return None
    return fig

def analyze_chi_squared(
    df: int,
    chi_stat: Optional[float] = None,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    show_plot: bool = True,
    figure_size: Tuple[int, int] = (10, 6),
) -> Tuple[Dict[str, Any], Optional[Figure]]:
    """Complete chi-squared analysis with visualization."""
    results = calculate_chi_squared(df, chi_stat, alpha, alternative)
    fig = None
    if show_plot:
        fig = visualize_chi_squared(results, show_plot=show_plot, figure_size=figure_size)
    return results, fig