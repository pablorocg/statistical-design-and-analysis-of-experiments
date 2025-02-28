"""
F-Distribution Functions.
"""

from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from matplotlib.figure import Figure

# Set seaborn style
sns.set_theme(style="whitegrid")


def calculate_f_distribution(
    df1: int,
    df2: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    f_stat: Optional[float] = None,
) -> dict:
    """Calculate F-distribution critical values, p-values and confidence intervals."""
    # Validate inputs
    if not (isinstance(df1, int) and isinstance(df2, int) and df1 > 0 and df2 > 0):
        raise TypeError("Degrees of freedom must be positive integers")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")

    # Calculate critical values
    critical_values = {}
    if alternative == "two-sided":
        critical_values = {
            "upper": stats.f.ppf(1 - alpha / 2, df1, df2),
            "lower": stats.f.ppf(alpha / 2, df1, df2),
        }
    elif alternative == "greater":
        critical_values = {"upper": stats.f.ppf(1 - alpha, df1, df2)}
    else:  # less
        critical_values = {"lower": stats.f.ppf(alpha, df1, df2)}

    # Prepare results
    results = {
        "critical_values": critical_values,
        "parameters": {
            "df1": df1,
            "df2": df2,
            "alpha": alpha,
            "alternative": alternative,
            "confidence_level": (1 - alpha) * 100,
        },
    }

    # Calculate p-value and CI if F-statistic provided
    if f_stat is not None:
        if not isinstance(f_stat, (int, float)) or f_stat <= 0:
            raise ValueError("F-statistic must be a positive number")

        results["f_stat"] = f_stat

        # Calculate p-value
        if alternative == "two-sided":
            p_value = 2 * min(
                1 - stats.f.cdf(f_stat, df1, df2), stats.f.cdf(f_stat, df1, df2)
            )
        elif alternative == "greater":
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        else:  # less
            p_value = stats.f.cdf(f_stat, df1, df2)

        results["p_value"] = p_value
        results["reject_null"] = p_value < alpha

        # Calculate confidence interval
        if alternative == "two-sided":
            ci = (f_stat / critical_values["upper"], f_stat / critical_values["lower"])
        elif alternative == "greater":
            ci = (f_stat / critical_values["upper"], float("inf"))
        else:  # less
            ci = (0, f_stat / critical_values["lower"])

        results["confidence_interval"] = ci

    return results


def format_f_results(results: dict, decimals: int = 4) -> str:
    """Format the results into a readable string."""
    params = results["parameters"]
    crit_vals = results["critical_values"]

    lines = [
        "F-Distribution Analysis Results:",
        "--------------------------------",
        f"Test Type: {params['alternative']}",
        "\nParameters:",
        f"  - Numerator df (df1): {params['df1']}",
        f"  - Denominator df (df2): {params['df2']}",
        f"  - Alpha: {params['alpha']}",
        f"  - Confidence Level: {params['confidence_level']}%",
        "\nCritical Values:",
    ]

    if "upper" in crit_vals:
        lines.append(f"  - Upper: {crit_vals['upper']:.{decimals}f}")
    if "lower" in crit_vals:
        lines.append(f"  - Lower: {crit_vals['lower']:.{decimals}f}")

    if "p_value" in results:
        lines.extend(
            [
                f"\nP-value: {results['p_value']:.{decimals}f}",
                f"F-statistic: {results['f_stat']:.{decimals}f}",
                "\nTest Interpretation:",
                "  Reject the null hypothesis"
                if results["p_value"] < params["alpha"]
                else "  Fail to reject the null hypothesis",
            ]
        )

    if "confidence_interval" in results:
        ci = results["confidence_interval"]
        lower = "0" if ci[0] == 0 else f"{ci[0]:.{decimals}f}"
        upper = "∞" if ci[1] == float("inf") else f"{ci[1]:.{decimals}f}"
        lines.extend(["\nConfidence Interval:", f"  ({lower}, {upper})"])

    return "\n".join(lines)


def visualize_f_distribution(
    results: dict, show_plot: bool = True, figure_size: Tuple[int, int] = (10, 6)
) -> Optional[Figure]:
    """Visualize the F-distribution analysis results using Seaborn."""
    # Extract parameters
    params = results["parameters"]
    df1, df2 = params["df1"], params["df2"]
    alpha = params["alpha"]
    alternative = params["alternative"]
    critical_values = results["critical_values"]
    f_stat = results.get("f_stat")

    # Create figure with Seaborn styling
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figure_size)

    # X-range
    if alternative == "two-sided":
        x_max = max(
            critical_values.get("upper", 0) * 1.5, f_stat * 1.2 if f_stat else 0
        )
    elif alternative == "greater":
        x_max = max(
            critical_values.get("upper", 0) * 1.5, f_stat * 1.2 if f_stat else 0
        )
    else:  # less
        x_max = max(critical_values.get("lower", 0) * 3, f_stat * 1.2 if f_stat else 0)
    x_max = max(x_max, 5)
    x = np.linspace(0.001, x_max, 1000)

    # Plot PDF with Seaborn
    y = stats.f.pdf(x, df1, df2)
    sns.lineplot(
        x=x,
        y=y,
        ax=ax,
        color=sns.color_palette()[0],
        linewidth=2,
        label=f"F({df1}, {df2})",
    )

    # Use Seaborn color palette
    colors = sns.color_palette("muted")
    reject_color = colors[3]  # Usually red/orange in muted palette
    stat_color = colors[2]  # Usually green in muted palette

    # Shade regions and add critical lines
    if alternative == "two-sided":
        lower, upper = critical_values["lower"], critical_values["upper"]
        # Lower tail
        x_lower = np.linspace(0.001, lower, 100)
        y_lower = stats.f.pdf(x_lower, df1, df2)
        ax.fill_between(
            x_lower,
            y_lower,
            alpha=0.3,
            color=reject_color,
            label=f"Rejection region (α/2={alpha / 2:.3f})",
        )
        # Upper tail
        x_upper = np.linspace(upper, x_max, 100)
        y_upper = stats.f.pdf(x_upper, df1, df2)
        ax.fill_between(x_upper, y_upper, alpha=0.3, color=reject_color)
        # Critical lines
        sns.lineplot(
            x=[lower, lower],
            y=[0, max(y) * 1.1],
            color=reject_color,
            linestyle="--",
            ax=ax,
            label=f"Critical values: {lower:.4f}, {upper:.4f}",
        )
        sns.lineplot(
            x=[upper, upper],
            y=[0, max(y) * 1.1],
            color=reject_color,
            linestyle="--",
            ax=ax,
        )

    elif alternative == "greater":
        upper = critical_values["upper"]
        # Upper tail
        x_upper = np.linspace(upper, x_max, 100)
        y_upper = stats.f.pdf(x_upper, df1, df2)
        ax.fill_between(
            x_upper,
            y_upper,
            alpha=0.3,
            color=reject_color,
            label=f"Rejection region (α={alpha:.3f})",
        )
        # Critical line
        sns.lineplot(
            x=[upper, upper],
            y=[0, max(y) * 1.1],
            color=reject_color,
            linestyle="--",
            ax=ax,
            label=f"Critical value: {upper:.4f}",
        )

    else:  # less
        lower = critical_values["lower"]
        # Lower tail
        x_lower = np.linspace(0.001, lower, 100)
        y_lower = stats.f.pdf(x_lower, df1, df2)
        ax.fill_between(
            x_lower,
            y_lower,
            alpha=0.3,
            color=reject_color,
            label=f"Rejection region (α={alpha:.3f})",
        )
        # Critical line
        sns.lineplot(
            x=[lower, lower],
            y=[0, max(y) * 1.1],
            color=reject_color,
            linestyle="--",
            ax=ax,
            label=f"Critical value: {lower:.4f}",
        )

    # Add F-statistic if provided
    if f_stat is not None:
        sns.lineplot(
            x=[f_stat, f_stat],
            y=[0, max(y) * 1.1],
            color=stat_color,
            linestyle="-",
            linewidth=1.5,
            ax=ax,
            label=f"F-statistic: {f_stat:.4f} (p={results['p_value']:.4f})",
        )

        # Highlight confidence interval
        if "confidence_interval" in results:
            ci = results["confidence_interval"]
            ci_label = f"{params['confidence_level']}% CI: "

            if ci[0] > 0 and ci[1] < float("inf"):
                ax.axvspan(
                    ci[0],
                    ci[1],
                    alpha=0.2,
                    color=stat_color,
                    label=f"{ci_label}({ci[0]:.4f}, {ci[1]:.4f})",
                )
            elif ci[0] > 0:
                ax.axvspan(
                    ci[0],
                    x_max,
                    alpha=0.2,
                    color=stat_color,
                    label=f"{ci_label}({ci[0]:.4f}, ∞)",
                )
            elif ci[1] < float("inf"):
                ax.axvspan(
                    0,
                    ci[1],
                    alpha=0.2,
                    color=stat_color,
                    label=f"{ci_label}(0, {ci[1]:.4f})",
                )

    # Labels and formatting with Seaborn styling
    title_suffix = {
        "two-sided": f" - Two-sided Test (α={alpha:.3f})",
        "greater": f" - Right-tailed Test (α={alpha:.3f})",
        "less": f" - Left-tailed Test (α={alpha:.3f})",
    }

    ax.set_title(
        f"F-Distribution (df1={df1}, df2={df2}){title_suffix[alternative]}", fontsize=13
    )
    ax.set_xlabel("F Value", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(loc="best", frameon=True, framealpha=0.7)
    ax.set_ylim(bottom=0)

    # Add despine for cleaner look
    sns.despine(left=False, bottom=False)
    plt.tight_layout()

    if show_plot:
        plt.show()
        return None
    return fig


def analyze_f_distribution(
    df1: int,
    df2: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    f_stat: Optional[float] = None,
    visualize: bool = True,
    figure_size: Tuple[int, int] = (10, 6),
) -> Tuple[dict, Optional[Figure]]:
    """Analyze F-distribution with calculations and optional visualization."""
    results = calculate_f_distribution(df1, df2, alpha, alternative, f_stat)
    fig = (
        visualize_f_distribution(results, show_plot=visualize, figure_size=figure_size)
        if visualize
        else None
    )
    return results, fig
