"""
Chi-Squared Distribution Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from typing import Dict, Optional, Tuple, Literal
from scipy import stats

# Set seaborn theme
sns.set_theme(style="whitegrid")


def calculate_chi_squared_critical(
    df: int,
    chi_stat: Optional[float] = None,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    observed_values: Optional[np.ndarray] = None,
    expected_values: Optional[np.ndarray] = None,
) -> Dict:
    """Calculate chi-squared distribution critical values and related statistics."""
    # Input validation
    if not isinstance(df, int) or df <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid input parameters")
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")

    # Calculate chi-squared statistic if observed and expected values are provided
    if observed_values is not None and expected_values is not None:
        if len(observed_values) != len(expected_values):
            raise ValueError("Observed and expected arrays must have the same length")
        contributions = (observed_values - expected_values) ** 2 / expected_values
        chi_stat = np.sum(contributions)

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

        # Add contributions if available
        if observed_values is not None and expected_values is not None:
            results.update(
                {
                    "contributions": contributions,
                    "observed_values": observed_values,
                    "expected_values": expected_values,
                }
            )

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


def format_chi_squared_critical_results(results: Dict, decimals: int = 4) -> str:
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
        "\nCritical Values:",
    ]

    # Add critical values
    if "upper" in crit_vals:
        lines.append(f"  Upper: {crit_vals['upper']:.{decimals}f}")
    if "lower" in crit_vals:
        lines.append(f"  Lower: {crit_vals['lower']:.{decimals}f}")

    # Add test statistic and p-value if available
    if "test_statistic" in results:
        lines.append(
            f"\nChi-squared statistic: {results['test_statistic']:.{decimals}f}"
        )

    if "p_value" in results:
        lines.extend(
            [
                f"P-value: {results['p_value']:.{decimals}f}",
                "\nTest Interpretation:",
                f"  {'Reject' if results['p_value'] < params['alpha'] else 'Fail to reject'} "
                + f"the null hypothesis (p={results['p_value']:.{decimals}f} "
                + f"{('<' if results['p_value'] < params['alpha'] else '≥')} α={params['alpha']})",
            ]
        )

    # Add contribution details if available
    if "contributions" in results:
        lines.append("\nContributions to Chi-squared statistic:")
        for i, (obs, exp, cont) in enumerate(
            zip(
                results["observed_values"],
                results["expected_values"],
                results["contributions"],
            )
        ):
            lines.append(
                f"  Category {i + 1}: Observed={obs}, Expected={exp:.{decimals}f}, "
                + f"Contribution={cont:.{decimals}f}"
            )

    return "\n".join(lines)


def visualize_chi_squared_distribution(
    results: Dict, show_plot: bool = True, figure_size: Tuple[int, int] = (10, 6)
) -> Optional[Figure]:
    """Visualize the chi-squared distribution analysis results using Seaborn."""
    # Extract parameters
    params = results["parameters"]
    df, alpha = params["df"], params["alpha"]
    alternative = params["alternative"]
    critical_values = results["critical_values"]
    chi_stat = results.get("test_statistic")
    has_contributions = "contributions" in results

    # Set up Seaborn style and color palette
    sns.set_style("whitegrid")
    colors = sns.color_palette("muted")
    main_color = colors[0]  # Blue
    reject_color = colors[3]  # Red/orange
    stat_color = colors[2]  # Green

    # Create figure with appropriate layout
    if has_contributions:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    else:
        fig, ax1 = plt.subplots(figsize=figure_size)

    # Calculate appropriate x-range
    x_max = df * 2  # Base range on degrees of freedom
    if "upper" in critical_values:
        x_max = max(x_max, critical_values["upper"] * 1.5)
    if chi_stat is not None:
        x_max = max(x_max, chi_stat * 1.2)

    x = np.linspace(0.001, x_max, 1000)
    y = stats.chi2.pdf(x, df)

    # Plot the chi-squared PDF
    sns.lineplot(x=x, y=y, color=main_color, linewidth=2, ax=ax1, label=f"χ²({df})")

    # Shade rejection regions and add critical lines
    if alternative == "two-sided":
        # Lower and upper tails
        if "lower" in critical_values:
            lower = critical_values["lower"]
            x_lower = np.linspace(0.001, lower, 100)
            y_lower = stats.chi2.pdf(x_lower, df)
            ax1.fill_between(
                x_lower,
                y_lower,
                alpha=0.3,
                color=reject_color,
                label=f"Rejection region (α/2={alpha / 2:.3f})",
            )
            sns.lineplot(
                x=[lower, lower],
                y=[0, max(y) * 1.1],
                color=reject_color,
                linestyle="--",
                ax=ax1,
            )

        if "upper" in critical_values:
            upper = critical_values["upper"]
            x_upper = np.linspace(upper, x_max, 100)
            y_upper = stats.chi2.pdf(x_upper, df)
            ax1.fill_between(x_upper, y_upper, alpha=0.3, color=reject_color)

            # Critical lines label
            critical_label = f"Critical values: "
            if "lower" in critical_values:
                critical_label += f"{lower:.4f}, {upper:.4f}"
            else:
                critical_label += f"{upper:.4f}"

            sns.lineplot(
                x=[upper, upper],
                y=[0, max(y) * 1.1],
                color=reject_color,
                linestyle="--",
                ax=ax1,
                label=critical_label,
            )

    elif alternative == "greater":
        # Upper tail only
        upper = critical_values["upper"]
        x_upper = np.linspace(upper, x_max, 100)
        y_upper = stats.chi2.pdf(x_upper, df)
        ax1.fill_between(
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
            ax=ax1,
            label=f"Critical value: {upper:.4f}",
        )

    else:  # less
        # Lower tail only
        lower = critical_values["lower"]
        x_lower = np.linspace(0.001, lower, 100)
        y_lower = stats.chi2.pdf(x_lower, df)
        ax1.fill_between(
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
            ax=ax1,
            label=f"Critical value: {lower:.4f}",
        )

    # Add chi-squared statistic if provided
    if chi_stat is not None:
        sns.lineplot(
            x=[chi_stat, chi_stat],
            y=[0, max(y) * 1.1],
            color=stat_color,
            linestyle="-",
            linewidth=1.5,
            ax=ax1,
            label=f"χ² stat: {chi_stat:.4f} (p={results['p_value']:.4f})",
        )

    # Set title and labels with Seaborn styling
    title_suffix = {
        "two-sided": f" - Two-sided Test (α={alpha:.3f})",
        "greater": f" - Right-tailed Test (α={alpha:.3f})",
        "less": f" - Left-tailed Test (α={alpha:.3f})",
    }

    ax1.set_title(
        f"Chi-Squared Distribution (df={df}){title_suffix[alternative]}", fontsize=12
    )
    ax1.set_xlabel("χ² Value", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.legend(loc="best", frameon=True, framealpha=0.7)
    ax1.set_ylim(bottom=0)

    # Add contribution plot if data is available
    if has_contributions:
        contributions = results["contributions"]
        observed = results["observed_values"]
        expected = results["expected_values"]
        categories = np.arange(len(contributions))

        # Create a DataFrame for Seaborn
        data = []
        for i, (obs, exp) in enumerate(zip(observed, expected)):
            data.extend(
                [
                    {"Category": f"Cat {i + 1}", "Value": obs, "Type": "Observed"},
                    {"Category": f"Cat {i + 1}", "Value": exp, "Type": "Expected"},
                ]
            )

        # Create the grouped bar chart with Seaborn
        bar_plot = sns.barplot(
            x="Category",
            y="Value",
            hue="Type",
            data=data,
            palette={"Observed": colors[0], "Expected": colors[1]},
            ax=ax2,
        )

        # Add text annotations for contributions
        for i, contribution in enumerate(contributions):
            ax2.annotate(
                f"{contribution:.2f}",
                xy=(i, max(observed[i], expected[i]) + 0.5),
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5),
            )

        # Add labels and title
        ax2.set_xlabel("Categories", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.set_title(
            "Observed vs Expected Values\n(with χ² contributions)", fontsize=12
        )
        ax2.legend(frameon=True, framealpha=0.7)

    # Add despine for cleaner look
    sns.despine(ax=ax1, left=False, bottom=False)
    if has_contributions:
        sns.despine(ax=ax2, left=False, bottom=False)

    plt.tight_layout()

    if show_plot:
        plt.show()
        return None
    return fig


def visualize_chi_squared(
    df: int,
    chi_stat: Optional[float] = None,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    observed_values: Optional[np.ndarray] = None,
    expected_values: Optional[np.ndarray] = None,
    show_plot: bool = True,
    figure_size: Tuple[int, int] = (10, 6),
) -> Tuple[Dict, Optional[Figure]]:
    """Complete chi-squared analysis with visualization."""
    results = calculate_chi_squared_critical(
        df, chi_stat, alpha, alternative, observed_values, expected_values
    )
    fig = (
        visualize_chi_squared_distribution(results, show_plot, figure_size)
        if show_plot
        else None
    )
    return results, fig
