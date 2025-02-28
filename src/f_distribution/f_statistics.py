"""
F-Distribution Functions.
"""

from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.figure import Figure


def calculate_f_distribution(
    df1: int,
    df2: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    f_stat: Optional[float] = None,
) -> dict:
    """
    Calculate F-distribution critical values, p-values and confidence intervals.

    Parameters:
    -----------
    df1 : int
        Degrees of freedom for the numerator (first group)
    df2 : int
        Degrees of freedom for the denominator (second group)
    alpha : float, optional
        Significance level (default is 0.05 for 95% confidence)
    alternative : str, optional
        Type of test: 'two-sided' (default), 'greater', or 'less'
    f_stat : float, optional
        F-statistic value for p-value and confidence interval calculation

    Returns:
    --------
    dict
        Dictionary containing:
        - 'critical_values': Critical value(s) for the F-distribution
        - 'p_value': P-value if f_stat is provided
        - 'confidence_interval': Tuple of (lower, upper) bounds if f_stat is provided
        - 'parameters': Dictionary of input parameters
    """
    # Validate inputs
    if not (isinstance(df1, int) and isinstance(df2, int)):
        raise TypeError("Degrees of freedom must be integers")
    if not (df1 > 0 and df2 > 0):
        raise ValueError("Degrees of freedom must be positive")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1")
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")

    # Calculate critical values based on alternative hypothesis
    critical_values = {}
    if alternative == "two-sided":
        critical_values["upper"] = stats.f.ppf(1 - alpha / 2, df1, df2)
        critical_values["lower"] = stats.f.ppf(alpha / 2, df1, df2)
    elif alternative == "greater":
        critical_values["upper"] = stats.f.ppf(1 - alpha, df1, df2)
    else:  # less
        critical_values["lower"] = stats.f.ppf(alpha, df1, df2)

    # Prepare results dictionary
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

    # Calculate p-value and confidence interval if F-statistic is provided
    if f_stat is not None:
        if not isinstance(f_stat, (int, float)):
            raise TypeError("F-statistic must be a number")
        if f_stat <= 0:
            raise ValueError("F-statistic must be positive")

        # Store F-statistic for visualization
        results["f_stat"] = f_stat

        # Calculate p-value based on alternative hypothesis
        if alternative == "two-sided":
            p_value = 2 * min(
                1 - stats.f.cdf(f_stat, df1, df2), stats.f.cdf(f_stat, df1, df2)
            )
        elif alternative == "greater":
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        else:  # less
            p_value = stats.f.cdf(f_stat, df1, df2)

        results["p_value"] = p_value

        # Calculate confidence interval
        if alternative == "two-sided":
            # Corrected calculation for confidence intervals
            ci_lower = f_stat / critical_values["upper"]
            ci_upper = f_stat / critical_values["lower"]
        elif alternative == "greater":
            ci_lower = f_stat / critical_values["upper"]
            ci_upper = float("inf")
        else:  # less
            ci_lower = 0
            ci_upper = f_stat / critical_values["lower"]

        results["confidence_interval"] = (ci_lower, ci_upper)
        
        # Add reject_null flag
        results["reject_null"] = p_value < alpha

    return results


def format_f_results(results: dict, decimals: int = 4) -> str:
    """
    Format the results from calculate_f_distribution into a readable string.

    Parameters:
    -----------
    results : dict
        Output dictionary from calculate_f_distribution
    decimals : int, optional
        Number of decimal places to round to (default is 4)

    Returns:
    --------
    str
        Formatted string with the results
    """
    output = []
    params = results["parameters"]

    output.append("F-Distribution Analysis Results:")
    output.append("--------------------------------")
    output.append(f"Test Type: {params['alternative']}")
    output.append("\nParameters:")
    output.append(f"  - Numerator df (df1): {params['df1']}")
    output.append(f"  - Denominator df (df2): {params['df2']}")
    output.append(f"  - Alpha: {params['alpha']}")
    output.append(f"  - Confidence Level: {params['confidence_level']}%")

    output.append("\nCritical Values:")
    if "upper" in results["critical_values"]:
        output.append(f"  - Upper: {results['critical_values']['upper']:.{decimals}f}")
    if "lower" in results["critical_values"]:
        output.append(f"  - Lower: {results['critical_values']['lower']:.{decimals}f}")

    if "p_value" in results:
        output.append(f"\nP-value: {results['p_value']:.{decimals}f}")
        output.append(f"F-statistic: {results['f_stat']:.{decimals}f}")

        # Add interpretation
        output.append("\nTest Interpretation:")
        if results["p_value"] < params["alpha"]:
            output.append("  Reject the null hypothesis")
        else:
            output.append("  Fail to reject the null hypothesis")

    if "confidence_interval" in results:
        ci = results["confidence_interval"]
        output.append("\nConfidence Interval:")
        if ci[0] == 0:
            lower = "0"
        else:
            lower = f"{ci[0]:.{decimals}f}"
        if ci[1] == float("inf"):
            upper = "∞"
        else:
            upper = f"{ci[1]:.{decimals}f}"
        output.append(f"  ({lower}, {upper})")

    return "\n".join(output)


def visualize_f_distribution(
    results: dict, show_plot: bool = True, figure_size: Tuple[int, int] = (10, 6)
) -> Optional[Figure]:
    """
    Visualize the F-distribution analysis results.

    Parameters:
    -----------
    results : dict
        Output dictionary from calculate_f_distribution
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
    df1 = params["df1"]
    df2 = params["df2"]
    alpha = params["alpha"]
    alternative = params["alternative"]
    critical_values = results["critical_values"]
    f_stat = results.get("f_stat")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figure_size)

    # Calculate appropriate x-range for the F-distribution
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

    x_max = max(x_max, 5)  # Ensure a reasonable x-range
    x = np.linspace(0.001, x_max, 1000)

    # Plot the F-distribution PDF
    y = stats.f.pdf(x, df1, df2)
    ax.plot(x, y, "b-", lw=2, label=f"F({df1}, {df2})")

    # Shade rejection regions based on alternative hypothesis
    if alternative == "two-sided":
        # Lower tail
        x_lower = np.linspace(0.001, critical_values["lower"], 100)
        y_lower = stats.f.pdf(x_lower, df1, df2)
        ax.fill_between(
            x_lower,
            y_lower,
            alpha=0.3,
            color="r",
            label=f"Rejection region (α/2={alpha / 2:.3f})",
        )

        # Upper tail
        x_upper = np.linspace(critical_values["upper"], x_max, 100)
        y_upper = stats.f.pdf(x_upper, df1, df2)
        ax.fill_between(x_upper, y_upper, alpha=0.3, color="r")

        # Add vertical lines for critical values
        ax.axvline(
            critical_values["lower"],
            color="r",
            linestyle="--",
            label=f"Critical values: {critical_values['lower']:.4f}, {critical_values['upper']:.4f}",
        )
        ax.axvline(critical_values["upper"], color="r", linestyle="--")

    elif alternative == "greater":
        # Upper tail only
        x_upper = np.linspace(critical_values["upper"], x_max, 100)
        y_upper = stats.f.pdf(x_upper, df1, df2)
        ax.fill_between(
            x_upper,
            y_upper,
            alpha=0.3,
            color="r",
            label=f"Rejection region (α={alpha:.3f})",
        )

        # Add vertical line for critical value
        ax.axvline(
            critical_values["upper"],
            color="r",
            linestyle="--",
            label=f"Critical value: {critical_values['upper']:.4f}",
        )

    else:  # less
        # Lower tail only
        x_lower = np.linspace(0.001, critical_values["lower"], 100)
        y_lower = stats.f.pdf(x_lower, df1, df2)
        ax.fill_between(
            x_lower,
            y_lower,
            alpha=0.3,
            color="r",
            label=f"Rejection region (α={alpha:.3f})",
        )

        # Add vertical line for critical value
        ax.axvline(
            critical_values["lower"],
            color="r",
            linestyle="--",
            label=f"Critical value: {critical_values['lower']:.4f}",
        )

    # Add F-statistic if provided
    if f_stat is not None:
        ax.axvline(
            f_stat,
            color="g",
            linestyle="-",
            linewidth=1.5,
            label=f"F-statistic: {f_stat:.4f} (p={results['p_value']:.4f})",
        )

        # Highlight confidence interval if provided
        if "confidence_interval" in results:
            ci = results["confidence_interval"]
            if ci[0] > 0 and ci[1] < float("inf"):
                ax.axvspan(
                    ci[0],
                    ci[1],
                    alpha=0.2,
                    color="g",
                    label=f"{params['confidence_level']}% CI: ({ci[0]:.4f}, {ci[1]:.4f})",
                )
            elif ci[0] > 0:
                ax.axvspan(
                    ci[0],
                    x_max,
                    alpha=0.2,
                    color="g",
                    label=f"{params['confidence_level']}% CI: ({ci[0]:.4f}, ∞)",
                )
            elif ci[1] < float("inf"):
                ax.axvspan(
                    0,
                    ci[1],
                    alpha=0.2,
                    color="g",
                    label=f"{params['confidence_level']}% CI: (0, {ci[1]:.4f})",
                )

    # Add title and labels
    title = f"F-Distribution (df1={df1}, df2={df2})"
    if alternative == "two-sided":
        title += f" - Two-sided Test (α={alpha:.3f})"
    elif alternative == "greater":
        title += f" - Right-tailed Test (α={alpha:.3f})"
    else:  # less
        title += f" - Left-tailed Test (α={alpha:.3f})"

    ax.set_title(title)
    ax.set_xlabel("F Value")
    ax.set_ylabel("Probability Density")

    # Add legend
    ax.legend(loc="best")

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Tight layout for better spacing
    plt.tight_layout()

    if show_plot:
        plt.show()
        return None
    else:
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
    """
    Analyze F-distribution with calculations and optional visualization.

    Parameters:
    -----------
    df1 : int
        Degrees of freedom for the numerator (first group)
    df2 : int
        Degrees of freedom for the denominator (second group)
    alpha : float, optional
        Significance level (default is 0.05 for 95% confidence)
    alternative : str, optional
        Type of test: 'two-sided' (default), 'greater', or 'less'
    f_stat : float, optional
        F-statistic value for p-value and confidence interval calculation
    visualize : bool, optional
        Whether to create a visualization (default is True)
    figure_size : tuple, optional
        Size of the figure (width, height) in inches (default is (10, 6))

    Returns:
    --------
    tuple
        (results_dict, figure) where:
        - results_dict: Output dictionary from calculate_f_distribution
        - figure: matplotlib Figure object or None if visualize is False

    Examples:
    --------
    >>> # Analyze with visualization
    >>> results, fig = analyze_f_distribution(7, 9, alpha=0.05, f_stat=3.0625)
    >>> print(format_f_results(results))
    >>> plt.figure(fig.number)
    >>> plt.show()
    >>>
    >>> # Analyze without visualization
    >>> results, _ = analyze_f_distribution(7, 9, visualize=False)
    """
    # Calculate F-distribution results
    results = calculate_f_distribution(df1, df2, alpha, alternative, f_stat)

    # Create visualization if requested
    fig = None
    if visualize:
        fig = visualize_f_distribution(
            results, show_plot=True, figure_size=figure_size
        )

    return results, fig
