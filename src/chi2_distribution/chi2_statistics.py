"""
Chi-Squared Distribution Functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Optional, Tuple, Literal
from scipy import stats


def calculate_chi_squared_critical(
    df: int,
    chi_stat: Optional[float] = None,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    observed_values: Optional[np.ndarray] = None,
    expected_values: Optional[np.ndarray] = None,
) -> Dict:
    """
    Calculate chi-squared distribution critical values and related statistics.

    Parameters:
    -----------
    df : int
        Degrees of freedom
    chi_stat : float, optional
        Pre-calculated chi-squared statistic for p-value calculation
    alpha : float, optional
        Significance level (default is 0.05)
    alternative : str, optional
        Type of test: 'two-sided', 'greater', or 'less' (default is 'two-sided')
    observed_values : numpy.ndarray, optional
        Observed frequency values for goodness of fit or independence test
    expected_values : numpy.ndarray, optional
        Expected frequency values for goodness of fit or independence test

    Returns:
    --------
    dict
        Dictionary containing:
        - 'critical_values': Critical value(s) for the chi-squared distribution
        - 'p_value': P-value if chi_stat is provided
        - 'test_statistic': Calculated chi-squared statistic if observed and expected values are provided
        - 'parameters': Dictionary of input parameters
        - 'contributions': Individual contributions to the chi-squared statistic if observed and expected values are provided

    Examples:
    --------
    >>> # Variance test with pre-calculated chi-squared statistic
    >>> results = calculate_chi_squared_critical(df=19, chi_stat=38.0, alternative='greater')
    >>> print(results['p_value'])

    >>> # Goodness of fit test
    >>> observed = np.array([89, 37, 30, 28, 16])
    >>> expected = np.array([80, 40, 30, 30, 20])
    >>> results = calculate_chi_squared_critical(df=4, observed_values=observed, expected_values=expected)
    >>> print(results['test_statistic'])
    """
    # Input validation
    if not isinstance(df, int) or df <= 0:
        raise ValueError("Degrees of freedom must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")

    # Calculate chi-squared statistic if observed and expected values are provided
    if observed_values is not None and expected_values is not None:
        if len(observed_values) != len(expected_values):
            raise ValueError("Observed and expected arrays must have the same length")

        # Calculate chi-squared statistic and contributions
        contributions = (observed_values - expected_values) ** 2 / expected_values
        chi_stat = np.sum(contributions)

    # Calculate critical values based on alternative hypothesis
    critical_values = {}

    if alternative == "two-sided":
        critical_values["upper"] = stats.chi2.ppf(1 - alpha / 2, df)
        critical_values["lower"] = stats.chi2.ppf(alpha / 2, df)
    elif alternative == "greater":
        critical_values["upper"] = stats.chi2.ppf(1 - alpha, df)
    else:  # less
        critical_values["lower"] = stats.chi2.ppf(alpha, df)

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
            results["contributions"] = contributions
            results["observed_values"] = observed_values
            results["expected_values"] = expected_values

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

        # Add test result
        results["reject_null"] = p_value < alpha

    return results


def format_chi_squared_critical_results(results: Dict, decimals: int = 4) -> str:
    """
    Format the results from calculate_chi_squared_critical into a readable string.

    Parameters:
    -----------
    results : dict
        Output dictionary from calculate_chi_squared_critical
    decimals : int, optional
        Number of decimal places to round to (default is 4)

    Returns:
    --------
    str
        Formatted string with the results
    """
    output = []
    params = results["parameters"]

    output.append("Chi-Squared Distribution Analysis Results:")
    output.append("----------------------------------------")
    output.append(f"Test Type: {params['alternative']}")
    output.append(f"Degrees of Freedom: {params['df']}")
    output.append(f"Alpha: {params['alpha']}")
    output.append(f"Confidence Level: {params['confidence_level']}%")

    output.append("\nCritical Values:")
    if "upper" in results["critical_values"]:
        output.append(f"  Upper: {results['critical_values']['upper']:.{decimals}f}")
    if "lower" in results["critical_values"]:
        output.append(f"  Lower: {results['critical_values']['lower']:.{decimals}f}")

    if "test_statistic" in results:
        output.append(
            f"\nChi-squared statistic: {results['test_statistic']:.{decimals}f}"
        )

    if "p_value" in results:
        output.append(f"P-value: {results['p_value']:.{decimals}f}")

        # Add interpretation
        output.append("\nTest Interpretation:")
        if results["p_value"] < params["alpha"]:
            output.append(
                f"  Reject the null hypothesis (p={results['p_value']:.{decimals}f} < α={params['alpha']})"
            )
        else:
            output.append(
                f"  Fail to reject the null hypothesis (p={results['p_value']:.{decimals}f} ≥ α={params['alpha']})"
            )

    # Add contribution details if available
    if "contributions" in results:
        output.append("\nContributions to Chi-squared statistic:")
        for i, (obs, exp, cont) in enumerate(
            zip(
                results["observed_values"],
                results["expected_values"],
                results["contributions"],
            )
        ):
            output.append(
                f"  Category {i + 1}: Observed={obs}, Expected={exp:.{decimals}f}, Contribution={cont:.{decimals}f}"
            )

    return "\n".join(output)


def visualize_chi_squared_distribution(
    results: Dict, show_plot: bool = True, figure_size: Tuple[int, int] = (10, 6)
) -> Optional[Figure]:
    """
    Visualize the chi-squared distribution analysis results.

    Parameters:
    -----------
    results : dict
        Output dictionary from calculate_chi_squared_critical
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
    chi_stat = results.get("test_statistic")

    # Create a figure with multiple panels if we have contribution data
    has_contributions = "contributions" in results

    if has_contributions:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    else:
        fig, ax1 = plt.subplots(figsize=figure_size)

    # Calculate appropriate x-range for the chi-squared distribution
    if alternative == "two-sided":
        x_max = max(
            critical_values.get("upper", 0) * 1.5, chi_stat * 1.2 if chi_stat else 0
        )
    elif alternative == "greater":
        x_max = max(
            critical_values.get("upper", 0) * 1.5, chi_stat * 1.2 if chi_stat else 0
        )
    else:  # less
        x_max = max(
            critical_values.get("lower", 0) * 3, chi_stat * 1.2 if chi_stat else 0
        )

    x_max = max(x_max, df * 2)  # Ensure a reasonable x-range based on df
    x = np.linspace(0.001, x_max, 1000)

    # Plot the chi-squared PDF
    y = stats.chi2.pdf(x, df)
    ax1.plot(x, y, "b-", lw=2, label=f"χ²({df})")

    # Shade rejection regions based on alternative hypothesis
    if alternative == "two-sided":
        # Lower tail
        if "lower" in critical_values:
            x_lower = np.linspace(0.001, critical_values["lower"], 100)
            y_lower = stats.chi2.pdf(x_lower, df)
            ax1.fill_between(
                x_lower,
                y_lower,
                alpha=0.3,
                color="r",
                label=f"Rejection region (α/2={alpha / 2:.3f})",
            )

        # Upper tail
        if "upper" in critical_values:
            x_upper = np.linspace(critical_values["upper"], x_max, 100)
            y_upper = stats.chi2.pdf(x_upper, df)
            ax1.fill_between(x_upper, y_upper, alpha=0.3, color="r")

            # Add vertical lines for critical values
            if "lower" in critical_values:
                ax1.axvline(
                    critical_values["lower"],
                    color="r",
                    linestyle="--",
                    label=f"Critical values: {critical_values['lower']:.4f}, {critical_values['upper']:.4f}",
                )
            ax1.axvline(critical_values["upper"], color="r", linestyle="--")

    elif alternative == "greater":
        # Upper tail only
        x_upper = np.linspace(critical_values["upper"], x_max, 100)
        y_upper = stats.chi2.pdf(x_upper, df)
        ax1.fill_between(
            x_upper,
            y_upper,
            alpha=0.3,
            color="r",
            label=f"Rejection region (α={alpha:.3f})",
        )

        # Add vertical line for critical value
        ax1.axvline(
            critical_values["upper"],
            color="r",
            linestyle="--",
            label=f"Critical value: {critical_values['upper']:.4f}",
        )

    else:  # less
        # Lower tail only
        x_lower = np.linspace(0.001, critical_values["lower"], 100)
        y_lower = stats.chi2.pdf(x_lower, df)
        ax1.fill_between(
            x_lower,
            y_lower,
            alpha=0.3,
            color="r",
            label=f"Rejection region (α={alpha:.3f})",
        )

        # Add vertical line for critical value
        ax1.axvline(
            critical_values["lower"],
            color="r",
            linestyle="--",
            label=f"Critical value: {critical_values['lower']:.4f}",
        )

    # Add chi-squared statistic if provided
    if chi_stat is not None:
        ax1.axvline(
            chi_stat,
            color="g",
            linestyle="-",
            linewidth=1.5,
            label=f"χ² stat: {chi_stat:.4f} (p={results['p_value']:.4f})",
        )

    # Add title and labels
    title = f"Chi-Squared Distribution (df={df})"
    if alternative == "two-sided":
        title += f" - Two-sided Test (α={alpha:.3f})"
    elif alternative == "greater":
        title += f" - Right-tailed Test (α={alpha:.3f})"
    else:  # less
        title += f" - Left-tailed Test (α={alpha:.3f})"

    ax1.set_title(title)
    ax1.set_xlabel("χ² Value")
    ax1.set_ylabel("Probability Density")

    # Add legend
    ax1.legend(loc="best")

    # Set y-axis to start at 0
    ax1.set_ylim(bottom=0)

    # Add contribution plot if data is available
    if has_contributions:
        contributions = results["contributions"]
        observed = results["observed_values"]
        expected = results["expected_values"]

        # Set up bar positions
        categories = np.arange(len(contributions))
        width = 0.35

        # Create the grouped bar chart
        ax2.bar(
            categories - width / 2, observed, width, label="Observed", color="royalblue"
        )
        ax2.bar(
            categories + width / 2,
            expected,
            width,
            label="Expected",
            color="lightgreen",
        )

        # Add text annotations for contributions
        for i, contribution in enumerate(contributions):
            ax2.annotate(
                f"{contribution:.2f}",
                xy=(i, max(observed[i], expected[i]) + 0.5),
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
            )

        # Add labels and title
        ax2.set_xlabel("Categories")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Observed vs Expected Values\n(with χ² contributions)")
        ax2.set_xticks(categories)
        ax2.set_xticklabels([f"Cat {i + 1}" for i in categories])
        ax2.legend()

    # Tight layout for better spacing
    plt.tight_layout()

    if show_plot:
        plt.show()
        return None
    else:
        return fig
