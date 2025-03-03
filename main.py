"""
Statistical Testing Library - Main Interface

This module provides simplified access to the statistical testing functions
with examples demonstrating the use of each test.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import (
    calculate_chi_squared_critical, format_chi_squared_critical_results, visualize_chi_squared,
    calculate_f_distribution, format_f_results, analyze_f_distribution,
    calculate_t_critical, format_t_results, analyze_t_distribution,
    calculate_z_critical, format_z_results, analyze_z_distribution
)

# Set global visualization theme
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def run_z_test_example():
    """Run an example Z-test for a population mean with known standard deviation."""
    print("\n=== Z-Test Example ===")
    print("Testing if a sample mean differs from a hypothesized population mean")
    
    # Example data
    mu_0 = 100       # Null hypothesis: population mean = 100
    x_bar = 102.5    # Sample mean
    sigma = 15       # Known population standard deviation
    n = 64           # Sample size
    alpha = 0.05     # Significance level
    
    # Run analysis
    results, fig = analyze_z_distribution(
        mu_0=mu_0, x_bar=x_bar, sigma=sigma, n=n, 
        alpha=alpha, alternative="two-sided"
    )
    
    # Print results
    print(format_z_results(results))
    return results, fig


def run_t_test_example():
    """Run an example t-test for a population mean with unknown standard deviation."""
    print("\n=== t-Test Example ===")
    print("Testing if a sample mean differs from a hypothesized population mean")
    
    # Example data
    df = 24          # Degrees of freedom (n-1)
    t_stat = 2.3     # Calculated t-statistic
    mean_diff = 1.5  # Observed mean difference
    std_error = 0.65 # Standard error of the mean
    alpha = 0.05     # Significance level
    
    # Run analysis
    results, fig = analyze_t_distribution(
        df=df, t_stat=t_stat, mean_diff=mean_diff, std_error=std_error,
        alpha=alpha, alternative="two-sided"
    )
    
    # Print results
    print(format_t_results(results))
    return results, fig


def run_f_test_example():
    """Run an example F-test for comparing two population variances."""
    print("\n=== F-Test Example ===")
    print("Testing if two population variances are equal")
    
    # Example data
    df1 = 15         # Degrees of freedom for numerator (n₁-1)
    df2 = 20         # Degrees of freedom for denominator (n₂-1)
    f_stat = 2.25    # Calculated F-statistic
    alpha = 0.05     # Significance level
    
    # Run analysis
    results, fig = analyze_f_distribution(
        df1=df1, df2=df2, f_stat=f_stat,
        alpha=alpha, alternative="two-sided"
    )
    
    # Print results
    print(format_f_results(results))
    return results, fig


def run_chi_squared_example():
    """Run an example Chi-squared test for goodness of fit."""
    print("\n=== Chi-squared Test Example ===")
    print("Testing if observed frequencies match expected frequencies")
    
    # Example data
    observed = np.array([89, 37, 30, 28, 16])  # Observed frequencies
    expected = np.array([80, 40, 30, 30, 20])  # Expected frequencies
    df = 4                                      # Degrees of freedom
    alpha = 0.05                                # Significance level
    
    # Run analysis
    results, fig = visualize_chi_squared(
        df=df, observed_values=observed, expected_values=expected,
        alpha=alpha, alternative="two-sided"
    )
    
    # Print results
    print(format_chi_squared_critical_results(results))
    return results, fig


def run_all_examples():
    """Run all statistical test examples."""
    z_results, z_fig = run_z_test_example()
    t_results, t_fig = run_t_test_example()
    f_results, f_fig = run_f_test_example()
    chi2_results, chi2_fig = run_chi_squared_example()
    
    print("\n=== All examples completed successfully ===")
    
    return {
        "z_test": (z_results, z_fig),
        "t_test": (t_results, t_fig),
        "f_test": (f_results, f_fig),
        "chi2_test": (chi2_results, chi2_fig)
    }


if __name__ == "__main__":
    results = run_all_examples()
    plt.show()  # Display all figures