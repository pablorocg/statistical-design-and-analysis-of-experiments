# Statistical Testing Library

A comprehensive library for statistical hypothesis testing with visualizations powered by Seaborn.

## Features

- **Z-tests** - For population means with known standard deviation
- **T-tests** - For population means with unknown standard deviation
- **F-tests** - For comparing population variances
- **Chi-squared tests** - For goodness of fit and independence testing

All tests include:
- Calculation of test statistics and critical values
- P-value computation
- Confidence interval estimation
- Beautiful visualizations using Seaborn
- Formatted text output for easy interpretation

## Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/username/stat-testing-library.git
cd stat-testing-library
pip install -r requirements.txt
```

## Quick Start

```python
from src import analyze_t_distribution, format_t_results

# Example t-test
results, fig = analyze_t_distribution(
    df=24, 
    t_stat=2.3, 
    mean_diff=1.5, 
    std_error=0.65, 
    alpha=0.05
)

# Display results
print(format_t_results(results))
fig.show()
```

## Documentation

### Z-Test (Known Variance)

#### Test Statistic
$Z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$

#### Critical Value
* Two-tailed test: $±z_{α/2}$
* One-tailed test: $z_α$ or $-z_α$

### T-Test (Unknown Variance)

#### Test Statistic
$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$

#### Degrees of Freedom
$df = n - 1$

### F-Test for Comparing Two Variances

#### Test Statistic
$F = \frac{s_1^2}{s_2^2}$ where $s_1^2$ is the larger sample variance

#### Degrees of Freedom
* $df_1 = n_1 - 1$ (numerator)
* $df_2 = n_2 - 1$ (denominator)

### Chi-Squared Tests

#### Goodness of Fit Test Statistic
$\chi^2 = \sum_{i=1}^n \frac{(O_i - E_i)^2}{E_i}$

#### Degrees of Freedom
For goodness of fit: $df = n - 1 - m$

## Examples

Check the `main.ipynb` notebook for detailed examples or run the examples directly:

```bash
python main.py
```

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- Seaborn

## License

MIT