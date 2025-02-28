# Hypothesis Testing Formulas for Two-Sample Tests

## F-Test for Comparing Two Variances

### Test Statistic
$F = \frac{s_1^2}{s_2^2}$ where $s_1^2$ is the larger sample variance

### Degrees of Freedom
* $df_1 = n_1 - 1$ (numerator)
* $df_2 = n_2 - 1$ (denominator)

### Critical Value
* Two-tailed test: $F_{α/2,df_1,df_2}$ and $F_{1-α/2,df_1,df_2}$
* One-tailed test: $F_{α,df_1,df_2}$

### P-value Calculation
For $H_1: σ_1^2 > σ_2^2$:
* P-value = $P(F > F_{obs})$ where $F_{obs}$ is the observed F-statistic

## Two-Sample t-Test for Means

### Equal Variances Assumed

#### Pooled Standard Deviation
$s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$

#### Test Statistic
$t = \frac{\bar{x_1} - \bar{x_2}}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$

#### Degrees of Freedom
$df = n_1 + n_2 - 2$

### Unequal Variances (Welch's t-test)

#### Test Statistic
$t = \frac{\bar{x_1} - \bar{x_2}}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$

#### Welch-Satterthwaite Degrees of Freedom
$df = \frac{(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2})^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$

### P-value Calculation
For two-tailed test:
* P-value = $2 \times P(T > |t_{obs}|)$ where $t_{obs}$ is the observed t-statistic

For one-tailed test:
* P-value = $P(T > t_{obs})$ for right-tailed test
* P-value = $P(T < t_{obs})$ for left-tailed test

## Power Analysis

### For Comparing Means

#### Effect Size (Cohen's d)
$d = \frac{|\mu_1 - \mu_2|}{\sigma}$

#### Sample Size for Desired Power
For equal sample sizes ($n_1 = n_2 = n$):
$n = \frac{2(z_{α/2} + z_β)^2}{d^2}$

Where:
* $z_{α/2}$ is the critical value for Type I error rate
* $z_β$ is the critical value for Type II error rate
* $d$ is the effect size

### Power Calculation
$Power = 1 - β = Φ(\frac{|μ_1 - μ_2|}{\sqrt{\frac{σ_1^2}{n_1} + \frac{σ_2^2}{n_2}}} - z_{α/2})$

## Confidence Intervals

### For Difference in Means (Equal Variances)
$(\bar{x_1} - \bar{x_2}) ± t_{α/2,df} \times s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$

### For Difference in Means (Unequal Variances)
$(\bar{x_1} - \bar{x_2}) ± t_{α/2,df} \times \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$

### For Ratio of Variances
$[\frac{s_1^2}{s_2^2} \cdot \frac{1}{F_{α/2,df_1,df_2}}, \frac{s_1^2}{s_2^2} \cdot F_{α/2,df_2,df_1}]$

## Special Considerations

### Paired Design
Use differences: $d_i = x_{1i} - x_{2i}$
Test statistic: $t = \frac{\bar{d}}{s_d/\sqrt{n}}$
Degrees of freedom: $df = n - 1$

### Assumptions to Check
1. Independence of observations
2. Normality of populations
3. Equal variances (when assumed)
4. Random sampling

### Common Decision Rules
* Reject $H_0$ if p-value < α
* Reject $H_0$ if test statistic > critical value
* Reject $H_0$ if confidence interval doesn't contain hypothesized value








# Chi-Squared Distribution Formulas and Applications

## Fundamental Properties

### Probability Density Function
The chi-squared probability density function with k degrees of freedom is:

$f(x;k) = \frac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}e^{-x/2}$

Where:
- x ≥ 0 is the random variable
- k is the degrees of freedom
- Γ is the gamma function

### Key Properties
- Mean: $E(X) = k$
- Variance: $Var(X) = 2k$
- Skewness: $\sqrt{8/k}$
- Kurtosis: $12/k$

## Goodness of Fit Test

### Test Statistic
$\chi^2 = \sum_{i=1}^n \frac{(O_i - E_i)^2}{E_i}$

Where:
- $O_i$ = Observed frequency in category i
- $E_i$ = Expected frequency in category i
- n = Number of categories

### Degrees of Freedom
For goodness of fit: $df = n - 1 - m$
Where:
- n = number of categories
- m = number of parameters estimated from the data

## Test of Independence

### Test Statistic
$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$

Where:
- $O_{ij}$ = Observed frequency in cell (i,j)
- $E_{ij}$ = Expected frequency in cell (i,j)
- $E_{ij} = \frac{(row_i total)(column_j total)}{grand total}$

### Degrees of Freedom
For independence test: $df = (r-1)(c-1)$
Where:
- r = number of rows
- c = number of columns

## Test of Homogeneity

### Test Statistic
Same as independence test:
$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$

### Degrees of Freedom
Same as independence test: $df = (r-1)(c-1)$

## Sample Variance Test

### Test Statistic
$\chi^2 = \frac{(n-1)s^2}{\sigma_0^2}$

Where:
- s² = sample variance
- σ₀² = hypothesized population variance
- n = sample size

### Degrees of Freedom
$df = n - 1$

## Common Decision Rules

### Critical Value Method
Reject H₀ if $\chi^2 > \chi^2_{\alpha,df}$ (right-tailed test)
Where:
- α = significance level
- df = degrees of freedom

### P-value Method
P-value = $P(\chi^2 > \chi^2_{obs})$
Reject H₀ if p-value < α

## Confidence Intervals

### For Population Variance
$[\frac{(n-1)s^2}{\chi^2_{\alpha/2,n-1}}, \frac{(n-1)s^2}{\chi^2_{1-\alpha/2,n-1}}]$

Where:
- s² = sample variance
- n = sample size
- α = significance level

## Special Cases and Adjustments

### Yates' Correction for 2×2 Tables
$\chi^2 = \sum_{i=1}^2 \sum_{j=1}^2 \frac{(|O_{ij} - E_{ij}| - 0.5)^2}{E_{ij}}$

### Conditions for Valid Chi-Square Tests
1. Independence of observations
2. Expected frequencies ≥ 5 (recommended)
3. Random sampling
4. Mutually exclusive categories

## Power Analysis

### Effect Size (w)
$w = \sqrt{\sum_{i=1}^k \frac{(p_{1i} - p_{0i})^2}{p_{0i}}}$

Where:
- $p_{1i}$ = alternative probability for category i
- $p_{0i}$ = null probability for category i

### Sample Size Determination
$n = \frac{\chi^2_{\alpha,df}(\lambda)}{w^2}$

Where:
- λ = non-centrality parameter
- w = effect size
- α = significance level
- df = degrees of freedom