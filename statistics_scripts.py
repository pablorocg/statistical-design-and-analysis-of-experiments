









def compare_alpha_levels(
    distribution_type: Literal["f", "t", "chi2", "z"],
    alpha_levels: list = [0.01, 0.025, 0.05, 0.1],
    **kwargs
) -> Tuple[list, Figure]:
    """
    Compare test results across multiple alpha levels.
    
    Parameters:
    -----------
    distribution_type : str
        Type of distribution: 'f', 't', 'chi2', or 'z'
    alpha_levels : list, optional
        List of alpha values to compare (default is [0.01, 0.025, 0.05, 0.1])
    **kwargs :
        Additional parameters required for the specific distribution
        
    Returns:
    --------
    tuple
        (results_list, figure) where:
        - results_list: List of result dictionaries for each alpha level
        - figure: matplotlib Figure containing comparison visualization
    
    Examples:
    --------
    >>> # Compare F-test results with different alpha levels
    >>> results_list, fig = compare_alpha_levels(
    ...     "f", alpha_levels=[0.01, 0.05, 0.1],
    ...     df1=7, df2=9, alternative="two-sided", f_stat=3.0625
    ... )
    """
    # Validate distribution type
    if distribution_type not in ["f", "t", "chi2", "z"]:
        raise ValueError("Distribution type must be 'f', 't', 'chi2', or 'z'")
    
    # Select the appropriate analysis function
    if distribution_type == "f":
        analyze_func = analyze_f_distribution
    elif distribution_type == "t":
        analyze_func = analyze_t_distribution
    elif distribution_type == "chi2":
        analyze_func = analyze_chi_squared_distribution
    else:  # z
        analyze_func = analyze_z_distribution
    
    # Run analysis for each alpha level
    results_list = []
    
    for alpha in alpha_levels:
        # Create a copy of kwargs with the current alpha
        current_kwargs = kwargs.copy()
        current_kwargs["alpha"] = alpha
        current_kwargs["visualize"] = False  # No individual visualizations
        
        # Run the analysis
        results, _ = analyze_func(**current_kwargs)
        results_list.append(results)
    
    # Create comparison visualization
    fig = create_alpha_comparison_plot(results_list, distribution_type)
    
    return results_list, fig

def create_alpha_comparison_plot(
    results_list: list, 
    distribution_type: str,
    figure_size: Tuple[int, int] = (14, 8)
) -> Figure:
    """
    Create a comparison plot for multiple alpha levels.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries for each alpha level
    distribution_type : str
        Type of distribution: 'f', 't', 'chi2', or 'z'
    figure_size : tuple, optional
        Size of the figure (width, height) in inches (default is (14, 8))
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the comparison visualization
    """
    # Extract common parameters
    alpha_levels = [r["parameters"]["alpha"] for r in results_list]
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    
    # Create color map for different alpha levels
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_levels)))
    
    # Plot 1: Distribution with critical values for each alpha
    if distribution_type == "f":
        df1 = results_list[0]["parameters"]["df1"]
        df2 = results_list[0]["parameters"]["df2"]
        
        # For F-distribution
        x_max = max([
            cv.get("upper", 0) for r in results_list 
            for cv in [r["critical_values"]]
        ]) * 1.5
        
        x = np.linspace(0.001, x_max, 1000)
        y = stats.f.pdf(x, df1, df2)
        
        ax1.plot(x, y, 'k-', lw=2, label=f'F({df1}, {df2})')
        
        for i, results in enumerate(results_list):
            alpha = results["parameters"]["alpha"]
            cv = results["critical_values"]
            
            if "upper" in cv:
                ax1.axvline(
                    cv["upper"], 
                    color=colors[i], 
                    linestyle='--',
                    label=f"α={alpha}: Critical = {cv['upper']:.4f}"
                )
                
            if "lower" in cv and distribution_type == "f":
                ax1.axvline(
                    cv["lower"], 
                    color=colors[i], 
                    linestyle=':'
                )
        
        # Add test statistic if present
        if "f_stat" in results_list[0]:
            f_stat = results_list[0]["f_stat"]
            ax1.axvline(
                f_stat,
                color='r',
                linestyle='-',
                linewidth=2,
                label=f"F-statistic: {f_stat:.4f}"
            )
            
        ax1.set_title(f"F-Distribution Critical Values Comparison\n(df1={df1}, df2={df2})")
        ax1.set_xlabel('F Value')
    
    elif distribution_type == "t":
        df = results_list[0]["parameters"]["df"]
        
        # For t-distribution
        x = np.linspace(-5, 5, 1000)
        y = stats.t.pdf(x, df)
        
        ax1.plot(x, y, 'k-', lw=2, label=f't({df})')
        
        for i, results in enumerate(results_list):
            alpha = results["parameters"]["alpha"]
            cv = results["critical_values"]
            
            if "upper" in cv:
                ax1.axvline(
                    cv["upper"], 
                    color=colors[i], 
                    linestyle='--',
                    label=f"α={alpha}: Upper = {cv['upper']:.4f}"
                )
                
            if "lower" in cv:
                ax1.axvline(
                    cv["lower"], 
                    color=colors[i], 
                    linestyle=':'
                )
        
        # Add test statistic if present
        if "t_stat" in results_list[0]:
            t_stat = results_list[0]["t_stat"]
            ax1.axvline(
                t_stat,
                color='r',
                linestyle='-',
                linewidth=2,
                label=f"t-statistic: {t_stat:.4f}"
            )
            
        ax1.set_title(f"t-Distribution Critical Values Comparison\n(df={df})")
        ax1.set_xlabel('t Value')
    
    elif distribution_type == "chi2":
        df = results_list[0]["parameters"]["df"]
        
        # For chi-squared distribution
        x_max = max([
            cv.get("upper", 0) for r in results_list 
            for cv in [r["critical_values"]]
        ]) * 1.5
        
        x = np.linspace(0.001, x_max, 1000)
        y = stats.chi2.pdf(x, df)
        
        ax1.plot(x, y, 'k-', lw=2, label=f'χ²({df})')
        
        for i, results in enumerate(results_list):
            alpha = results["parameters"]["alpha"]
            cv = results["critical_values"]
            
            if "upper" in cv:
                ax1.axvline(
                    cv["upper"], 
                    color=colors[i], 
                    linestyle='--',
                    label=f"α={alpha}: Upper = {cv['upper']:.4f}"
                )
                
            if "lower" in cv:
                ax1.axvline(
                    cv["lower"], 
                    color=colors[i], 
                    linestyle=':'
                )
        
        # Add test statistic if present
        if "test_statistic" in results_list[0]:
            chi_stat = results_list[0]["test_statistic"]
            ax1.axvline(
                chi_stat,
                color='r',
                linestyle='-',
                linewidth=2,
                label=f"χ² statistic: {chi_stat:.4f}"
            )
            
        ax1.set_title(f"Chi-Squared Distribution Critical Values Comparison\n(df={df})")
        ax1.set_xlabel('χ² Value')
    
    else:  # z-distribution
        # For standard normal distribution
        x = np.linspace(-4, 4, 1000)
        y = stats.norm.pdf(x)
        
        ax1.plot(x, y, 'k-', lw=2, label='N(0, 1)')
        
        for i, results in enumerate(results_list):
            alpha = results["parameters"]["alpha"]
            cv = results["critical_values"]
            
            if "upper" in cv:
                ax1.axvline(
                    cv["upper"], 
                    color=colors[i], 
                    linestyle='--',
                    label=f"α={alpha}: Upper = {cv['upper']:.4f}"
                )
                
            if "lower" in cv:
                ax1.axvline(
                    cv["lower"], 
                    color=colors[i], 
                    linestyle=':'
                )
        
        # Add test statistic if present
        if "z_statistic" in results_list[0]:
            z_stat = results_list[0]["z_statistic"]
            ax1.axvline(
                z_stat,
                color='r',
                linestyle='-',
                linewidth=2,
                label=f"Z-statistic: {z_stat:.4f}"
            )
            
        ax1.set_title(f"Standard Normal Distribution Critical Values Comparison")
        ax1.set_xlabel('Z Value')
    
    ax1.set_ylabel('Probability Density')
    ax1.legend(loc='best')
    
    # Plot 2: P-values and rejection decisions
    # Extract p-values if available
    if distribution_type != "f" or "p_value" in results_list[0]:
        p_values = []
        rejection_decisions = []
        
        for results in results_list:
            if "p_value" in results:
                p_values.append(results["p_value"])
                alpha = results["parameters"]["alpha"]
                rejection_decisions.append("Reject H₀" if results["p_value"] < alpha else "Fail to Reject H₀")
        
        if p_values:
            # Bar chart for rejection decisions
            y_pos = np.arange(len(alpha_levels))
            
            bars = ax2.barh(
                y_pos, 
                [1] * len(alpha_levels), 
                color=[colors[i] if p_values[i] < alpha_levels[i] else 'lightgray' for i in range(len(alpha_levels))]
            )
            
            # Add alpha and p-value annotations
            for i, (alpha, p_value, decision) in enumerate(zip(alpha_levels, p_values, rejection_decisions)):
                ax2.text(
                    0.5, i, 
                    f"α={alpha:.4f}, p={p_value:.4f}\n{decision}",
                    ha='center', 
                    va='center',
                    color='black' if decision == "Reject H₀" else 'darkgray'
                )
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([f"α = {alpha:.4f}" for alpha in alpha_levels])
            ax2.set_xlim(0, 1)
            ax2.set_xticks([])
            
            if distribution_type == "f":
                test_name = "F-test"
            elif distribution_type == "t":
                test_name = "t-test"
            elif distribution_type == "chi2":
                test_name = "Chi-Squared Test"
            else:
                test_name = "Z-test"
                
            ax2.set_title(f"{test_name} Results Comparison\nAcross Different Alpha Levels")
    else:
        ax2.text(
            0.5, 0.5,
            "No p-values available for comparison",
            ha='center',
            va='center',
            transform=ax2.transAxes,
            fontsize=12
        )
    
    # Add confidence intervals comparison if available
    if "confidence_interval" in results_list[0]:
        # Create a new axis for confidence intervals
        ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.2])
        
        ci_data = []
        labels = []
        
        for i, results in enumerate(results_list):
            alpha = results["parameters"]["alpha"]
            ci = results["confidence_interval"]
            
            ci_data.append([ci[0] if ci[0] != float("-inf") else None, 
                            ci[1] if ci[1] != float("inf") else None])
            labels.append(f"{(1-alpha)*100:.1f}% CI")
        
        # Find reasonable x-limits
        all_values = [v for ci in ci_data for v in ci if v is not None]
        if all_values:
            x_min = min(all_values) - 0.1 * (max(all_values) - min(all_values))
            x_max = max(all_values) + 0.1 * (max(all_values) - min(all_values))
        else:
            # Fallback if all CIs are one-sided
            if distribution_type == "z":
                mu_0 = results_list[0]["parameters"]["mu_0"]
                x_bar = results_list[0]["parameters"]["x_bar"]
                x_min = min(mu_0, x_bar) - 1
                x_max = max(mu_0, x_bar) + 1
            else:
                x_min = -1
                x_max = 1
        
        # Plot confidence intervals
        y_pos = np.arange(len(labels))
        
        for i, (ci, color) in enumerate(zip(ci_data, colors)):
            if ci[0] is not None and ci[1] is not None:
                # Two-sided CI
                ax3.plot([ci[0], ci[1]], [i, i], 'o-', color=color, linewidth=2, markersize=6)
            elif ci[0] is not None:
                # Lower bound only
                ax3.plot([ci[0], x_max], [i, i], '-', color=color, linewidth=2)
                ax3.plot(ci[0], i, 'o', color=color, markersize=6)
                ax3.annotate(
                    '∞', 
                    xy=(x_max, i), 
                    xytext=(-10, 0), 
                    textcoords='offset points',
                    ha='right',
                    va='center',
                    color=color,
                    fontsize=12
                )
            elif ci[1] is not None:
                # Upper bound only
                ax3.plot([x_min, ci[1]], [i, i], '-', color=color, linewidth=2)
                ax3.plot(ci[1], i, 'o', color=color, markersize=6)
                ax3.annotate(
                    '-∞', 
                    xy=(x_min, i), 
                    xytext=(10, 0), 
                    textcoords='offset points',
                    ha='left',
                    va='center',
                    color=color,
                    fontsize=12
                )
        
        # Plot special values
        if distribution_type == "z":
            # Add null hypothesis value
            mu_0 = results_list[0]["parameters"]["mu_0"]
            ax3.axvline(
                mu_0,
                color='r',
                linestyle='--',
                linewidth=1,
                label=f"H₀: μ = {mu_0:.4f}"
            )
            
            # Add sample mean
            x_bar = results_list[0]["parameters"]["x_bar"]
            ax3.axvline(
                x_bar,
                color='g',
                linestyle='-',
                linewidth=1,
                label=f"Sample Mean: {x_bar:.4f}"
            )
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels)
        ax3.set_xlim(x_min, x_max)
        ax3.set_title("Confidence Intervals Comparison")
        ax3.legend(loc='lower right')
        
        # Adjust main layout
        plt.subplots_adjust(bottom=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


#---------------------------------------------------------
# Power Analysis Functions
#---------------------------------------------------------

def calculate_power(
    distribution_type: Literal["f", "t", "chi2", "z"],
    alpha: float = 0.05,
    effect_size: Optional[float] = None,
    sample_size: Optional[int] = None,
    power: Optional[float] = None,
    **kwargs
) -> Dict:
    """
    Calculate statistical power, required sample size, or minimum detectable effect size.

    Parameters:
    -----------
    distribution_type : str
        Type of distribution: 'f', 't', 'chi2', or 'z'
    alpha : float, optional
        Significance level (default is 0.05)
    effect_size : float, optional
        Standardized effect size (required if calculating power or sample size)
    sample_size : int, optional
        Sample size (required if calculating power or effect size)
    power : float, optional
        Statistical power (required if calculating sample size or effect size)
    **kwargs : 
        Additional parameters required for the specific distribution
        
    Returns:
    --------
    dict
        Dictionary containing the calculation results
    """
    # Check that we have exactly two of the three parameters
    params_count = sum(p is not None for p in [effect_size, sample_size, power])
    if params_count != 2:
        raise ValueError("Exactly two of effect_size, sample_size, and power must be provided")
    
    # Get the alternative hypothesis type
    alternative = kwargs.get("alternative", "two-sided")
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    # Initialize results dictionary
    results = {
        "distribution_type": distribution_type,
        "alpha": alpha,
        "alternative": alternative,
    }
    
    # Calculate the missing parameter based on distribution type
    if distribution_type == "t":
        df = kwargs.get("df", None)
        if df is None and sample_size is not None:
            # For one-sample or paired t-test
            df = sample_size - 1
        elif df is None:
            raise ValueError("For t-distribution, either df or sample_size must be provided")
        
        results["df"] = df
        
        # Calculate power if effect size and sample size are provided
        if sample_size is not None and effect_size is not None:
            # Calculate non-centrality parameter
            ncp = effect_size * np.sqrt(sample_size)
            
            # Calculate power
            if alternative == "two-sided":
                crit = stats.t.ppf(1 - alpha/2, df)
                computed_power = 1 - stats.nct.cdf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
            elif alternative == "greater":
                crit = stats.t.ppf(1 - alpha, df)
                computed_power = 1 - stats.nct.cdf(crit, df, ncp)
            else:  # less
                crit = stats.t.ppf(alpha, df)
                computed_power = stats.nct.cdf(crit, df, ncp)
            
            results["power"] = computed_power
            results["sample_size"] = sample_size
            results["effect_size"] = effect_size
        
        # Calculate effect size if sample size and power are provided
        elif sample_size is not None and power is not None:
            # Define objective function for finding effect size
            def objective(es):
                ncp_temp = es * np.sqrt(sample_size)
                if alternative == "two-sided":
                    crit = stats.t.ppf(1 - alpha/2, df)
                    pwr = 1 - stats.nct.cdf(crit, df, ncp_temp) + stats.nct.cdf(-crit, df, ncp_temp)
                elif alternative == "greater":
                    crit = stats.t.ppf(1 - alpha, df)
                    pwr = 1 - stats.nct.cdf(crit, df, ncp_temp)
                else:  # less
                    crit = stats.t.ppf(alpha, df)
                    pwr = stats.nct.cdf(crit, df, ncp_temp)
                return pwr - power
            
            # Find effect size that gives desired power
            try:
                effect_size = brentq(objective, 0.01, 5.0)
                results["effect_size"] = effect_size
                results["sample_size"] = sample_size
                results["power"] = power
            except ValueError:
                # If no solution found in range, provide a reasonable estimate
                results["effect_size"] = "Could not be solved in range [0.01, 5.0]"
                results["sample_size"] = sample_size
                results["power"] = power
        
        # Calculate sample size if effect size and power are provided
        elif effect_size is not None and power is not None:
            # Define objective function for finding sample size
            def objective(n):
                n = max(2, int(n))  # Ensure n is an integer ≥ 2
                df_temp = n - 1
                ncp_temp = effect_size * np.sqrt(n)
                if alternative == "two-sided":
                    crit = stats.t.ppf(1 - alpha/2, df_temp)
                    pwr = 1 - stats.nct.cdf(crit, df_temp, ncp_temp) + stats.nct.cdf(-crit, df_temp, ncp_temp)
                elif alternative == "greater":
                    crit = stats.t.ppf(1 - alpha, df_temp)
                    pwr = 1 - stats.nct.cdf(crit, df_temp, ncp_temp)
                else:  # less
                    crit = stats.t.ppf(alpha, df_temp)
                    pwr = stats.nct.cdf(crit, df_temp, ncp_temp)
                return pwr - power
            
            # Find sample size that gives desired power
            try:
                sample_size = max(2, int(brentq(objective, 2, 1000)))
                results["sample_size"] = sample_size
                results["effect_size"] = effect_size
                results["power"] = power
            except ValueError:
                # If no solution found in range, provide a reasonable estimate
                results["sample_size"] = "Could not be solved in range [2, 1000]"
                results["effect_size"] = effect_size
                results["power"] = power
    
    elif distribution_type == "z":
        # For z-test (known variance)
        if sample_size is not None and effect_size is not None:
            # Calculate power
            if alternative == "two-sided":
                crit = stats.norm.ppf(1 - alpha/2)
                computed_power = 1 - stats.norm.cdf(crit - effect_size * np.sqrt(sample_size)) + stats.norm.cdf(-crit - effect_size * np.sqrt(sample_size))
            elif alternative == "greater":
                crit = stats.norm.ppf(1 - alpha)
                computed_power = 1 - stats.norm.cdf(crit - effect_size * np.sqrt(sample_size))
            else:  # less
                crit = stats.norm.ppf(alpha)
                computed_power = stats.norm.cdf(crit - effect_size * np.sqrt(sample_size))
            
            results["power"] = computed_power
            results["sample_size"] = sample_size
            results["effect_size"] = effect_size
            
        elif sample_size is not None and power is not None:
            # Solve for effect size
            if alternative == "two-sided":
                crit = stats.norm.ppf(1 - alpha/2)
                z_power = stats.norm.ppf(power)
                effect_size = (crit + z_power) / np.sqrt(sample_size)
            elif alternative == "greater":
                crit = stats.norm.ppf(1 - alpha)
                z_power = stats.norm.ppf(power)
                effect_size = (crit + z_power) / np.sqrt(sample_size)
            else:  # less
                crit = stats.norm.ppf(alpha)
                z_power = stats.norm.ppf(power)
                effect_size = (crit - z_power) / np.sqrt(sample_size)
            
            results["effect_size"] = effect_size
            results["sample_size"] = sample_size
            results["power"] = power
            
        elif effect_size is not None and power is not None:
            # Solve for sample size
            if alternative == "two-sided":
                crit = stats.norm.ppf(1 - alpha/2)
                z_power = stats.norm.ppf(power)
                sample_size = ((crit + z_power) / effect_size) ** 2
            elif alternative == "greater":
                crit = stats.norm.ppf(1 - alpha)
                z_power = stats.norm.ppf(power)
                sample_size = ((crit + z_power) / effect_size) ** 2
            else:  # less
                crit = stats.norm.ppf(alpha)
                z_power = stats.norm.ppf(power)
                sample_size = ((crit - z_power) / effect_size) ** 2
            
            # Round up to the nearest integer
            sample_size = int(np.ceil(sample_size))
            results["sample_size"] = sample_size
            results["effect_size"] = effect_size
            results["power"] = power
    
    elif distribution_type == "f":
        # For F-test (ANOVA or regression)
        df1 = kwargs.get("df1", None)
        df2 = kwargs.get("df2", None)
        
        if df1 is None or df2 is None:
            raise ValueError("For F-distribution, both df1 and df2 must be provided")
        
        results["df1"] = df1
        results["df2"] = df2
        
        if sample_size is not None and effect_size is not None:
            # Calculate non-centrality parameter
            ncp = effect_size * sample_size
            
            # Calculate power
            if alternative == "two-sided":
                crit_upper = stats.f.ppf(1 - alpha/2, df1, df2)
                crit_lower = stats.f.ppf(alpha/2, df1, df2)
                computed_power = 1 - stats.ncf.cdf(crit_upper, df1, df2, ncp) + stats.ncf.cdf(crit_lower, df1, df2, ncp)
            elif alternative == "greater":
                crit = stats.f.ppf(1 - alpha, df1, df2)
                computed_power = 1 - stats.ncf.cdf(crit, df1, df2, ncp)
            else:  # less
                crit = stats.f.ppf(alpha, df1, df2)
                computed_power = stats.ncf.cdf(crit, df1, df2, ncp)
            
            results["power"] = computed_power
            results["sample_size"] = sample_size
            results["effect_size"] = effect_size
            
        elif sample_size is not None and power is not None:
            # Define objective function for finding effect size
            def objective(es):
                ncp_temp = es * sample_size
                if alternative == "two-sided":
                    crit_upper = stats.f.ppf(1 - alpha/2, df1, df2)
                    crit_lower = stats.f.ppf(alpha/2, df1, df2)
                    pwr = 1 - stats.ncf.cdf(crit_upper, df1, df2, ncp_temp) + stats.ncf.cdf(crit_lower, df1, df2, ncp_temp)
                elif alternative == "greater":
                    crit = stats.f.ppf(1 - alpha, df1, df2)
                    pwr = 1 - stats.ncf.cdf(crit, df1, df2, ncp_temp)
                else:  # less
                    crit = stats.f.ppf(alpha, df1, df2)
                    pwr = stats.ncf.cdf(crit, df1, df2, ncp_temp)
                return pwr - power
            
            # Find effect size that gives desired power
            try:
                effect_size = brentq(objective, 0.01, 5.0)
                results["effect_size"] = effect_size
                results["sample_size"] = sample_size
                results["power"] = power
            except ValueError:
                results["effect_size"] = "Could not be solved in range [0.01, 5.0]"
                results["sample_size"] = sample_size
                results["power"] = power
                
        elif effect_size is not None and power is not None:
            # Define objective function for finding sample size
            def objective(n):
                n = max(5, int(n))  # Ensure n is a reasonable integer
                ncp_temp = effect_size * n
                if alternative == "two-sided":
                    crit_upper = stats.f.ppf(1 - alpha/2, df1, df2)
                    crit_lower = stats.f.ppf(alpha/2, df1, df2)
                    pwr = 1 - stats.ncf.cdf(crit_upper, df1, df2, ncp_temp) + stats.ncf.cdf(crit_lower, df1, df2, ncp_temp)
                elif alternative == "greater":
                    crit = stats.f.ppf(1 - alpha, df1, df2)
                    pwr = 1 - stats.ncf.cdf(crit, df1, df2, ncp_temp)
                else:  # less
                    crit = stats.f.ppf(alpha, df1, df2)
                    pwr = stats.ncf.cdf(crit, df1, df2, ncp_temp)
                return pwr - power
            
            # Find sample size that gives desired power
            try:
                sample_size = max(5, int(brentq(objective, 5, 1000)))
                results["sample_size"] = sample_size
                results["effect_size"] = effect_size
                results["power"] = power
            except ValueError:
                results["sample_size"] = "Could not be solved in range [5, 1000]"
                results["effect_size"] = effect_size
                results["power"] = power
    
    elif distribution_type == "chi2":
        # For chi-squared test
        df = kwargs.get("df", None)
        if df is None:
            raise ValueError("For chi-squared distribution, df must be provided")
        
        results["df"] = df
        
        if sample_size is not None and effect_size is not None:
            # Calculate non-centrality parameter
            ncp = effect_size * sample_size
            
            # Calculate power
            if alternative == "two-sided":
                crit_upper = stats.chi2.ppf(1 - alpha/2, df)
                crit_lower = stats.chi2.ppf(alpha/2, df)
                computed_power = 1 - stats.ncx2.cdf(crit_upper, df, ncp) + stats.ncx2.cdf(crit_lower, df, ncp)
            elif alternative == "greater":
                crit = stats.chi2.ppf(1 - alpha, df)
                computed_power = 1 - stats.ncx2.cdf(crit, df, ncp)
            else:  # less
                crit = stats.chi2.ppf(alpha, df)
                computed_power = stats.ncx2.cdf(crit, df, ncp)
            
            results["power"] = computed_power
            results["sample_size"] = sample_size
            results["effect_size"] = effect_size
            
        elif sample_size is not None and power is not None:
            # Define objective function for finding effect size
            def objective(es):
                ncp_temp = es * sample_size
                if alternative == "two-sided":
                    crit_upper = stats.chi2.ppf(1 - alpha/2, df)
                    crit_lower = stats.chi2.ppf(alpha/2, df)
                    pwr = 1 - stats.ncx2.cdf(crit_upper, df, ncp_temp) + stats.ncx2.cdf(crit_lower, df, ncp_temp)
                elif alternative == "greater":
                    crit = stats.chi2.ppf(1 - alpha, df)
                    pwr = 1 - stats.ncx2.cdf(crit, df, ncp_temp)
                else:  # less
                    crit = stats.chi2.ppf(alpha, df)
                    pwr = stats.ncx2.cdf(crit, df, ncp_temp)
                return pwr - power
            
            # Find effect size that gives desired power
            try:
                effect_size = brentq(objective, 0.01, 5.0)
                results["effect_size"] = effect_size
                results["sample_size"] = sample_size
                results["power"] = power
            except ValueError:
                results["effect_size"] = "Could not be solved in range [0.01, 5.0]"
                results["sample_size"] = sample_size
                results["power"] = power
                
        elif effect_size is not None and power is not None:
            # Define objective function for finding sample size
            def objective(n):
                n = max(5, int(n))  # Ensure n is a reasonable integer
                ncp_temp = effect_size * n
                if alternative == "two-sided":
                    crit_upper = stats.chi2.ppf(1 - alpha/2, df)
                    crit_lower = stats.chi2.ppf(alpha/2, df)
                    pwr = 1 - stats.ncx2.cdf(crit_upper, df, ncp_temp) + stats.ncx2.cdf(crit_lower, df, ncp_temp)
                elif alternative == "greater":
                    crit = stats.chi2.ppf(1 - alpha, df)
                    pwr = 1 - stats.ncx2.cdf(crit, df, ncp_temp)
                else:  # less
                    crit = stats.chi2.ppf(alpha, df)
                    pwr = stats.ncx2.cdf(crit, df, ncp_temp)
                return pwr - power
            
            # Find sample size that gives desired power
            try:
                sample_size = max(5, int(brentq(objective, 5, 1000)))
                results["sample_size"] = sample_size
                results["effect_size"] = effect_size
                results["power"] = power
            except ValueError:
                results["sample_size"] = "Could not be solved in range [5, 1000]"
                results["effect_size"] = effect_size
                results["power"] = power
    
    return results


def format_power_results(results: Dict, decimals: int = 4) -> str:
    """
    Format the results from calculate_power into a readable string.

    Parameters:
    -----------
    results : dict
        Output dictionary from calculate_power
    decimals : int, optional
        Number of decimal places to round to (default is 4)

    Returns:
    --------
    str
        Formatted string with the results
    """
    output = []
    
    # Add header based on distribution type
    dist_type = results["distribution_type"]
    if dist_type == "t":
        output.append("T-Test Power Analysis Results:")
    elif dist_type == "z":
        output.append("Z-Test Power Analysis Results:")
    elif dist_type == "f":
        output.append("F-Test Power Analysis Results:")
    elif dist_type == "chi2":
        output.append("Chi-Squared Test Power Analysis Results:")
    
    output.append("-" * len(output[0]))
    
    # Add general parameters
    output.append(f"Test Type: {results['alternative']}")
    output.append(f"Alpha: {results['alpha']}")
    
    # Add distribution-specific parameters
    if dist_type == "t":
        output.append(f"Degrees of Freedom: {results['df']}")
    elif dist_type == "f":
        output.append(f"Numerator df (df1): {results['df1']}")
        output.append(f"Denominator df (df2): {results['df2']}")
    elif dist_type == "chi2":
        output.append(f"Degrees of Freedom: {results['df']}")
    
    # Add the power analysis results
    output.append("\nPower Analysis:")
    
    if "power" in results:
        if isinstance(results["power"], (int, float)):
            output.append(f"  Power: {results['power']:.{decimals}f}")
        else:
            output.append(f"  Power: {results['power']}")
    
    if "effect_size" in results:
        if isinstance(results["effect_size"], (int, float)):
            output.append(f"  Effect Size: {results['effect_size']:.{decimals}f}")
        else:
            output.append(f"  Effect Size: {results['effect_size']}")
    
    if "sample_size" in results:
        if isinstance(results["sample_size"], (int, float)):
            output.append(f"  Sample Size: {results['sample_size']}")
        else:
            output.append(f"  Sample Size: {results['sample_size']}")
    
    # Add interpretation
    output.append("\nInterpretation:")
    if "power" in results and isinstance(results["power"], (int, float)):
        if results["power"] < 0.8:
            output.append(f"  The power of {results['power']:.{decimals}f} is less than the commonly recommended level of 0.8.")
            output.append("  This suggests the test may not be sensitive enough to detect meaningful effects.")
            
            if "sample_size" in results and isinstance(results["sample_size"], (int, float)):
                output.append("  Consider increasing the sample size or seeking a larger effect.")
        else:
            output.append(f"  The power of {results['power']:.{decimals}f} is adequate (≥ 0.8).")
            output.append("  This indicates a good chance of detecting the specified effect size.")
    
    return "\n".join(output)


def visualize_power_analysis(
    results: Dict, 
    show_plot: bool = True, 
    figure_size: Tuple[int, int] = (10, 6)
) -> Optional[Figure]:
    """
    Visualize the power analysis results.

    Parameters:
    -----------
    results : dict
        Output dictionary from calculate_power
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
    dist_type = results["distribution_type"]
    alpha = results["alpha"]
    alternative = results["alternative"]
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
    
    # First panel: Power curve vs. effect size or sample size
    if "sample_size" in results and isinstance(results["sample_size"], (int, float)):
        # Plot power vs. effect size
        sample_size = results["sample_size"]
        effect_sizes = np.linspace(0.1, 1.0, 100)
        powers = []
        
        for es in effect_sizes:
            # Calculate power for this effect size
            try:
                if dist_type == "t":
                    temp_results = calculate_power(
                        distribution_type="t",
                        alpha=alpha,
                        effect_size=es,
                        sample_size=sample_size,
                        df=results.get("df"),
                        alternative=alternative
                    )
                elif dist_type == "z":
                    temp_results = calculate_power(
                        distribution_type="z",
                        alpha=alpha,
                        effect_size=es,
                        sample_size=sample_size,
                        alternative=alternative
                    )
                elif dist_type == "f":
                    temp_results = calculate_power(
                        distribution_type="f",
                        alpha=alpha,
                        effect_size=es,
                        sample_size=sample_size,
                        df1=results.get("df1"),
                        df2=results.get("df2"),
                        alternative=alternative
                    )
                elif dist_type == "chi2":
                    temp_results = calculate_power(
                        distribution_type="chi2",
                        alpha=alpha,
                        effect_size=es,
                        sample_size=sample_size,
                        df=results.get("df"),
                        alternative=alternative
                    )
                powers.append(temp_results["power"])
            except:
                powers.append(np.nan)
        
        # Remove any NaN values
        valid_indices = ~np.isnan(powers)
        effect_sizes = effect_sizes[valid_indices]
        powers = np.array(powers)[valid_indices]
        
        # Plot power vs. effect size
        ax1.plot(effect_sizes, powers, 'b-', linewidth=2)
        
        # Add the point for the current effect size and power
        if "power" in results and "effect_size" in results and isinstance(results["power"], (int, float)) and isinstance(results["effect_size"], (int, float)):
            ax1.plot(results["effect_size"], results["power"], 'ro', markersize=8)
            
            # Add a horizontal line at the current power
            ax1.axhline(y=results["power"], color='r', linestyle='--', alpha=0.5)
            
            # Add a vertical line at the current effect size
            ax1.axvline(x=results["effect_size"], color='r', linestyle='--', alpha=0.5)
        
        # Add a horizontal line at power = 0.8
        ax1.axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label="Power = 0.8")
        
        ax1.set_xlabel('Effect Size')
        ax1.set_ylabel('Power')
        ax1.set_title(f'Power vs. Effect Size\n(n={sample_size}, α={alpha})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
    elif "effect_size" in results and isinstance(results["effect_size"], (int, float)):
        # Plot power vs. sample size
        effect_size = results["effect_size"]
        
        # Determine a reasonable range of sample sizes
        if "sample_size" in results and isinstance(results["sample_size"], (int, float)):
            max_n = max(100, int(results["sample_size"] * 2))
        else:
            max_n = 100
            
        sample_sizes = np.arange(5, max_n + 1, 5)
        powers = []
        
        for n in sample_sizes:
            # Calculate power for this sample size
            try:
                if dist_type == "t":
                    temp_results = calculate_power(
                        distribution_type="t",
                        alpha=alpha,
                        effect_size=effect_size,
                        sample_size=n,
                        df=n-1,  # Assuming one-sample or paired t-test
                        alternative=alternative
                    )
                elif dist_type == "z":
                    temp_results = calculate_power(
                        distribution_type="z",
                        alpha=alpha,
                        effect_size=effect_size,
                        sample_size=n,
                        alternative=alternative
                    )
                elif dist_type == "f":
                    temp_results = calculate_power(
                        distribution_type="f",
                        alpha=alpha,
                        effect_size=effect_size,
                        sample_size=n,
                        df1=results.get("df1"),
                        df2=results.get("df2"),
                        alternative=alternative
                    )
                elif dist_type == "chi2":
                    temp_results = calculate_power(
                        distribution_type="chi2",
                        alpha=alpha,
                        effect_size=effect_size,
                        sample_size=n,
                        df=results.get("df"),
                        alternative=alternative
                    )
                powers.append(temp_results["power"])
            except:
                powers.append(np.nan)
                
        # Remove any NaN values
        valid_indices = ~np.isnan(powers)
        sample_sizes = sample_sizes[valid_indices]
        powers = np.array(powers)[valid_indices]
        
        # Plot power vs. sample size
        ax1.plot(sample_sizes, powers, 'b-', linewidth=2)
        
        # Add the point for the current sample size and power
        if "power" in results and "sample_size" in results and isinstance(results["power"], (int, float)) and isinstance(results["sample_size"], (int, float)):
            ax1.plot(results["sample_size"], results["power"], 'ro', markersize=8)
            
            # Add a horizontal line at the current power
            ax1.axhline(y=results["power"], color='r', linestyle='--', alpha=0.5)
            
            # Add a vertical line at the current sample size
            ax1.axvline(x=results["sample_size"], color='r', linestyle='--', alpha=0.5)
        
        # Add a horizontal line at power = 0.8
        ax1.axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label="Power = 0.8")
        
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('Power')
        ax1.set_title(f'Power vs. Sample Size\n(ES={effect_size:.2f}, α={alpha})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Second panel: Distribution with critical values and effect
    if dist_type == "t":
        df = results.get("df")
        
        # For t-distribution
        x = np.linspace(-5, 5, 1000)
        y_null = stats.t.pdf(x, df)
        
        # Plot null distribution
        ax2.plot(x, y_null, 'b-', lw=2, label='Null Distribution')
        
        # Calculate critical values
        if alternative == "two-sided":
            crit_upper = stats.t.ppf(1 - alpha/2, df)
            crit_lower = -crit_upper
            
            # Shade rejection regions
            x_lower = np.linspace(-5, crit_lower, 100)
            y_lower = stats.t.pdf(x_lower, df)
            ax2.fill_between(x_lower, y_lower, alpha=0.3, color='r', label=f"Rejection Region (α/2={alpha/2:.3f})")
            
            x_upper = np.linspace(crit_upper, 5, 100)
            y_upper = stats.t.pdf(x_upper, df)
            ax2.fill_between(x_upper, y_upper, alpha=0.3, color='r')
            
            # Add critical value lines
            ax2.axvline(crit_lower, color='r', linestyle='--')
            ax2.axvline(crit_upper, color='r', linestyle='--')
            
        elif alternative == "greater":
            crit = stats.t.ppf(1 - alpha, df)
            
            # Shade rejection region
            x_upper = np.linspace(crit, 5, 100)
            y_upper = stats.t.pdf(x_upper, df)
            ax2.fill_between(x_upper, y_upper, alpha=0.3, color='r', label=f"Rejection Region (α={alpha:.3f})")
            
            # Add critical value line
            ax2.axvline(crit, color='r', linestyle='--')
            
        else:  # less
            crit = stats.t.ppf(alpha, df)
            
            # Shade rejection region
            x_lower = np.linspace(-5, crit, 100)
            y_lower = stats.t.pdf(x_lower, df)
            ax2.fill_between(x_lower, y_lower, alpha=0.3, color='r', label=f"Rejection Region (α={alpha:.3f})")
            
            # Add critical value line
            ax2.axvline(crit, color='r', linestyle='--')
        
        # Add alternative distribution if effect size is available
        if "effect_size" in results and isinstance(results["effect_size"], (int, float)) and "sample_size" in results and isinstance(results["sample_size"], (int, float)):
            es = results["effect_size"]
            n = results["sample_size"]
            
            # Non-central parameter
            ncp = es * np.sqrt(n)
            
            # Alternative distribution
            y_alt = stats.nct.pdf(x, df, ncp)
            
            # Plot alternative distribution
            ax2.plot(x, y_alt, 'g-', lw=2, label='Alternative Distribution')
            
            # Add effect size annotation
            if alternative == "two-sided":
                # Shade the power region (area under alternative distribution in rejection region)
                x_lower_power = np.linspace(-5, crit_lower, 100)
                y_lower_power = stats.nct.pdf(x_lower_power, df, ncp)
                ax2.fill_between(x_lower_power, y_lower_power, alpha=0.2, color='g')
                
                x_upper_power = np.linspace(crit_upper, 5, 100)
                y_upper_power = stats.nct.pdf(x_upper_power, df, ncp)
                ax2.fill_between(x_upper_power, y_upper_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
                
            elif alternative == "greater":
                # Shade the power region
                x_upper_power = np.linspace(crit, 5, 100)
                y_upper_power = stats.nct.pdf(x_upper_power, df, ncp)
                ax2.fill_between(x_upper_power, y_upper_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
                
            else:  # less
                # Shade the power region
                x_lower_power = np.linspace(-5, crit, 100)
                y_lower_power = stats.nct.pdf(x_lower_power, df, ncp)
                ax2.fill_between(x_lower_power, y_lower_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
        
        ax2.set_title(f't-Distribution (df={df})\nPower Analysis Visualization')
        
    elif dist_type == "z":
        # For standard normal distribution
        x = np.linspace(-5, 5, 1000)
        y_null = stats.norm.pdf(x)
        
        # Plot null distribution
        ax2.plot(x, y_null, 'b-', lw=2, label='Null Distribution')
        
        # Calculate critical values
        if alternative == "two-sided":
            crit_upper = stats.norm.ppf(1 - alpha/2)
            crit_lower = -crit_upper
            
            # Shade rejection regions
            x_lower = np.linspace(-5, crit_lower, 100)
            y_lower = stats.norm.pdf(x_lower)
            ax2.fill_between(x_lower, y_lower, alpha=0.3, color='r', label=f"Rejection Region (α/2={alpha/2:.3f})")
            
            x_upper = np.linspace(crit_upper, 5, 100)
            y_upper = stats.norm.pdf(x_upper)
            ax2.fill_between(x_upper, y_upper, alpha=0.3, color='r')
            
            # Add critical value lines
            ax2.axvline(crit_lower, color='r', linestyle='--')
            ax2.axvline(crit_upper, color='r', linestyle='--')
            
        elif alternative == "greater":
            crit = stats.norm.ppf(1 - alpha)
            
            # Shade rejection region
            x_upper = np.linspace(crit, 5, 100)
            y_upper = stats.norm.pdf(x_upper)
            ax2.fill_between(x_upper, y_upper, alpha=0.3, color='r', label=f"Rejection Region (α={alpha:.3f})")
            
            # Add critical value line
            ax2.axvline(crit, color='r', linestyle='--')
            
        else:  # less
            crit = stats.norm.ppf(alpha)
            
            # Shade rejection region
            x_lower = np.linspace(-5, crit, 100)
            y_lower = stats.norm.pdf(x_lower)
            ax2.fill_between(x_lower, y_lower, alpha=0.3, color='r', label=f"Rejection Region (α={alpha:.3f})")
            
            # Add critical value line
            ax2.axvline(crit, color='r', linestyle='--')
        
        # Add alternative distribution if effect size is available
        if "effect_size" in results and isinstance(results["effect_size"], (int, float)) and "sample_size" in results and isinstance(results["sample_size"], (int, float)):
            es = results["effect_size"]
            n = results["sample_size"]
            
            # Shift for the alternative distribution
            shift = es * np.sqrt(n)
            
            # Alternative distribution
            y_alt = stats.norm.pdf(x - shift)
            
            # Plot alternative distribution
            ax2.plot(x, y_alt, 'g-', lw=2, label='Alternative Distribution')
            
            # Add effect size annotation
            if alternative == "two-sided":
                # Shade the power region (area under alternative distribution in rejection region)
                x_lower_power = np.linspace(-5, crit_lower, 100)
                y_lower_power = stats.norm.pdf(x_lower_power - shift)
                ax2.fill_between(x_lower_power, y_lower_power, alpha=0.2, color='g')
                
                x_upper_power = np.linspace(crit_upper, 5, 100)
                y_upper_power = stats.norm.pdf(x_upper_power - shift)
                ax2.fill_between(x_upper_power, y_upper_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
                
            elif alternative == "greater":
                # Shade the power region
                x_upper_power = np.linspace(crit, 5, 100)
                y_upper_power = stats.norm.pdf(x_upper_power - shift)
                ax2.fill_between(x_upper_power, y_upper_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
                
            else:  # less
                # Shade the power region
                x_lower_power = np.linspace(-5, crit, 100)
                y_lower_power = stats.norm.pdf(x_lower_power - shift)
                ax2.fill_between(x_lower_power, y_lower_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
        
        ax2.set_title('Standard Normal Distribution\nPower Analysis Visualization')
        
    elif dist_type == "f":
        df1 = results.get("df1")
        df2 = results.get("df2")
        
        # For F-distribution
        x_max = 10  # Reasonable upper limit
        x = np.linspace(0.001, x_max, 1000)
        y_null = stats.f.pdf(x, df1, df2)
        
        # Plot null distribution
        ax2.plot(x, y_null, 'b-', lw=2, label='Null Distribution')
        
        # Calculate critical values
        if alternative == "two-sided":
            crit_upper = stats.f.ppf(1 - alpha/2, df1, df2)
            crit_lower = stats.f.ppf(alpha/2, df1, df2)
            
            # Shade rejection regions
            x_lower = np.linspace(0.001, crit_lower, 100)
            y_lower = stats.f.pdf(x_lower, df1, df2)
            ax2.fill_between(x_lower, y_lower, alpha=0.3, color='r', label=f"Rejection Region (α/2={alpha/2:.3f})")
            
            x_upper = np.linspace(crit_upper, x_max, 100)
            y_upper = stats.f.pdf(x_upper, df1, df2)
            ax2.fill_between(x_upper, y_upper, alpha=0.3, color='r')
            
            # Add critical value lines
            ax2.axvline(crit_lower, color='r', linestyle='--')
            ax2.axvline(crit_upper, color='r', linestyle='--')
            
        elif alternative == "greater":
            crit = stats.f.ppf(1 - alpha, df1, df2)
            
            # Shade rejection region
            x_upper = np.linspace(crit, x_max, 100)
            y_upper = stats.f.pdf(x_upper, df1, df2)
            ax2.fill_between(x_upper, y_upper, alpha=0.3, color='r', label=f"Rejection Region (α={alpha:.3f})")
            
            # Add critical value line
            ax2.axvline(crit, color='r', linestyle='--')
            
        else:  # less
            crit = stats.f.ppf(alpha, df1, df2)
            
            # Shade rejection region
            x_lower = np.linspace(0.001, crit, 100)
            y_lower = stats.f.pdf(x_lower, df1, df2)
            ax2.fill_between(x_lower, y_lower, alpha=0.3, color='r', label=f"Rejection Region (α={alpha:.3f})")
            
            # Add critical value line
            ax2.axvline(crit, color='r', linestyle='--')
        
        # Add alternative distribution if effect size is available
        if "effect_size" in results and isinstance(results["effect_size"], (int, float)) and "sample_size" in results and isinstance(results["sample_size"], (int, float)):
            es = results["effect_size"]
            n = results["sample_size"]
            
            # Non-centrality parameter
            ncp = es * n
            
            # Alternative distribution
            y_alt = stats.ncf.pdf(x, df1, df2, ncp)
            
            # Plot alternative distribution
            ax2.plot(x, y_alt, 'g-', lw=2, label='Alternative Distribution')
            
            # Add effect size annotation
            if alternative == "two-sided":
                # Shade the power region (area under alternative distribution in rejection region)
                x_lower_power = np.linspace(0.001, crit_lower, 100)
                y_lower_power = stats.ncf.pdf(x_lower_power, df1, df2, ncp)
                ax2.fill_between(x_lower_power, y_lower_power, alpha=0.2, color='g')
                
                x_upper_power = np.linspace(crit_upper, x_max, 100)
                y_upper_power = stats.ncf.pdf(x_upper_power, df1, df2, ncp)
                ax2.fill_between(x_upper_power, y_upper_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
                
            elif alternative == "greater":
                # Shade the power region
                x_upper_power = np.linspace(crit, x_max, 100)
                y_upper_power = stats.ncf.pdf(x_upper_power, df1, df2, ncp)
                ax2.fill_between(x_upper_power, y_upper_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
                
            else:  # less
                # Shade the power region
                x_lower_power = np.linspace(0.001, crit, 100)
                y_lower_power = stats.ncf.pdf(x_lower_power, df1, df2, ncp)
                ax2.fill_between(x_lower_power, y_lower_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
        
        ax2.set_title(f'F-Distribution (df1={df1}, df2={df2})\nPower Analysis Visualization')
        
    elif dist_type == "chi2":
        df = results.get("df")
        
        # For chi-squared distribution
        x_max = max(20, df * 3)
        x = np.linspace(0.001, x_max, 1000)
        y_null = stats.chi2.pdf(x, df)
        
        # Plot null distribution
        ax2.plot(x, y_null, 'b-', lw=2, label='Null Distribution')
        
        # Calculate critical values
        if alternative == "two-sided":
            crit_upper = stats.chi2.ppf(1 - alpha/2, df)
            crit_lower = stats.chi2.ppf(alpha/2, df)
            
            # Shade rejection regions
            x_lower = np.linspace(0.001, crit_lower, 100)
            y_lower = stats.chi2.pdf(x_lower, df)
            ax2.fill_between(x_lower, y_lower, alpha=0.3, color='r', label=f"Rejection Region (α/2={alpha/2:.3f})")
            
            x_upper = np.linspace(crit_upper, x_max, 100)
            y_upper = stats.chi2.pdf(x_upper, df)
            ax2.fill_between(x_upper, y_upper, alpha=0.3, color='r')
            
            # Add critical value lines
            ax2.axvline(crit_lower, color='r', linestyle='--')
            ax2.axvline(crit_upper, color='r', linestyle='--')
            
        elif alternative == "greater":
            crit = stats.chi2.ppf(1 - alpha, df)
            
            # Shade rejection region
            x_upper = np.linspace(crit, x_max, 100)
            y_upper = stats.chi2.pdf(x_upper, df)
            ax2.fill_between(x_upper, y_upper, alpha=0.3, color='r', label=f"Rejection Region (α={alpha:.3f})")
            
            # Add critical value line
            ax2.axvline(crit, color='r', linestyle='--')
            
        else:  # less
            crit = stats.chi2.ppf(alpha, df)
            
            # Shade rejection region
            x_lower = np.linspace(0.001, crit, 100)
            y_lower = stats.chi2.pdf(x_lower, df)
            ax2.fill_between(x_lower, y_lower, alpha=0.3, color='r', label=f"Rejection Region (α={alpha:.3f})")
            
            # Add critical value line
            ax2.axvline(crit, color='r', linestyle='--')
        
        # Add alternative distribution if effect size is available
        if "effect_size" in results and isinstance(results["effect_size"], (int, float)) and "sample_size" in results and isinstance(results["sample_size"], (int, float)):
            es = results["effect_size"]
            n = results["sample_size"]
            
            # Non-centrality parameter
            ncp = es * n
            
            # Alternative distribution
            y_alt = stats.ncx2.pdf(x, df, ncp)
            
            # Plot alternative distribution
            ax2.plot(x, y_alt, 'g-', lw=2, label='Alternative Distribution')
            
            # Add effect size annotation
            if alternative == "two-sided":
                # Shade the power region (area under alternative distribution in rejection region)
                x_lower_power = np.linspace(0.001, crit_lower, 100)
                y_lower_power = stats.ncx2.pdf(x_lower_power, df, ncp)
                ax2.fill_between(x_lower_power, y_lower_power, alpha=0.2, color='g')
                
                x_upper_power = np.linspace(crit_upper, x_max, 100)
                y_upper_power = stats.ncx2.pdf(x_upper_power, df, ncp)
                ax2.fill_between(x_upper_power, y_upper_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
                
            elif alternative == "greater":
                # Shade the power region
                x_upper_power = np.linspace(crit, x_max, 100)
                y_upper_power = stats.ncx2.pdf(x_upper_power, df, ncp)
                ax2.fill_between(x_upper_power, y_upper_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
                
            else:  # less
                # Shade the power region
                x_lower_power = np.linspace(0.001, crit, 100)
                y_lower_power = stats.ncx2.pdf(x_lower_power, df, ncp)
                ax2.fill_between(x_lower_power, y_lower_power, alpha=0.2, color='g', label=f"Power Region ({results['power']:.3f})")
        
        ax2.set_title(f'Chi-Squared Distribution (df={df})\nPower Analysis Visualization')
    
    ax2.set_xlabel('Test Statistic Value')
    ax2.set_ylabel('Probability Density')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        return None
    else:
        return fig


def analyze_power(
    distribution_type: Literal["f", "t", "chi2", "z"],
    alpha: float = 0.05,
    effect_size: Optional[float] = None,
    sample_size: Optional[int] = None,
    power: Optional[float] = None,
    visualize: bool = True,
    figure_size: Tuple[int, int] = (12, 6),
    **kwargs
) -> Tuple[Dict, Optional[Figure]]:
    """
    Perform power analysis and optional visualization.
    
    Parameters:
    -----------
    distribution_type : str
        Type of distribution: 'f', 't', 'chi2', or 'z'
    alpha : float, optional
        Significance level (default is 0.05)
    effect_size : float, optional
        Standardized effect size (required if calculating power or sample size)
    sample_size : int, optional
        Sample size (required if calculating power or effect size)
    power : float, optional
        Statistical power (required if calculating sample size or effect size)
    visualize : bool, optional
        Whether to create a visualization (default is True)
    figure_size : tuple, optional
        Size of the figure (width, height) in inches (default is (12, 6))
    **kwargs : 
        Additional parameters required for the specific distribution
        
    Returns:
    --------
    tuple
        (results_dict, figure) where:
        - results_dict: Output dictionary from calculate_power
        - figure: matplotlib Figure object or None if visualize is False
        
    Examples:
    --------
    >>> # Calculate and visualize power for a t-test
    >>> results, fig = analyze_power(
    ...     distribution_type="t", 
    ...     alpha=0.05, 
    ...     effect_size=0.5, 
    ...     sample_size=30, 
    ...     alternative="two-sided"
    ... )
    >>> print(format_power_results(results))
    >>> plt.figure(fig.number)
    >>> plt.show()
    """
    # Calculate power analysis results
    results = calculate_power(
        distribution_type=distribution_type,
        alpha=alpha,
        effect_size=effect_size,
        sample_size=sample_size,
        power=power,
        **kwargs
    )
    
    # Create visualization if requested
    fig = None
    if visualize:
        try:
            fig = visualize_power_analysis(results, show_plot=False, figure_size=figure_size)
        except Exception as e:
            print(f"Warning: Could not create power analysis visualization. Error: {e}")
    
    return results, fig


# Example usage and demonstration of the enhanced statistics library
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example 1: F-distribution analysis with multiple alpha levels
    print("Example 1: F-test with Multiple Alpha Levels")
    results_list, fig1 = compare_alpha_levels(
        "f", alpha_levels=[0.01, 0.025, 0.05, 0.1],
        df1=9, df2=9, alternative="two-sided", f_stat=3.754
    )
    print(format_f_results(results_list[2]))  # Show results for alpha=0.05
    plt.figure(fig1.number)
    plt.savefig("f_test_alpha_comparison.png")
    plt.show()
    
    # Example 2: T-test analysis
    print("\nExample 2: One-sample t-test")
    results2, fig2 = analyze_t_distribution(
        df=20, 
        alpha=0.05, 
        alternative="two-sided", 
        t_stat=2.5,
        mean_diff=1.2,
        std_error=0.48
    )
    print(format_t_results(results2))
    plt.figure(fig2.number)
    plt.savefig("t_test_visualization.png")
    plt.show()
    
    # Example 3: Chi-squared goodness of fit test
    print("\nExample 3: Chi-squared Goodness of Fit Test")
    observed = np.array([89, 37, 30, 28, 16])
    expected = np.array([80, 40, 30, 30, 20])
    results3, fig3 = analyze_chi_squared_distribution(
        df=4,
        alpha=0.05,
        alternative="two-sided",
        observed_values=observed,
        expected_values=expected
    )
    print(format_chi_squared_critical_results(results3))
    plt.figure(fig3.number)
    plt.savefig("chi_squared_visualization.png")
    plt.show()
    
    # Example 4: Z-test
    print("\nExample 4: Z-test for Population Mean")
    results4, fig4 = analyze_z_distribution(
        mu_0=3.0,
        x_bar=3.1,
        sigma=0.2,
        n=20,
        alpha=0.05,
        alternative="greater"
    )
    print(format_z_results(results4))
    plt.figure(fig4.number)
    plt.savefig("z_test_visualization.png")
    plt.show()
    
    # Example 5: Power analysis for t-test
    print("\nExample 5: Power Analysis for t-test")
    power_results, power_fig = analyze_power(
        distribution_type="t",
        alpha=0.05,
        effect_size=0.5,
        sample_size=30,
        alternative="two-sided",
        df=29
    )
    print(format_power_results(power_results))
    plt.figure(power_fig.number)
    plt.savefig("t_test_power_analysis.png")
    plt.show()
    
    # Example 6: Sample size calculation for desired power
    print("\nExample 6: Sample Size Calculation for Z-test")
    sample_size_results, sample_size_fig = analyze_power(
        distribution_type="z",
        alpha=0.05,
        effect_size=0.3,
        power=0.8,
        alternative="two-sided"
    )
    print(format_power_results(sample_size_results))
    plt.figure(sample_size_fig.number)
    plt.savefig("z_test_sample_size_calculation.png")
    plt.show()
    