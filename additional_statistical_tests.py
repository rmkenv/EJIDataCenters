#!/usr/bin/env python3
"""
Additional Statistical Tests and Advanced Analysis
Supplementary analysis for the Data Center Environmental Justice study
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import normaltest, levene, bartlett, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_prepare_data():
    """Load and prepare data for additional tests"""
    df = pd.read_csv('/home/ubuntu/Uploads/us_datacenters_with_eji.csv')
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace(-999, np.nan)
    
    # Create additional categorical variables
    df['EJI_Tertile'] = pd.qcut(df['RPL_EJI'], q=3, labels=['Low', 'Medium', 'High'])
    df['SVM_High'] = (df['RPL_SVM'] > df['RPL_SVM'].median()).astype(int)
    df['EBM_High'] = (df['RPL_EBM'] > df['RPL_EBM'].median()).astype(int)
    
    # Company size categories
    company_counts = df['company_name'].value_counts()
    df['Company_Size'] = df['company_name'].map(company_counts)
    df['Company_Category'] = pd.cut(df['Company_Size'], 
                                   bins=[0, 5, 15, float('inf')], 
                                   labels=['Small', 'Medium', 'Large'])
    
    return df

def normality_tests(df):
    """Test normality of key variables"""
    print("=== NORMALITY TESTS ===")
    
    key_vars = ['RPL_EJI', 'RPL_SVM', 'RPL_EBM', 'RPL_HVM', 'RPL_CBM']
    normality_results = {}
    
    for var in key_vars:
        data = df[var].dropna()
        
        # Shapiro-Wilk test (for smaller samples)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = shapiro(data)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # D'Agostino's normality test
        dagostino_stat, dagostino_p = normaltest(data)
        
        normality_results[var] = {
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'dagostino_statistic': dagostino_stat,
            'dagostino_p_value': dagostino_p,
            'normal_shapiro': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None,
            'normal_dagostino': dagostino_p > 0.05
        }
        
        print(f"\n{var}:")
        print(f"  D'Agostino: stat={dagostino_stat:.4f}, p={dagostino_p:.6f}, Normal={dagostino_p > 0.05}")
        if not np.isnan(shapiro_p):
            print(f"  Shapiro-Wilk: stat={shapiro_stat:.4f}, p={shapiro_p:.6f}, Normal={shapiro_p > 0.05}")
    
    return normality_results

def homogeneity_tests(df):
    """Test homogeneity of variances"""
    print("\n=== HOMOGENEITY OF VARIANCE TESTS ===")
    
    # Test variance equality across EJI tertiles
    groups = [group['RPL_EJI'].dropna().values for name, group in df.groupby('EJI_Tertile')]
    groups = [g for g in groups if len(g) > 0]
    
    # Levene's test
    levene_stat, levene_p = levene(*groups)
    
    # Bartlett's test
    bartlett_stat, bartlett_p = bartlett(*groups)
    
    print(f"Levene's test: stat={levene_stat:.4f}, p={levene_p:.6f}, Equal variances={levene_p > 0.05}")
    print(f"Bartlett's test: stat={bartlett_stat:.4f}, p={bartlett_p:.6f}, Equal variances={bartlett_p > 0.05}")
    
    return {
        'levene': {'statistic': levene_stat, 'p_value': levene_p},
        'bartlett': {'statistic': bartlett_stat, 'p_value': bartlett_p}
    }

def effect_size_analysis(df):
    """Calculate effect sizes for key relationships"""
    print("\n=== EFFECT SIZE ANALYSIS ===")
    
    # Cohen's d for EJI differences between high/low SVM
    high_svm = df[df['SVM_High'] == 1]['RPL_EJI'].dropna()
    low_svm = df[df['SVM_High'] == 0]['RPL_EJI'].dropna()
    
    # Cohen's d calculation
    pooled_std = np.sqrt(((len(high_svm) - 1) * high_svm.var() + 
                         (len(low_svm) - 1) * low_svm.var()) / 
                        (len(high_svm) + len(low_svm) - 2))
    cohens_d = (high_svm.mean() - low_svm.mean()) / pooled_std
    
    # Eta squared for ANOVA
    # EJI across company categories
    company_groups = [group['RPL_EJI'].dropna().values 
                     for name, group in df.groupby('Company_Category')]
    company_groups = [g for g in company_groups if len(g) > 0]
    
    if len(company_groups) > 1:
        f_stat, p_val = stats.f_oneway(*company_groups)
        
        # Calculate eta squared
        ss_between = sum([len(g) * (np.mean(g) - df['RPL_EJI'].mean())**2 for g in company_groups])
        ss_total = sum([(x - df['RPL_EJI'].mean())**2 for g in company_groups for x in g])
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
    else:
        eta_squared = 0
    
    print(f"Cohen's d (High vs Low SVM): {cohens_d:.4f}")
    print(f"Eta squared (Company Category): {eta_squared:.4f}")
    
    # Effect size interpretation
    def interpret_cohens_d(d):
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
    
    def interpret_eta_squared(eta):
        if eta < 0.01:
            return "negligible"
        elif eta < 0.06:
            return "small"
        elif eta < 0.14:
            return "medium"
        else:
            return "large"
    
    print(f"Cohen's d interpretation: {interpret_cohens_d(cohens_d)}")
    print(f"Eta squared interpretation: {interpret_eta_squared(eta_squared)}")
    
    return {
        'cohens_d': cohens_d,
        'eta_squared': eta_squared,
        'cohens_d_interpretation': interpret_cohens_d(cohens_d),
        'eta_squared_interpretation': interpret_eta_squared(eta_squared)
    }

def advanced_correlation_analysis(df):
    """Advanced correlation analysis with confidence intervals"""
    print("\n=== ADVANCED CORRELATION ANALYSIS ===")
    
    # Key variables for correlation
    vars_of_interest = ['RPL_EJI', 'RPL_SVM', 'RPL_EBM', 'RPL_HVM', 'EPL_POV200', 
                       'EPL_MINRTY', 'EPL_OZONE', 'EPL_PM']
    
    correlation_results = {}
    
    for i, var1 in enumerate(vars_of_interest):
        for var2 in vars_of_interest[i+1:]:
            data1 = df[var1].dropna()
            data2 = df[var2].dropna()
            
            # Find common indices
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) > 10:
                x = df.loc[common_idx, var1]
                y = df.loc[common_idx, var2]
                
                # Pearson correlation with confidence interval
                r, p = stats.pearsonr(x, y)
                
                # Bootstrap confidence interval for correlation
                n_bootstrap = 1000
                bootstrap_rs = []
                
                for _ in range(n_bootstrap):
                    indices = np.random.choice(len(x), len(x), replace=True)
                    boot_r, _ = stats.pearsonr(x.iloc[indices], y.iloc[indices])
                    bootstrap_rs.append(boot_r)
                
                ci_lower = np.percentile(bootstrap_rs, 2.5)
                ci_upper = np.percentile(bootstrap_rs, 97.5)
                
                correlation_results[f"{var1}_vs_{var2}"] = {
                    'correlation': r,
                    'p_value': p,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'significant': p < 0.05
                }
    
    # Display top correlations
    sorted_corrs = sorted(correlation_results.items(), 
                         key=lambda x: abs(x[1]['correlation']), reverse=True)
    
    print("Top 10 Correlations with 95% Confidence Intervals:")
    for pair, results in sorted_corrs[:10]:
        print(f"{pair}: r={results['correlation']:.4f} "
              f"[{results['ci_lower']:.4f}, {results['ci_upper']:.4f}], "
              f"p={results['p_value']:.6f}")
    
    return correlation_results

def power_analysis(df):
    """Statistical power analysis"""
    print("\n=== POWER ANALYSIS ===")
    
    # Sample size for current study
    n = len(df.dropna(subset=['RPL_EJI']))
    
    # Effect size from correlation between EJI and SVM
    r_eji_svm = df['RPL_EJI'].corr(df['RPL_SVM'])
    
    # Convert correlation to Cohen's f²
    r_squared = r_eji_svm ** 2
    cohens_f2 = r_squared / (1 - r_squared)
    
    # Approximate power calculation for correlation
    # Using Fisher's z-transformation
    z_r = 0.5 * np.log((1 + r_eji_svm) / (1 - r_eji_svm))
    se_z = 1 / np.sqrt(n - 3)
    
    # Critical value for α = 0.05, two-tailed
    z_critical = 1.96
    
    # Power approximation
    z_beta = abs(z_r) / se_z - z_critical
    power = stats.norm.cdf(z_beta)
    
    print(f"Sample size: {n}")
    print(f"Observed correlation (EJI-SVM): {r_eji_svm:.4f}")
    print(f"Cohen's f²: {cohens_f2:.4f}")
    print(f"Estimated power: {power:.4f}")
    
    return {
        'sample_size': n,
        'observed_correlation': r_eji_svm,
        'cohens_f2': cohens_f2,
        'estimated_power': power
    }

def create_advanced_visualizations(df):
    """Create advanced statistical visualizations"""
    print("\n=== CREATING ADVANCED VISUALIZATIONS ===")
    
    # 1. Q-Q plots for normality assessment
    fig_qq = make_subplots(
        rows=2, cols=3,
        subplot_titles=['EJI Q-Q Plot', 'SVM Q-Q Plot', 'EBM Q-Q Plot',
                       'HVM Q-Q Plot', 'CBM Q-Q Plot', 'Residuals Q-Q Plot']
    )
    
    vars_to_plot = ['RPL_EJI', 'RPL_SVM', 'RPL_EBM', 'RPL_HVM', 'RPL_CBM']
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for i, var in enumerate(vars_to_plot):
        data = df[var].dropna()
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
        sample_quantiles = np.sort(data)
        
        row, col = positions[i]
        fig_qq.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles, 
                      mode='markers', name=var, showlegend=False),
            row=row, col=col
        )
        
        # Add reference line
        min_val, max_val = min(theoretical_quantiles), max(theoretical_quantiles)
        fig_qq.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', line=dict(color='red', dash='dash'),
                      showlegend=False),
            row=row, col=col
        )
    
    fig_qq.update_layout(height=800, title_text="Q-Q Plots for Normality Assessment")
    fig_qq.write_html("/home/ubuntu/qq_plots.html")
    
    # 2. Box plots by categories
    fig_box = make_subplots(
        rows=2, cols=2,
        subplot_titles=['EJI by Company Category', 'EJI by EJI Tertile',
                       'SVM by Company Category', 'EBM by Company Category']
    )
    
    # EJI by Company Category
    for category in df['Company_Category'].dropna().unique():
        data = df[df['Company_Category'] == category]['RPL_EJI'].dropna()
        fig_box.add_trace(
            go.Box(y=data, name=category, showlegend=False),
            row=1, col=1
        )
    
    # EJI by EJI Tertile
    for tertile in df['EJI_Tertile'].dropna().unique():
        data = df[df['EJI_Tertile'] == tertile]['RPL_EJI'].dropna()
        fig_box.add_trace(
            go.Box(y=data, name=tertile, showlegend=False),
            row=1, col=2
        )
    
    fig_box.update_layout(height=800, title_text="Distribution Analysis by Categories")
    fig_box.write_html("/home/ubuntu/box_plots.html")
    
    # 3. Correlation network plot
    correlation_matrix = df[vars_to_plot].corr()
    
    # Create network-style correlation plot
    fig_network = go.Figure()
    
    # Add correlation strength as edge weights
    threshold = 0.3  # Only show correlations above this threshold
    
    for i, var1 in enumerate(vars_to_plot):
        for j, var2 in enumerate(vars_to_plot):
            if i < j and abs(correlation_matrix.loc[var1, var2]) > threshold:
                corr_val = correlation_matrix.loc[var1, var2]
                
                # Position variables in a circle
                angle1 = 2 * np.pi * i / len(vars_to_plot)
                angle2 = 2 * np.pi * j / len(vars_to_plot)
                
                x1, y1 = np.cos(angle1), np.sin(angle1)
                x2, y2 = np.cos(angle2), np.sin(angle2)
                
                # Add edge
                fig_network.add_trace(
                    go.Scatter(x=[x1, x2], y=[y1, y2],
                              mode='lines',
                              line=dict(width=abs(corr_val)*10,
                                       color='red' if corr_val > 0 else 'blue'),
                              showlegend=False,
                              hovertemplate=f'{var1} - {var2}: {corr_val:.3f}')
                )
    
    # Add nodes
    for i, var in enumerate(vars_to_plot):
        angle = 2 * np.pi * i / len(vars_to_plot)
        x, y = np.cos(angle), np.sin(angle)
        
        fig_network.add_trace(
            go.Scatter(x=[x], y=[y],
                      mode='markers+text',
                      marker=dict(size=20, color='lightblue'),
                      text=var,
                      textposition='middle center',
                      showlegend=False)
        )
    
    fig_network.update_layout(
        title="Correlation Network (|r| > 0.3)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=800, height=800
    )
    fig_network.write_html("/home/ubuntu/correlation_network.html")
    
    print("Advanced visualizations created:")
    print("- qq_plots.html")
    print("- box_plots.html") 
    print("- correlation_network.html")

def main():
    """Run all additional statistical tests"""
    print("ADDITIONAL STATISTICAL TESTS AND ADVANCED ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Run all tests
    normality_results = normality_tests(df)
    homogeneity_results = homogeneity_tests(df)
    effect_size_results = effect_size_analysis(df)
    correlation_results = advanced_correlation_analysis(df)
    power_results = power_analysis(df)
    create_advanced_visualizations(df)
    
    # Save results
    import json
    
    all_results = {
        'normality_tests': normality_results,
        'homogeneity_tests': homogeneity_results,
        'effect_sizes': effect_size_results,
        'power_analysis': power_results,
        'sample_info': {
            'total_datacenters': len(df),
            'complete_eji_cases': len(df.dropna(subset=['RPL_EJI'])),
            'companies': df['company_name'].nunique(),
            'states': df['GEOID'].astype(str).str[:2].nunique()
        }
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    json_results = {}
    for key, value in all_results.items():
        if isinstance(value, dict):
            json_results[key] = {k: convert_numpy(v) for k, v in value.items()}
        else:
            json_results[key] = convert_numpy(value)
    
    with open('/home/ubuntu/additional_statistical_results.json', 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSIS COMPLETE")
    print("Generated files:")
    print("- additional_statistical_results.json")
    print("- qq_plots.html")
    print("- box_plots.html")
    print("- correlation_network.html")
    print("=" * 60)

if __name__ == "__main__":
    main()
