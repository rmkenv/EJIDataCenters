#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of US Data Centers and Environmental Justice Index (EJI)
A rigorous scientific assessment examining relationships between data center locations 
and environmental justice indicators.

Author: AI Research Assistant
Date: June 19, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, mannwhitneyu, kruskal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataCenterEJIAnalysis:
    """
    Comprehensive analysis class for data center and environmental justice relationships
    """
    
    def __init__(self, data_path):
        """Initialize with data loading and preprocessing"""
        self.df = pd.read_csv(data_path)
        self.results = {}
        self.figures = {}
        self.preprocess_data()
        
    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        print("=== DATA PREPROCESSING ===")
        
        # Basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Data centers: {len(self.df)}")
        
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].replace(-999, np.nan)
        
        # Create categorical variables for analysis
        self.df['EJI_Quartile'] = pd.qcut(self.df['RPL_EJI'], q=4, labels=['Q1_Low', 'Q2_Med_Low', 'Q3_Med_High', 'Q4_High'])
        self.df['SVM_Quartile'] = pd.qcut(self.df['RPL_SVM'], q=4, labels=['Q1_Low', 'Q2_Med_Low', 'Q3_Med_High', 'Q4_High'])
        self.df['EBM_Quartile'] = pd.qcut(self.df['RPL_EBM'], q=4, labels=['Q1_Low', 'Q2_Med_Low', 'Q3_Med_High', 'Q4_High'])
        
        # Define key variable groups
        self.eji_vars = ['RPL_EJI', 'RPL_SER', 'RPL_EJI_CBM']
        self.svm_vars = ['RPL_SVM', 'EPL_MINRTY', 'EPL_POV200', 'EPL_NOHSDP', 'EPL_UNEMP', 
                        'EPL_RENTER', 'EPL_HOUBDN', 'EPL_UNINSUR', 'EPL_NOINT', 'EPL_AGE65', 
                        'EPL_AGE17', 'EPL_DISABL', 'EPL_LIMENG', 'EPL_MOBILE', 'EPL_GROUPQ']
        self.ebm_vars = ['RPL_EBM', 'EPL_OZONE', 'EPL_PM', 'EPL_DSLPM', 'EPL_TOTCR', 
                        'EPL_NPL', 'EPL_TRI', 'EPL_TSD', 'EPL_RMP', 'EPL_COAL', 'EPL_LEAD']
        self.hvm_vars = ['RPL_HVM', 'EPL_ASTHMA', 'EPL_CANCER', 'EPL_CHD', 'EPL_MHLTH', 'EPL_DIABETES']
        self.cbm_vars = ['RPL_CBM', 'EPL_BURN', 'EPL_SMOKE', 'EPL_CFLD', 'EPL_DRGT', 
                        'EPL_HRCN', 'EPL_RFLD', 'EPL_SWND', 'EPL_TRND']
        
        print("Data preprocessing completed.")
        
    def exploratory_analysis(self):
        """Comprehensive exploratory data analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        desc_stats = self.df[self.eji_vars + ['longitude', 'latitude']].describe()
        print("\nDescriptive Statistics for Key Variables:")
        print(desc_stats)
        
        # Geographic distribution
        # Filter out NaN values for plotting
        plot_df = self.df.dropna(subset=['RPL_EJI', 'latitude', 'longitude'])
        
        fig_geo = px.scatter_mapbox(
            plot_df, 
            lat="latitude", 
            lon="longitude",
            color="RPL_EJI",
            size="RPL_EJI",
            hover_name="name",
            hover_data=["company_name", "RPL_SVM", "RPL_EBM"],
            color_continuous_scale="Viridis",
            size_max=15,
            zoom=3,
            mapbox_style="open-street-map",
            title="Geographic Distribution of Data Centers by Environmental Justice Index",
            width=1200,
            height=800
        )
        fig_geo.write_html("/home/ubuntu/datacenter_geographic_distribution.html")
        self.figures['geographic'] = fig_geo
        
        # Distribution plots
        fig_dist = make_subplots(
            rows=2, cols=3,
            subplot_titles=['EJI Overall', 'Social Vulnerability', 'Environmental Burden', 
                          'Health Vulnerability', 'Climate Burden', 'EJI vs SVM'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add histograms
        fig_dist.add_trace(go.Histogram(x=self.df['RPL_EJI'], name='EJI', nbinsx=30), row=1, col=1)
        fig_dist.add_trace(go.Histogram(x=self.df['RPL_SVM'], name='SVM', nbinsx=30), row=1, col=2)
        fig_dist.add_trace(go.Histogram(x=self.df['RPL_EBM'], name='EBM', nbinsx=30), row=1, col=3)
        fig_dist.add_trace(go.Histogram(x=self.df['RPL_HVM'], name='HVM', nbinsx=30), row=2, col=1)
        fig_dist.add_trace(go.Histogram(x=self.df['RPL_CBM'], name='CBM', nbinsx=30), row=2, col=2)
        fig_dist.add_trace(go.Scatter(x=self.df['RPL_EJI'], y=self.df['RPL_SVM'], 
                                    mode='markers', name='EJI vs SVM'), row=2, col=3)
        
        fig_dist.update_layout(height=800, title_text="Distribution of Environmental Justice Indicators")
        fig_dist.write_html("/home/ubuntu/eji_distributions.html")
        self.figures['distributions'] = fig_dist
        
        return desc_stats
    
    def research_hypotheses(self):
        """Define and test multiple research hypotheses"""
        print("\n=== RESEARCH HYPOTHESES TESTING ===")
        
        hypotheses = {
            'H1': 'Data centers are disproportionately located in areas with higher environmental justice burdens',
            'H2': 'Data centers are more likely to be in socially vulnerable communities',
            'H3': 'Data centers correlate with higher environmental burden indicators',
            'H4': 'Data center density varies by geographic region and EJI levels',
            'H5': 'Company type influences environmental justice impact patterns',
            'H6': 'Data center size correlates with environmental justice indicators'
        }
        
        print("Research Hypotheses:")
        for h, desc in hypotheses.items():
            print(f"{h}: {desc}")
        
        # Test H1: Overall EJI burden
        us_median_eji = 0.5  # National median percentile
        dc_median_eji = self.df['RPL_EJI'].median()
        dc_mean_eji = self.df['RPL_EJI'].mean()
        
        # One-sample t-test against national median
        t_stat, p_val = stats.ttest_1samp(self.df['RPL_EJI'].dropna(), us_median_eji)
        
        print(f"\nH1 Testing:")
        print(f"US Median EJI: {us_median_eji}")
        print(f"Data Center Median EJI: {dc_median_eji:.4f}")
        print(f"Data Center Mean EJI: {dc_mean_eji:.4f}")
        print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.6f}")
        
        self.results['H1'] = {
            'hypothesis': hypotheses['H1'],
            'dc_median': dc_median_eji,
            'dc_mean': dc_mean_eji,
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
        
        return hypotheses
    
    def correlation_analysis(self):
        """Comprehensive correlation analysis"""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Select numeric variables for correlation
        numeric_vars = self.eji_vars + self.svm_vars + self.ebm_vars + self.hvm_vars + self.cbm_vars
        corr_data = self.df[numeric_vars].dropna()
        
        # Pearson correlations
        pearson_corr = corr_data.corr(method='pearson')
        
        # Spearman correlations (non-parametric)
        spearman_corr = corr_data.corr(method='spearman')
        
        # Create correlation heatmap
        fig_corr = px.imshow(
            pearson_corr,
            title="Pearson Correlation Matrix: Environmental Justice Indicators",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            width=1200,
            height=1000
        )
        fig_corr.write_html("/home/ubuntu/correlation_matrix.html")
        self.figures['correlation'] = fig_corr
        
        # Key correlations with EJI
        eji_correlations = []
        for var in numeric_vars:
            if var != 'RPL_EJI':
                r, p = pearsonr(corr_data['RPL_EJI'], corr_data[var])
                eji_correlations.append({
                    'variable': var,
                    'correlation': r,
                    'p_value': p,
                    'significant': p < 0.05
                })
        
        eji_corr_df = pd.DataFrame(eji_correlations).sort_values('correlation', key=abs, ascending=False)
        print("\nTop 10 Correlations with Overall EJI:")
        print(eji_corr_df.head(10))
        
        self.results['correlations'] = {
            'pearson_matrix': pearson_corr,
            'spearman_matrix': spearman_corr,
            'eji_correlations': eji_corr_df
        }
        
        return eji_corr_df
    
    def statistical_tests(self):
        """Comprehensive statistical significance testing"""
        print("\n=== STATISTICAL SIGNIFICANCE TESTING ===")
        
        test_results = {}
        
        # 1. ANOVA: EJI differences across quartiles
        quartile_groups = [group['RPL_EJI'].values for name, group in self.df.groupby('EJI_Quartile')]
        f_stat, p_val = stats.f_oneway(*quartile_groups)
        test_results['anova_eji_quartiles'] = {'f_statistic': f_stat, 'p_value': p_val}
        
        # 2. Kruskal-Wallis test (non-parametric alternative)
        h_stat, p_val_kw = stats.kruskal(*quartile_groups)
        test_results['kruskal_wallis'] = {'h_statistic': h_stat, 'p_value': p_val_kw}
        
        # 3. Mann-Whitney U tests for pairwise comparisons
        companies = self.df['company_name'].value_counts().head(5).index
        company_tests = {}
        
        for i, comp1 in enumerate(companies):
            for comp2 in companies[i+1:]:
                group1 = self.df[self.df['company_name'] == comp1]['RPL_EJI'].dropna()
                group2 = self.df[self.df['company_name'] == comp2]['RPL_EJI'].dropna()
                
                if len(group1) > 0 and len(group2) > 0:
                    u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                    company_tests[f"{comp1}_vs_{comp2}"] = {
                        'u_statistic': u_stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05
                    }
        
        test_results['company_comparisons'] = company_tests
        
        # 4. Chi-square test for categorical associations
        # Create high/low EJI categories
        self.df['EJI_High'] = (self.df['RPL_EJI'] > self.df['RPL_EJI'].median()).astype(int)
        
        # Test association with top companies
        top_companies = self.df['company_name'].value_counts().head(5).index
        company_eji_crosstab = pd.crosstab(
            self.df[self.df['company_name'].isin(top_companies)]['company_name'],
            self.df[self.df['company_name'].isin(top_companies)]['EJI_High']
        )
        
        chi2, p_val_chi2, dof, expected = chi2_contingency(company_eji_crosstab)
        test_results['chi_square_company_eji'] = {
            'chi2_statistic': chi2,
            'p_value': p_val_chi2,
            'degrees_of_freedom': dof,
            'crosstab': company_eji_crosstab
        }
        
        print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_val:.6f}")
        print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}, p-value: {p_val_kw:.6f}")
        print(f"Chi-square statistic: {chi2:.4f}, p-value: {p_val_chi2:.6f}")
        
        self.results['statistical_tests'] = test_results
        return test_results
    
    def regression_modeling(self):
        """Multiple regression modeling approaches"""
        print("\n=== REGRESSION MODELING ===")
        
        # Prepare data for modeling
        feature_vars = self.svm_vars + self.ebm_vars + self.hvm_vars + self.cbm_vars
        feature_vars = [var for var in feature_vars if var in self.df.columns]
        
        # Remove target variable from features
        feature_vars = [var for var in feature_vars if var != 'RPL_EJI']
        
        # Create modeling dataset
        model_data = self.df[feature_vars + ['RPL_EJI']].dropna()
        X = model_data[feature_vars]
        y = model_data['RPL_EJI']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        
        models['linear_regression'] = {
            'model': lr,
            'r2_score': lr_r2,
            'rmse': lr_rmse,
            'coefficients': dict(zip(feature_vars, lr.coef_))
        }
        
        # 2. Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        ridge_pred = ridge.predict(X_test_scaled)
        ridge_r2 = r2_score(y_test, ridge_pred)
        ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
        
        models['ridge_regression'] = {
            'model': ridge,
            'r2_score': ridge_r2,
            'rmse': ridge_rmse,
            'coefficients': dict(zip(feature_vars, ridge.coef_))
        }
        
        # 3. Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        models['random_forest'] = {
            'model': rf,
            'r2_score': rf_r2,
            'rmse': rf_rmse,
            'feature_importance': dict(zip(feature_vars, rf.feature_importances_))
        }
        
        # Feature importance plot
        importance_df = pd.DataFrame({
            'feature': feature_vars,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df.tail(15),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Feature Importances (Random Forest)',
            labels={'importance': 'Feature Importance', 'feature': 'Environmental Justice Indicators'}
        )
        fig_importance.write_html("/home/ubuntu/feature_importance.html")
        self.figures['feature_importance'] = fig_importance
        
        print(f"Linear Regression R²: {lr_r2:.4f}, RMSE: {lr_rmse:.4f}")
        print(f"Ridge Regression R²: {ridge_r2:.4f}, RMSE: {ridge_rmse:.4f}")
        print(f"Random Forest R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4f}")
        
        self.results['regression_models'] = models
        return models
    
    def clustering_analysis(self):
        """K-means clustering analysis"""
        print("\n=== K-MEANS CLUSTERING ANALYSIS ===")
        
        # Prepare clustering data
        cluster_vars = ['RPL_EJI', 'RPL_SVM', 'RPL_EBM', 'RPL_HVM', 'RPL_CBM']
        cluster_data = self.df[cluster_vars].dropna()
        
        # Scale data
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(cluster_data_scaled)
            inertias.append(kmeans.inertia_)
        
        # Elbow plot
        fig_elbow = px.line(
            x=list(k_range),
            y=inertias,
            title='Elbow Method for Optimal Number of Clusters',
            labels={'x': 'Number of Clusters (k)', 'y': 'Inertia'}
        )
        fig_elbow.write_html("/home/ubuntu/elbow_plot.html")
        
        # Perform clustering with optimal k (let's use k=4)
        optimal_k = 4
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
        
        # Add cluster labels to dataframe
        cluster_df = cluster_data.copy()
        cluster_df['Cluster'] = cluster_labels
        
        # Cluster characteristics
        cluster_summary = cluster_df.groupby('Cluster')[cluster_vars].mean()
        print("\nCluster Characteristics (Mean Values):")
        print(cluster_summary)
        
        # 3D visualization
        fig_3d = px.scatter_3d(
            cluster_df,
            x='RPL_EJI',
            y='RPL_SVM',
            z='RPL_EBM',
            color='Cluster',
            title='3D Clustering of Data Centers by Environmental Justice Indicators',
            labels={
                'RPL_EJI': 'Overall EJI Percentile',
                'RPL_SVM': 'Social Vulnerability Percentile',
                'RPL_EBM': 'Environmental Burden Percentile'
            }
        )
        fig_3d.write_html("/home/ubuntu/3d_clusters.html")
        self.figures['clustering_3d'] = fig_3d
        
        # Geographic distribution of clusters
        cluster_geo_df = self.df.loc[cluster_data.index].copy()
        cluster_geo_df['Cluster'] = cluster_labels
        
        fig_cluster_geo = px.scatter_mapbox(
            cluster_geo_df,
            lat="latitude",
            lon="longitude",
            color="Cluster",
            hover_name="name",
            hover_data=["company_name", "RPL_EJI", "RPL_SVM"],
            title="Geographic Distribution of Data Center Clusters",
            mapbox_style="open-street-map",
            zoom=3,
            width=1200,
            height=800
        )
        fig_cluster_geo.write_html("/home/ubuntu/cluster_geographic.html")
        self.figures['cluster_geographic'] = fig_cluster_geo
        
        self.results['clustering'] = {
            'optimal_k': optimal_k,
            'cluster_labels': cluster_labels,
            'cluster_summary': cluster_summary,
            'inertias': inertias
        }
        
        return cluster_summary
    
    def geographic_analysis(self):
        """Geographic and spatial analysis"""
        print("\n=== GEOGRAPHIC ANALYSIS ===")
        
        # State-level analysis
        self.df['State'] = self.df['GEOID'].astype(str).str[:2]
        state_analysis = self.df.groupby('State').agg({
            'RPL_EJI': ['count', 'mean', 'median', 'std'],
            'RPL_SVM': 'mean',
            'RPL_EBM': 'mean',
            'latitude': 'mean',
            'longitude': 'mean'
        }).round(4)
        
        state_analysis.columns = ['_'.join(col).strip() for col in state_analysis.columns]
        state_analysis = state_analysis.sort_values('RPL_EJI_count', ascending=False)
        
        print("Top 10 States by Data Center Count:")
        print(state_analysis.head(10))
        
        # Regional patterns
        # Define regions based on longitude
        self.df['Region'] = pd.cut(
            self.df['longitude'],
            bins=[-180, -100, -80, 180],
            labels=['West', 'Central', 'East']
        )
        
        regional_analysis = self.df.groupby('Region').agg({
            'RPL_EJI': ['count', 'mean', 'std'],
            'RPL_SVM': 'mean',
            'RPL_EBM': 'mean',
            'RPL_HVM': 'mean'
        }).round(4)
        
        print("\nRegional Analysis:")
        print(regional_analysis)
        
        # Test for regional differences
        regional_groups = [group['RPL_EJI'].values for name, group in self.df.groupby('Region')]
        f_stat, p_val = stats.f_oneway(*regional_groups)
        
        print(f"\nRegional ANOVA - F-statistic: {f_stat:.4f}, p-value: {p_val:.6f}")
        
        self.results['geographic'] = {
            'state_analysis': state_analysis,
            'regional_analysis': regional_analysis,
            'regional_anova': {'f_statistic': f_stat, 'p_value': p_val}
        }
        
        return state_analysis
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n=== COMPREHENSIVE ANALYSIS SUMMARY ===")
        
        summary = {
            'dataset_info': {
                'total_datacenters': len(self.df),
                'unique_companies': self.df['company_name'].nunique(),
                'geographic_coverage': f"{self.df['State'].nunique()} states",
                'analysis_date': '2025-06-19'
            },
            'key_findings': {},
            'statistical_significance': {},
            'recommendations': []
        }
        
        # Key findings
        summary['key_findings'] = {
            'mean_eji': self.df['RPL_EJI'].mean(),
            'median_eji': self.df['RPL_EJI'].median(),
            'eji_above_national_median': (self.df['RPL_EJI'] > 0.5).mean(),
            'highest_correlation_with_eji': self.results['correlations']['eji_correlations'].iloc[0]['variable'],
            'best_model_r2': max([model['r2_score'] for model in self.results['regression_models'].values()]),
            'optimal_clusters': self.results['clustering']['optimal_k']
        }
        
        # Statistical significance summary
        if 'H1' in self.results:
            summary['statistical_significance']['h1_significant'] = self.results['H1']['significant']
        
        # Recommendations
        summary['recommendations'] = [
            "Implement environmental justice screening for new data center locations",
            "Prioritize community engagement in high EJI areas",
            "Develop mitigation strategies for environmental burden reduction",
            "Consider social vulnerability factors in site selection",
            "Establish monitoring programs for health and environmental impacts"
        ]
        
        self.results['summary'] = summary
        
        # Save comprehensive results
        import json
        with open('/home/ubuntu/analysis_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                return obj
            
            json_results = {}
            for key, value in self.results.items():
                if key not in ['correlations']:  # Skip large matrices
                    json_results[key] = convert_numpy(value)
            
            json.dump(json_results, f, indent=2, default=str)
        
        print("Analysis Summary:")
        print(f"Total Data Centers Analyzed: {summary['dataset_info']['total_datacenters']}")
        print(f"Mean EJI Percentile: {summary['key_findings']['mean_eji']:.4f}")
        print(f"Percentage Above National Median: {summary['key_findings']['eji_above_national_median']:.2%}")
        print(f"Best Model R²: {summary['key_findings']['best_model_r2']:.4f}")
        
        return summary

def main():
    """Main analysis execution"""
    print("COMPREHENSIVE STATISTICAL ANALYSIS OF US DATA CENTERS AND ENVIRONMENTAL JUSTICE INDEX")
    print("=" * 90)
    
    # Initialize analysis
    analyzer = DataCenterEJIAnalysis('/home/ubuntu/Uploads/us_datacenters_with_eji.csv')
    
    # Execute all analyses
    analyzer.exploratory_analysis()
    analyzer.research_hypotheses()
    analyzer.correlation_analysis()
    analyzer.statistical_tests()
    analyzer.regression_modeling()
    analyzer.clustering_analysis()
    analyzer.geographic_analysis()
    analyzer.generate_summary_report()
    
    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("Generated files:")
    print("- datacenter_geographic_distribution.html")
    print("- eji_distributions.html")
    print("- correlation_matrix.html")
    print("- feature_importance.html")
    print("- elbow_plot.html")
    print("- 3d_clusters.html")
    print("- cluster_geographic.html")
    print("- analysis_results.json")
    print("=" * 90)

if __name__ == "__main__":
    main()
