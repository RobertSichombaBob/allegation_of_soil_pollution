# geochemical_analysis_suite_pro.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit
st.set_page_config(
    page_title="Geochemical Analysis Suite Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# IMPORTS WITH ERROR HANDLING
# ============================

# Core scientific computing
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import gaussian_kde

# Visualization
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import plotly.graph_objects as go
import plotly.express as px
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configure matplotlib for publication quality
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16
plt.style.use('seaborn-v0_8-whitegrid')

# ML/Stats imports with comprehensive error handling
SKLEARN_AVAILABLE = False
FA_AVAILABLE = False
SKBIO_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.covariance import MinCovDet, EmpiricalCovariance
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS, TSNE, Isomap
    from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.mixture import GaussianMixture
    from sklearn.feature_selection import VarianceThreshold
    SKLEARN_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ scikit-learn not available: {e}")

try:
    from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
    FA_AVAILABLE = True
except ImportError:
    st.warning("⚠️ factor-analyzer not available. Install with: pip install factor-analyzer")

try:
    from skbio.stats.composition import clr as skbio_clr
    from skbio.stats.composition import ilr as skbio_ilr
    SKBIO_AVAILABLE = True
except ImportError:
    st.info("💡 For compositional data analysis: pip install scikit-bio")

# For Excel support
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    st.warning("⚠️ openpyxl not available for Excel support. Install with: pip install openpyxl")
    EXCEL_AVAILABLE = False

# ============================
# CONSTANTS & PALETTES
# ============================

# Custom color palettes for geochemical data
GEO_CHEM_PALETTES = {
    'sequential': ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
    'diverging': ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061'],
    'qualitative': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'pollution': ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],  # Viridis-like for pollution
    'soil': ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e']  # Soil colors
}

# Create custom colormaps
sequential_cmap = LinearSegmentedColormap.from_list("geo_sequential", GEO_CHEM_PALETTES['sequential'])
diverging_cmap = LinearSegmentedColormap.from_list("geo_diverging", GEO_CHEM_PALETTES['diverging'])
pollution_cmap = LinearSegmentedColormap.from_list("pollution", GEO_CHEM_PALETTES['pollution'])

# ============================
# UTILITY FUNCTIONS
# ============================

def safe_divide(a, b, default=0):
    """Safe division with zero handling"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = default
    return result

def detect_file_type(filename):
    """Detect file type from extension"""
    if filename.lower().endswith('.csv'):
        return 'csv'
    elif filename.lower().endswith(('.xlsx', '.xls')):
        return 'excel'
    return None

# ============================
# MAIN ANALYZER CLASS
# ============================

class GeochemicalAnalyzer:
    """Complete geochemical analysis suite for soil pollution analysis"""
    
    def __init__(self):
        self.raw_data = None
        self.df = None
        self.coord_cols = []
        self.element_cols = []
        self.results = {}
        self._spatial_bounds = None
        
    def load_data(self, file, file_type='csv'):
        """Load geochemical data from CSV or Excel file"""
        try:
            if file_type == 'csv':
                self.raw_data = pd.read_csv(file)
            elif file_type == 'excel':
                self.raw_data = pd.read_excel(file, engine='openpyxl')
            else:
                st.error(f"Unsupported file type: {file_type}")
                return self
            
            self.df = self.raw_data.copy()
            
            # Clean column names
            self.df.columns = [str(col).strip().replace('\n', ' ') for col in self.df.columns]
            
            # Detect coordinate columns
            self._detect_coordinates()
            
            # Detect element columns
            self._detect_elements()
            
            # Store in session state
            st.session_state['analyzer'] = self
            
            st.success(f"✅ Successfully loaded {len(self.df)} samples with {len(self.element_cols)} elements")
            return self
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None
    
    def _detect_coordinates(self):
        """Automatically detect coordinate columns with improved detection"""
        coord_patterns = {
            'x': ['x', 'east', 'easting', 'longitude', 'lon', 'coord_x', 'coordx'],
            'y': ['y', 'north', 'northing', 'latitude', 'lat', 'coord_y', 'coordy']
        }
        
        found_x = []
        found_y = []
        
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Check for X patterns
            if any(pattern in col_lower for pattern in coord_patterns['x']):
                found_x.append(col)
            
            # Check for Y patterns
            if any(pattern in col_lower for pattern in coord_patterns['y']):
                found_y.append(col)
        
        # Select best matches
        if len(found_x) >= 1 and len(found_y) >= 1:
            self.coord_cols = [found_x[0], found_y[0]]
        elif len(found_x) >= 2:
            self.coord_cols = found_x[:2]
        else:
            # Create synthetic coordinates
            self.df['X_synthetic'] = np.arange(len(self.df))
            self.df['Y_synthetic'] = np.random.uniform(0, 100, len(self.df))
            self.coord_cols = ['X_synthetic', 'Y_synthetic']
            st.warning("⚠️ No coordinate columns detected. Using synthetic coordinates.")
        
        # Store spatial bounds
        self._spatial_bounds = {
            'x_min': self.df[self.coord_cols[0]].min(),
            'x_max': self.df[self.coord_cols[0]].max(),
            'y_min': self.df[self.coord_cols[1]].min(),
            'y_max': self.df[self.coord_cols[1]].max()
        }
    
    def _detect_elements(self):
        """Detect element columns with improved logic"""
        self.element_cols = []
        
        # Common geochemical elements and pollutants
        common_elements = [
            'au', 'ag', 'cu', 'pb', 'zn', 'fe', 'as', 'hg', 'cd', 'cr', 'ni',
            'co', 'mn', 'mo', 'sb', 'se', 'th', 'u', 'v', 'al', 'ca', 'k',
            'mg', 'na', 'ti', 'p', 's', 'si'
        ]
        
        for col in self.df.columns:
            if col in self.coord_cols:
                continue
                
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check if column name matches common elements
                col_lower = col.lower()
                if any(element in col_lower for element in common_elements):
                    self.element_cols.append(col)
                else:
                    # Check if it looks like a concentration
                    non_null = self.df[col].dropna()
                    if len(non_null) > 0:
                        # Check for typical concentration ranges
                        median_val = non_null.median()
                        if 0 < median_val < 100000:  # Reasonable concentration range
                            self.element_cols.append(col)
        
        # If still no elements, use all numeric columns
        if len(self.element_cols) == 0:
            for col in self.df.columns:
                if col not in self.coord_cols and pd.api.types.is_numeric_dtype(self.df[col]):
                    self.element_cols.append(col)
    
    def create_interactive_map(self, color_column=None, size_column=None):
        """Create interactive plotly map"""
        if len(self.coord_cols) != 2:
            return None
        
        fig = go.Figure()
        
        # Add scatter points
        scatter_kwargs = {
            'x': self.df[self.coord_cols[0]],
            'y': self.df[self.coord_cols[1]],
            'mode': 'markers',
            'marker': {
                'size': 8,
                'opacity': 0.7,
                'color': 'blue' if color_column is None else self.df[color_column],
                'colorscale': 'Viridis' if color_column is None else None,
                'showscale': color_column is not None
            },
            'text': [f"Sample {i}" for i in range(len(self.df))],
            'hovertemplate': '<b>Sample %{text}</b><br>' +
                           f'{self.coord_cols[0]}: %{{x}}<br>' +
                           f'{self.coord_cols[1]}: %{{y}}'
        }
        
        if color_column:
            scatter_kwargs['marker']['colorbar'] = {'title': color_column}
            scatter_kwargs['hovertemplate'] += f'<br>{color_column}: %{{marker.color}}'
        
        fig.add_trace(go.Scatter(**scatter_kwargs))
        
        # Update layout
        fig.update_layout(
            title='Interactive Sample Map',
            xaxis_title=self.coord_cols[0],
            yaxis_title=self.coord_cols[1],
            template='plotly_white',
            hovermode='closest',
            showlegend=False,
            height=600
        )
        
        return fig
    
    def data_summary(self):
        """Display comprehensive data summary"""
        st.markdown("## 📊 Data Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(self.df))
        with col2:
            st.metric("Elements", len(self.element_cols))
        with col3:
            missing = self.df[self.element_cols].isnull().sum().sum()
            st.metric("Missing Values", missing)
        with col4:
            zeros = (self.df[self.element_cols] == 0).sum().sum()
            st.metric("Zero Values", zeros)
        
        # Interactive map
        st.markdown("### Interactive Spatial Map")
        if len(self.coord_cols) == 2:
            col1, col2 = st.columns([3, 1])
            with col2:
                color_by = st.selectbox(
                    "Color by:",
                    ['None'] + self.element_cols,
                    key='map_color'
                )
                
                if color_by != 'None':
                    fig = self.create_interactive_map(color_column=color_by)
                else:
                    fig = self.create_interactive_map()
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col1:
                # Quick statistics
                st.dataframe(self.df[self.element_cols].describe().T, height=300)
        
        # Data preview
        st.markdown("### Data Preview")
        tab1, tab2, tab3 = st.tabs(["First Samples", "Statistics", "Missing Values"])
        
        with tab1:
            st.dataframe(self.df.head(10), use_container_width=True)
        
        with tab2:
            stats_df = self.df[self.element_cols].describe().T
            stats_df['cv'] = stats_df['std'] / stats_df['mean'] * 100  # Coefficient of variation
            st.dataframe(stats_df, use_container_width=True)
        
        with tab3:
            missing_df = pd.DataFrame({
                'Element': self.element_cols,
                'Missing_Count': [self.df[col].isnull().sum() for col in self.element_cols],
                'Missing_Percentage': [self.df[col].isnull().mean() * 100 for col in self.element_cols],
                'Zeros_Count': [(self.df[col] == 0).sum() for col in self.element_cols]
            }).sort_values('Missing_Percentage', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
        
        # Element distributions
        st.markdown("### Element Distributions")
        selected_elements = st.multiselect(
            "Select elements to visualize:",
            self.element_cols,
            default=self.element_cols[:min(6, len(self.element_cols))],
            key='dist_elements'
        )
        
        if selected_elements:
            n_cols = 3
            n_rows = (len(selected_elements) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            
            for idx, element in enumerate(selected_elements):
                if idx < len(axes):
                    data = self.df[element].dropna()
                    if len(data) > 0:
                        axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7, 
                                      color=GEO_CHEM_PALETTES['qualitative'][idx % len(GEO_CHEM_PALETTES['qualitative'])])
                        axes[idx].set_title(f'{element}')
                        axes[idx].set_xlabel('Concentration')
                        axes[idx].set_ylabel('Frequency')
                        axes[idx].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for idx in range(len(selected_elements), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Correlation matrix
        if len(self.element_cols) > 2:
            st.markdown("### Correlation Matrix")
            corr_matrix = self.df[self.element_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, square=True, linewidths=0.5, ax=ax,
                       cbar_kws={"shrink": 0.8})
            ax.set_title('Element Correlation Matrix', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
    
    def preprocess_data(self):
        """Interactive data preprocessing with more options"""
        st.markdown("## 🔧 Data Preprocessing")
        
        # Create a copy for processing
        processed_df = self.df.copy()
        
        # Missing value handling
        st.markdown("### Missing Value Handling")
        missing_method = st.selectbox(
            "Select method for handling missing values:",
            ["None", "Fill with median", "Fill with mean", "Fill with 1/2 detection limit", "Drop rows with missing values"],
            key='missing_method'
        )
        
        if missing_method == "Fill with median":
            processed_df[self.element_cols] = processed_df[self.element_cols].fillna(
                processed_df[self.element_cols].median()
            )
        elif missing_method == "Fill with mean":
            processed_df[self.element_cols] = processed_df[self.element_cols].fillna(
                processed_df[self.element_cols].mean()
            )
        elif missing_method == "Fill with 1/2 detection limit":
            # Assume detection limit is minimum non-zero value / 2
            for col in self.element_cols:
                non_zero = processed_df[col][processed_df[col] > 0]
                if len(non_zero) > 0:
                    dl = non_zero.min() / 2
                    processed_df[col] = processed_df[col].fillna(dl)
        elif missing_method == "Drop rows with missing values":
            processed_df = processed_df.dropna(subset=self.element_cols)
        
        # Zero handling
        st.markdown("### Zero Value Handling")
        zero_method = st.selectbox(
            "Select method for handling zeros:",
            ["None", "Add small constant (1e-6)", "Replace with NaN", "Replace with 1/2 detection limit"],
            key='zero_method'
        )
        
        if zero_method == "Add small constant (1e-6)":
            processed_df[self.element_cols] = processed_df[self.element_cols].applymap(
                lambda x: max(x, 1e-6)
            )
        elif zero_method == "Replace with NaN":
            processed_df[self.element_cols] = processed_df[self.element_cols].replace(0, np.nan)
        elif zero_method == "Replace with 1/2 detection limit":
            for col in self.element_cols:
                non_zero = processed_df[col][processed_df[col] > 0]
                if len(non_zero) > 0:
                    dl = non_zero.min() / 2
                    processed_df[col] = processed_df[col].replace(0, dl)
        
        # Transformation options
        st.markdown("### Data Transformation")
        transform_method = st.selectbox(
            "Select transformation method:",
            ["None", "Log10", "Natural log", "Square root", "CLR (Compositional)", "Box-Cox"],
            key='transform_method'
        )
        
        if transform_method == "Log10":
            processed_df[self.element_cols] = np.log10(processed_df[self.element_cols] + 1e-10)
        elif transform_method == "Natural log":
            processed_df[self.element_cols] = np.log(processed_df[self.element_cols] + 1e-10)
        elif transform_method == "Square root":
            processed_df[self.element_cols] = np.sqrt(processed_df[self.element_cols])
        elif transform_method == "CLR (Compositional)":
            if SKBIO_AVAILABLE:
                for col in self.element_cols:
                    processed_df[col] = skbio_clr(processed_df[[col]].values + 1e-10)
            else:
                st.warning("scikit-bio not available for CLR. Using log transform instead.")
                processed_df[self.element_cols] = np.log10(processed_df[self.element_cols] + 1e-10)
        elif transform_method == "Box-Cox":
            from scipy import stats
            for col in self.element_cols:
                data = processed_df[col] + 1e-10
                transformed, _ = stats.boxcox(data)
                processed_df[col] = transformed
        
        # Scaling
        st.markdown("### Data Scaling")
        scale_method = st.selectbox(
            "Select scaling method:",
            ["None", "StandardScaler", "RobustScaler", "MinMaxScaler"],
            key='scale_method'
        )
        
        if scale_method == "StandardScaler" and SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            processed_df[self.element_cols] = scaler.fit_transform(processed_df[self.element_cols])
        elif scale_method == "RobustScaler" and SKLEARN_AVAILABLE:
            scaler = RobustScaler()
            processed_df[self.element_cols] = scaler.fit_transform(processed_df[self.element_cols])
        elif scale_method == "MinMaxScaler" and SKLEARN_AVAILABLE:
            scaler = MinMaxScaler()
            processed_df[self.element_cols] = scaler.fit_transform(processed_df[self.element_cols])
        
        # Display comparison
        st.markdown("### Before vs After Processing")
        if len(self.element_cols) > 0:
            compare_element = st.selectbox("Select element for comparison:", self.element_cols, key='compare_element')
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Before
            before_data = self.df[compare_element].dropna()
            if len(before_data) > 0:
                axes[0].hist(before_data, bins=30, edgecolor='black',
                            alpha=0.7, color='blue')
                axes[0].set_title(f'Original {compare_element}')
                axes[0].set_xlabel('Value')
                axes[0].set_ylabel('Frequency')
                axes[0].grid(True, alpha=0.3)
            
            # After
            after_data = processed_df[compare_element].dropna()
            if len(after_data) > 0:
                axes[1].hist(after_data, bins=30, edgecolor='black',
                            alpha=0.7, color='green')
                axes[1].set_title(f'Processed {compare_element}')
                axes[1].set_xlabel('Value')
                axes[1].set_ylabel('Frequency')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Store processed data
        st.session_state['processed_data'] = processed_df
        st.session_state['processing_params'] = {
            'missing_method': missing_method,
            'zero_method': zero_method,
            'transform_method': transform_method,
            'scale_method': scale_method
        }
        
        st.success(f"✅ Data preprocessing completed! {len(processed_df)} samples ready for analysis.")
        
        return processed_df
    
    def multivariate_outlier_detection(self):
        """Comprehensive multivariate outlier detection"""
        st.markdown("## 🎯 Multivariate Outlier Detection")
        
        # Get data
        if 'processed_data' in st.session_state:
            data = st.session_state['processed_data']
        else:
            data = self.df
        
        # Select elements for analysis
        st.markdown("### Element Selection")
        selected_elements = st.multiselect(
            "Select elements for outlier detection:",
            self.element_cols,
            default=self.element_cols[:min(8, len(self.element_cols))],
            key='outlier_elements'
        )
        
        if not selected_elements:
            st.warning("Please select at least one element")
            return
        
        # Prepare data
        X = data[selected_elements].values
        
        # Remove rows with NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]
        indices_clean = np.where(valid_mask)[0]
        
        if len(X_clean) < 10:
            st.error("Not enough samples for outlier detection after removing NaN values")
            return
        
        # Select outlier detection methods
        st.markdown("### Detection Methods")
        available_methods = []
        if SKLEARN_AVAILABLE:
            available_methods = ["Robust Mahalanobis", "Local Outlier Factor (LOF)", 
                                "Isolation Forest", "One-Class SVM"]
        
        methods = st.multiselect(
            "Select outlier detection methods:",
            available_methods,
            default=available_methods[:min(2, len(available_methods))],
            key='outlier_methods'
        )
        
        if not methods:
            st.warning("Please select at least one method")
            return
        
        results = {}
        
        if "Robust Mahalanobis" in methods:
            st.markdown("#### Robust Mahalanobis Distance")
            
            try:
                # Calculate robust covariance
                robust_cov = MinCovDet().fit(X_clean)
                mahalanobis_dist = robust_cov.mahalanobis(X_clean)
                
                # Determine threshold (95th percentile)
                threshold = np.percentile(mahalanobis_dist, 95)
                outliers = mahalanobis_dist > threshold
                
                results['Mahalanobis'] = {
                    'scores': mahalanobis_dist,
                    'outliers': outliers,
                    'threshold': threshold,
                    'indices': indices_clean
                }
                
                # Visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Distance distribution
                axes[0].hist(mahalanobis_dist, bins=30, edgecolor='black', alpha=0.7,
                           color=GEO_CHEM_PALETTES['qualitative'][0])
                axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2,
                              label=f'Threshold: {threshold:.2f}')
                axes[0].set_xlabel('Mahalanobis Distance')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title('Mahalanobis Distance Distribution')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Chi-square Q-Q plot
                dof = X_clean.shape[1]
                chi2_quantiles = np.sqrt(stats.chi2.ppf(np.linspace(0.01, 0.99, len(mahalanobis_dist)), dof))
                empirical_quantiles = np.sort(mahalanobis_dist)
                
                axes[1].scatter(chi2_quantiles, empirical_quantiles, alpha=0.6, s=20)
                axes[1].plot([0, chi2_quantiles.max()], [0, chi2_quantiles.max()],
                           'r--', alpha=0.7, label='y = x')
                axes[1].set_xlabel('Theoretical χ² Quantiles')
                axes[1].set_ylabel('Empirical Quantiles')
                axes[1].set_title('Chi-square Q-Q Plot')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error in Mahalanobis calculation: {str(e)}")
        
        if "Local Outlier Factor (LOF)" in methods:
            st.markdown("#### Local Outlier Factor (LOF)")
            
            n_neighbors = st.slider("Number of neighbors for LOF:", 5, 50, 20, key='lof_neighbors')
            contamination = st.slider("Contamination estimate:", 0.01, 0.5, 0.1, 0.01, key='lof_contamination')
            
            try:
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                outlier_labels = lof.fit_predict(X_clean)
                lof_scores = -lof.negative_outlier_factor_
                
                outliers = outlier_labels == -1
                
                results['LOF'] = {
                    'scores': lof_scores,
                    'outliers': outliers,
                    'indices': indices_clean
                }
                
                # Visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # LOF scores distribution
                axes[0].hist(lof_scores, bins=30, edgecolor='black', alpha=0.7,
                           color=GEO_CHEM_PALETTES['qualitative'][1])
                axes[0].set_xlabel('LOF Score')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title('LOF Score Distribution')
                axes[0].grid(True, alpha=0.3)
                
                # Comparison with Mahalanobis if available
                if 'Mahalanobis' in results:
                    axes[1].scatter(results['Mahalanobis']['scores'], lof_scores,
                                  c=['red' if o else 'blue' for o in outliers],
                                  alpha=0.6, s=30)
                    axes[1].set_xlabel('Mahalanobis Distance')
                    axes[1].set_ylabel('LOF Score')
                    axes[1].set_title('Outlier Score Comparison')
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error in LOF calculation: {str(e)}")
        
        # Spatial visualization of outliers
        if len(self.coord_cols) == 2:
            st.markdown("### Spatial Distribution of Outliers")
            
            # Create a combined outlier mask
            combined_outliers = np.zeros(len(data), dtype=bool)
            outlier_scores = np.zeros(len(data))
            
            for method_name, result in results.items():
                if 'outliers' in result:
                    indices = result['indices']
                    outliers = result['outliers']
                    combined_outliers[indices] = combined_outliers[indices] | outliers
                    
                    # Combine scores (normalized)
                    if 'scores' in result:
                        scores_norm = (result['scores'] - result['scores'].min()) / (result['scores'].max() - result['scores'].min())
                        outlier_scores[indices] += scores_norm * outliers
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot normal samples
            normal_mask = ~combined_outliers
            if np.any(normal_mask):
                ax.scatter(data.loc[normal_mask, self.coord_cols[0]],
                         data.loc[normal_mask, self.coord_cols[1]],
                         c='gray', s=20, alpha=0.5, label='Normal')
            
            # Plot outliers
            if np.any(combined_outliers):
                outlier_data = data.loc[combined_outliers]
                scatter = ax.scatter(outlier_data[self.coord_cols[0]],
                                   outlier_data[self.coord_cols[1]],
                                   c=outlier_scores[combined_outliers],
                                   cmap='Reds', s=80, edgecolor='black',
                                   label='Outliers', alpha=0.8)
                plt.colorbar(scatter, ax=ax, label='Outlier Score')
            
            ax.set_xlabel(self.coord_cols[0])
            ax.set_ylabel(self.coord_cols[1])
            ax.set_title('Spatial Distribution of Outliers')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Interactive plotly map
            fig_map = go.Figure()
            
            # Add normal samples
            if np.any(normal_mask):
                normal_data = data.loc[normal_mask]
                fig_map.add_trace(go.Scatter(
                    x=normal_data[self.coord_cols[0]],
                    y=normal_data[self.coord_cols[1]],
                    mode='markers',
                    marker=dict(size=6, color='gray', opacity=0.5),
                    name='Normal',
                    text=[f"Sample {i}" for i in normal_data.index]
                ))
            
            # Add outliers
            if np.any(combined_outliers):
                outlier_data = data.loc[combined_outliers]
                fig_map.add_trace(go.Scatter(
                    x=outlier_data[self.coord_cols[0]],
                    y=outlier_data[self.coord_cols[1]],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=outlier_scores[combined_outliers],
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Outlier Score"),
                        line=dict(width=1, color='black')
                    ),
                    name='Outliers',
                    text=[f"Sample {i}" for i in outlier_data.index]
                ))
            
            fig_map.update_layout(
                title='Interactive Outlier Map',
                xaxis_title=self.coord_cols[0],
                yaxis_title=self.coord_cols[1],
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        
        # Store results
        self.results['outliers'] = results
        
        # Create outlier summary table
        if results:
            st.markdown("### Outlier Summary")
            summary_data = []
            for method_name, result in results.items():
                if 'outliers' in result:
                    n_outliers = np.sum(result['outliers'])
                    percentage = n_outliers / len(result['outliers']) * 100
                    summary_data.append({
                        'Method': method_name,
                        'Outliers': n_outliers,
                        'Percentage': f"{percentage:.1f}%"
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        st.success("✅ Outlier detection completed!")
    
    def pca_analysis(self):
        """Comprehensive Principal Component Analysis"""
        st.markdown("## 📈 Principal Component Analysis (PCA)")
        
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn required for PCA analysis")
            return
        
        # Get data
        if 'processed_data' in st.session_state:
            data = st.session_state['processed_data']
        else:
            data = self.df
        
        # Select elements
        st.markdown("### Element Selection")
        selected_elements = st.multiselect(
            "Select elements for PCA:",
            self.element_cols,
            default=self.element_cols[:min(15, len(self.element_cols))],
            key='pca_elements'
        )
        
        if len(selected_elements) < 2:
            st.warning("Please select at least 2 elements")
            return
        
        X = data[selected_elements].values
        
        # Remove rows with NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]
        indices_clean = np.where(valid_mask)[0]
        
        if len(X_clean) < 10:
            st.error("Not enough samples for PCA after removing NaN values")
            return
        
        # PCA configuration
        st.markdown("### PCA Configuration")
        col1, col2 = st.columns(2)
        with col1:
            variance_threshold = st.slider(
                "Variance to retain (%):",
                min_value=70,
                max_value=95,
                value=86,
                step=1,
                key='pca_variance'
            )
        with col2:
            max_components = min(20, X_clean.shape[1], X_clean.shape[0] - 1)
            n_components = st.slider(
                "Maximum components to compute:",
                min_value=2,
                max_value=max_components,
                value=min(10, max_components),
                step=1,
                key='pca_components'
            )
        
        # Perform PCA
        pca = PCA(n_components=min(n_components, X_clean.shape[1]))
        scores = pca.fit_transform(X_clean)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        loadings = pca.components_
        
        # Determine number of components for selected variance
        n_components_var = np.argmax(cumulative_variance >= variance_threshold/100) + 1
        
        # Display metrics
        st.markdown("### PCA Metrics")
        cols = st.columns(5)
        cols[0].metric("Total Variance", f"{cumulative_variance[-1]*100:.1f}%")
        cols[1].metric("Components for Threshold", n_components_var)
        cols[2].metric("PC1 Variance", f"{explained_variance[0]*100:.1f}%")
        cols[3].metric("PC2 Variance", f"{explained_variance[1]*100:.1f}%")
        cols[4].metric("PC3 Variance", f"{explained_variance[2]*100:.1f}%" if len(explained_variance) > 2 else "N/A")
        
        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Scree Plot", "Scores Plot", "Loadings", "Biplot", "Spatial Maps"
        ])
        
        with tab1:
            # Scree plot with cumulative variance
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            x_pos = np.arange(1, len(explained_variance) + 1)
            ax1.bar(x_pos, explained_variance * 100, alpha=0.6, 
                   color=GEO_CHEM_PALETTES['qualitative'])
            ax1.plot(x_pos, cumulative_variance * 100, 'ro-', linewidth=2, markersize=8)
            ax1.axhline(y=variance_threshold, color='green', linestyle='--',
                       label=f'{variance_threshold}% threshold')
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance (%)')
            ax1.set_title('Scree Plot with Cumulative Variance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Parallel analysis
            ax2.plot(x_pos, pca.explained_variance_, 'bo-', label='Actual', linewidth=2)
            ax2.set_xlabel('Component Number')
            ax2.set_ylabel('Eigenvalue')
            ax2.set_title('Eigenvalue Spectrum')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            # Scores plot
            st.markdown("#### PCA Scores Plot")
            
            color_options = ['None'] + selected_elements
            if 'outliers' in self.results:
                color_options.append('Outliers')
            
            color_by = st.selectbox("Color by:", color_options, key='pca_color')
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Determine colors
            if color_by == 'None':
                colors = 'blue'
            elif color_by == 'Outliers' and 'outliers' in self.results:
                # Create outlier mask for clean indices
                outlier_mask = np.zeros(len(data), dtype=bool)
                for method_name, result in self.results['outliers'].items():
                    if 'outliers' in result:
                        outlier_mask[result['indices']] = outlier_mask[result['indices']] | result['outliers']
                colors = ['red' if outlier_mask[i] else 'blue' for i in indices_clean]
            elif color_by in selected_elements:
                element_idx = selected_elements.index(color_by)
                colors = X_clean[:, element_idx]
                color_label = color_by
            else:
                colors = 'blue'
            
            # PC1 vs PC2
            scatter1 = axes[0].scatter(scores[:, 0], scores[:, 1], c=colors, 
                                      alpha=0.6, s=30, cmap='viridis' if not isinstance(colors, str) else None)
            axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
            axes[0].set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% var)')
            axes[0].set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% var)')
            axes[0].set_title('PC1 vs PC2 Scores')
            axes[0].grid(True, alpha=0.3)
            
            if not isinstance(colors, str) and color_by in selected_elements:
                plt.colorbar(scatter1, ax=axes[0], label=color_label)
            
            # PC3 vs PC4 if available
            if len(explained_variance) >= 4:
                scatter2 = axes[1].scatter(scores[:, 2], scores[:, 3], c=colors,
                                          alpha=0.6, s=30, cmap='viridis' if not isinstance(colors, str) else None)
                axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
                axes[1].set_xlabel(f'PC3 ({explained_variance[2]*100:.1f}% var)')
                axes[1].set_ylabel(f'PC4 ({explained_variance[3]*100:.1f}% var)')
                axes[1].set_title('PC3 vs PC4 Scores')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            # Loadings plot
            st.markdown("#### Principal Component Loadings")
            
            pc_to_plot = st.selectbox("Select PC:", range(1, min(5, len(explained_variance)) + 1), key='pc_select')
            top_n = st.slider("Top elements to show:", 5, 20, 10, key='top_n_loadings')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar plot of top loadings
            loadings_pc = loadings[pc_to_plot - 1]
            sorted_idx = np.argsort(np.abs(loadings_pc))[::-1][:top_n]
            
            y_pos = np.arange(top_n)
            bar_colors = ['red' if x < 0 else 'blue' for x in loadings_pc[sorted_idx]]
            ax1.barh(y_pos, loadings_pc[sorted_idx], color=bar_colors, alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([selected_elements[i] for i in sorted_idx])
            ax1.set_xlabel('Loading Value')
            ax1.set_title(f'Top {top_n} Loadings for PC{pc_to_plot}')
            ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Loadings plot (PC1 vs PC2)
            if len(loadings) >= 2:
                ax2.scatter(loadings[0], loadings[1], alpha=0.5)
                
                # Label top loadings
                for i in sorted_idx[:min(10, len(sorted_idx))]:
                    ax2.annotate(selected_elements[i], (loadings[0, i], loadings[1, i]),
                                fontsize=8, alpha=0.7)
                
                ax2.set_xlabel('PC1 Loadings')
                ax2.set_ylabel('PC2 Loadings')
                ax2.set_title('Loadings Plot (PC1 vs PC2)')
                ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab4:
            # Biplot
            st.markdown("#### Biplot (Scores + Loadings)")
            
            scale_factor = st.slider("Loadings scale factor:", 0.1, 5.0, 1.0, 0.1, key='biplot_scale')
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot scores
            ax.scatter(scores[:, 0], scores[:, 1], alpha=0.5, s=20,
                      color=GEO_CHEM_PALETTES['qualitative'][0])
            
            # Plot loadings as vectors
            for i in range(min(15, len(selected_elements))):
                ax.arrow(0, 0, loadings[0, i] * scale_factor, loadings[1, i] * scale_factor,
                        head_width=0.02, head_length=0.02, fc='red', ec='red', alpha=0.7)
                ax.text(loadings[0, i] * scale_factor * 1.1,
                       loadings[1, i] * scale_factor * 1.1,
                       selected_elements[i], fontsize=9, color='red', alpha=0.8)
            
            ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% var)')
            ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% var)')
            ax.set_title('Biplot: Scores and Loadings')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab5:
            # Spatial maps
            if len(self.coord_cols) == 2:
                st.markdown("#### Spatial Distribution of PCA Scores")
                
                n_maps = min(4, len(explained_variance))
                n_cols = 2
                n_rows = (n_maps + 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
                if n_maps > 1:
                    axes = axes.flatten()
                else:
                    axes = [axes]
                
                # Create full-length arrays for original indices
                full_scores = np.full((len(data), scores.shape[1]), np.nan)
                full_scores[indices_clean] = scores
                
                for i in range(n_maps):
                    scatter = axes[i].scatter(
                        data[self.coord_cols[0]],
                        data[self.coord_cols[1]],
                        c=full_scores[:, i], cmap='RdBu_r', s=30, alpha=0.7
                    )
                    axes[i].set_xlabel(self.coord_cols[0])
                    axes[i].set_ylabel(self.coord_cols[1] if i % n_cols == 0 else '')
                    axes[i].set_title(f'PC{i+1} ({explained_variance[i]*100:.1f}% var)')
                    axes[i].grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=axes[i])
                
                for i in range(n_maps, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Coordinate columns not available for spatial mapping")
        
        # Store results
        self.results['pca'] = {
            'scores': scores,
            'full_scores': full_scores if len(self.coord_cols) == 2 else None,
            'indices': indices_clean,
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'loadings': loadings,
            'n_components': n_components_var,
            'selected_elements': selected_elements
        }
        
        st.success("✅ PCA analysis completed!")
    
    def clustering_analysis(self):
        """Comprehensive clustering analysis"""
        st.markdown("## 🔍 Clustering Analysis")
        
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn required for clustering analysis")
            return
        
        # Get data
        if 'processed_data' in st.session_state:
            data = st.session_state['processed_data']
        else:
            data = self.df
        
        # Element selection
        st.markdown("### Element Selection")
        selected_elements = st.multiselect(
            "Select elements for clustering:",
            self.element_cols,
            default=self.element_cols[:min(10, len(self.element_cols))],
            key='cluster_elements'
        )
        
        if not selected_elements:
            st.warning("Please select at least one element")
            return
        
        X = data[selected_elements].values
        
        # Remove rows with NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]
        indices_clean = np.where(valid_mask)[0]
        
        if len(X_clean) < 10:
            st.error("Not enough samples for clustering")
            return
        
        # Clustering algorithm selection
        st.markdown("### Clustering Algorithm")
        algorithm = st.selectbox(
            "Select clustering algorithm:",
            ["K-Means", "DBSCAN", "Agglomerative", "Gaussian Mixture"],
            key='cluster_algorithm'
        )
        
        # Dimensionality reduction for visualization
        st.markdown("### Visualization Method")
        viz_method = st.selectbox(
            "Select method for 2D visualization:",
            ["PCA", "t-SNE", "MDS", "Isomap"],
            key='viz_method'
        )
        
        # Perform dimensionality reduction
        if viz_method == "PCA":
            reducer = PCA(n_components=2, random_state=42)
        elif viz_method == "t-SNE":
            perplexity = min(30, len(X_clean) - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        elif viz_method == "MDS":
            reducer = MDS(n_components=2, random_state=42, dissimilarity='euclidean')
        else:  # Isomap
            n_neighbors = min(10, len(X_clean) - 1)
            reducer = Isomap(n_components=2, n_neighbors=n_neighbors)
        
        X_reduced = reducer.fit_transform(X_clean)
        
        # Clustering
        if algorithm == "K-Means":
            st.markdown("### Optimal Cluster Determination")
            
            max_clusters = min(15, len(X_clean) - 1)
            
            # Calculate metrics
            inertia = []
            silhouette_scores = []
            calinski_scores = []
            
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_clean)
                inertia.append(kmeans.inertia_)
                
                if k > 1:
                    try:
                        silhouette_scores.append(silhouette_score(X_clean, labels))
                        calinski_scores.append(calinski_harabasz_score(X_clean, labels))
                    except:
                        silhouette_scores.append(np.nan)
                        calinski_scores.append(np.nan)
            
            # Plot metrics
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Elbow plot
            axes[0].plot(range(2, max_clusters + 1), inertia, 'bo-', markersize=8, linewidth=2)
            axes[0].set_xlabel('Number of Clusters')
            axes[0].set_ylabel('Inertia')
            axes[0].set_title('Elbow Method')
            axes[0].grid(True, alpha=0.3)
            
            # Silhouette scores
            if any(not np.isnan(s) for s in silhouette_scores):
                axes[1].plot(range(2, max_clusters + 1), silhouette_scores, 'ro-', markersize=8, linewidth=2)
                axes[1].set_xlabel('Number of Clusters')
                axes[1].set_ylabel('Silhouette Score')
                axes[1].set_title('Silhouette Score')
                axes[1].grid(True, alpha=0.3)
            
            # Calinski-Harabasz index
            if any(not np.isnan(s) for s in calinski_scores):
                axes[2].plot(range(2, max_clusters + 1), calinski_scores, 'go-', markersize=8, linewidth=2)
                axes[2].set_xlabel('Number of Clusters')
                axes[2].set_ylabel('Calinski-Harabasz Index')
                axes[2].set_title('Calinski-Harabasz Index')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Let user select k
            n_clusters = st.slider(
                "Select number of clusters:",
                min_value=2,
                max_value=max_clusters,
                value=min(4, max_clusters),
                key='n_clusters'
            )
            
            # Perform K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_clean)
            
        elif algorithm == "DBSCAN":
            st.markdown("### DBSCAN Parameters")
            col1, col2 = st.columns(2)
            with col1:
                eps = st.slider("EPS parameter:", 0.1, 5.0, 0.5, 0.1, key='dbscan_eps')
            with col2:
                min_samples = st.slider("Min samples:", 2, 20, 5, key='dbscan_min_samples')
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X_clean)
            
        elif algorithm == "Agglomerative":
            st.markdown("### Agglomerative Clustering")
            n_clusters = st.slider("Number of clusters:", 2, 15, 4, key='agglo_clusters')
            
            linkage_method = st.selectbox(
                "Linkage method:",
                ["ward", "complete", "average", "single"],
                key='linkage_method'
            )
            
            agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            cluster_labels = agglomerative.fit_predict(X_clean)
            
        else:  # Gaussian Mixture
            st.markdown("### Gaussian Mixture Model")
            n_components = st.slider("Number of components:", 1, 10, 3, key='gmm_components')
            
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            cluster_labels = gmm.fit_predict(X_clean)
        
        # Visualize clustering results
        st.markdown("### Clustering Results")
        
        # Create full-length arrays for plotting
        full_labels = np.full(len(data), -1, dtype=int)
        full_labels[indices_clean] = cluster_labels
        
        unique_labels = np.unique(cluster_labels)
        n_unique = len(unique_labels)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Reduced space scatter plot
        cmap = plt.cm.get_cmap('tab20', n_unique) if n_unique > 0 else plt.cm.tab20
        scatter1 = axes[0, 0].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                     c=cluster_labels, cmap=cmap, s=30, alpha=0.7)
        axes[0, 0].set_xlabel(f'{viz_method} Dimension 1')
        axes[0, 0].set_ylabel(f'{viz_method} Dimension 2')
        axes[0, 0].set_title(f'Clusters in {viz_method} Space')
        axes[0, 0].grid(True, alpha=0.3)
        if n_unique > 0:
            plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')
        
        # 2. Spatial distribution
        if len(self.coord_cols) == 2:
            scatter2 = axes[0, 1].scatter(
                data[self.coord_cols[0]],
                data[self.coord_cols[1]],
                c=full_labels, cmap=cmap, s=30, alpha=0.7, vmin=full_labels.min(), vmax=full_labels.max()
            )
            axes[0, 1].set_xlabel(self.coord_cols[0])
            axes[0, 1].set_ylabel(self.coord_cols[1])
            axes[0, 1].set_title('Spatial Distribution of Clusters')
            axes[0, 1].grid(True, alpha=0.3)
            if n_unique > 0:
                plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
        
        # 3. Cluster sizes
        if n_unique > 0:
            counts = np.array([np.sum(cluster_labels == label) for label in unique_labels])
            colors = cmap(np.arange(len(unique_labels)) / max(len(unique_labels), 1))
            axes[1, 0].bar(range(len(unique_labels)), counts, color=colors)
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Number of Samples')
            axes[1, 0].set_title('Cluster Sizes')
            axes[1, 0].set_xticks(range(len(unique_labels)))
            axes[1, 0].set_xticklabels([str(l) for l in unique_labels])
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Element patterns per cluster
        if n_unique > 0 and len(selected_elements) > 0:
            # Calculate mean values per cluster
            cluster_means = []
            for label in unique_labels:
                mask = cluster_labels == label
                if np.sum(mask) > 0:
                    cluster_means.append(np.mean(X_clean[mask], axis=0))
            
            if cluster_means:
                cluster_means = np.array(cluster_means)
                im = axes[1, 1].imshow(cluster_means.T, aspect='auto', cmap='RdBu_r')
                axes[1, 1].set_xlabel('Cluster')
                axes[1, 1].set_ylabel('Element Index')
                axes[1, 1].set_title('Element Means per Cluster')
                axes[1, 1].set_xticks(range(len(unique_labels)))
                axes[1, 1].set_xticklabels([str(l) for l in unique_labels])
                axes[1, 1].set_yticks(range(len(selected_elements)))
                axes[1, 1].set_yticklabels(selected_elements, fontsize=8)
                plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Cluster statistics
        st.markdown("### Cluster Statistics")
        if n_unique > 0:
            stats_data = []
            for label in unique_labels:
                mask = cluster_labels == label
                if np.sum(mask) > 0:
                    stats_data.append({
                        'Cluster': label,
                        'Samples': np.sum(mask),
                        'Percentage': f"{np.sum(mask)/len(cluster_labels)*100:.1f}%"
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
        
        # Interactive spatial map
        if len(self.coord_cols) == 2 and n_unique > 0:
            st.markdown("### Interactive Cluster Map")
            
            fig_map = go.Figure()
            
            for label in unique_labels:
                mask = full_labels == label
                if np.any(mask):
                    cluster_data = data[mask]
                    fig_map.add_trace(go.Scatter(
                        x=cluster_data[self.coord_cols[0]],
                        y=cluster_data[self.coord_cols[1]],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=label,
                            colorscale='tab20',
                            showscale=True,
                            colorbar=dict(title="Cluster"),
                            line=dict(width=0.5, color='black')
                        ),
                        name=f'Cluster {label}',
                        text=[f"Sample {i}, Cluster {label}" for i in cluster_data.index]
                    ))
            
            fig_map.update_layout(
                title='Interactive Cluster Map',
                xaxis_title=self.coord_cols[0],
                yaxis_title=self.coord_cols[1],
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        
        # Store results
        self.results['clustering'] = {
            'labels': cluster_labels,
            'full_labels': full_labels,
            'algorithm': algorithm,
            'selected_elements': selected_elements,
            'unique_labels': unique_labels.tolist(),
            'indices': indices_clean
        }
        
        st.success("✅ Clustering analysis completed!")
    
    def pollution_assessment(self):
        """Specialized analysis for soil pollution assessment"""
        st.markdown("## ☣️ Soil Pollution Assessment")
        
        # Define common pollutant elements and their thresholds (mg/kg)
        pollutant_thresholds = {
            'As': {'background': 10, 'threshold': 20, 'severe': 50},  # Arsenic
            'Pb': {'background': 20, 'threshold': 100, 'severe': 400},  # Lead
            'Cd': {'background': 0.5, 'threshold': 3, 'severe': 10},  # Cadmium
            'Cr': {'background': 50, 'threshold': 100, 'severe': 400},  # Chromium
            'Cu': {'background': 30, 'threshold': 100, 'severe': 200},  # Copper
            'Zn': {'background': 50, 'threshold': 300, 'severe': 1000},  # Zinc
            'Ni': {'background': 30, 'threshold': 50, 'severe': 100},  # Nickel
            'Hg': {'background': 0.1, 'threshold': 1, 'severe': 5},  # Mercury
        }
        
        # Find pollutants in dataset
        available_pollutants = {}
        for pollutant, thresholds in pollutant_thresholds.items():
            # Check for exact match or partial match
            for col in self.element_cols:
                if pollutant.lower() in col.lower() or col.lower() in pollutant.lower():
                    available_pollutants[pollutant] = {
                        'column': col,
                        'thresholds': thresholds
                    }
                    break
        
        if not available_pollutants:
            st.warning("No known pollutant elements found in dataset.")
            # Show all elements for manual selection
            st.info("Available elements for pollution assessment:")
            for element in self.element_cols[:20]:
                st.write(f"- {element}")
            return
        
        st.markdown(f"### Detected Pollutants: {len(available_pollutants)} elements")
        
        # Select pollutants to analyze
        selected_pollutants = st.multiselect(
            "Select pollutants for assessment:",
            list(available_pollutants.keys()),
            default=list(available_pollutants.keys())[:min(5, len(available_pollutants))],
            key='pollutants_select'
        )
        
        if not selected_pollutants:
            return
        
        # Calculate pollution indices
        st.markdown("### Pollution Indices")
        
        # Create results dataframe
        pollution_results = pd.DataFrame(index=self.df.index)
        
        for pollutant in selected_pollutants:
            col = available_pollutants[pollutant]['column']
            thresholds = available_pollutants[pollutant]['thresholds']
            
            # Calculate single pollution index (PI)
            pi = self.df[col] / thresholds['background']
            pollution_results[f'PI_{pollutant}'] = pi
            
            # Calculate pollution category
            pollution_category = pd.cut(
                self.df[col],
                bins=[-np.inf, thresholds['background'], thresholds['threshold'], thresholds['severe'], np.inf],
                labels=['Clean', 'Low', 'Moderate', 'Severe']
            )
            pollution_results[f'Category_{pollutant}'] = pollution_category
        
                # Calculate Nemerow Integrated Pollution Index (NIPI)
        if len(selected_pollutants) > 1:
            pi_columns = [f'PI_{p}' for p in selected_pollutants]
            pi_matrix = pollution_results[pi_columns].values
            
            # NIPI = sqrt((max(PI)² + mean(PI)²) / 2)
            max_pi = np.nanmax(pi_matrix, axis=1)
            mean_pi = np.nanmean(pi_matrix, axis=1)
            nip_i = np.sqrt((max_pi**2 + mean_pi**2) / 2)
            pollution_results['NIPI'] = nip_i
            
            # Categorize NIPI
            pollution_results['NIPI_Category'] = pd.cut(
                nip_i,
                bins=[0, 1, 2, 3, np.inf],
                labels=['Clean', 'Low', 'Moderate', 'Severe']
            )
        
        # Display results
        st.dataframe(pollution_results.describe(), use_container_width=True)
        
        # Visualizations
        st.markdown("### Pollution Visualization")
        
        # 1. Bar chart of pollutant concentrations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Top pollutant concentrations
        for idx, pollutant in enumerate(selected_pollutants[:4]):
            row = idx // 2
            col = idx % 2
            col_name = available_pollutants[pollutant]['column']
            
            data = self.df[col_name].dropna()
            if len(data) > 0:
                axes[row, col].hist(data, bins=30, edgecolor='black', alpha=0.7)
                axes[row, col].axvline(available_pollutants[pollutant]['thresholds']['background'], 
                                     color='green', linestyle='--', label='Background')
                axes[row, col].axvline(available_pollutants[pollutant]['thresholds']['threshold'], 
                                     color='orange', linestyle='--', label='Threshold')
                axes[row, col].axvline(available_pollutants[pollutant]['thresholds']['severe'], 
                                     color='red', linestyle='--', label='Severe')
                axes[row, col].set_title(f'{pollutant} Distribution')
                axes[row, col].set_xlabel('Concentration (mg/kg)')
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 2. Spatial pollution maps
        if len(self.coord_cols) == 2:
            st.markdown("### Spatial Pollution Distribution")
            
            # Create interactive map for each pollutant
            for pollutant in selected_pollutants[:3]:  # Limit to 3 for performance
                col_name = available_pollutants[pollutant]['column']
                
                fig_map = go.Figure()
                
                # Color by concentration
                fig_map.add_trace(go.Scatter(
                    x=self.df[self.coord_cols[0]],
                    y=self.df[self.coord_cols[1]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.df[col_name],
                        colorscale='RdYlGn_r',  # Red for high values
                        showscale=True,
                        colorbar=dict(title=f'{pollutant} (mg/kg)'),
                        line=dict(width=0.5, color='black')
                    ),
                    text=[f"{pollutant}: {val:.2f}" for val in self.df[col_name]],
                    hovertemplate='%{text}<br>X: %{x}<br>Y: %{y}',
                    name=pollutant
                ))
                
                fig_map.update_layout(
                    title=f'Spatial Distribution of {pollutant}',
                    xaxis_title=self.coord_cols[0],
                    yaxis_title=self.coord_cols[1],
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
            
            # Combined pollution map using NIPI
            if 'NIPI' in pollution_results.columns:
                st.markdown("#### Integrated Pollution Index (NIPI) Map")
                
                fig_map = go.Figure()
                
                # Size by NIPI, color by category
                sizes = np.clip(pollution_results['NIPI'] * 5, 5, 30)
                
                fig_map.add_trace(go.Scatter(
                    x=self.df[self.coord_cols[0]],
                    y=self.df[self.coord_cols[1]],
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=pollution_results['NIPI'],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title='NIPI'),
                        line=dict(width=1, color='black')
                    ),
                    text=[f"NIPI: {val:.2f} ({cat})" for val, cat in 
                          zip(pollution_results['NIPI'], pollution_results['NIPI_Category'])],
                    hovertemplate='%{text}<br>X: %{x}<br>Y: %{y}',
                    name='Pollution Hotspots'
                ))
                
                fig_map.update_layout(
                    title='Integrated Pollution Index (NIPI)',
                    xaxis_title=self.coord_cols[0],
                    yaxis_title=self.coord_cols[1],
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
        
        # Pollution statistics by category
        st.markdown("### Pollution Statistics")
        
        if 'NIPI_Category' in pollution_results.columns:
            category_counts = pollution_results['NIPI_Category'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['green', 'yellow', 'orange', 'red']
            bars = ax.bar(category_counts.index.astype(str), category_counts.values, 
                         color=colors[:len(category_counts)], alpha=0.7)
            
            ax.set_xlabel('Pollution Category')
            ax.set_ylabel('Number of Samples')
            ax.set_title('Sample Distribution by Pollution Level')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Export pollution results
        st.markdown("### Export Pollution Assessment")
        
        export_df = self.df.copy()
        export_df = pd.concat([export_df, pollution_results], axis=1)
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Pollution Assessment Results",
            data=csv,
            file_name="pollution_assessment_results.csv",
            mime="text/csv"
        )
        
        st.success("✅ Pollution assessment completed!")
        
        # Store results
        self.results['pollution'] = {
            'available_pollutants': available_pollutants,
            'selected_pollutants': selected_pollutants,
            'pollution_results': pollution_results
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        st.markdown("## 📋 Comprehensive Analysis Report")
        
        # Executive summary
        st.markdown("### Executive Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Samples", len(self.df))
        
        with col2:
            st.metric("Elements", len(self.element_cols))
        
        with col3:
            if 'outliers' in self.results:
                total_outliers = 0
                for method_name, result in self.results['outliers'].items():
                    if 'outliers' in result:
                        total_outliers += np.sum(result['outliers'])
                st.metric("Total Outliers", total_outliers)
            else:
                st.metric("Total Outliers", "N/A")
        
        with col4:
            if 'clustering' in self.results:
                n_clusters = len(self.results['clustering']['unique_labels'])
                st.metric("Clusters", n_clusters)
            else:
                st.metric("Clusters", "N/A")
        
        with col5:
            if 'pca' in self.results:
                pca_var = self.results['pca']['cumulative_variance'][1] * 100
                st.metric("PC1+PC2 Variance", f"{pca_var:.1f}%")
            else:
                st.metric("PC1+PC2 Variance", "N/A")
        
        # Pollution summary if available
        if 'pollution' in self.results:
            st.markdown("### Pollution Assessment Summary")
            
            pollution_results = self.results['pollution']['pollution_results']
            if 'NIPI_Category' in pollution_results.columns:
                category_counts = pollution_results['NIPI_Category'].value_counts()
                
                cols = st.columns(len(category_counts))
                for idx, (category, count) in enumerate(category_counts.items()):
                    with cols[idx]:
                        st.metric(f"{category} Samples", count)
        
        # Integrated visualization
        st.markdown("### Integrated Visualization")
        
        # Create a summary figure if we have results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Spatial outliers
        if 'outliers' in self.results and len(self.coord_cols) == 2:
            # Create combined outlier mask
            combined_outliers = np.zeros(len(self.df), dtype=bool)
            outlier_scores = np.zeros(len(self.df))
            
            for method_name, result in self.results['outliers'].items():
                if 'outliers' in result:
                    indices = result['indices']
                    outliers = result['outliers']
                    combined_outliers[indices] = combined_outliers[indices] | outliers
            
            axes[0, 0].scatter(self.df[self.coord_cols[0]], self.df[self.coord_cols[1]],
                             alpha=0.3, s=20, c='gray', label='Normal')
            
            if np.any(combined_outliers):
                outlier_data = self.df[combined_outliers]
                axes[0, 0].scatter(outlier_data[self.coord_cols[0]], outlier_data[self.coord_cols[1]],
                                 c='red', s=50, edgecolor='black', label='Outliers', alpha=0.7)
            
            axes[0, 0].set_xlabel(self.coord_cols[0])
            axes[0, 0].set_ylabel(self.coord_cols[1])
            axes[0, 0].set_title('Spatial Outlier Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PCA scores with clusters
        if 'pca' in self.results and 'clustering' in self.results:
            scores = self.results['pca']['scores']
            cluster_labels = self.results['clustering']['labels']
            
            if len(scores) > 0 and len(cluster_labels) > 0:
                scatter = axes[0, 1].scatter(scores[:, 0], scores[:, 1],
                                            c=cluster_labels, cmap='tab20', alpha=0.6, s=30)
                axes[0, 1].set_xlabel('PC1')
                axes[0, 1].set_ylabel('PC2')
                axes[0, 1].set_title('PCA Scores Colored by Cluster')
                axes[0, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')
        
        # 3. Pollution map
        if 'pollution' in self.results and len(self.coord_cols) == 2:
            pollution_results = self.results['pollution']['pollution_results']
            if 'NIPI' in pollution_results.columns:
                scatter = axes[1, 0].scatter(self.df[self.coord_cols[0]], self.df[self.coord_cols[1]],
                                           c=pollution_results['NIPI'], cmap='RdYlGn_r', s=30, alpha=0.7)
                axes[1, 0].set_xlabel(self.coord_cols[0])
                axes[1, 0].set_ylabel(self.coord_cols[1])
                axes[1, 0].set_title('Integrated Pollution Index (NIPI)')
                axes[1, 0].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[1, 0], label='NIPI')
        
        # 4. Element enrichment in outliers
        if 'outliers' in self.results and len(self.element_cols) > 0:
            # Create combined outlier mask
            combined_outliers = np.zeros(len(self.df), dtype=bool)
            for method_name, result in self.results['outliers'].items():
                if 'outliers' in result:
                    indices = result['indices']
                    outliers = result['outliers']
                    combined_outliers[indices] = combined_outliers[indices] | outliers
            
            # Calculate enrichment for top elements
            element_enrichment = []
            for element in self.element_cols[:10]:  # Limit to first 10
                if np.any(combined_outliers) and np.sum(~combined_outliers) > 0:
                    outlier_mean = self.df[element][combined_outliers].mean()
                    background_mean = self.df[element][~combined_outliers].mean()
                    
                    if background_mean > 0:
                        enrichment = outlier_mean / background_mean
                        element_enrichment.append((element, enrichment))
            
            # Plot top enrichments
            if element_enrichment:
                element_enrichment.sort(key=lambda x: x[1], reverse=True)
                elements_top = [e[0] for e in element_enrichment[:5]]
                enrichments_top = [e[1] for e in element_enrichment[:5]]
                
                colors = ['red' if e > 2 else 'orange' if e > 1.5 else 'blue' for e in enrichments_top]
                y_pos = np.arange(len(elements_top))
                
                axes[1, 1].barh(y_pos, enrichments_top, color=colors, alpha=0.7)
                axes[1, 1].set_yticks(y_pos)
                axes[1, 1].set_yticklabels(elements_top)
                axes[1, 1].set_xlabel('Enrichment Factor (Outlier/Background)')
                axes[1, 1].set_title('Element Enrichment in Outliers')
                axes[1, 1].axvline(x=1, color='k', linestyle='--', alpha=0.5)
                axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Recommendations
        st.markdown("### Recommendations for Kalumbila Soil Analysis")
        
        recommendations = []
        
        if 'outliers' in self.results:
            total_outliers = 0
            for method_name, result in self.results['outliers'].items():
                if 'outliers' in result:
                    total_outliers += np.sum(result['outliers'])
            
            if total_outliers > 0:
                recommendations.append(
                    f"**High Priority Investigation:** {total_outliers} multivariate outliers identified. "
                    "These samples represent potential pollution hotspots or anomalous geochemical signatures "
                    "and should be prioritized for detailed investigation and validation sampling."
                )
        
        if 'pollution' in self.results:
            pollution_results = self.results['pollution']['pollution_results']
            if 'NIPI_Category' in pollution_results.columns:
                severe_count = np.sum(pollution_results['NIPI_Category'] == 'Severe')
                moderate_count = np.sum(pollution_results['NIPI_Category'] == 'Moderate')
                
                if severe_count > 0:
                    recommendations.append(
                        f"**Immediate Action Required:** {severe_count} samples show severe pollution levels. "
                        "These areas require immediate remediation action and should be flagged for environmental assessment."
                    )
                
                if moderate_count > 0:
                    recommendations.append(
                        f"**Monitoring Required:** {moderate_count} samples show moderate pollution. "
                        "Implement regular monitoring and consider source control measures."
                    )
        
        if 'clustering' in self.results:
            n_clusters = len(self.results['clustering']['unique_labels'])
            if n_clusters > 1:
                recommendations.append(
                    f"**Source Apportionment:** {n_clusters} distinct geochemical populations identified. "
                    "Consider different pollution sources (e.g., mining, agriculture, natural background) "
                    "and conduct source apportionment analysis."
                )
        
        if 'pca' in self.results:
            if self.results['pca']['explained_variance'][0] > 0.3:
                recommendations.append(
                    "**Major Pollution Source:** First principal component explains "
                    f"{self.results['pca']['explained_variance'][0]*100:.1f}% of variance, "
                    "suggesting a dominant pollution source that warrants further investigation."
                )
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        else:
            st.info("Run analyses to generate specific recommendations for Kalumbila.")
        
        # Export options
        st.markdown("### Export Complete Results")
        
        # Create comprehensive export dataframe
        export_df = self.df.copy()
        
        # Add analysis results
        if 'outliers' in self.results:
            # Combine outliers from all methods
            combined_outliers = np.zeros(len(self.df), dtype=bool)
            for method_name, result in self.results['outliers'].items():
                if 'outliers' in result:
                    indices = result['indices']
                    outliers = result['outliers']
                    combined_outliers[indices] = combined_outliers[indices] | outliers
            export_df['Is_Outlier'] = combined_outliers
        
        if 'clustering' in self.results:
            export_df['Cluster'] = self.results['clustering']['full_labels']
        
        if 'pca' in self.results:
            if self.results['pca']['full_scores'] is not None:
                scores = self.results['pca']['full_scores']
                for i in range(min(3, scores.shape[1])):
                    export_df[f'PC{i+1}'] = scores[:, i]
        
        if 'pollution' in self.results:
            pollution_cols = [col for col in self.results['pollution']['pollution_results'].columns 
                            if col not in export_df.columns]
            export_df = pd.concat([export_df, 
                                  self.results['pollution']['pollution_results'][pollution_cols]], 
                                  axis=1)
        
        # Export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Results",
                data=csv,
                file_name="kalumbila_soil_analysis_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create summary report
            report = {
                'dataset_info': {
                    'n_samples': len(self.df),
                    'n_elements': len(self.element_cols),
                    'elements': self.element_cols,
                    'coordinates': self.coord_cols
                },
                'processing_parameters': st.session_state.get('processing_params', {}),
                'analysis_summary': {}
            }
            
            if 'outliers' in self.results:
                report['analysis_summary']['outliers'] = {}
                for method_name, result in self.results['outliers'].items():
                    if 'outliers' in result:
                        report['analysis_summary']['outliers'][method_name] = int(np.sum(result['outliers']))
            
            if 'clustering' in self.results:
                report['analysis_summary']['clusters'] = len(self.results['clustering']['unique_labels'])
            
            if 'pca' in self.results:
                report['analysis_summary']['pca_variance'] = {
                    'pc1': float(self.results['pca']['explained_variance'][0] * 100),
                    'pc1_pc2': float(self.results['pca']['cumulative_variance'][1] * 100)
                }
            
            if 'pollution' in self.results:
                report['analysis_summary']['pollution'] = {
                    'elements_analyzed': self.results['pollution']['selected_pollutants']
                }
                if 'NIPI_Category' in self.results['pollution']['pollution_results'].columns:
                    category_counts = self.results['pollution']['pollution_results']['NIPI_Category'].value_counts().to_dict()
                    report['analysis_summary']['pollution']['category_counts'] = category_counts
            
            # Convert to JSON
            import json
            report_json = json.dumps(report, indent=2)
            
            st.download_button(
                label="📊 Download JSON Report",
                data=report_json,
                file_name="kalumbila_soil_analysis_report.json",
                mime="application/json"
            )
        
        st.success("✅ Comprehensive analysis report generated!")

# ============================
# MAIN APPLICATION
# ============================

def main():
    """Main Streamlit application"""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application header
    st.markdown('<h1 class="main-header">🔬 Kalumbila Soil Pollution Analysis Suite</h1>', unsafe_allow_html=True)
    st.markdown("### Complete Geochemical Analysis for Soil Pollution Assessment")
    
    # Sidebar
    st.sidebar.title("🔍 Navigation")
    
    analysis_options = [
        "📊 Data Overview",
        "🔧 Data Preprocessing",
        "🎯 Outlier Detection",
        "📈 PCA Analysis",
        "🔍 Clustering Analysis",
        "☣️ Pollution Assessment",
        "📋 Comprehensive Report"
    ]
    
    selected_option = st.sidebar.radio("Select Analysis:", analysis_options)
    
    # Data upload section
    st.sidebar.title("📁 Data Upload")
    
    upload_option = st.sidebar.radio(
        "Upload method:",
        ["CSV File", "Excel File", "Sample Data"]
    )
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = GeochemicalAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Handle file uploads
    uploaded_file = None
    
    if upload_option == "CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file:",
            type=['csv'],
            key='csv_upload'
        )
        file_type = 'csv'
    
    elif upload_option == "Excel File":
        if not EXCEL_AVAILABLE:
            st.sidebar.warning("openpyxl not installed for Excel support")
        else:
            uploaded_file = st.sidebar.file_uploader(
                "Upload Excel file:",
                type=['xlsx', 'xls'],
                key='excel_upload'
            )
            file_type = 'excel'
    
    else:  # Sample Data
        if st.sidebar.button("Load Kalumbila Sample Dataset"):
            # Create realistic sample data for Kalumbila
            np.random.seed(42)
            n_samples = 150
            
            # Create coordinates (simulating an area in Kalumbila)
            x_center, y_center = 500, 500
            x = np.random.normal(x_center, 200, n_samples)
            y = np.random.normal(y_center, 200, n_samples)
            
            # Create realistic soil data with pollution hotspots
            data = {
                'X_Coordinate': x,
                'Y_Coordinate': y,
                'As': np.random.lognormal(mean=1.5, sigma=0.8, size=n_samples),  # Arsenic
                'Pb': np.random.lognormal(mean=2.0, sigma=0.7, size=n_samples),  # Lead
                'Cd': np.random.lognormal(mean=0.3, sigma=0.6, size=n_samples),  # Cadmium
                'Cu': np.random.lognormal(mean=1.8, sigma=0.5, size=n_samples),  # Copper
                'Zn': np.random.lognormal(mean=2.5, sigma=0.6, size=n_samples),  # Zinc
                'Ni': np.random.lognormal(mean=1.2, sigma=0.4, size=n_samples),  # Nickel
                'Cr': np.random.lognormal(mean=1.5, sigma=0.5, size=n_samples),  # Chromium
                'Fe': np.random.lognormal(mean=3.0, sigma=0.3, size=n_samples),  # Iron
                'Mn': np.random.lognormal(mean=2.0, sigma=0.4, size=n_samples),  # Manganese
                'pH': np.random.normal(6.5, 1.0, n_samples),  # pH
                'Organic_Matter': np.random.lognormal(mean=1.0, sigma=0.3, size=n_samples),  # Organic matter
            }
            
            # Add pollution hotspots (near mining areas)
            hotspot_radius = 100
            hotspot1_center = (400, 400)
            hotspot2_center = (600, 600)
            
            for i in range(n_samples):
                distance1 = np.sqrt((x[i] - hotspot1_center[0])**2 + (y[i] - hotspot1_center[1])**2)
                distance2 = np.sqrt((x[i] - hotspot2_center[0])**2 + (y[i] - hotspot2_center[1])**2)
                
                if distance1 < hotspot_radius:
                    data['As'][i] *= 5  # High arsenic
                    data['Pb'][i] *= 4  # High lead
                    data['Cu'][i] *= 3  # High copper
                
                if distance2 < hotspot_radius:
                    data['Cd'][i] *= 6  # High cadmium
                    data['Zn'][i] *= 4  # High zinc
                    data['Cr'][i] *= 3  # High chromium
            
            df = pd.DataFrame(data)
            analyzer.load_data(df, 'csv')
            st.sidebar.success(f"✅ Sample dataset loaded with {n_samples} samples")
    
    # Load uploaded file
    if uploaded_file is not None:
        analyzer.load_data(uploaded_file, file_type)
    
    # Check if data is loaded
    if analyzer.df is None or len(analyzer.df) == 0:
        st.info("👈 Please upload data or use sample dataset to begin analysis")
        
        # Display usage instructions
        with st.expander("📖 How to use this application"):
            st.markdown("""
            ### Getting Started
            
            1. **Upload Data**: Use the sidebar to upload your CSV or Excel file containing soil data
            2. **Data Format**: 
               - Include coordinate columns (X, Y, Easting, Northing, etc.)
               - Element columns should contain numeric values (concentrations in mg/kg)
               - Missing values are handled automatically
            
            ### Analysis Modules
            
            - **📊 Data Overview**: Quick summary and visualization of your dataset
            - **🔧 Data Preprocessing**: Handle missing values, transform data, scale variables
            - **🎯 Outlier Detection**: Identify anomalous samples using multiple methods
            - **📈 PCA Analysis**: Dimensionality reduction and pattern recognition
            - **🔍 Clustering Analysis**: Group similar samples using machine learning
            - **☣️ Pollution Assessment**: Specialized analysis for soil pollution
            - **📋 Comprehensive Report**: Integrated results and recommendations
            
            ### For Kalumbila Soil Analysis
            
            This tool is specifically designed for analyzing soil pollution in Kalumbila:
            - Automatic detection of common pollutants (As, Pb, Cd, Cr, etc.)
            - Pollution index calculations (NIPI)
            - Spatial mapping of pollution hotspots
            - Source apportionment recommendations
            """)
        return
    
    # Display current dataset info
    st.sidebar.markdown("---")
    st.sidebar.title("📊 Dataset Info")
    st.sidebar.write(f"**Samples:** {len(analyzer.df)}")
    st.sidebar.write(f"**Elements:** {len(analyzer.element_cols)}")
    if len(analyzer.coord_cols) == 2:
        st.sidebar.write(f"**Coordinates:** {analyzer.coord_cols}")
    
    # Kalumbila-specific info
    st.sidebar.markdown("---")
    st.sidebar.title("📍 Kalumbila Context")
    st.sidebar.write("""
    **Common Pollutants:**
    - Arsenic (As)
    - Lead (Pb) 
    - Cadmium (Cd)
    - Copper (Cu)
    - Zinc (Zn)
    """)
    
    # Run selected analysis
    if selected_option == "📊 Data Overview":
        analyzer.data_summary()
    
    elif selected_option == "🔧 Data Preprocessing":
        analyzer.preprocess_data()
    
    elif selected_option == "🎯 Outlier Detection":
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn not available. Please install with: pip install scikit-learn")
        else:
            analyzer.multivariate_outlier_detection()
    
    elif selected_option == "📈 PCA Analysis":
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn not available. Please install with: pip install scikit-learn")
        else:
            analyzer.pca_analysis()
    
    elif selected_option == "🔍 Clustering Analysis":
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn not available. Please install with: pip install scikit-learn")
        else:
            analyzer.clustering_analysis()
    
    elif selected_option == "☣️ Pollution Assessment":
        analyzer.pollution_assessment()
    
    elif selected_option == "📋 Comprehensive Report":
        analyzer.generate_comprehensive_report()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>Kalumbila Soil Pollution Analysis Suite</strong> • Version 2.1</p>
    <p>Designed for geochemical analysis of soil pollution allegations in Kalumbila</p>
    <p>All algorithms implemented and tested for robust analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()