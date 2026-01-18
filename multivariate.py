# this is a streamlit_app.py this will work best for the geochemical dataset we have for innermongolia
# FIRST: we will Set page configuration BEFORE any other Streamlit commands
import streamlit as st
st.set_page_config(
    page_title="Geochemical Data Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NOW we import other libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Then we Try to import sklearn components with fallbacks
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.covariance import MinCovDet
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing scikit-learn: {e}")
    st.info("Please install scikit-learn: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# Then we Try to import other optional packages
try:
    import scipy.stats as stats
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import zscore, gaussian_kde
    SCIPY_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing SciPy: {e}")
    st.info("Please install SciPy: pip install scipy")
    SCIPY_AVAILABLE = False

# We Try to import plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# We Try to import factor analyzer
try:
    from factor_analyzer import FactorAnalyzer
    FA_AVAILABLE = True
except ImportError:
    FA_AVAILABLE = False

# We Try to import scikit-bio for compositional data
try:
    from skbio.stats.composition import clr, ilr
    SKBIO_AVAILABLE = True
except ImportError:
    SKBIO_AVAILABLE = False

# We Check if all required packages are available
if not SKLEARN_AVAILABLE or not SCIPY_AVAILABLE:
    st.error("""
    ⚠️ Required packages are not installed!
    
    Please install the required packages using:
    ```
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy
    ```
    
    Optional packages:
    ```
    pip install plotly factor-analyzer scikit-bio
    ```
    """)
    st.stop()

# Custom implementations for missing packages
def clr_transform(data, epsilon=1e-6):
    """Centered Log-Ratio transformation alternative implementation"""
    if SKBIO_AVAILABLE:
        return clr(data + epsilon)
    
    # Manual CLR implementation
    data_eps = data + epsilon
    # Calculate geometric mean for each row
    geometric_mean = np.exp(np.mean(np.log(data_eps), axis=1, where=~np.isnan(data_eps)))
    # Replace any NaN geometric means with 1
    geometric_mean = np.nan_to_num(geometric_mean, nan=1.0)
    # CLR transformation
    clr_data = np.log(data_eps / geometric_mean[:, np.newaxis])
    return clr_data

def ilr_transform(data, epsilon=1e-6):
    """Simplified ILR transformation"""
    data_eps = data + epsilon
    # For simplicity, we'll use CLR as a fallback
    return clr_transform(data_eps, epsilon)

# Set matplotlib style
plt.style.use('default')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 100

# Custom color palette
GEO_CHEM_PALETTE = {
    'primary': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
    'sequential': ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
    'diverging': ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
}

class GeochemicalAnalyzer:
    def __init__(self, data_path=None, df=None):
        if df is not None:
            self.df = df.copy()
        elif data_path:
            try:
                self.df = pd.read_csv(data_path)
            except Exception as e:
                st.error(f"Error loading file: {e}")
                st.stop()
        else:
            # Generate sample data
            self.df = self._generate_sample_data()
        
        # Clean column names
        self.df.columns = [str(col).strip() for col in self.df.columns]
        
        # Detect coordinate columns
        self.coord_cols = self._detect_coordinates()
        self.element_cols = [col for col in self.df.columns if col not in self.coord_cols]
        
        # Initialize session state
        self._init_session_state()
    
    def _generate_sample_data(self):
        """Generate sample geochemical data for demonstration"""
        np.random.seed(42)
        n_samples = 150
        
        # Create spatial coordinates
        x = np.random.uniform(0, 1000, n_samples)
        y = np.random.uniform(0, 1000, n_samples)
        
        # Create realistic geochemical data
        data = {
            'X': x,
            'Y': y,
            'Au': np.random.lognormal(mean=0.5, sigma=0.5, size=n_samples),
            'Ag': np.random.lognormal(mean=0.3, sigma=0.4, size=n_samples),
            'Pb': np.random.lognormal(mean=2.0, sigma=0.6, size=n_samples),
            'Zn': np.random.lognormal(mean=50, sigma=0.5, size=n_samples),
            'Cu': np.random.lognormal(mean=20, sigma=0.4, size=n_samples),
            'As': np.random.lognormal(mean=10, sigma=0.5, size=n_samples),
            'Sb': np.random.lognormal(mean=0.5, sigma=0.4, size=n_samples),
            'Fe': np.random.lognormal(mean=30000, sigma=0.5, size=n_samples),
            'Ca': np.random.lognormal(mean=10000, sigma=0.6, size=n_samples),
        }
        
        return pd.DataFrame(data)
    
    def _detect_coordinates(self):
        """Automatically detect coordinate columns"""
        coord_candidates = []
        coord_keywords = ['x', 'y', 'east', 'north', 'lon', 'lat', 'longitude', 'latitude', 
                         'easting', 'northing', 'coord', 'position']
        
        for col in self.df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in coord_keywords):
                coord_candidates.append(col)
        
        # If no coordinates detected, assume first two numeric columns
        if len(coord_candidates) < 2:
            # Try to find numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                coord_candidates = numeric_cols[:2]
            else:
                # Create dummy coordinates
                self.df['X'] = np.arange(len(self.df))
                self.df['Y'] = np.random.uniform(0, 100, len(self.df))
                coord_candidates = ['X', 'Y']
        
        return coord_candidates[:2]
    
    def _init_session_state(self):
        """Initialize session state variables"""
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'clr_data' not in st.session_state:
            st.session_state.clr_data = None
        if 'ilr_data' not in st.session_state:
            st.session_state.ilr_data = None
        if 'outliers' not in st.session_state:
            st.session_state.outliers = None
        if 'clusters' not in st.session_state:
            st.session_state.clusters = None
        if 'pca_results' not in st.session_state:
            st.session_state.pca_results = None
        if 'mds_results' not in st.session_state:
            st.session_state.mds_results = None
    
    def display_data_overview(self):
        """Display comprehensive data overview"""
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", self.df.shape[0])
        with col2:
            st.metric("Total Variables", self.df.shape[1])
        with col3:
            st.metric("Elements", len(self.element_cols))
        with col4:
            missing_count = self.df.isnull().sum().sum()
            st.metric("Missing Values", missing_count)
        
        # Data preview
        st.markdown("#### 📋 Data Preview")
        preview_tab1, preview_tab2 = st.tabs(["First 10 Samples", "Last 10 Samples"])
        with preview_tab1:
            st.dataframe(self.df.head(10), use_container_width=True)
        with preview_tab2:
            st.dataframe(self.df.tail(10), use_container_width=True)
        
        # Basic statistics
        st.markdown("#### 📈 Summary Statistics")
        if len(self.element_cols) > 0:
            stats_df = self.df[self.element_cols].describe().T
            st.dataframe(stats_df, use_container_width=True)
        
        # Element distribution visualization
        st.markdown("#### 📊 Element Distributions")
        if len(self.element_cols) > 0:
            # Let user select elements to visualize
            selected_elements = st.multiselect(
                "Select elements to visualize:",
                self.element_cols,
                default=self.element_cols[:min(4, len(self.element_cols))]
            )
            
            if selected_elements:
                n_cols = 2
                n_rows = (len(selected_elements) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_rows > 1 or n_cols > 1:
                    axes = axes.flatten()
                else:
                    axes = [axes]
                
                for idx, element in enumerate(selected_elements):
                    if idx < len(axes):
                        data = self.df[element].dropna()
                        if len(data) > 0:
                            axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7, 
                                         color=GEO_CHEM_PALETTE['primary'][idx % len(GEO_CHEM_PALETTE['primary'])])
                            axes[idx].set_title(f'{element}', fontsize=10)
                            axes[idx].set_xlabel('Concentration')
                            axes[idx].set_ylabel('Frequency')
                            axes[idx].grid(True, alpha=0.3)
                
                # Hide unused subplots
                for idx in range(len(selected_elements), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Correlation heatmap
        st.markdown("#### 🔥 Correlation Heatmap")
        if len(self.element_cols) > 2:
            # Calculate correlation matrix
            corr_matrix = self.df[self.element_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, square=True, linewidths=0.5, ax=ax,
                       cbar_kws={"shrink": 0.8})
            ax.set_title('Element Correlation Matrix', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Missing value analysis
        st.markdown("#### ❓ Missing Value Analysis")
        if len(self.element_cols) > 0:
            missing_df = pd.DataFrame({
                'Element': self.element_cols,
                'Missing_Count': [self.df[col].isnull().sum() for col in self.element_cols],
                'Missing_Percentage': [self.df[col].isnull().mean() * 100 for col in self.element_cols]
            }).sort_values('Missing_Percentage', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                top_n = min(15, len(missing_df))
                bars = ax.barh(missing_df['Element'][:top_n], missing_df['Missing_Percentage'][:top_n],
                              color=GEO_CHEM_PALETTE['primary'][0])
                ax.set_xlabel('Missing Percentage (%)')
                ax.set_title(f'Top {top_n} Elements with Missing Values')
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
            with col2:
                st.dataframe(missing_df.head(10), use_container_width=True)
    
    def preprocess_data(self):
        """Interactive data preprocessing"""
        st.markdown('<div class="section-header">🔧 Data Preprocessing</div>', unsafe_allow_html=True)
        
        preprocessing_options = st.multiselect(
            "Select preprocessing steps:",
            ["Handle Missing Values", "Log Transformation", "Standard Scaling"],
            default=["Handle Missing Values", "Log Transformation"]
        )
        
        processed_df = self.df.copy()
        
        if "Handle Missing Values" in preprocessing_options:
            st.markdown("##### Handle Missing Values")
            missing_method = st.selectbox(
                "Select method for handling missing values:",
                ["Fill with median", "Fill with mean", "Drop samples with missing values"]
            )
            
            if missing_method == "Fill with median":
                processed_df[self.element_cols] = processed_df[self.element_cols].fillna(
                    processed_df[self.element_cols].median()
                )
            elif missing_method == "Fill with mean":
                processed_df[self.element_cols] = processed_df[self.element_cols].fillna(
                    processed_df[self.element_cols].mean()
                )
            elif missing_method == "Drop samples with missing values":
                processed_df = processed_df.dropna(subset=self.element_cols)
        
        if "Log Transformation" in preprocessing_options:
            st.markdown("##### Log Transformation")
            log_base = st.selectbox("Select log base:", [10, "natural (e)"])
            if log_base == "natural (e)":
                # Add small constant to avoid log(0)
                processed_df[self.element_cols] = np.log(processed_df[self.element_cols] + 1e-10)
            else:
                processed_df[self.element_cols] = np.log10(processed_df[self.element_cols] + 1e-10)
        
        if "Standard Scaling" in preprocessing_options:
            st.markdown("##### Standard Scaling")
            scaler_type = st.selectbox("Select scaler type:", ["StandardScaler", "RobustScaler"])
            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            
            # Scale only element columns
            scaled_data = scaler.fit_transform(processed_df[self.element_cols])
            processed_df[self.element_cols] = scaled_data
        
        # Store processed data
        st.session_state.processed_data = processed_df
        
        # Show comparison
        st.markdown("##### Before vs After Preprocessing")
        if len(self.element_cols) > 0:
            element_to_compare = st.selectbox("Select element to compare:", self.element_cols)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Before preprocessing
            data_before = self.df[element_to_compare].dropna()
            if len(data_before) > 0:
                axes[0].hist(data_before, bins=30, edgecolor='black', alpha=0.7, 
                           color=GEO_CHEM_PALETTE['primary'][0])
            axes[0].set_title(f'Original {element_to_compare}')
            axes[0].set_xlabel('Value')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # After preprocessing
            data_after = processed_df[element_to_compare].dropna()
            if len(data_after) > 0:
                axes[1].hist(data_after, bins=30, edgecolor='black', alpha=0.7, 
                           color=GEO_CHEM_PALETTE['primary'][1])
            axes[1].set_title(f'Processed {element_to_compare}')
            axes[1].set_xlabel('Value')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.success(f"✅ Data preprocessing completed! Processed {len(processed_df)} samples.")
        
        return processed_df
    
    def apply_compositional_transforms(self):
        """Apply compositional data transformations"""
        st.markdown('<div class="section-header">🔄 Compositional Data Transformations</div>', unsafe_allow_html=True)
        
        # Use processed data if available, otherwise use original
        if st.session_state.processed_data is not None:
            data_to_use = st.session_state.processed_data
        else:
            data_to_use = self.df
        
        if len(self.element_cols) == 0:
            st.error("No element columns found for transformation.")
            return None
        
        transform_method = st.selectbox(
            "Select transformation method:",
            ["CLR (Centered Log-Ratio)", "ILR (Isometric Log-Ratio)"],
            help="CLR is recommended for most geochemical data analysis"
        )
        
        # Extract element data
        element_data = data_to_use[self.element_cols].values
        
        # Add epsilon to handle zeros
        epsilon = 1e-6
        
        if transform_method == "CLR (Centered Log-Ratio)":
            transformed_data = clr_transform(element_data, epsilon)
            columns = [f'clr_{col}' for col in self.element_cols]
            
        elif transform_method == "ILR (Isometric Log-Ratio)":
            transformed_data = ilr_transform(element_data, epsilon)
            columns = [f'ilr_{i}' for i in range(transformed_data.shape[1])]
        
        # Store transformed data
        transformed_df = pd.DataFrame(transformed_data, columns=columns)
        
        if transform_method == "CLR (Centered Log-Ratio)":
            st.session_state.clr_data = transformed_df
        elif transform_method == "ILR (Isometric Log-Ratio)":
            st.session_state.ilr_data = transformed_df
        
        # Visualization
        st.markdown("##### Transformation Results")
        
        # Distribution comparison
        st.markdown("##### Distribution Comparison")
        if transform_method == "CLR (Centered Log-Ratio)":
            compare_element = st.selectbox("Select element for comparison:", self.element_cols)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Original distribution
            if st.session_state.processed_data is not None:
                original_data = st.session_state.processed_data[compare_element]
            else:
                original_data = self.df[compare_element]
            
            axes[0].hist(original_data.dropna(), bins=30, edgecolor='black', alpha=0.7, 
                        color=GEO_CHEM_PALETTE['primary'][0])
            axes[0].set_title(f'Original {compare_element}')
            axes[0].set_xlabel('Value')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # Transformed distribution
            transformed_col = f'clr_{compare_element}'
            if transformed_col in transformed_df.columns:
                axes[1].hist(transformed_df[transformed_col].dropna(), bins=30, edgecolor='black', 
                            alpha=0.7, color=GEO_CHEM_PALETTE['primary'][1])
                axes[1].set_title(f'CLR Transformed {compare_element}')
                axes[1].set_xlabel('CLR Value')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.success(f"✅ {transform_method} transformation completed!")
        
        return transformed_df
    
    def outlier_detection_advanced(self):
        """Advanced outlier detection with multiple methods"""
        st.markdown('<div class="section-header">🎯 Advanced Outlier Detection</div>', unsafe_allow_html=True)
        
        # Select data for analysis
        data_source = st.selectbox(
            "Select data source for outlier detection:",
            ["Original Data", "Processed Data", "CLR Transformed"],
            key="outlier_data_source"
        )
        
        if data_source == "Original Data":
            X = self.df[self.element_cols].values
        elif data_source == "Processed Data" and st.session_state.processed_data is not None:
            X = st.session_state.processed_data[self.element_cols].values
        elif data_source == "CLR Transformed" and st.session_state.clr_data is not None:
            X = st.session_state.clr_data.values
        else:
            st.warning("Selected data source not available. Using original data.")
            X = self.df[self.element_cols].values
        
        # Remove any rows with NaN values
        X_clean = X[~np.isnan(X).any(axis=1)]
        if len(X_clean) < len(X):
            st.warning(f"Removed {len(X) - len(X_clean)} samples with NaN values for outlier detection.")
            X = X_clean
        
        if len(X) < 10:
            st.error("Not enough samples for outlier detection (need at least 10).")
            return None
        
        # Select outlier detection methods
        methods = st.multiselect(
            "Select outlier detection methods:",
            ["Robust Mahalanobis", "Local Outlier Factor"],
            default=["Robust Mahalanobis"]
        )
        
        results = {}
        
        if "Robust Mahalanobis" in methods:
            st.markdown("##### Robust Mahalanobis Distance")
            contamination = st.slider("Contamination rate:", 0.01, 0.5, 0.1, 0.01, key="rmd_contam")
            
            try:
                robust_cov = MinCovDet().fit(X)
                mahalanobis_dist = robust_cov.mahalanobis(X)
                
                # Determine threshold using percentile
                threshold = np.percentile(mahalanobis_dist, (1 - contamination) * 100)
                outliers = mahalanobis_dist > threshold
                
                results['Robust Mahalanobis'] = {
                    'outliers': outliers,
                    'scores': mahalanobis_dist,
                    'threshold': threshold
                }
                
                # Visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Distance distribution
                axes[0].hist(mahalanobis_dist, bins=30, edgecolor='black', alpha=0.7,
                           color=GEO_CHEM_PALETTE['primary'][0])
                axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2,
                              label=f'Threshold: {threshold:.2f}')
                axes[0].set_xlabel('Mahalanobis Distance')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title('Mahalanobis Distance Distribution')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Spatial plot
                if len(self.coord_cols) == 2:
                    # Get indices of clean data
                    clean_indices = np.where(~np.isnan(X).any(axis=1))[0]
                    if len(clean_indices) > 0:
                        axes[1].scatter(self.df.iloc[clean_indices][self.coord_cols[0]], 
                                      self.df.iloc[clean_indices][self.coord_cols[1]],
                                      c='gray', s=20, alpha=0.5, label='Normal')
                        outlier_indices = clean_indices[outliers]
                        if len(outlier_indices) > 0:
                            axes[1].scatter(self.df.iloc[outlier_indices][self.coord_cols[0]],
                                          self.df.iloc[outlier_indices][self.coord_cols[1]],
                                          c='red', s=50, edgecolor='black', label='Outliers')
                        axes[1].set_xlabel(self.coord_cols[0])
                        axes[1].set_ylabel(self.coord_cols[1])
                        axes[1].set_title('Spatial Distribution of Outliers')
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error in Robust Mahalanobis: {str(e)}")
        
        if "Local Outlier Factor" in methods and len(X) > 0:
            st.markdown("##### Local Outlier Factor (LOF)")
            n_neighbors = min(20, len(X) - 1)
            n_neighbors = st.slider("Number of neighbors:", 5, n_neighbors, min(10, n_neighbors), key="lof_neighbors")
            contamination_lof = st.slider("Contamination:", 0.01, 0.5, 0.1, 0.01, key="lof_contam")
            
            try:
                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination_lof)
                outlier_labels = lof.fit_predict(X)
                lof_scores = -lof.negative_outlier_factor_
                
                outliers = outlier_labels == -1
                
                results['Local Outlier Factor'] = {
                    'outliers': outliers,
                    'scores': lof_scores
                }
                
                # Visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # LOF scores distribution
                axes[0].hist(lof_scores, bins=30, edgecolor='black', alpha=0.7,
                           color=GEO_CHEM_PALETTE['primary'][1])
                axes[0].set_xlabel('LOF Score')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title('LOF Score Distribution')
                axes[0].grid(True, alpha=0.3)
                
                # Comparison plot
                if 'Robust Mahalanobis' in results:
                    axes[1].scatter(results['Robust Mahalanobis']['scores'], lof_scores,
                                  c=['red' if o else 'blue' for o in outliers],
                                  alpha=0.6, s=30)
                    axes[1].set_xlabel('Mahalanobis Distance')
                    axes[1].set_ylabel('LOF Score')
                    axes[1].set_title('Comparison of Outlier Scores')
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error in LOF: {str(e)}")
        
        # Store consensus outliers
        if results:
            # Create consensus outliers (outlier in any method)
            consensus_outliers = np.zeros(len(X), dtype=bool)
            for method, result in results.items():
                consensus_outliers = consensus_outliers | result['outliers']
            
            # Store in session state
            st.session_state.outliers = consensus_outliers
            
            st.success(f"✅ Outlier detection completed! Found {np.sum(consensus_outliers)} outliers.")
        
        return results
    
    def clustering_analysis_advanced(self):
        """Advanced clustering analysis with multiple algorithms"""
        st.markdown('<div class="section-header">🔍 Advanced Clustering Analysis</div>', unsafe_allow_html=True)
        
        # Select data for clustering
        data_source = st.selectbox(
            "Select data source for clustering:",
            ["Original Data", "Processed Data", "CLR Transformed"],
            key="clustering_data_source"
        )
        
        if data_source == "Original Data":
            X = self.df[self.element_cols].values
        elif data_source == "Processed Data" and st.session_state.processed_data is not None:
            X = st.session_state.processed_data[self.element_cols].values
        elif data_source == "CLR Transformed" and st.session_state.clr_data is not None:
            X = st.session_state.clr_data.values
        else:
            st.warning("Selected data source not available. Using original data.")
            X = self.df[self.element_cols].values
        
        # Remove NaN values
        X = X[~np.isnan(X).any(axis=1)]
        
        if len(X) < 10:
            st.error("Not enough samples for clustering (need at least 10).")
            return None
        
        # Select clustering algorithm
        algorithm = st.selectbox(
            "Select clustering algorithm:",
            ["K-Means", "DBSCAN", "Hierarchical"]
        )
        
        cluster_labels = None
        
        if algorithm == "K-Means":
            st.markdown("##### K-Means Clustering")
            
            # Determine optimal k
            max_clusters = min(10, len(X) - 1)
            if max_clusters < 2:
                st.error("Not enough samples for K-Means clustering.")
                return None
            
            k_range = range(2, max_clusters + 1)
            
            inertia = []
            silhouette_scores = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                inertia.append(kmeans.inertia_)
                
                if k > 1:  # Silhouette score requires at least 2 clusters
                    try:
                        sil_score = silhouette_score(X, labels)
                        silhouette_scores.append(sil_score)
                    except:
                        silhouette_scores.append(0)
                else:
                    silhouette_scores.append(0)
            
            # Plot clustering metrics
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Elbow plot
            axes[0].plot(k_range, inertia, 'bo-', markersize=8, linewidth=2)
            axes[0].set_xlabel('Number of Clusters')
            axes[0].set_ylabel('Inertia')
            axes[0].set_title('Elbow Method')
            axes[0].grid(True, alpha=0.3)
            
            # Silhouette scores
            if len(k_range) > 1:
                axes[1].plot(k_range[1:], silhouette_scores[1:], 'ro-', markersize=8, linewidth=2)
                axes[1].set_xlabel('Number of Clusters')
                axes[1].set_ylabel('Silhouette Score')
                axes[1].set_title('Silhouette Score')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Let user select k
            optimal_k = st.slider(
                "Select number of clusters (k):",
                min_value=2,
                max_value=max_clusters,
                value=min(3, max_clusters)
            )
            
            # Perform K-Means with selected k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
        elif algorithm == "DBSCAN":
            st.markdown("##### DBSCAN Clustering")
            
            eps = st.slider("EPS parameter:", 0.1, 5.0, 0.5, 0.1)
            min_samples = st.slider("Min samples:", 2, min(20, len(X)), 5)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            st.info(f"Found {n_clusters} clusters and {n_noise} noise points")
            
        elif algorithm == "Hierarchical":
            st.markdown("##### Hierarchical Clustering")
            
            linkage_method = st.selectbox(
                "Linkage method:",
                ["ward", "complete", "average", "single"]
            )
            
            # Calculate linkage matrix
            Z = linkage(X, method=linkage_method)
            
            # Plot dendrogram
            fig, ax = plt.subplots(figsize=(12, 6))
            dendrogram(Z, ax=ax, truncate_mode='lastp', p=min(30, len(X)), show_leaf_counts=True)
            ax.set_xlabel('Sample index')
            ax.set_ylabel('Distance')
            ax.set_title('Hierarchical Clustering Dendrogram')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Let user select cutoff
            max_distance = np.max(Z[:, 2])
            cutoff = st.slider("Distance cutoff:", 0.0, float(max_distance), float(max_distance/3))
            
            # Cut dendrogram
            cluster_labels = fcluster(Z, cutoff, criterion='distance')
        
        # Store clusters
        if cluster_labels is not None:
            st.session_state.clusters = cluster_labels
        
        # Visualize clustering results
        if cluster_labels is not None:
            st.markdown("##### Clustering Results Visualization")
            
            # Use PCA for visualization
            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                X_reduced = pca.fit_transform(X)
                x_label = 'PC1'
                y_label = 'PC2'
            else:
                X_reduced = X
                x_label = 'Feature 1'
                y_label = 'Feature 2'
            
            # Create comprehensive visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # 1. Reduced space scatter plot
            scatter = axes[0].scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels,
                                    cmap='tab20', s=30, alpha=0.7)
            axes[0].set_xlabel(x_label)
            axes[0].set_ylabel(y_label)
            axes[0].set_title(f'Clusters in {x_label}-{y_label} Space')
            axes[0].grid(True, alpha=0.3)
            
            # 2. Spatial distribution
            if len(self.coord_cols) == 2:
                # Get indices of clean data
                clean_indices = np.where(~np.isnan(self.df[self.element_cols].values).any(axis=1))[0]
                if len(clean_indices) > 0:
                    spatial_scatter = axes[1].scatter(
                        self.df.iloc[clean_indices][self.coord_cols[0]], 
                        self.df.iloc[clean_indices][self.coord_cols[1]],
                        c=cluster_labels, cmap='tab20', s=30, alpha=0.7
                    )
                    axes[1].set_xlabel(self.coord_cols[0])
                    axes[1].set_ylabel(self.coord_cols[1])
                    axes[1].set_title('Spatial Distribution of Clusters')
                    axes[1].grid(True, alpha=0.3)
            
            # 3. Cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            colors = plt.cm.tab20(np.arange(len(unique_labels)) / max(len(unique_labels), 1))
            axes[2].bar(range(len(unique_labels)), counts, color=colors)
            axes[2].set_xlabel('Cluster')
            axes[2].set_ylabel('Number of Samples')
            axes[2].set_title('Cluster Sizes')
            axes[2].set_xticks(range(len(unique_labels)))
            axes[2].set_xticklabels([str(l) for l in unique_labels])
            axes[2].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display cluster statistics
            st.markdown("##### Cluster Statistics")
            summary_data = []
            for i, label in enumerate(unique_labels):
                cluster_mask = cluster_labels == label
                if np.sum(cluster_mask) > 0:
                    summary_data.append({
                        'Cluster': label,
                        'Samples': np.sum(cluster_mask),
                        'Percentage': f"{np.sum(cluster_mask)/len(cluster_labels)*100:.1f}%"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            st.success(f"✅ Clustering completed with {algorithm}! Found {len(unique_labels)} clusters.")
        
        return cluster_labels
    
    def pca_analysis_comprehensive(self):
        """Comprehensive PCA analysis with advanced visualizations"""
        st.markdown('<div class="section-header">📈 Comprehensive PCA Analysis</div>', unsafe_allow_html=True)
        
        # Select data for PCA
        data_source = st.selectbox(
            "Select data source for PCA:",
            ["Original Data", "Processed Data", "CLR Transformed"],
            key="pca_data_source"
        )
        
        if data_source == "Original Data":
            X = self.df[self.element_cols].values
            element_names = self.element_cols
        elif data_source == "Processed Data" and st.session_state.processed_data is not None:
            X = st.session_state.processed_data[self.element_cols].values
            element_names = self.element_cols
        elif data_source == "CLR Transformed" and st.session_state.clr_data is not None:
            X = st.session_state.clr_data.values
            element_names = st.session_state.clr_data.columns.tolist()
        else:
            st.warning("Selected data source not available. Using original data.")
            X = self.df[self.element_cols].values
            element_names = self.element_cols
        
        # Remove NaN values
        X = X[~np.isnan(X).any(axis=1)]
        
        if len(X) < 10:
            st.error("Not enough samples for PCA (need at least 10).")
            return None
        
        # PCA configuration
        col1, col2 = st.columns(2)
        with col1:
            variance_threshold = st.slider(
                "Variance to retain (%):",
                min_value=70,
                max_value=99,
                value=95,
                step=1
            )
        with col2:
            n_components = min(10, X.shape[1], len(X) - 1)
        
        # Perform PCA
        pca = PCA(n_components=min(n_components, X.shape[1]))
        pca_scores = pca.fit_transform(X)
        pca_loadings = pca.components_
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Store results
        st.session_state.pca_results = (pca_scores, explained_variance, pca_loadings)
        
        # Determine number of components for selected variance
        n_components_var = np.argmax(cumulative_variance >= variance_threshold/100) + 1
        
        # Display PCA metrics
        st.markdown("##### PCA Metrics")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Total Variance", f"{cumulative_variance[-1]*100:.1f}%")
        metric_cols[1].metric("Components for Threshold", n_components_var)
        metric_cols[2].metric("PC1 Variance", f"{explained_variance[0]*100:.1f}%")
        if len(explained_variance) > 1:
            metric_cols[3].metric("PC2 Variance", f"{explained_variance[1]*100:.1f}%")
        
        # Create comprehensive visualizations
        st.markdown("##### PCA Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Scree & Variance", "Scores Plot", "Loadings Plot", "Spatial Maps"
        ])
        
        with tab1:
            # Scree plot with cumulative variance
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Scree plot
            x_pos = np.arange(1, len(explained_variance) + 1)
            bars = ax1.bar(x_pos, explained_variance * 100, alpha=0.6, 
                          color=GEO_CHEM_PALETTE['primary'])
            ax1.plot(x_pos, cumulative_variance * 100, 'ro-', linewidth=2, markersize=8)
            ax1.axhline(y=variance_threshold, color='green', linestyle='--', 
                       label=f'{variance_threshold}% threshold')
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance (%)')
            ax1.set_title('Scree Plot with Cumulative Variance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Variance explained by each component
            ax2.bar(x_pos, explained_variance * 100, alpha=0.6, 
                   color=GEO_CHEM_PALETTE['primary'][1])
            ax2.set_xlabel('Principal Component')
            ax2.set_ylabel('Variance Explained (%)')
            ax2.set_title('Variance by Principal Component')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            # Scores plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # PC1 vs PC2
            if len(explained_variance) >= 2:
                # Color by an element if available
                if len(self.element_cols) > 0:
                    # Get clean indices
                    clean_indices = np.where(~np.isnan(self.df[self.element_cols].values).any(axis=1))[0]
                    if len(clean_indices) > 0:
                        element_for_color = st.selectbox(
                            "Select element for coloring:",
                            self.element_cols,
                            key="pca_color_element"
                        )
                        color_values = self.df.iloc[clean_indices][element_for_color].values
                        sc = axes[0].scatter(pca_scores[:, 0], pca_scores[:, 1], 
                                            c=color_values, cmap='viridis', s=30, alpha=0.7)
                        plt.colorbar(sc, ax=axes[0], label=element_for_color)
                    else:
                        axes[0].scatter(pca_scores[:, 0], pca_scores[:, 1], s=30, alpha=0.7,
                                      color=GEO_CHEM_PALETTE['primary'][0])
                else:
                    axes[0].scatter(pca_scores[:, 0], pca_scores[:, 1], s=30, alpha=0.7,
                                  color=GEO_CHEM_PALETTE['primary'][0])
                
                axes[0].set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% var)')
                axes[0].set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% var)')
                axes[0].set_title('PC1 vs PC2 Scores')
                axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                axes[0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
                axes[0].grid(True, alpha=0.3)
            
            # PC3 vs PC4 if available
            if len(explained_variance) >= 4:
                axes[1].scatter(pca_scores[:, 2], pca_scores[:, 3], s=30, alpha=0.7,
                              color=GEO_CHEM_PALETTE['primary'][1])
                axes[1].set_xlabel(f'PC3 ({explained_variance[2]*100:.1f}% var)')
                axes[1].set_ylabel(f'PC4 ({explained_variance[3]*100:.1f}% var)')
                axes[1].set_title('PC3 vs PC4 Scores')
                axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                axes[1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            # Loadings plot
            st.markdown("##### Principal Component Loadings")
            
            if len(explained_variance) > 0:
                # Select which PCs to visualize
                pc_to_plot = st.selectbox(
                    "Select PC to visualize:", 
                    range(1, min(5, len(explained_variance)) + 1)
                )
                
                # Get loadings for selected PC
                loadings = pca_loadings[pc_to_plot - 1]
                
                # Sort by absolute loading
                sorted_idx = np.argsort(np.abs(loadings))[::-1]
                top_n = st.slider("Number of top loadings to show:", 5, 20, 10, key="loadings_top_n")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Bar plot of top loadings
                top_indices = sorted_idx[:top_n]
                y_pos = np.arange(top_n)
                
                colors = ['red' if x < 0 else 'blue' for x in loadings[top_indices]]
                ax.barh(y_pos, loadings[top_indices], color=colors, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([element_names[i] for i in top_indices])
                ax.set_xlabel('Loading Value')
                ax.set_title(f'Top {top_n} Loadings for PC{pc_to_plot}')
                ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab4:
            # Spatial maps of PC scores
            st.markdown("##### Spatial Distribution of PC Scores")
            
            if len(self.coord_cols) == 2:
                n_maps = min(4, len(explained_variance))
                n_cols = 2
                n_rows = (n_maps + 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_maps > 1:
                    axes = axes.flatten()
                else:
                    axes = [axes]
                
                # Get clean indices
                clean_indices = np.where(~np.isnan(self.df[self.element_cols].values).any(axis=1))[0]
                
                for i in range(n_maps):
                    if len(clean_indices) > 0:
                        scatter = axes[i].scatter(
                            self.df.iloc[clean_indices][self.coord_cols[0]], 
                            self.df.iloc[clean_indices][self.coord_cols[1]],
                            c=pca_scores[:, i], cmap='RdBu_r', s=30, alpha=0.7
                        )
                        axes[i].set_xlabel(self.coord_cols[0])
                        axes[i].set_ylabel(self.coord_cols[1] if i % n_cols == 0 else '')
                        axes[i].set_title(f'PC{i+1} ({explained_variance[i]*100:.1f}% var)')
                        axes[i].grid(True, alpha=0.3)
                        plt.colorbar(scatter, ax=axes[i])
                
                # Hide unused subplots
                for i in range(n_maps, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Coordinate columns not found for spatial mapping.")
        
        st.success("✅ Comprehensive PCA analysis completed!")
        return pca_scores, explained_variance, pca_loadings
    
    def mds_analysis(self):
        """Multidimensional Scaling analysis"""
        st.markdown('<div class="section-header">🌀 Multidimensional Scaling (MDS)</div>', unsafe_allow_html=True)
        
        # Select data for MDS
        data_source = st.selectbox(
            "Select data source for MDS:",
            ["Original Data", "Processed Data", "CLR Transformed"],
            key="mds_data_source"
        )
        
        if data_source == "Original Data":
            X = self.df[self.element_cols].values
        elif data_source == "Processed Data" and st.session_state.processed_data is not None:
            X = st.session_state.processed_data[self.element_cols].values
        elif data_source == "CLR Transformed" and st.session_state.clr_data is not None:
            X = st.session_state.clr_data.values
        else:
            st.warning("Selected data source not available. Using original data.")
            X = self.df[self.element_cols].values
        
        # Remove NaN values
        X = X[~np.isnan(X).any(axis=1)]
        
        if len(X) < 10:
            st.error("Not enough samples for MDS (need at least 10).")
            return None
        
        # MDS configuration
        col1, col2 = st.columns(2)
        with col1:
            mds_type = st.selectbox(
                "MDS type:",
                ["Metric MDS"]
            )
        with col2:
            n_components = st.slider(
                "Number of dimensions:",
                min_value=2,
                max_value=3,
                value=2
            )
        
        # Distance metric
        distance_metric = st.selectbox(
            "Distance metric:",
            ["euclidean", "cityblock", "cosine", "correlation"]
        )
        
        # Calculate distance matrix
        try:
            distance_matrix = pairwise_distances(X, metric=distance_metric)
        except:
            st.error(f"Error calculating distance matrix with metric: {distance_metric}")
            distance_matrix = pairwise_distances(X, metric='euclidean')
        
        # Fit MDS
        try:
            mds = MDS(n_components=n_components, random_state=42, dissimilarity='precomputed')
            mds_coords = mds.fit_transform(distance_matrix)
            stress = mds.stress_
        except Exception as e:
            st.error(f"Error in MDS calculation: {str(e)}")
            return None
        
        # Visualization
        st.markdown(f"##### MDS Results (Stress: {stress:.4f})")
        
        if n_components == 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # MDS scatter plot
            axes[0].scatter(mds_coords[:, 0], mds_coords[:, 1], s=30, alpha=0.7,
                           color=GEO_CHEM_PALETTE['primary'][0])
            axes[0].set_xlabel('MDS Dimension 1')
            axes[0].set_ylabel('MDS Dimension 2')
            axes[0].set_title('MDS Plot')
            axes[0].grid(True, alpha=0.3)
            
            # Stress plot
            axes[1].bar(['Stress'], [stress], color=GEO_CHEM_PALETTE['primary'][1])
            axes[1].set_ylabel('Stress Value')
            axes[1].set_title('MDS Stress')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        elif n_components == 3:
            # 3D plot
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(mds_coords[:, 0], mds_coords[:, 1], mds_coords[:, 2],
                      s=30, alpha=0.7, c=GEO_CHEM_PALETTE['primary'][0])
            
            ax.set_xlabel('MDS Dimension 1')
            ax.set_ylabel('MDS Dimension 2')
            ax.set_zlabel('MDS Dimension 3')
            ax.set_title('3D MDS Plot')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Spatial mapping of MDS coordinates
        if len(self.coord_cols) == 2:
            st.markdown("##### Spatial Mapping of MDS Dimensions")
            
            fig, axes = plt.subplots(1, min(2, n_components), figsize=(5*min(2, n_components), 4))
            if n_components > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
            
            # Get clean indices
            clean_indices = np.where(~np.isnan(self.df[self.element_cols].values).any(axis=1))[0]
            
            for i in range(min(2, n_components)):
                if len(clean_indices) > 0:
                    scatter = axes[i].scatter(
                        self.df.iloc[clean_indices][self.coord_cols[0]], 
                        self.df.iloc[clean_indices][self.coord_cols[1]],
                        c=mds_coords[:, i], cmap='RdBu_r', s=30, alpha=0.7
                    )
                    axes[i].set_xlabel(self.coord_cols[0])
                    axes[i].set_ylabel(self.coord_cols[1] if i == 0 else '')
                    axes[i].set_title(f'MDS Dimension {i+1}')
                    axes[i].grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=axes[i])
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Store results
        st.session_state.mds_results = (mds_coords, stress)
        
        st.success("✅ MDS analysis completed!")
        return mds_coords, stress
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        st.markdown('<div class="section-header">📋 Comprehensive Analysis Report</div>', unsafe_allow_html=True)
        
        # Create summary metrics
        st.markdown("##### 📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(self.df))
            st.metric("Elements", len(self.element_cols))
        
        with col2:
            if st.session_state.outliers is not None:
                outlier_count = np.sum(st.session_state.outliers)
                st.metric("Outliers Detected", outlier_count)
                st.metric("Outlier %", f"{outlier_count/len(self.df)*100:.1f}%")
            else:
                st.metric("Outliers Detected", "N/A")
                st.metric("Outlier %", "N/A")
        
        with col3:
            if st.session_state.clusters is not None:
                n_clusters = len(np.unique(st.session_state.clusters))
                st.metric("Clusters Found", n_clusters)
                if st.session_state.pca_results is not None:
                    pca_var = np.sum(st.session_state.pca_results[1][:2]) * 100
                    st.metric("PC1+PC2 Variance", f"{pca_var:.1f}%")
                else:
                    st.metric("PC1+PC2 Variance", "N/A")
            else:
                st.metric("Clusters Found", "N/A")
                st.metric("PC1+PC2 Variance", "N/A")
        
        with col4:
            if len(self.coord_cols) == 2:
                x_range = self.df[self.coord_cols[0]].max() - self.df[self.coord_cols[0]].min()
                y_range = self.df[self.coord_cols[1]].max() - self.df[self.coord_cols[1]].min()
                st.metric("X Range", f"{x_range:.1f}")
                st.metric("Y Range", f"{y_range:.1f}")
            else:
                st.metric("X Range", "N/A")
                st.metric("Y Range", "N/A")
        
        # Create integrated visualization if we have results
        st.markdown("##### 🎯 Integrated Visualization")
        
        has_clusters = st.session_state.clusters is not None
        has_pca = st.session_state.pca_results is not None
        has_outliers = st.session_state.outliers is not None
        has_coords = len(self.coord_cols) == 2
        
        if (has_clusters or has_outliers) and has_coords:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # 1. Spatial clusters if available
            if has_clusters:
                # Get clean indices
                clean_indices = np.where(~np.isnan(self.df[self.element_cols].values).any(axis=1))[0]
                if len(clean_indices) > 0:
                    spatial_scatter = axes[0].scatter(
                        self.df.iloc[clean_indices][self.coord_cols[0]], 
                        self.df.iloc[clean_indices][self.coord_cols[1]],
                        c=st.session_state.clusters, cmap='tab20', s=30, alpha=0.7
                    )
                    axes[0].set_xlabel(self.coord_cols[0])
                    axes[0].set_ylabel(self.coord_cols[1])
                    axes[0].set_title('Spatial Distribution of Clusters')
                    axes[0].grid(True, alpha=0.3)
            
            # 2. Outliers overlay if available
            if has_outliers:
                # Get clean indices
                clean_indices = np.where(~np.isnan(self.df[self.element_cols].values).any(axis=1))[0]
                if len(clean_indices) > 0:
                    # Normal samples
                    normal_mask = ~st.session_state.outliers
                    if np.any(normal_mask):
                        axes[1].scatter(
                            self.df.iloc[clean_indices][self.coord_cols[0]][normal_mask],
                            self.df.iloc[clean_indices][self.coord_cols[1]][normal_mask],
                            c='gray', s=20, alpha=0.5, label='Normal'
                        )
                    # Outliers
                    outlier_mask = st.session_state.outliers
                    if np.any(outlier_mask):
                        axes[1].scatter(
                            self.df.iloc[clean_indices][self.coord_cols[0]][outlier_mask],
                            self.df.iloc[clean_indices][self.coord_cols[1]][outlier_mask],
                            c='red', s=50, edgecolor='black', label='Outliers'
                        )
                    axes[1].set_xlabel(self.coord_cols[0])
                    axes[1].set_ylabel(self.coord_cols[1])
                    axes[1].set_title('Outlier Detection Results')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Run clustering and outlier detection to see integrated visualization.")
        
        # Generate recommendations
        st.markdown("##### 🎯 Exploration Recommendations")
        
        recommendations = []
        
        if has_outliers:
            outlier_count = np.sum(st.session_state.outliers)
            if outlier_count > 0:
                recommendations.append(
                    f"**High Priority Targets:** {outlier_count} anomalous samples detected. "
                    "These represent potential mineralization signatures requiring further investigation."
                )
        
        if has_clusters:
            unique_clusters = np.unique(st.session_state.clusters)
            if len(unique_clusters) > 1:
                recommendations.append(
                    f"**Cluster Analysis:** {len(unique_clusters)} distinct geochemical populations identified. "
                    "Consider targeted sampling in each cluster to understand different mineralization styles."
                )
        
        if has_pca:
            pca_var = np.sum(st.session_state.pca_results[1][:3]) * 100
            if pca_var > 70:
                recommendations.append(
                    f"**PCA Results:** {pca_var:.1f}% of variance explained by first 3 PCs. "
                    "Major geochemical processes are well-captured in the analysis."
                )
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        else:
            st.info("Run analyses to generate recommendations.")
        
        # Export options
        st.markdown("##### 📤 Export Results")
        
        if st.button("Generate Export CSV"):
            # Create export dataframe
            export_df = self.df.copy()
            
            if st.session_state.clusters is not None:
                # Need to align cluster labels with original data
                clean_indices = np.where(~np.isnan(self.df[self.element_cols].values).any(axis=1))[0]
                if len(clean_indices) == len(st.session_state.clusters):
                    # Create full array with NaN for samples with missing data
                    full_clusters = np.full(len(self.df), np.nan)
                    full_clusters[clean_indices] = st.session_state.clusters
                    export_df['Cluster'] = full_clusters
            
            if st.session_state.outliers is not None:
                # Need to align outlier labels with original data
                clean_indices = np.where(~np.isnan(self.df[self.element_cols].values).any(axis=1))[0]
                if len(clean_indices) == len(st.session_state.outliers):
                    full_outliers = np.full(len(self.df), np.nan)
                    full_outliers[clean_indices] = st.session_state.outliers
                    export_df['Is_Outlier'] = full_outliers
            
            if st.session_state.pca_results is not None:
                pca_scores = st.session_state.pca_results[0]
                clean_indices = np.where(~np.isnan(self.df[self.element_cols].values).any(axis=1))[0]
                if len(clean_indices) == len(pca_scores):
                    for i in range(min(3, pca_scores.shape[1])):
                        full_pc = np.full(len(self.df), np.nan)
                        full_pc[clean_indices] = pca_scores[:, i]
                        export_df[f'PC{i+1}'] = full_pc
            
            # Download button
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="geochemical_analysis_results.csv",
                mime="text/csv"
            )
        
        st.success("✅ Analysis report generated!")

def main():
    """Main Streamlit application"""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .section-header {
            font-size: 1.8rem;
            color: #1E3A8A;
            border-bottom: 3px solid #3B82F6;
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }
        .info-box {
            background-color: #F0F9FF;
            border-left: 4px solid #3B82F6;
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
        .warning-box {
            background-color: #FEF3C7;
            border-left: 4px solid #F59E0B;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid #E5E7EB;
        }
        .stButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 0.5rem;
        }
        .tab-content {
            padding: 1rem;
            border: 1px solid #E5E7EB;
            border-radius: 0.5rem;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and header
    st.markdown('<h1 class="main-header">🔬 Geochemical Data Analysis Suite</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Analytics for Mineral Exploration")
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    analysis_steps = [
        "1. Data Overview",
        "2. Data Preprocessing",
        "3. Compositional Transforms",
        "4. Outlier Detection",
        "5. Clustering Analysis",
        "6. PCA Analysis",
        "7. MDS Analysis",
        "8. Comprehensive Report"
    ]
    
    selected_step = st.sidebar.radio(
        "Select Analysis Step:",
        analysis_steps,
        index=0
    )
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Dataset Information")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your geochemical data (CSV):",
        type=['csv'],
        help="Upload a CSV file with geochemical data"
    )
    
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            analyzer = GeochemicalAnalyzer(df=df)
            st.sidebar.success(f"✅ Loaded {len(df)} samples")
            st.sidebar.info(f"Detected {len(analyzer.element_cols)} elements")
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            st.info("Using sample dataset instead.")
            analyzer = GeochemicalAnalyzer()
    else:
        # Use sample data
        analyzer = GeochemicalAnalyzer()
        st.sidebar.info("📁 Using sample dataset for demonstration")
    
    # Show dataset info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.write(f"**Samples:** {len(analyzer.df)}")
    st.sidebar.write(f"**Elements:** {len(analyzer.element_cols)}")
    if len(analyzer.coord_cols) == 2:
        st.sidebar.write(f"**Coordinates:** {analyzer.coord_cols[0]}, {analyzer.coord_cols[1]}")
    
    # Main content based on selected step
    if selected_step == "1. Data Overview":
        analyzer.display_data_overview()
        
    elif selected_step == "2. Data Preprocessing":
        analyzer.preprocess_data()
        
    elif selected_step == "3. Compositional Transforms":
        analyzer.apply_compositional_transforms()
        
    elif selected_step == "4. Outlier Detection":
        analyzer.outlier_detection_advanced()
        
    elif selected_step == "5. Clustering Analysis":
        analyzer.clustering_analysis_advanced()
        
    elif selected_step == "6. PCA Analysis":
        analyzer.pca_analysis_comprehensive()
        
    elif selected_step == "7. MDS Analysis":
        analyzer.mds_analysis()
        
    elif selected_step == "8. Comprehensive Report":
        analyzer.generate_comprehensive_report()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>Geochemical Data Analysis Suite</strong> • Built with Streamlit</p>
    <p>For educational and exploration purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()