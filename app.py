import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import requests
from io import StringIO
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Function to safely load example datasets with fallback to local cache
def load_example_dataset(dataset_name, dataset_urls):
    """
    Load an example dataset with error handling and local caching.
    
    Args:
        dataset_name: Name of the dataset to load
        dataset_urls: Dictionary mapping dataset names to URLs
        
    Returns:
        pandas DataFrame or None if loading failed
    """
    # Create a data directory if it doesn't exist
    data_dir = "example_datasets"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Define local file path
    file_extension = ".json" if "JSON" in dataset_name else ".csv"
    file_name = dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + file_extension
    local_path = os.path.join(data_dir, file_name)
    
    # Try to load from URL first
    try:
        with st.spinner(f"Downloading {dataset_name} dataset..."):
            # Use requests with timeout instead of pandas direct URL access
            response = requests.get(dataset_urls[dataset_name], timeout=10)
            response.raise_for_status()  # Raise error for bad responses
            
            # Parse data based on format
            if file_extension == ".json":
                import json
                json_data = response.json()
                
                # Handle different JSON structures
                if isinstance(json_data, list):
                    data = pd.json_normalize(json_data)
                elif isinstance(json_data, dict):
                    if any(isinstance(value, list) for value in json_data.values()):
                        data = pd.DataFrame(json_data)
                    else:
                        data = pd.DataFrame([json_data])
                else:
                    raise ValueError("Unsupported JSON structure")
                    
                # Cache as CSV for consistency
                data.to_csv(local_path.replace('.json', '.csv'), index=False)
            else:
                # Parse CSV from response content
                data = pd.read_csv(StringIO(response.text))
                # Cache the dataset locally for future use
                data.to_csv(local_path, index=False)
            
            st.success(f"{dataset_name} dataset loaded successfully!")
            
            return data
            
    except Exception as e:
        st.warning(f"Could not download {dataset_name} dataset: {str(e)}")
        
        # Try to load from local cache if available
        cache_path = local_path.replace('.json', '.csv') if file_extension == ".json" else local_path
        if os.path.exists(cache_path):
            st.info(f"Loading {dataset_name} dataset from local cache...")
            try:
                return pd.read_csv(cache_path)
            except Exception as e2:
                st.error(f"Error loading from local cache: {str(e2)}")
        
        # If everything fails
        return None

# Set page configuration
st.set_page_config(
    page_title="Advanced ML Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title with styling
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown("""
    <div style='background-color:#0066cc;padding:10px;border-radius:10px'>
    <h1 style='color:white;text-align:center;'>Advanced ML Analytics Platform</h1>
    </div>
    """, unsafe_allow_html=True)
with col2:
    # Add a reset button to clear all session state
    if st.button("🔄 Reset All", type="primary", help="Reset the entire workflow and clear all session state"):
        # Store the current reset counter value
        current_reset_counter = st.session_state.get('reset_counter', 0)
        
        # Clear all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Initialize critical session state variables to ensure clean reset
        st.session_state.data_loaded = False
        st.session_state.df = None
        
        # Increment reset counter to force file uploader to use a new key
        st.session_state.reset_counter = current_reset_counter + 1
        
        # Force the page to reload completely
        st.rerun()

# Add description that clearly articulates the problem statement
st.markdown("""
### 📝 Problem Statement

This application addresses the challenge of efficiently analyzing datasets and building predictive models without extensive coding. 
Data science tasks typically require significant manual coding for exploration, preprocessing, and modeling, which can be time-consuming 
and error-prone, especially for non-technical stakeholders.

**Significance**:
1. Democratizes machine learning by allowing non-technical users to perform advanced analytics
2. Accelerates data-driven decision making through automated insights generation
3. Provides comprehensive data understanding through interactive visualizations
4. Enables comparative model evaluation for optimal predictive performance

The platform implements best practices in machine learning workflow from data exploration to model evaluation, making advanced analytics accessible to all.
""")


# Enhanced sidebar with instructions and additional options
st.sidebar.markdown("""
<div style='background-color:#0066cc;padding:10px;border-radius:10px'>
<h2 style='text-align:center;color:white;'>Navigation Panel</h2>
</div>
""", unsafe_allow_html=True)

# Add a timer to track how long the session has been active
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
    st.session_state.data_loaded = False

elapsed_time = int(time.time() - st.session_state.start_time)
st.sidebar.markdown(f"**Session Time:** {elapsed_time//60} mins {elapsed_time%60} secs")

# Sidebar for instructions with improved formatting
st.sidebar.markdown("### 📋 Instructions")
st.sidebar.markdown("""
1. **Upload Data**: Select a CSV dataset for analysis
2. **Explore Data**: Review statistics and visualizations
3. **Preprocess**: Clean and prepare your data
4. **Model Selection**: Choose and tune ML algorithms
5. **Evaluate Results**: Interpret performance metrics
""")

# Add example datasets for users to try
st.sidebar.markdown("### 🔍 Example Datasets")
example_datasets = {
    "Iris Classification": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Titanic Survival": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    # Use GitHub raw content instead of direct UCI link for more reliability
    "Wine Quality": "https://raw.githubusercontent.com/SiddharthChabukswar/UCI-ML-Repository-Datasets/master/Wine%20Quality/winequality-red.csv",
    "Countries Data (JSON)": "https://restcountries.com/v3.1/all?fields=name,capital,population,area,region"
}

# Generate a unique key for the selectbox based on the reset counter
example_dataset_key = f"example_dataset_{st.session_state.get('reset_counter', 0)}"

# Force empty selection after reset by setting index=0 (which is the empty string option)
selected_example = st.sidebar.selectbox("Try an example dataset:", 
                                       [""] + list(example_datasets.keys()),
                                       index=0,  # Always default to empty selection
                                       key=example_dataset_key)

# Add option to use built-in scikit-learn datasets as a fallback
st.sidebar.markdown("### 📊 Offline Datasets")
# Create a unique key for the checkbox
checkbox_key = f"use_builtin_{st.session_state.get('reset_counter', 0)}"

# Always default to False after reset
use_builtin = st.sidebar.checkbox("Use built-in scikit-learn datasets (works offline)", 
                                 value=False,
                                 key=checkbox_key)

if use_builtin:
    # Create a unique key for the radio button based on the reset counter
    radio_key = f"builtin_dataset_{st.session_state.get('reset_counter', 0)}"
    builtin_dataset = st.sidebar.radio(
        "Select built-in dataset:",
        ["Iris", "Wine", "Breast Cancer", "Diabetes"],
        key=radio_key
    )

# Add reference and resources section
st.sidebar.markdown("### 📚 Resources")
st.sidebar.markdown("""
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Data Visualization Guide](https://matplotlib.org/)
- [Feature Engineering Tips](https://www.kaggle.com/learn/feature-engineering)
""")

# Step 1: Upload Dataset with enhanced options
st.markdown("## 📤 Data Acquisition")

# Let user choose file format
st.markdown("### File Format Selection")
# Create a unique key for the radio button
format_key = f"file_format_{st.session_state.get('reset_counter', 0)}"
file_format = st.radio(
    "Select your data file format:",
    ["CSV", "JSON"],
    horizontal=True,
    help="Choose the format of your data file",
    key=format_key
)

# File uploader based on selected format
# Using a dynamic key to ensure it resets properly when "Reset All" is clicked
uploader_key = f"file_uploader_{file_format}_{st.session_state.get('reset_counter', 0)}"
if file_format == "CSV":
    uploaded_file = st.file_uploader(
        "Upload your CSV dataset", 
        type=["csv"],
        help="Upload a comma-separated values file (.csv)",
        key=uploader_key
    )
else:
    uploaded_file = st.file_uploader(
        "Upload your JSON dataset", 
        type=["json"],
        help="Upload a JSON file (.json) - should contain an array of objects or a single object with arrays",
        key=uploader_key
    )

# Load example dataset if selected
if selected_example and not use_builtin:
    # Use our custom function to load the dataset with error handling and caching
    df = load_example_dataset(selected_example, example_datasets)
    
    if df is not None:
        st.session_state.data_loaded = True
    else:
        st.error(f"Could not load {selected_example} dataset")
        st.markdown("""
        ### Troubleshooting Tips:
        1. Check your internet connection
        2. Try uploading the dataset manually instead
        3. Try another example dataset
        4. Try the built-in scikit-learn datasets (offline option)
        
        If problems persist, you can download the datasets directly from these sources:
        - [Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
        - [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
        - [Wine Quality Dataset](https://raw.githubusercontent.com/SiddharthChabukswar/UCI-ML-Repository-Datasets/master/Wine%20Quality/winequality-red.csv)
        """)
        st.session_state.data_loaded = False

# Use scikit-learn built-in datasets if selected
elif use_builtin:
    try:
        if builtin_dataset == "Iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            st.success("Iris dataset loaded successfully!")
            st.info("This dataset contains measurements of iris flowers with 3 different species.")
            
        elif builtin_dataset == "Wine":
            from sklearn.datasets import load_wine
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            st.success("Wine dataset loaded successfully!")
            st.info("This dataset contains chemical analyses of wines from 3 different cultivars.")
            
        elif builtin_dataset == "Breast Cancer":
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            st.success("Breast Cancer dataset loaded successfully!")
            st.info("This dataset contains features of breast mass and classification as malignant/benign.")
            
        elif builtin_dataset == "Diabetes":
            from sklearn.datasets import load_diabetes
            data = load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            st.success("Diabetes dataset loaded successfully!")
            st.info("This dataset contains patient data and a quantitative measure of disease progression.")
        
        st.session_state.data_loaded = True
        
    except Exception as e:
        st.error(f"Error loading built-in dataset: {str(e)}")
        st.session_state.data_loaded = False
        
elif uploaded_file is not None:
    try:
        if file_format == "CSV":
            df = pd.read_csv(uploaded_file)
            st.success("CSV dataset uploaded successfully!")
        else:  # JSON format
            import json
            
            # Read JSON file
            json_data = json.load(uploaded_file)
            
            # Handle different JSON structures
            if isinstance(json_data, list):
                # List of objects - direct conversion
                df = pd.json_normalize(json_data)
                st.success("JSON dataset (array format) uploaded successfully!")
                
            elif isinstance(json_data, dict):
                # Check if it's a nested structure or flat dictionary
                if any(isinstance(value, list) for value in json_data.values()):
                    # Dictionary with arrays - assume column-oriented data
                    df = pd.DataFrame(json_data)
                    st.success("JSON dataset (object with arrays) uploaded successfully!")
                else:
                    # Single object - convert to single-row DataFrame
                    df = pd.DataFrame([json_data])
                    st.success("JSON dataset (single object) uploaded successfully!")
                    st.info("Note: Single object converted to one-row dataset")
            else:
                raise ValueError("Unsupported JSON structure. Please provide an array of objects or an object with arrays.")
                
        st.session_state.data_loaded = True
        
        # Display JSON structure info if JSON was uploaded
        if file_format == "JSON":
            st.info(f"JSON structure detected and converted to DataFrame with {df.shape[0]} rows and {df.shape[1]} columns")
            
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {str(e)}")
        st.markdown("Please ensure your JSON file is properly formatted.")
        st.session_state.data_loaded = False
    except Exception as e:
        st.error(f"Error reading uploaded file: {str(e)}")
        if file_format == "CSV":
            st.markdown("Please ensure your file is a valid CSV format.")
        else:
            st.markdown("""
            **JSON Format Requirements:**
            - Array of objects: `[{"col1": "val1", "col2": "val2"}, ...]`
            - Object with arrays: `{"col1": ["val1", "val2"], "col2": ["val3", "val4"]}`
            - Single object: `{"col1": "val1", "col2": "val2"}`
            """)
        st.session_state.data_loaded = False
else:
    st.info("Please upload a CSV/JSON file or select an example dataset to begin analysis.")
    st.session_state.data_loaded = False

if st.session_state.data_loaded:
    # Dataset overview in expandable section
    with st.expander("📊 Dataset Overview", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df.head(10), width="stretch")  # Show first 10 rows with full width
        with col2:
            st.markdown("### Dataset Profile")
            st.markdown(f"**Rows:** {df.shape[0]}")
            st.markdown(f"**Columns:** {df.shape[1]}")
            st.markdown(f"**Numeric Features:** {len(df.select_dtypes(include=np.number).columns)}")
            st.markdown(f"**Categorical Features:** {len(df.select_dtypes(include='object').columns)}")
            st.markdown(f"**Missing Values:** {df.isnull().sum().sum()} ({(df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100):.2f}%)")
    
    # Step 2: Comprehensive Exploratory Data Analysis (EDA)
    st.markdown("## 🔍 Exploratory Data Analysis")
    
    # Interactive tabs for different EDA aspects
    eda_tabs = st.tabs(["Summary Statistics", "Data Distribution", "Correlation Analysis", "Missing Values", "Outlier Detection"])
    
    # Tab 1: Summary Statistics
    with eda_tabs[0]:
        st.markdown("### Comprehensive Data Summary")
        
        # Summary metrics for both numerical and categorical features
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include='object').columns
        
        # Better formatting for summary statistics
        if len(num_cols) > 0:
            st.markdown("#### Numerical Features Statistics")
            st.dataframe(df[num_cols].describe().T.style.highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='#ffcccc'))
            
            # Advanced statistics
            skew_data = df[num_cols].skew().reset_index()
            skew_data.columns = ['Feature', 'Skewness']
            kurt_data = df[num_cols].kurtosis().reset_index()
            kurt_data.columns = ['Feature', 'Kurtosis']
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Skewness** (measure of asymmetry)")
                st.dataframe(skew_data)
            with col2:
                st.markdown("**Kurtosis** (measure of tailedness)")
                st.dataframe(kurt_data)
            
            st.markdown("""
            **Interpretation**:
            - **Skewness**: Values > 0.5 or < -0.5 indicate moderate skewness; > 1 or < -1 indicate high skewness
            - **Kurtosis**: Values > 3 indicate heavy tails (more outliers); < 3 indicate light tails
            
            **Actionable Insight**: Features with high skewness may need transformation (e.g., log, sqrt) before modeling.
            """)
            
        if len(cat_cols) > 0:
            st.markdown("#### Categorical Features Overview")
            cat_summary = pd.DataFrame({
                'Feature': cat_cols,
                'Unique Values': [df[col].nunique() for col in cat_cols],
                'Most Common': [df[col].value_counts().index[0] if not df[col].value_counts().empty else 'N/A' for col in cat_cols],
                'Frequency (%)': [df[col].value_counts(normalize=True).max() * 100 if not df[col].value_counts().empty else 0 for col in cat_cols]
            })
            st.dataframe(cat_summary)
    
    # Tab 2: Data Distribution
    with eda_tabs[1]:
        st.markdown("### Data Distribution Analysis")
        
        # Enhanced Visualizations for Numerical Columns
        if len(num_cols) > 0:
            st.markdown("#### 📈 Numerical Features Distribution")
            
            # Let user select columns to visualize
            selected_num_cols = st.multiselect("Select numerical features to visualize:", options=list(num_cols), default=list(num_cols)[:min(3, len(num_cols))])
            
            if selected_num_cols:
                # Distribution plots with both histogram and KDE
                for i in range(0, len(selected_num_cols), 2):
                    cols_subset = selected_num_cols[i:i+2]
                    if cols_subset:
                        cols = st.columns(len(cols_subset))
                        for j, col_name in enumerate(cols_subset):
                            with cols[j]:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(df[col_name], kde=True, ax=ax)
                                ax.set_title(f"Distribution of {col_name}")
                                st.pyplot(fig)
                                
                                # Add quantitative analysis
                                st.markdown(f"""
                                **Distribution metrics for {col_name}**:
                                - **Mean**: {df[col_name].mean():.2f}
                                - **Median**: {df[col_name].median():.2f}
                                - **Std Dev**: {df[col_name].std():.2f}
                                - **IQR**: {df[col_name].quantile(0.75) - df[col_name].quantile(0.25):.2f}
                                """)
            
            # Advanced option: Box plots
            st.markdown("#### 📊 Box Plots (Identify Outliers)")
            selected_box_cols = st.multiselect("Select features for box plots:", options=list(num_cols), default=list(num_cols)[:min(3, len(num_cols))])
            
            if selected_box_cols:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.boxplot(data=df[selected_box_cols], ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                st.markdown("""
                **Interpretation**: 
                - Box shows IQR (middle 50% of data)
                - Line inside box is median
                - Whiskers extend to 1.5 * IQR
                - Points beyond whiskers are potential outliers
                
                **Actionable Insight**: Consider treating outliers before modeling through capping, removing, or using robust models.
                """)
        
        # Enhanced Visualizations for Categorical Columns
        if len(cat_cols) > 0:
            st.markdown("#### 📊 Categorical Features Distribution")
            
            # Let user select columns to visualize
            selected_cat_cols = st.multiselect("Select categorical features to visualize:", options=list(cat_cols), default=list(cat_cols)[:min(2, len(cat_cols))])
            
            if selected_cat_cols:
                for col_name in selected_cat_cols:
                    # Calculate value counts and percentages
                    value_counts = df[col_name].value_counts()
                    total = len(df[col_name])
                    
                    # Display counts and percentages
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.countplot(x=df[col_name], order=value_counts.index, ax=ax)
                        ax.set_title(f"Distribution of {col_name}")
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig)
                    
                    with col2:
                        # Show percentages in a table
                        percentages = (value_counts / total * 100).reset_index()
                        percentages.columns = [col_name, 'Percentage (%)']
                        percentages['Percentage (%)'] = percentages['Percentage (%)'].round(2)
                        st.dataframe(percentages)
                        
                        # Highlight imbalance if present
                        max_pct = percentages['Percentage (%)'].max()
                        min_pct = percentages['Percentage (%)'].min()
                        if max_pct > 75 or min_pct < 5:
                            st.warning(f"⚠️ Imbalance detected in {col_name}. This may affect model performance.")
                            st.markdown("""
                            **Recommendation**: Consider resampling techniques:
                            - Oversampling minority class
                            - Undersampling majority class
                            - SMOTE for synthetic data generation
                            """)
    
    # Tab 3: Correlation Analysis
    with eda_tabs[2]:
        st.markdown("### 🔗 Feature Relationships & Correlation Analysis")
        
        # Correlation Heatmap (for numerical features)
        if len(num_cols) > 1:
            st.markdown("#### Correlation Heatmap")
            
            # Let user choose correlation method
            corr_method = st.radio("Select correlation method:", ["Pearson (linear)", "Spearman (monotonic)", "Kendall (rank)"], horizontal=True)
            method_map = {"Pearson (linear)": "pearson", "Spearman (monotonic)": "spearman", "Kendall (rank)": "kendall"}
            
            # Generate correlation matrix
            corr = df[num_cols].corr(method=method_map[corr_method])
            
            # Create mask for upper triangle to improve readability
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # Plot correlation heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            heatmap = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, ax=ax, 
                                  annot_kws={"size": 8}, linewidths=0.5)
            ax.set_title(f"{corr_method} Correlation Matrix", fontsize=14)
            st.pyplot(fig)
            
            # Identify and display strong correlations
            strong_corr = pd.DataFrame(columns=['Feature 1', 'Feature 2', 'Correlation'])
            for i in range(len(corr.columns)):
                for j in range(i):
                    if abs(corr.iloc[i, j]) > 0.5:  # Threshold for strong correlation
                        strong_corr = pd.concat([strong_corr, pd.DataFrame({
                            'Feature 1': [corr.columns[i]], 
                            'Feature 2': [corr.columns[j]],
                            'Correlation': [corr.iloc[i, j]]
                        })])
            
            if not strong_corr.empty:
                st.markdown("#### Strong Feature Correlations (|r| > 0.5)")
                st.dataframe(strong_corr.sort_values(by='Correlation', key=abs, ascending=False))
                
                st.markdown("""
                **Interpretation**:
                - **Strong positive correlation (close to 1)**: Features increase together
                - **Strong negative correlation (close to -1)**: As one feature increases, the other decreases
                
                **Actionable Insights**:
                - Consider removing highly correlated features to reduce multicollinearity
                - For feature pairs with |r| > 0.8, consider keeping only one feature
                - Look for unexpected correlations that might reveal hidden patterns
                """)
            
            # Scatter plot for selected features
            st.markdown("#### Feature Relationship Scatter Plot")
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("Select X-axis feature:", options=num_cols)
            with col2:
                y_feature = st.selectbox("Select Y-axis feature:", options=[col for col in num_cols if col != x_feature], index=0)
            
            # Add option for hue if categorical columns exist
            hue_feature = None
            if len(cat_cols) > 0:
                hue_feature = st.selectbox("Color points by (optional):", options=["None"] + list(cat_cols))
                if hue_feature == "None":
                    hue_feature = None
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df[x_feature], y=df[y_feature], hue=df[hue_feature] if hue_feature else None, ax=ax)
            ax.set_title(f"Relationship between {x_feature} and {y_feature}")
            st.pyplot(fig)
            
            # Display correlation and regression line
            if hue_feature is None:
                corr_val = df[[x_feature, y_feature]].corr().iloc[0, 1]
                st.markdown(f"**Correlation between {x_feature} and {y_feature}**: {corr_val:.4f}")
                
                # Add regression plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.regplot(x=df[x_feature], y=df[y_feature], ax=ax, line_kws={"color": "red"})
                ax.set_title(f"Regression Plot: {x_feature} vs {y_feature}")
                st.pyplot(fig)
    
    # Tab 4: Missing Values Analysis
    with eda_tabs[3]:
        st.markdown("### 🧩 Missing Data Analysis")
        
        # Calculate missing values
        missing_data = df.isnull().sum().reset_index()
        missing_data.columns = ['Feature', 'Missing Count']
        missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(df) * 100).round(2)
        missing_data = missing_data.sort_values('Missing Percentage', ascending=False)
        
        # Display missing values
        if missing_data['Missing Count'].sum() > 0:
            st.markdown("#### Missing Values Overview")
            
            # Visualize missing data
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Feature', y='Missing Percentage', data=missing_data[missing_data['Missing Count'] > 0], ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.title('Missing Values by Feature')
            st.pyplot(fig)
            
            # Show missing data table
            st.dataframe(missing_data[missing_data['Missing Count'] > 0])
            
            # Missing data patterns visualization (missingno)
            st.markdown("#### Missing Value Patterns")
            
            # Create a heatmap-style visualization of missing patterns
            if len(df.columns) <= 30:  # Limit to avoid overloading
                fig, ax = plt.subplots(figsize=(12, 8))
                ax = sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
                plt.title('Missing Value Patterns (Yellow indicates missing)')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Recommendations for handling missing values
            st.markdown("""
            #### Strategies for Handling Missing Data
            
            1. **For numerical features**:
               - Replace with mean/median (for symmetric/skewed distributions)
               - Use KNN imputation for better accuracy
               - Create 'missing' indicator feature if missingness is informative
            
            2. **For categorical features**:
               - Replace with mode (most frequent category)
               - Create a new 'Unknown' or 'Missing' category
               - Use model-based imputation
            
            3. **Row removal**:
               - Consider removing rows if missing percentage is low
               - Not recommended if missing data exceeds 5-10%
            
            **Advanced Technique**: Use models to predict missing values based on other features
            """)
            
            # Interactive imputation demo
            st.markdown("#### Try Different Imputation Methods")
            feature_to_impute = st.selectbox(
                "Select a feature to impute:", 
                options=[col for col in df.columns if df[col].isnull().sum() > 0],
                index=0 if any(df[col].isnull().sum() > 0 for col in df.columns) else None
            )
            
            if feature_to_impute:
                method = "mean" if df[feature_to_impute].dtype in ['int64', 'float64'] else "most_frequent"
                imputation_method = st.radio(
                    "Select imputation method:",
                    ["Mean/Mode", "Median", "KNN Imputation"],
                    horizontal=True
                )
                
                # Create a copy of the dataframe for demonstration
                df_imputed = df.copy()
                
                # Apply selected imputation
                if imputation_method == "Mean/Mode":
                    if df[feature_to_impute].dtype in ['int64', 'float64']:
                        fill_value = df[feature_to_impute].mean()
                        df_imputed[feature_to_impute].fillna(fill_value, inplace=True)
                        st.write(f"Filled missing values with mean: {fill_value:.2f}")
                    else:
                        fill_value = df[feature_to_impute].mode()[0]
                        df_imputed[feature_to_impute].fillna(fill_value, inplace=True)
                        st.write(f"Filled missing values with mode: {fill_value}")
                        
                elif imputation_method == "Median":
                    if df[feature_to_impute].dtype in ['int64', 'float64']:
                        fill_value = df[feature_to_impute].median()
                        df_imputed[feature_to_impute].fillna(fill_value, inplace=True)
                        st.write(f"Filled missing values with median: {fill_value:.2f}")
                    else:
                        st.write("Median imputation only applies to numerical features")
                        
                elif imputation_method == "KNN Imputation":
                    st.write("KNN Imputation (using 5 nearest neighbors)")
                    # This is a simplified demo - in practice, would handle categorical features properly
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    if feature_to_impute in numeric_cols and len(numeric_cols) > 1:
                        imputer = KNNImputer(n_neighbors=5)
                        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                        st.write("KNN imputation completed for numeric features")
                    else:
                        st.write("KNN imputation requires multiple numeric features")
                
                # Show before/after comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Before Imputation**")
                    st.write(df[feature_to_impute].describe())
                with col2:
                    st.markdown("**After Imputation**")
                    st.write(df_imputed[feature_to_impute].describe())
        else:
            st.success("✅ No missing values found in the dataset!")
            
    # Tab 5: Outlier Detection
    with eda_tabs[4]:
        st.markdown("### 🔎 Outlier Detection & Analysis")
        
        if len(num_cols) > 0:
            # Let user select columns for outlier detection
            selected_cols_outliers = st.multiselect(
                "Select numerical features for outlier analysis:",
                options=num_cols,
                default=list(num_cols)[:min(3, len(num_cols))]
            )
            
            if selected_cols_outliers:
                # Z-score method
                st.markdown("#### Z-score Method (identifies values > 3 standard deviations from mean)")
                
                # Calculate and display outliers using Z-score
                outliers_z = {}
                for col in selected_cols_outliers:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers_z[col] = np.sum(z_scores > 3)
                
                outliers_df_z = pd.DataFrame({
                    'Feature': list(outliers_z.keys()),
                    'Outliers Count (Z-score > 3)': list(outliers_z.values()),
                    'Percentage (%)': [count / len(df) * 100 for count in outliers_z.values()]
                })
                
                # IQR method
                st.markdown("#### IQR Method (identifies values outside 1.5 * IQR from quartiles)")
                
                # Calculate and display outliers using IQR
                outliers_iqr = {}
                for col in selected_cols_outliers:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers_iqr[col] = np.sum((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
                
                outliers_df_iqr = pd.DataFrame({
                    'Feature': list(outliers_iqr.keys()),
                    'Outliers Count (IQR method)': list(outliers_iqr.values()),
                    'Percentage (%)': [count / len(df) * 100 for count in outliers_iqr.values()]
                })
                
                # Display results side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(outliers_df_z)
                with col2:
                    st.dataframe(outliers_df_iqr)
                
                # Visualize outliers for a selected feature
                st.markdown("#### Visualize Outliers")
                selected_feature = st.selectbox("Select a feature to visualize outliers:", options=selected_cols_outliers)
                
                if selected_feature:
                    # Box plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x=df[selected_feature], ax=ax)
                    ax.set_title(f"Box Plot: {selected_feature}")
                    st.pyplot(fig)
                    
                    # Histogram with outlier boundaries
                    Q1 = df[selected_feature].quantile(0.25)
                    Q3 = df[selected_feature].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[selected_feature], kde=True, ax=ax)
                    ax.axvline(lower_bound, color='r', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
                    ax.axvline(upper_bound, color='r', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')
                    ax.set_title(f"Distribution with Outlier Boundaries: {selected_feature}")
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Show actual outlier values
                    outliers = df[(df[selected_feature] < lower_bound) | (df[selected_feature] > upper_bound)][selected_feature]
                    if not outliers.empty:
                        st.markdown(f"**Sample outlier values for {selected_feature}:**")
                        st.write(outliers.head(10))
                
                # Recommendations for handling outliers
                st.markdown("""
                #### Strategies for Handling Outliers
                
                1. **Investigate first**: Understand if outliers represent errors or valid extreme cases
                
                2. **Treatment options**:
                   - **Trimming**: Remove outliers (if they're errors or very few)
                   - **Capping**: Set upper/lower limits (Winsorization)
                   - **Transformation**: Apply log, square root, or Box-Cox to reduce influence
                   - **Binning**: Convert to categorical ranges
                
                3. **Modeling approaches**:
                   - Use robust algorithms less sensitive to outliers (Random Forest, Gradient Boosting)
                   - Use robust scaling methods (RobustScaler)
                
                **Actionable Insight**: For machine learning, consider how outliers affect your specific model. Linear models are more sensitive to outliers than tree-based models.
                """)
                
                # Demo outlier treatment
                st.markdown("#### Try Different Outlier Treatments")
                treatment_method = st.radio(
                    "Select treatment method:",
                    ["Original Data", "Trimming", "Capping (Winsorization)", "Log Transformation"],
                    horizontal=True
                )
                
                # Apply selected treatment
                df_treated = df.copy()
                
                if treatment_method == "Trimming":
                    for col in selected_cols_outliers:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df_treated = df_treated[(df_treated[col] >= lower_bound) & (df_treated[col] <= upper_bound)]
                    st.write(f"Removed {len(df) - len(df_treated)} rows with outliers")
                    
                elif treatment_method == "Capping (Winsorization)":
                    for col in selected_cols_outliers:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df_treated[col] = df_treated[col].clip(lower=lower_bound, upper=upper_bound)
                    st.write("Capped outliers to IQR boundaries")
                    
                elif treatment_method == "Log Transformation":
                    for col in selected_cols_outliers:
                        # Add small constant to handle zeros
                        if (df_treated[col] <= 0).any():
                            min_val = df_treated[col].min()
                            shift = abs(min_val) + 1 if min_val <= 0 else 0
                            df_treated[col] = np.log(df_treated[col] + shift)
                            st.write(f"Applied log(x + {shift}) transformation to {col}")
                        else:
                            df_treated[col] = np.log(df_treated[col])
                            st.write(f"Applied log transformation to {col}")
                
                # Compare distributions before and after treatment
                if treatment_method != "Original Data":
                    selected_viz_feature = st.selectbox(
                        "Select a feature to compare before and after treatment:",
                        options=selected_cols_outliers
                    )
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Before treatment
                    sns.boxplot(x=df[selected_viz_feature], ax=ax1)
                    ax1.set_title(f"Before: {selected_viz_feature}")
                    
                    # After treatment
                    sns.boxplot(x=df_treated[selected_viz_feature], ax=ax2)
                    ax2.set_title(f"After: {selected_viz_feature}")
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Compare statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Before Treatment**")
                        st.write(df[selected_viz_feature].describe())
                    with col2:
                        st.markdown("**After Treatment**")
                        st.write(df_treated[selected_viz_feature].describe())

    # Step 3: Advanced Machine Learning Pipeline
    st.markdown("## 🤖 Machine Learning Pipeline")
    
    if st.session_state.data_loaded:
        # Create tabs for preprocessing, model selection, evaluation, and prediction
        ml_tabs = st.tabs(["Data Preparation", "Feature Engineering", "Model Selection & Training", "Evaluation & Insights", "🔮 Make Predictions"])
        
        # Tab 1: Data Preparation
        with ml_tabs[0]:
            st.markdown("### 🔧 Data Preparation")
            
            # Select Target Column
            st.markdown("#### Target Selection")
            target_col = st.selectbox("Select the target column for prediction:", options=[""] + list(df.columns))
            
            if not target_col:
                st.info("⬆️ Please select a target column to continue")
            else:
                # Check if target column has changed
                if 'target_col' in st.session_state and st.session_state.target_col != target_col:
                    # Target has changed, determine if type has changed as well (classification vs regression)
                    old_is_classification = False
                    new_is_classification = False
                    
                    # Check if old target was classification
                    if 'is_classification' in st.session_state:
                        old_is_classification = st.session_state.is_classification
                    
                    # Check if new target would be classification
                    if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                        new_is_classification = True
                    
                    # If task type changed (classification to regression or vice versa)
                    if old_is_classification != new_is_classification:
                        # Reset all ML-related session state because task type changed
                        keys_to_delete = [
                            # Data preparation
                            'X_original', 'y_original', 'X_train_raw', 'X_test_raw', 'y_train', 'y_test',
                            # Feature engineering
                            'X_train_processed', 'X_test_processed', 'preprocessing_done', 'preprocessing_summary',
                            'pca', 'pca_original_features', 'use_pca', 'n_components',
                            'scaler', 'encoders', 'ordinal_encoder', 'encoded_feature_names',
                            # Model training
                            'best_model', 'model_metrics', 'training_done', 'model_training_summary',
                            'models_trained', 'cv_results', 'feature_importance'
                        ]
                        for key in keys_to_delete:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        # Show message about task type change
                        st.warning(f"⚠️ Task type changed from {old_is_classification and 'Classification' or 'Regression'} " +
                                  f"to {new_is_classification and 'Classification' or 'Regression'}. All progress has been reset.")
                    else:
                        # Just show a message that target changed but type is the same
                        st.info(f"Target changed from {st.session_state.target_col} to {target_col}. Some progress may need to be redone.")
                
                # Store target in session state
                st.session_state.target_col = target_col
                
                # Analyze target variable
                if df[target_col].dtype == 'object' or df[target_col].nunique() < 10:
                    is_classification = True
                    task_type = "Classification"
                    
                    # Visualize target distribution
                    st.markdown(f"#### Target Distribution: {target_col}")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts = df[target_col].value_counts().sort_index()
                    if len(value_counts) <= 30:  # Don't plot if too many categories
                        sns.countplot(x=df[target_col], ax=ax, order=value_counts.index)
                        plt.xticks(rotation=45, ha='right')
                        st.pyplot(fig)
                    
                    # Check for class imbalance
                    class_counts = df[target_col].value_counts(normalize=True) * 100
                    st.dataframe(pd.DataFrame({'Class': class_counts.index, 'Percentage (%)': class_counts.values.round(2)}))
                    
                    if class_counts.max() > 75 or class_counts.min() < 10:
                        st.warning("⚠️ **Class Imbalance Detected**: This may affect model performance")
                        st.markdown("""
                        **Mitigation Strategies**:
                        - Oversampling minority classes
                        - Undersampling majority class
                        - Using class weights
                        - SMOTE (Synthetic Minority Over-sampling Technique)
                        """)
                        
                        # Allow user to select balancing strategy
                        balance_strategy = st.radio(
                            "Select a strategy for handling class imbalance:",
                            ["No adjustment", "Use class weights", "Oversampling (not implemented in demo)"],
                            horizontal=True
                        )
                        st.session_state.balance_strategy = balance_strategy
                else:
                    is_classification = False
                    task_type = "Regression"
                    
                    # Visualize target distribution for regression
                    st.markdown(f"#### Target Distribution: {target_col}")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[target_col], kde=True, ax=ax)
                    st.pyplot(fig)
                    
                    # Show statistics
                    st.dataframe(df[target_col].describe().to_frame().T)
                    
                    # Check for skewness
                    skewness = df[target_col].skew()
                    if abs(skewness) > 1:
                        st.warning(f"⚠️ Target is skewed (skewness={skewness:.2f}). Consider transformation.")
                
                # Store task type in session state
                st.session_state.is_classification = is_classification
                st.session_state.task_type = task_type
                st.info(f"📊 Task identified: **{task_type}**")
                
                # Data Splitting Options
                st.markdown("#### Train-Test Split Configuration")
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
                with col2:
                    random_seed = st.number_input("Random seed for reproducibility:", 0, 999, 42)
                
                # Option for stratified split in classification
                stratify = False
                if is_classification:
                    stratify = st.checkbox("Use stratified split (maintains class distribution)", True)
                
                st.session_state.split_params = {
                    'test_size': test_size,
                    'random_seed': random_seed,
                    'stratify': stratify
                }
                
                # Preview the split 
                if 'X' not in st.session_state:
                    # Prepare Data
                    X = df.drop(target_col, axis=1)
                    y = df[target_col]
                    st.session_state.X_original = X
                    st.session_state.y_original = y
                    
                    # Store column information
                    st.session_state.cat_cols = X.select_dtypes(include='object').columns.tolist()
                    st.session_state.num_cols = X.select_dtypes(include=np.number).columns.tolist()
                    
                    st.markdown(f"**Features**: {X.shape[1]} total ({len(st.session_state.num_cols)} numerical, {len(st.session_state.cat_cols)} categorical)")
                    
                    # If classification and stratify is selected, check if stratification is possible
                    if is_classification and stratify:
                        # Count occurrences of each class
                        class_counts = y.value_counts()
                        min_class_count = class_counts.min()
                        
                        # Warn if any class has fewer than 2 samples and disable stratification
                        if min_class_count < 2:
                            st.warning(f"⚠️ Stratified sampling not possible: The smallest class has only {min_class_count} sample(s). Using random sampling instead.")
                            stratify = False
                    
                    if stratify and is_classification:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_seed, stratify=y
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_seed
                        )
                    
                    st.session_state.X_train_raw = X_train
                    st.session_state.X_test_raw = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    st.success(f"✅ Data successfully split into training ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets")
                
                # Option to reset data preparation
                if st.button("Reset Data Preparation"):
                    for key in ['X_original', 'y_original', 'X_train_raw', 'X_test_raw', 'y_train', 'y_test']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        
        # Tab 2: Feature Engineering
        with ml_tabs[1]:
            st.markdown("### 🛠️ Feature Engineering")
            
            if 'X_train_raw' not in st.session_state:
                st.warning("⚠️ Please complete data preparation first")
            else:
                # Access split data
                X_train = st.session_state.X_train_raw.copy()
                X_test = st.session_state.X_test_raw.copy()
                
                # Preprocessing Options
                st.markdown("#### Handling Missing Values")
                missing_strategy = st.radio(
                    "Strategy for numerical missing values:",
                    ["Mean imputation", "Median imputation", "KNN imputation"],
                    horizontal=True
                )
                
                cat_missing_strategy = st.radio(
                    "Strategy for categorical missing values:",
                    ["Mode imputation", "Create 'Missing' category"],
                    horizontal=True
                )
                
                # Handling Categorical Features
                st.markdown("#### Encoding Categorical Features")
                encoding_strategy = st.radio(
                    "Encoding strategy for categorical features:",
                    ["Label Encoding", "One-Hot Encoding"],
                    horizontal=True
                )
                
                # Scaling Options
                st.markdown("#### Feature Scaling")
                scaling_strategy = st.radio(
                    "Scaling strategy for numerical features:",
                    ["StandardScaler (zero mean, unit variance)", 
                     "MinMaxScaler (0 to 1 range)",
                     "RobustScaler (robust to outliers)",
                     "No scaling"],
                    horizontal=True
                )
                
                # Feature Selection Options
                st.markdown("#### Feature Selection (Optional)")
                
                # Store PCA selection in session state for persistence when revisiting
                if 'use_pca' not in st.session_state:
                    st.session_state.use_pca = False
                
                # Use the session state value as the default for the checkbox
                use_pca = st.checkbox("Apply PCA for dimensionality reduction", st.session_state.use_pca)
                
                # Update session state when value changes
                st.session_state.use_pca = use_pca
                
                if use_pca:
                    if len(st.session_state.num_cols) < 2:
                        st.warning("PCA requires at least 2 numerical features")
                        st.session_state.use_pca = False
                        use_pca = False
                    else:
                        # Get default from session state if available
                        default_n_components = min(3, len(st.session_state.num_cols))
                        if 'n_components' in st.session_state:
                            default_n_components = st.session_state.n_components
                            
                        n_components = st.slider(
                            "Number of PCA components:", 
                            min_value=2, 
                            max_value=min(10, len(st.session_state.num_cols)), 
                            value=default_n_components
                        )
                        
                        # Store in session state
                        st.session_state.n_components = n_components
                
                # Apply preprocessing when user clicks this button
                if st.button("Apply Feature Engineering"):
                    with st.spinner("Preprocessing data..."):
                        # 1. Handle missing values for numerical features
                        if missing_strategy == "Mean imputation":
                            for col in st.session_state.num_cols:
                                if X_train[col].isnull().any():
                                    mean_val = X_train[col].mean()
                                    X_train[col].fillna(mean_val, inplace=True)
                                    X_test[col].fillna(mean_val, inplace=True)
                                    
                        elif missing_strategy == "Median imputation":
                            for col in st.session_state.num_cols:
                                if X_train[col].isnull().any():
                                    median_val = X_train[col].median()
                                    X_train[col].fillna(median_val, inplace=True)
                                    X_test[col].fillna(median_val, inplace=True)
                                    
                        elif missing_strategy == "KNN imputation":
                            if X_train[st.session_state.num_cols].isnull().any().any():
                                imputer = KNNImputer(n_neighbors=5)
                                X_train_imputed = pd.DataFrame(
                                    imputer.fit_transform(X_train[st.session_state.num_cols]),
                                    columns=st.session_state.num_cols,
                                    index=X_train.index
                                )
                                X_test_imputed = pd.DataFrame(
                                    imputer.transform(X_test[st.session_state.num_cols]),
                                    columns=st.session_state.num_cols,
                                    index=X_test.index
                                )
                                X_train[st.session_state.num_cols] = X_train_imputed
                                X_test[st.session_state.num_cols] = X_test_imputed
                        
                        # 2. Handle missing values for categorical features
                        if cat_missing_strategy == "Mode imputation":
                            for col in st.session_state.cat_cols:
                                if X_train[col].isnull().any():
                                    mode_val = X_train[col].mode()[0]
                                    X_train[col].fillna(mode_val, inplace=True)
                                    X_test[col].fillna(mode_val, inplace=True)
                                    
                        elif cat_missing_strategy == "Create 'Missing' category":
                            for col in st.session_state.cat_cols:
                                if X_train[col].isnull().any():
                                    X_train[col].fillna("Missing", inplace=True)
                                    X_test[col].fillna("Missing", inplace=True)
                        
                        # 3. Encode categorical features
                        if encoding_strategy == "Label Encoding":
                            from sklearn.preprocessing import OrdinalEncoder
                            # Create storage for individual encoders
                            st.session_state.encoders = {}
                            
                            # Apply ordinal encoding to each categorical column individually and store the encoder
                            for col in st.session_state.cat_cols:
                                ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                                X_train[col] = ordinal_encoder.fit_transform(X_train[[col]].astype(str))
                                X_test[col] = ordinal_encoder.transform(X_test[[col]].astype(str))
                                # Store the encoder for this specific column
                                st.session_state.encoders[col] = ordinal_encoder
                            
                            # Also store the main encoder for backward compatibility
                            ordinal_encoder_all = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                            # Just fit it to the data but don't transform again
                            ordinal_encoder_all.fit(X_train[st.session_state.cat_cols].astype(str))
                            st.session_state.ordinal_encoder = ordinal_encoder_all
                            
                        elif encoding_strategy == "One-Hot Encoding":
                            from sklearn.preprocessing import OneHotEncoder
                            # Create storage for individual encoders
                            st.session_state.encoders = {}
                            
                            # We'll store individual one-hot encoders, but still use pandas get_dummies for the actual transform
                            for col in st.session_state.cat_cols:
                                onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                                onehot_encoder.fit(X_train[[col]].astype(str))
                                # Store the encoder for future use in prediction
                                st.session_state.encoders[col] = onehot_encoder
                            
                            # Get dummies for train set
                            X_train_encoded = pd.get_dummies(X_train, columns=st.session_state.cat_cols, drop_first=True)
                            # Get dummies for test set and ensure same columns as training
                            X_test_encoded = pd.get_dummies(X_test, columns=st.session_state.cat_cols, drop_first=True)
                            # Handle potential column differences
                            missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
                            for col in missing_cols:
                                X_test_encoded[col] = 0
                            # Ensure same column order
                            X_test_encoded = X_test_encoded[X_train_encoded.columns]
                            X_train = X_train_encoded
                            X_test = X_test_encoded
                            
                            # Store the column names after one-hot encoding for later reference
                            st.session_state.encoded_feature_names = X_train.columns.tolist()
                        
                        # 4. Apply scaling
                        if scaling_strategy != "No scaling":
                            num_cols = X_train.select_dtypes(include=np.number).columns
                            
                            if scaling_strategy == "StandardScaler (zero mean, unit variance)":
                                scaler = StandardScaler()
                            elif scaling_strategy == "MinMaxScaler (0 to 1 range)":
                                from sklearn.preprocessing import MinMaxScaler
                                scaler = MinMaxScaler()
                            elif scaling_strategy == "RobustScaler (robust to outliers)":
                                scaler = RobustScaler()
                                
                            X_train_scaled = scaler.fit_transform(X_train[num_cols])
                            X_test_scaled = scaler.transform(X_test[num_cols])
                            
                            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=num_cols, index=X_train.index)
                            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=num_cols, index=X_test.index)
                            
                            X_train[num_cols] = X_train_scaled_df
                            X_test[num_cols] = X_test_scaled_df
                            
                            st.session_state.scaler = scaler
                        
                        # 5. Apply PCA if selected
                        if use_pca:
                            # Get numerical columns, excluding any PCA columns from previous runs
                            num_cols = [col for col in X_train.select_dtypes(include=np.number).columns 
                                       if not col.startswith('PCA_')]
                            
                            # Remove any previous PCA columns
                            pca_cols = [col for col in X_train.columns if col.startswith('PCA_')]
                            if pca_cols:
                                X_train = X_train.drop(columns=pca_cols)
                                X_test = X_test.drop(columns=pca_cols)
                            
                            pca = PCA(n_components=n_components)
                            pca_result_train = pca.fit_transform(X_train[num_cols])
                            pca_result_test = pca.transform(X_test[num_cols])
                            
                            # Store the original numerical column names (important for predictions later)
                            st.session_state.pca_original_features = num_cols
                            
                            # Replace numerical features with PCA components
                            for col in num_cols:
                                X_train.drop(col, axis=1, inplace=True)
                                X_test.drop(col, axis=1, inplace=True)
                                
                            # Add PCA components as new features
                            for i in range(n_components):
                                X_train[f'PCA_{i+1}'] = pca_result_train[:, i]
                                X_test[f'PCA_{i+1}'] = pca_result_test[:, i]
                                
                            st.session_state.pca = pca
                            st.session_state.pca_explained_variance = pca.explained_variance_ratio_
                            
                            # Visualize PCA results
                            if n_components >= 2:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                
                                if st.session_state.is_classification:
                                    # Handle categorical (string) labels by encoding them to numbers for visualization
                                    y_train_values = st.session_state.y_train.values
                                    
                                    # Check if y_train contains string values that need to be encoded
                                    if isinstance(y_train_values[0], (str, bool)) or not np.issubdtype(y_train_values.dtype, np.number):
                                        # Create a label encoder
                                        from sklearn.preprocessing import LabelEncoder
                                        le = LabelEncoder()
                                        color_values = le.fit_transform(y_train_values)
                                        
                                        # Create a mapping for the legend
                                        classes = le.classes_
                                        color_mapping = {i: class_name for i, class_name in enumerate(classes)}
                                        
                                        scatter = ax.scatter(
                                            pca_result_train[:, 0], 
                                            pca_result_train[:, 1],
                                            c=color_values,  # Use encoded values for colors 
                                            alpha=0.6,
                                            cmap='viridis'
                                        )
                                        
                                        # Create custom legend with original class names
                                        from matplotlib.lines import Line2D
                                        legend_elements = [
                                            Line2D([0], [0], marker='o', color='w', 
                                                   markerfacecolor=plt.cm.viridis(i / (len(classes) - 1)), 
                                                   markersize=10, label=class_name)
                                            for i, class_name in enumerate(classes)
                                        ]
                                        ax.legend(handles=legend_elements, title="Classes")
                                    else:
                                        # For numeric labels, use them directly
                                        scatter = ax.scatter(
                                            pca_result_train[:, 0], 
                                            pca_result_train[:, 1],
                                            c=y_train_values, 
                                            alpha=0.6,
                                            cmap='viridis'
                                        )
                                        legend = ax.legend(*scatter.legend_elements(), title="Classes")
                                        ax.add_artist(legend)
                                else:
                                    scatter = ax.scatter(
                                        pca_result_train[:, 0], 
                                        pca_result_train[:, 1],
                                        alpha=0.6
                                    )
                                    
                                ax.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                                ax.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                                ax.set_title('PCA Result: First Two Principal Components')
                                ax.grid(True)
                                
                                st.pyplot(fig)
                                
                                # Feature loadings
                                loadings = pd.DataFrame(
                                    pca.components_.T,
                                    columns=[f'PC{i+1}' for i in range(n_components)],
                                    index=num_cols
                                )
                                st.markdown("#### PCA Feature Loadings")
                                st.dataframe(loadings)
                                
                                # Explained variance
                                st.markdown("#### Explained Variance by Components")
                                explained_variance_df = pd.DataFrame({
                                    'Component': [f'PC{i+1}' for i in range(n_components)],
                                    'Explained Variance (%)': [v * 100 for v in pca.explained_variance_ratio_],
                                    'Cumulative Variance (%)': [sum(pca.explained_variance_ratio_[:i+1]) * 100 for i in range(n_components)]
                                })
                                st.dataframe(explained_variance_df)
                        
                        # Store preprocessed data
                        st.session_state.X_train_processed = X_train
                        st.session_state.X_test_processed = X_test
                        st.session_state.preprocessing_done = True
                        
                        # Create a short summary of preprocessing steps
                        preprocessing_summary = [
                            f"Missing numerical values: {missing_strategy}",
                            f"Missing categorical values: {cat_missing_strategy}",
                            f"Categorical encoding: {encoding_strategy}",
                            f"Scaling: {scaling_strategy}",
                            f"PCA: {'Applied' if use_pca else 'Not applied'}"
                        ]
                        st.session_state.preprocessing_summary = preprocessing_summary
                        
                        # If we're re-doing feature engineering, clear any existing model training
                        model_keys = ['best_model', 'model_metrics', 'training_done', 
                                     'model_training_summary', 'models_trained', 
                                     'cv_results', 'feature_importance']
                        for key in model_keys:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        st.success("✅ Feature engineering completed successfully!")
                        
                        # Preview processed data
                        st.markdown("#### Processed Data Preview")
                        st.dataframe(X_train.head())
                
                # Show preprocessing summary if preprocessing is done
                if 'preprocessing_done' in st.session_state and st.session_state.preprocessing_done:
                    st.markdown("#### Applied Preprocessing Steps")
                    for step in st.session_state.preprocessing_summary:
                        st.write(f"- {step}")
                        
                    # Option to reset preprocessing
                    if st.button("Reset Feature Engineering"):
                        # Clear all preprocessing related session state variables
                        keys_to_delete = [
                            'X_train_processed', 'X_test_processed', 
                            'preprocessing_done', 'preprocessing_summary',
                            'pca', 'pca_original_features', 'use_pca', 'n_components',
                            'scaler', 'encoders', 'ordinal_encoder', 'encoded_feature_names'
                        ]
                        for key in keys_to_delete:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
        
        # Tab 3: Model Selection & Training
        with ml_tabs[2]:
            st.markdown("### 🧠 Model Selection & Training")
            
            if 'preprocessing_done' not in st.session_state or not st.session_state.preprocessing_done:
                st.warning("⚠️ Please complete feature engineering first")
            else:
                # Access preprocessed data
                X_train = st.session_state.X_train_processed
                X_test = st.session_state.X_test_processed
                y_train = st.session_state.y_train
                y_test = st.session_state.y_test
                
                # Manual Model Type Selection Override
                st.markdown("#### 🎯 Choose Your Machine Learning Task")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info("**Classification**: Predicting categories/classes\n\n📋 Examples: \n• Email: Spam/Not Spam\n• Medical: Disease/Healthy\n• Business: Customer Segment\n• Finance: Loan Approval/Denial")
                with col2:
                    st.info("**Regression**: Predicting continuous numerical values\n\n📊 Examples: \n• Real Estate: House Prices\n• Business: Sales Revenue\n• Weather: Temperature Prediction\n• Finance: Stock Prices")
                
                # Allow manual override of automatic detection
                task_type = st.radio(
                    "Select the type of machine learning task:",
                    options=["Classification", "Regression"],
                    index=0 if st.session_state.is_classification else 1,
                    help="Choose the type of prediction task based on your target variable"
                )
                
                # Update task type if changed
                st.session_state.is_classification = (task_type == "Classification")
                
                # Model Selection based on chosen task type
                st.markdown("#### 🔧 Select Models to Compare")
                
                if st.session_state.is_classification:
                    st.markdown("**🎯 Classification Models Available:**")
                    model_options = {
                        "Random Forest": RandomForestClassifier(random_state=42),
                        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                        "SVM": SVC(probability=True, random_state=42),
                        "K-Nearest Neighbors": KNeighborsClassifier()
                    }
                    
                    # Justifications for model selection
                    model_justifications = {
                        "Random Forest": """
                        **Best for**: High accuracy, feature importance analysis, robust predictions
                        
                        **Strengths**:
                        - Excellent performance with minimal tuning
                        - Handles non-linear relationships naturally
                        - Built-in feature importance ranking
                        - Robust to outliers and missing values
                        - Works well with imbalanced datasets
                        
                        **Use When**: You need reliable predictions with interpretable feature importance
                        """,
                        "Gradient Boosting": """
                        **Best for**: Maximum predictive accuracy, complex pattern recognition
                        
                        **Strengths**:
                        - Often achieves highest accuracy in competitions
                        - Sequentially learns from previous errors
                        - Excellent with mixed data types
                        - Strong performance on tabular data
                        - Provides feature importance
                        
                        **Use When**: Accuracy is the top priority and you have sufficient data
                        """,
                        "Logistic Regression": """
                        **Best for**: Interpretability, baseline models, probability estimates
                        
                        **Strengths**:
                        - Highly interpretable coefficients
                        - Fast training and prediction
                        - Provides well-calibrated probabilities
                        - Works well with linear relationships
                        - Less prone to overfitting with regularization
                        
                        **Use When**: You need explainable predictions or a reliable baseline
                        """,
                        "SVM": """
                        **Best for**: High-dimensional data, complex decision boundaries
                        
                        **Strengths**:
                        - Excellent for high-dimensional spaces
                        - Memory efficient (uses support vectors)
                        - Versatile with different kernel functions
                        - Effective with limited training data
                        - Strong theoretical foundations
                        
                        **Use When**: You have high-dimensional features or complex patterns
                        """,
                        "K-Nearest Neighbors": """
                        **Best for**: Local patterns, non-parametric classification
                        
                        **Strengths**:
                        - Simple and intuitive approach
                        - No assumptions about data distribution
                        - Naturally handles multi-class problems
                        - Good for irregular decision boundaries
                        - No training phase required
                        
                        **Use When**: You have sufficient data and local similarity matters
                        """
                    }
                else:
                    st.markdown("**📈 Linear Regression Models Available:**")
                    st.info("🎯 **Focus on Linear Models**: These models work best when the relationship between features and target is approximately linear. They provide excellent interpretability and are perfect for understanding feature impacts.")
                    
                    # Pure regression models - for continuous target prediction
                    model_options = {
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
                        "Lasso Regression": Lasso(alpha=1.0, random_state=42),
                        "ElasticNet Regression": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
                    }
                    
                    # Justifications for regression models
                    model_justifications = {
                        "Linear Regression": """
                        **Best for**: Interpretable baseline, linear relationships
                        
                        **Strengths**:
                        - Highly interpretable coefficients
                        - Fast training and prediction
                        - No hyperparameters to tune
                        - Clear feature impact understanding
                        - Works well for linear relationships
                        
                        **Use When**: You need simple, explainable models with linear trends
                        """,
                        "Ridge Regression": """
                        **Best for**: Handling multicollinearity, regularized linear models
                        
                        **Strengths**:
                        - Prevents overfitting with L2 regularization
                        - Handles correlated features well
                        - Stable predictions with high-dimensional data
                        - Maintains all features (shrinks coefficients)
                        - Good bias-variance tradeoff
                        
                        **Use When**: You have many correlated features or risk of overfitting
                        """,
                        "Lasso Regression": """
                        **Best for**: Feature selection, sparse models
                        
                        **Strengths**:
                        - Automatic feature selection (L1 regularization)
                        - Creates sparse models (some coefficients = 0)
                        - Good for high-dimensional data
                        - Removes irrelevant features automatically
                        - Interpretable feature subset
                        
                        **Use When**: You want automatic feature selection and sparse models
                        """,
                        "ElasticNet Regression": """
                        **Best for**: Balanced regularization, grouped features
                        
                        **Strengths**:
                        - Combines Ridge and Lasso benefits
                        - Balances feature selection and multicollinearity
                        - More stable than Lasso for grouped features
                        - Good for correlated feature groups
                        - Tunable L1/L2 ratio parameter
                        
                        **Use When**: You have grouped correlated features and want balanced regularization
                        """
                    }
                
                # Enhanced Model Selection Interface
                st.markdown("---")
                
                # Display model selection with enhanced UI
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Select Models to Train:**")
                    selected_models = []
                    
                    # Create checkboxes for each model with better formatting
                    for model_name, model_obj in model_options.items():
                        # Default selection based on task type
                        if st.session_state.is_classification:
                            default_selected = (model_name == "Random Forest")
                        else:
                            default_selected = (model_name == "Linear Regression")
                        
                        if st.checkbox(
                            f"🔹 Train {model_name}", 
                            value=default_selected,
                            help=f"Click to select {model_name} for training"
                        ):
                            selected_models.append((model_name, model_obj))
                
                with col2:
                    if selected_models:
                        st.markdown("**Selected Models:**")
                        for model_name, _ in selected_models:
                            st.markdown(f"✅ {model_name}")
                        
                        st.markdown(f"**Total: {len(selected_models)} models**")
                    else:
                        st.markdown("**No models selected**")
                        st.warning("Please select at least one model")
                
                # Model Information Expander
                if selected_models:
                    with st.expander("📚 Learn About Your Selected Models", expanded=False):
                        for model_name, _ in selected_models:
                            if model_name in model_justifications:
                                st.markdown(f"### {model_name}")
                                st.markdown(model_justifications[model_name])
                                st.markdown("---")
                
                # Option for hyperparameter tuning
                st.markdown("#### Hyperparameter Tuning")
                use_grid_search = st.checkbox("Perform hyperparameter tuning (may take time)", False)
                
                # Train models when user clicks
                if st.button("Train Selected Models"):
                    if not selected_models:
                        st.error("⚠️ Please select at least one model to train")
                    else:
                        # Container for model results
                        results = []
                        models = {}
                        progress_bar = st.progress(0)
                        
                        # Apply class weights if needed
                        class_weight = None
                        if st.session_state.is_classification and hasattr(st.session_state, 'balance_strategy'):
                            if st.session_state.balance_strategy == "Use class weights":
                                class_weight = "balanced"
                        
                        # Train each selected model
                        for i, (model_name, model) in enumerate(selected_models):
                            st.markdown(f"**Training {model_name}...**")
                            
                            # Apply class weights if possible
                            if class_weight and hasattr(model, "class_weight"):
                                if hasattr(model, "set_params"):
                                    model.set_params(class_weight=class_weight)
                                
                            # Hyperparameter tuning if selected
                            if use_grid_search:
                                # Define parameter grids for each model
                                if st.session_state.is_classification:
                                    param_grids = {
                                        "Random Forest": {
                                            'n_estimators': [50, 100, 200],
                                            'max_depth': [None, 10, 20],
                                            'min_samples_split': [2, 5, 10]
                                        },
                                        "Gradient Boosting": {
                                            'n_estimators': [50, 100, 200],
                                            'learning_rate': [0.01, 0.1, 0.2],
                                            'max_depth': [3, 5, 7]
                                        },
                                        "Logistic Regression": {
                                            'C': [0.1, 1.0, 10.0],
                                            'solver': ['liblinear', 'lbfgs']
                                        },
                                        "SVM": {
                                            'C': [0.1, 1.0, 10.0],
                                            'kernel': ['linear', 'rbf']
                                        },
                                        "K-Nearest Neighbors": {
                                            'n_neighbors': [3, 5, 7, 9],
                                            'weights': ['uniform', 'distance']
                                        }
                                    }
                                else:
                                    # Regression-specific parameter grids
                                    param_grids = {
                                        "Linear Regression": {
                                            # Linear regression has no hyperparameters to tune
                                        },
                                        "Ridge Regression": {
                                            'alpha': [0.1, 1.0, 10.0, 100.0]
                                        },
                                        "Lasso Regression": {
                                            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                                        },
                                        "ElasticNet Regression": {
                                            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                                            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                                        }
                                    }
                                
                                # Get appropriate grid for the model
                                if model_name in param_grids and param_grids[model_name]:
                                    # Only perform grid search if there are parameters to tune
                                    grid_search = GridSearchCV(
                                        model, param_grids[model_name], 
                                        cv=5, scoring='accuracy' if st.session_state.is_classification else 'neg_mean_squared_error',
                                        n_jobs=-1
                                    )
                                    grid_search.fit(X_train, y_train)
                                    best_model = grid_search.best_estimator_
                                    best_params = grid_search.best_params_
                                    best_score = grid_search.best_score_
                                    
                                    st.write(f"✅ Best parameters: {best_params}")
                                    if st.session_state.is_classification:
                                        st.write(f"✅ Cross-validation accuracy: {best_score:.4f}")
                                    else:
                                        st.write(f"✅ Cross-validation RMSE: {np.sqrt(-best_score):.4f}")
                                    
                                    model = best_model
                                else:
                                    # No hyperparameters to tune (e.g., Linear Regression) or not in grid
                                    model.fit(X_train, y_train)
                                    if model_name == "Linear Regression":
                                        st.write("ℹ️ Linear Regression has no hyperparameters to tune")
                            else:
                                model.fit(X_train, y_train)
                            
                            # Make predictions
                            y_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            if st.session_state.is_classification:
                                accuracy = accuracy_score(y_test, y_pred)
                                try:
                                    y_prob = model.predict_proba(X_test)
                                    has_predict_proba = True
                                except:
                                    has_predict_proba = False
                                    
                                # Store results
                                results.append({
                                    'Model': model_name,
                                    'Accuracy': accuracy,
                                    'Has Probabilities': has_predict_proba
                                })
                                
                                st.write(f"Test Accuracy: {accuracy:.4f}")
                            else:
                                # Regression metrics
                                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                                
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                mae = mean_absolute_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)
                                
                                # Store results
                                results.append({
                                    'Model': model_name,
                                    'RMSE': rmse,
                                    'MAE': mae,
                                    'R²': r2
                                })
                                
                                st.write(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                                
                            # Store the trained model
                            models[model_name] = model
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(selected_models))
                        
                        # Store results and models in session state
                        st.session_state.model_results = results
                        st.session_state.trained_models = models
                        st.session_state.models_trained = True
                        
                        st.success("✅ All selected models trained successfully!")
                        
                        # Compare model performance
                        st.markdown("#### Model Comparison")
                        results_df = pd.DataFrame(results)
                        
                        # Check if we have any results
                        if len(results_df) == 0:
                            st.error("No models were successfully trained. Please check your data and try again.")
                        else:
                            if st.session_state.is_classification:
                                # Sort by accuracy (higher is better)
                                results_df = results_df.sort_values('Accuracy', ascending=False)
                                
                                # Visualize model comparison
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(x='Model', y='Accuracy', data=results_df, ax=ax)
                                ax.set_title('Model Accuracy Comparison')
                                ax.set_ylim(max(0, results_df['Accuracy'].min() - 0.1), 1)
                                plt.xticks(rotation=45, ha='right')
                                st.pyplot(fig)
                                
                                best_metric = 'Accuracy'
                                best_value = results_df.iloc[0]['Accuracy']
                            else:
                                # For regression, sort by R² (higher is better) as primary metric
                                results_df = results_df.sort_values('R²', ascending=False)
                                
                                # Visualize model comparison for regression
                                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                                
                                # R² Score (higher is better)
                                sns.barplot(x='Model', y='R²', data=results_df, ax=ax1)
                                ax1.set_title('R² Score Comparison (Higher is Better)')
                                ax1.tick_params(axis='x', rotation=45)
                                
                                # RMSE (lower is better)
                                sns.barplot(x='Model', y='RMSE', data=results_df, ax=ax2)
                                ax2.set_title('RMSE Comparison (Lower is Better)')
                                ax2.tick_params(axis='x', rotation=45)
                                
                                # MAE (lower is better)
                                sns.barplot(x='Model', y='MAE', data=results_df, ax=ax3)
                                ax3.set_title('MAE Comparison (Lower is Better)')
                                ax3.tick_params(axis='x', rotation=45)
                                
                                # Combined view - R² vs RMSE
                                ax4.scatter(results_df['R²'], results_df['RMSE'])
                                for i, model in enumerate(results_df['Model']):
                                    ax4.annotate(model, (results_df.iloc[i]['R²'], results_df.iloc[i]['RMSE']))
                                ax4.set_xlabel('R² Score')
                                ax4.set_ylabel('RMSE')
                                ax4.set_title('R² vs RMSE')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                best_metric = 'R²'
                                best_value = results_df.iloc[0]['R²']
                            
                            # Select best model only if we have results
                            best_model_name = results_df.iloc[0]['Model']
                            st.session_state.best_model_name = best_model_name
                            st.session_state.best_model = models[best_model_name]
                            
                            st.markdown(f"**Best Model: {best_model_name}** with {best_metric}: {best_value:.4f}")
                            
                            # Link to next tab
                            st.info("✓ Proceed to the Evaluation tab for detailed model assessment")
                
                # Show model results if models are trained
                if 'models_trained' in st.session_state and st.session_state.models_trained:
                    st.markdown("#### Trained Models Summary")
                    results_df = pd.DataFrame(st.session_state.model_results)
                    st.dataframe(results_df)
                    
                    # Option to reset model training
                    if st.button("Reset Model Training"):
                        # More comprehensive list of keys to clear when resetting model training
                        keys_to_delete = [
                            # Model training keys
                            'model_results', 'trained_models', 'models_trained', 'best_model_name', 'best_model',
                            'model_metrics', 'training_done', 'model_training_summary', 'cv_results', 'feature_importance',
                            # Do not clear feature engineering or data preparation
                        ]
                        for key in keys_to_delete:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                    
                    # Display justification for best model
                    if st.session_state.best_model_name in model_justifications:
                        st.markdown(f"**Why {st.session_state.best_model_name} works well for this problem:**")
                        st.markdown(model_justifications[st.session_state.best_model_name])
                    
                    # Show model architecture/parameters
                    st.markdown("**Best Model Parameters:**")
                    st.code(str(st.session_state.best_model))
        
        # Tab 4: Evaluation & Insights
        with ml_tabs[3]:
            st.markdown("### 📊 Model Evaluation & Insights")
            
            if 'models_trained' not in st.session_state or not st.session_state.models_trained:
                st.warning("⚠️ Please train models first")
            else:
                # Access data and best model
                X_test = st.session_state.X_test_processed
                y_test = st.session_state.y_test
                best_model = st.session_state.best_model
                best_model_name = st.session_state.best_model_name
                
                # Create tabs for different evaluation aspects - with appropriate names based on model type
                if st.session_state.is_classification:
                    # Classification-appropriate tab names
                    eval_tabs = st.tabs(["Performance Metrics", "Confusion Matrix", "ROC & PR Curves", "Feature Importance", "Insights & Recommendations"])
                else:
                    # Regression-appropriate tab names
                    eval_tabs = st.tabs(["Performance Metrics", "Prediction Analysis", "Regression Diagnostics", "Feature Importance", "Insights & Recommendations"])
                
                # Tab 1: Performance Metrics
                with eval_tabs[0]:
                    st.markdown("#### Detailed Performance Metrics")
                    
                    # Make predictions with best model
                    y_pred = best_model.predict(X_test)
                    
                    if st.session_state.is_classification:
                        # Classification metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Display classification report
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        
                        # Format the report
                        st.dataframe(report_df.style.format({
                            'precision': "{:.3f}",
                            'recall': "{:.3f}",
                            'f1-score': "{:.3f}",
                            'support': "{:.0f}"
                        }).highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))
                        
                        # Interpretation of metrics
                        st.markdown("""
                        **Metric Definitions:**
                        
                        - **Precision**: Proportion of positive identifications that were actually correct.
                          - *Interpretation*: High precision means few false positives.
                        
                        - **Recall**: Proportion of actual positives that were identified correctly.
                          - *Interpretation*: High recall means few false negatives.
                        
                        - **F1-Score**: Harmonic mean of precision and recall.
                          - *Interpretation*: Balances precision and recall, especially useful with imbalanced classes.
                        
                        - **Support**: Number of actual occurrences of the class in the test set.
                        
                        **Actionable Insights:**
                        - Classes with low precision may benefit from more discriminating features
                        - Classes with low recall may need more training examples or different model architecture
                        """)
                    else:
                        # Regression metrics for linear models
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
                        
                        # Calculate regression metrics
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        explained_var = explained_variance_score(y_test, y_pred)
                        
                        # Display metrics in columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("R² Score", f"{r2:.4f}", help="Proportion of variance explained by the model (1.0 is perfect)")
                            st.metric("RMSE", f"{rmse:.4f}", help="Root Mean Squared Error (lower is better)")
                            
                        with col2:
                            st.metric("MAE", f"{mae:.4f}", help="Mean Absolute Error (lower is better)")
                            st.metric("Explained Variance", f"{explained_var:.4f}", help="Explained variance (1.0 is perfect)")
                        
                        # Show regression equation if it's a linear model with coefficients
                        if hasattr(best_model, 'coef_'):
                            st.markdown("#### Linear Regression Equation")
                            
                            # Get feature names and coefficients
                            feature_names = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else [f"Feature {i}" for i in range(X_test.shape[1])]
                            coefficients = best_model.coef_
                            intercept = best_model.intercept_
                            
                            # Format equation
                            equation = f"y = {intercept:.4f}"
                            for name, coef in zip(feature_names, coefficients):
                                sign = '+' if coef >= 0 else ''
                                equation += f" {sign} {coef:.4f} × {name}"
                            
                            st.code(equation)
                            
                            st.markdown("""
                            **How to interpret your linear regression model:**
                            
                            - **R² Score**: Measures how well the model explains the variance in the data
                              - Values close to 1.0 indicate a good fit
                              - Values close to 0 indicate the model doesn't explain much variance
                            
                            - **RMSE (Root Mean Squared Error)**: Average prediction error in the same units as the target variable
                              - Penalizes larger errors more due to squaring
                              - Good for applications where outliers are particularly undesirable
                            
                            - **MAE (Mean Absolute Error)**: Average absolute prediction error
                              - Easier to interpret as it's in the same units as the target
                              - Less sensitive to outliers than RMSE
                            
                            - **Coefficients**: Show how much the target changes per unit increase in each feature
                              - Positive coefficients indicate positive relationships
                              - Negative coefficients indicate inverse relationships
                              - Larger absolute values indicate stronger effects
                            """)
                            
                            # Calculate standardized coefficients for better comparison
                            if isinstance(X_test, pd.DataFrame):
                                # Compute standard deviations of features
                                X_std = X_test.std()
                                y_std = y_test.std() if isinstance(y_test, pd.Series) else np.std(y_test)
                                
                                # Calculate standardized coefficients
                                std_coef = coefficients * (X_std / y_std)
                                
                                # Create DataFrame for comparison
                                coef_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Coefficient': coefficients,
                                    'Standardized Coefficient': std_coef
                                }).sort_values('Standardized Coefficient', key=abs, ascending=False)
                                
                                st.markdown("#### Standardized Coefficients")
                                st.markdown("Standardized coefficients allow direct comparison of feature importance regardless of scale:")
                                st.dataframe(coef_df.style.format({
                                    'Coefficient': "{:.4f}",
                                    'Standardized Coefficient': "{:.4f}"
                                }).background_gradient(cmap='coolwarm', subset=['Standardized Coefficient']))
                        
                        # Residual analysis
                        st.markdown("#### Residual Analysis")
                        residuals = y_test - y_pred
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Residual plot
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(y_pred, residuals)
                            ax.axhline(y=0, color='r', linestyle='-')
                            ax.set_xlabel('Predicted Values')
                            ax.set_ylabel('Residuals')
                            ax.set_title('Residual Plot')
                            st.pyplot(fig)
                            
                        with col2:
                            # QQ Plot for normality check
                            fig, ax = plt.subplots(figsize=(8, 6))
                            import scipy.stats as stats
                            stats.probplot(residuals, dist="norm", plot=ax)
                            ax.set_title('Q-Q Plot for Residuals')
                            st.pyplot(fig)
                        
                        # Residual distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(residuals, kde=True, ax=ax)
                        ax.axvline(x=0, color='r', linestyle='--')
                        ax.set_title('Residual Distribution')
                        st.pyplot(fig)
                        
                        st.markdown("""
                        **Residual Analysis Interpretation:**
                        
                        1. **Residual Plot**: 
                           - Look for random scatter around the zero line
                           - Patterns suggest the model is missing something
                           - Funnel shapes indicate heteroscedasticity (non-constant variance)
                        
                        2. **Q-Q Plot**: 
                           - Points following diagonal line suggest normally distributed residuals
                           - Deviations suggest non-normality
                           - S-shapes indicate skewness or heavy tails
                        
                        3. **Residual Distribution**:
                           - Should be approximately normal, centered at zero
                           - Skewness might indicate missing predictors or transformations needed
                        
                        **What Makes a Good Linear Regression Model:**
                        - Residuals randomly scattered around zero
                        - High R² value (close to 1)
                        - Low RMSE and MAE relative to the scale of your target
                        - Normally distributed residuals
                        - No systematic patterns in residuals
                        """)
                
                # Tab 2: Confusion Matrix
                with eval_tabs[1]:
                    if st.session_state.is_classification:
                        st.markdown("#### Confusion Matrix Analysis")
                        
                        # Generate confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        
                        # Normalize option
                        normalize = st.radio(
                            "Confusion Matrix Type:",
                            ["Raw Counts", "Normalized (by row)", "Normalized (by column)"],
                            horizontal=True
                        )
                        
                        # Apply normalization if selected
                        cm_display = cm.copy()
                        if normalize == "Normalized (by row)":
                            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                            fmt = '.2f'
                        elif normalize == "Normalized (by column)":
                            cm_display = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
                            fmt = '.2f'
                        else:
                            fmt = 'd'
                        
                        # Plot confusion matrix
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted Labels')
                        ax.set_ylabel('True Labels')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig)
                        
                        # Interpretation
                        st.markdown("""
                        **Confusion Matrix Interpretation:**
                        
                        - **Diagonal elements** represent correct predictions (True Positives for each class)
                        - **Off-diagonal elements** represent misclassifications
                          - Row i, column j: samples from class i predicted as class j
                        
                        **Advanced Analysis:**
                        - Look for patterns of misclassification between specific classes
                        - Classes frequently confused with each other may be similar in feature space
                        
                        **Improvement Actions:**
                        1. Focus on frequently misclassified classes by adding more training examples
                        2. Engineer features that better distinguish commonly confused classes
                        3. Consider hierarchical classification for similar classes
                        4. Adjust class weights if certain classes are consistently misclassified
                        """)
                        
                        # Detailed error analysis
                        st.markdown("#### Error Analysis")
                        
                        # Identify misclassified samples
                        misclassified = y_test != y_pred
                        
                        if misclassified.any():
                            st.write(f"Total misclassified samples: {misclassified.sum()} out of {len(y_test)} ({misclassified.sum()/len(y_test)*100:.2f}%)")
                            
                            # Store original indices of test data if available
                            if isinstance(st.session_state.X_test_raw, pd.DataFrame):
                                X_test_indices = st.session_state.X_test_raw.index
                                error_indices = X_test_indices[misclassified]
                                
                                # Most common misclassifications
                                misclassification_pairs = [(actual, pred) for actual, pred in zip(y_test[misclassified], y_pred[misclassified])]
                                misclassification_counts = pd.Series(misclassification_pairs).value_counts()
                                
                                if len(misclassification_counts) > 0:
                                    st.markdown("**Most Common Misclassifications:**")
                                    for (actual, pred), count in misclassification_counts.iloc[:5].items():
                                        st.write(f"- True: {actual}, Predicted: {pred} - {count} times")
                    else:
                        # Regression error analysis and visualization
                        st.markdown("#### Predicted vs Actual Values")
                        
                        # Create scatter plot of predicted vs actual
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.scatter(y_test, y_pred, alpha=0.6)
                        
                        # Add perfect prediction line
                        min_val = min(y_test.min(), y_pred.min())
                        max_val = max(y_test.max(), y_pred.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
                        
                        ax.set_xlabel('Actual Values')
                        ax.set_ylabel('Predicted Values')
                        ax.set_title('Predicted vs Actual Values')
                        
                        # Add R² annotation to plot
                        r2 = r2_score(y_test, y_pred)
                        ax.annotate(f'R² = {r2:.4f}', 
                                   xy=(0.05, 0.95), xycoords='axes fraction',
                                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                                   fontsize=12)
                        
                        st.pyplot(fig)
                        
                        # Error distribution analysis
                        st.markdown("#### Error Analysis")
                        
                        # Calculate errors
                        errors = y_test - y_pred
                        abs_errors = np.abs(errors)
                        
                        # Error statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Error", f"{np.mean(errors):.4f}")
                        with col2:
                            st.metric("Median Error", f"{np.median(errors):.4f}")
                        with col3:
                            st.metric("Error Std Dev", f"{np.std(errors):.4f}")
                        
                        # Error percentiles
                        percentiles = [25, 50, 75, 90, 95, 99]
                        error_percentiles = np.percentile(abs_errors, percentiles)
                        
                        error_df = pd.DataFrame({
                            'Percentile': [f"{p}%" for p in percentiles],
                            'Absolute Error': error_percentiles
                        })
                        
                        st.markdown("**Error Percentiles**")
                        st.markdown("This table shows the distribution of absolute errors:")
                        st.table(error_df.style.format({'Absolute Error': '{:.4f}'}))
                        
                        # Error distribution chart
                        st.markdown("**Error Distribution**")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(errors, bins=30, kde=True, ax=ax)
                        ax.axvline(0, color='red', linestyle='--', alpha=0.8)
                        ax.set_title('Error Distribution')
                        ax.set_xlabel('Error (Actual - Predicted)')
                        st.pyplot(fig)
                        
                        st.markdown("""
                        **Interpreting the Error Analysis:**
                        
                        1. **Predicted vs Actual Plot**:
                           - Points close to the diagonal line indicate good predictions
                           - Points above the line are underestimations
                           - Points below the line are overestimations
                           - Patterns may indicate non-linear relationships not captured by the model
                        
                        2. **Error Statistics**:
                           - Mean Error close to zero suggests unbiased predictions
                           - Large standard deviation indicates high prediction variability
                           - Error percentiles help understand the distribution of errors
                        
                        3. **Error Distribution**:
                           - Should be normally distributed around zero
                           - Skewness may indicate bias in predictions for certain ranges
                        
                        **Actionable Insights:**
                        - If errors increase with predicted values, consider logarithmic transformations
                        - If the model consistently over/under-predicts, review feature selection
                        - Consider removing outliers if they disproportionately impact predictions
                        - For non-linear patterns, try polynomial features or more complex models
                        """)
                
                # Tab 3: ROC & PR Curves for Classification / Regression Diagnostics for Regression
                with eval_tabs[2]:
                    if st.session_state.is_classification:
                        st.markdown("#### ROC and Precision-Recall Curves")
                        
                        # Check if model has predict_proba method
                        try:
                            y_prob = best_model.predict_proba(X_test)
                            has_predict_proba = True
                        except:
                            st.warning(f"{best_model_name} doesn't support probability predictions, so ROC and PR curves cannot be generated.")
                            has_predict_proba = False
                            
                        if has_predict_proba:
                            # For binary classification
                            if len(np.unique(y_test)) == 2:
                                # ROC curve
                                fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
                                roc_auc = auc(fpr, tpr)
                                
                                # Plot ROC curve
                                fig, ax = plt.subplots(figsize=(10, 8))
                                ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
                                ax.plot([0, 1], [0, 1], 'k--', label='Random')
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel('False Positive Rate')
                                ax.set_ylabel('True Positive Rate')
                                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                                ax.legend(loc="lower right")
                                ax.grid(True)
                                st.pyplot(fig)
                                
                                # Precision-Recall curve
                                precision, recall, _ = precision_recall_curve(y_test, y_prob[:, 1])
                                pr_auc = auc(recall, precision)
                                
                                # Plot PR curve
                                fig, ax = plt.subplots(figsize=(10, 8))
                                ax.plot(recall, precision, label=f'PR curve (area = {pr_auc:.3f})')
                                ax.set_xlim([0.0, 1.0])
                                ax.set_ylim([0.0, 1.05])
                                ax.set_xlabel('Recall')
                                ax.set_ylabel('Precision')
                                ax.set_title('Precision-Recall Curve')
                                ax.legend(loc="lower left")
                                ax.grid(True)
                                st.pyplot(fig)
                                
                                # Interpretation
                                st.markdown("""
                                **ROC Curve Interpretation:**
                                - The curve shows the tradeoff between true positive rate (sensitivity) and false positive rate (1-specificity)
                                - AUC (Area Under Curve) ranges from 0.5 (random) to 1.0 (perfect)
                                - Higher AUC indicates better discrimination ability
                                
                                **PR Curve Interpretation:**
                                - Shows the tradeoff between precision and recall
                                - Particularly useful for imbalanced datasets where ROC can be misleading
                                - Higher PR AUC indicates better performance on the positive class
                                
                                **Threshold Selection:**
                                - Different threshold values prioritize different metrics
                                - Lower threshold: higher recall, lower precision
                                - Higher threshold: higher precision, lower recall
                                - Select threshold based on business requirements and costs of false positives/negatives
                                """)
                                
                                # Threshold selection tool
                                st.markdown("#### Threshold Selection Tool")
                                threshold = st.slider("Select probability threshold:", 0.0, 1.0, 0.5, 0.01)
                                
                                # Calculate metrics for selected threshold
                                y_pred_threshold = (y_prob[:, 1] >= threshold).astype(int)
                                accuracy_threshold = accuracy_score(y_test, y_pred_threshold)
                                from sklearn.metrics import precision_score, recall_score, f1_score
                                precision_threshold = precision_score(y_test, y_pred_threshold, zero_division=0)
                                recall_threshold = recall_score(y_test, y_pred_threshold)
                                f1_threshold = f1_score(y_test, y_pred_threshold)
                                
                                # Display metrics for selected threshold
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Accuracy", f"{accuracy_threshold:.3f}")
                                col2.metric("Precision", f"{precision_threshold:.3f}")
                                col3.metric("Recall", f"{recall_threshold:.3f}")
                                col4.metric("F1 Score", f"{f1_threshold:.3f}")
                            else:
                                st.info("ROC and PR curves are shown for binary classification. For multi-class problems, these curves would be calculated per class (one-vs-rest).")
                                
                    else:
                        # Regression advanced diagnostics
                        st.markdown("#### Linear Regression Model Diagnostics")
                        
                        # Calculate residuals
                        y_pred = best_model.predict(X_test)
                        residuals = y_test - y_pred
                        
                        # Advanced regression diagnostics - Cook's distance
                        st.markdown("##### Influential Points Analysis")
                        
                        # Get feature names
                        X_cols = X_test.columns if isinstance(X_test, pd.DataFrame) else None
                        
                        try:
                            # Only compute for linear models with reasonable dataset size
                            if X_test.shape[0] < 10000:
                                # Compute Cook's distance
                                from statsmodels.stats.outliers_influence import OLSInfluence
                                import statsmodels.api as sm
                                
                                # Prepare data for statsmodels
                                X_with_const = sm.add_constant(X_test) if isinstance(X_test, pd.DataFrame) else sm.add_constant(X_test)
                                
                                # Fit the model
                                model = sm.OLS(y_test, X_with_const).fit()
                                
                                # Calculate influence metrics
                                influence = OLSInfluence(model)
                                cooks_d = influence.cooks_distance[0]
                                
                                # Plot Cook's distance
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.stem(cooks_d, markerfmt=",")
                                ax.set_title('Cook\'s Distance - Influential Points')
                                ax.set_xlabel('Observation Index')
                                ax.set_ylabel('Cook\'s Distance')
                                
                                # Add threshold line (4/n is common)
                                threshold = 4/len(X_test)
                                ax.axhline(y=threshold, color='red', linestyle='--')
                                ax.text(len(cooks_d)*0.9, threshold*1.1, f'Threshold: {threshold:.4f}', 
                                       color='red', ha='right')
                                
                                st.pyplot(fig)
                                
                                # Find influential points
                                influential_points = np.where(cooks_d > threshold)[0]
                                if len(influential_points) > 0:
                                    st.markdown(f"**{len(influential_points)} influential points detected** that may disproportionately affect model fit.")
                                    st.markdown("Consider reviewing these points and potentially treating them as outliers.")
                                else:
                                    st.success("No highly influential points detected.")
                                    
                            else:
                                st.info("Influence diagnostics skipped due to large dataset size (>10,000 rows).")
                                
                        except Exception as e:
                            st.warning(f"Could not compute influence metrics: {str(e)}")
                            st.info("This analysis is best suited for pure linear regression models with moderate dataset sizes.")
                        
                        # Heteroscedasticity test
                        st.markdown("##### Heteroscedasticity Analysis")
                        st.markdown("Testing if residual variance is constant across predicted values:")
                        
                        try:
                            import statsmodels.stats.api as sms
                            from statsmodels.compat import lzip
                            
                            # Prepare data for test
                            X_with_const = sm.add_constant(X_test) if isinstance(X_test, pd.DataFrame) else sm.add_constant(X_test)
                            
                            # Breusch-Pagan test
                            bp_test = sms.het_breuschpagan(residuals, X_with_const)
                            bp_labels = ['LM Statistic', 'LM p-value', 'F Statistic', 'F p-value']
                            
                            # Create test results
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Breusch-Pagan Test for Heteroscedasticity**")
                                bp_result = pd.DataFrame({
                                    'Metric': bp_labels,
                                    'Value': bp_test
                                })
                                st.dataframe(bp_result.style.format({'Value': '{:.4f}'}))
                                
                                # Interpret the test
                                if bp_test[1] < 0.05:
                                    st.warning("⚠️ **Heteroscedasticity detected** (p < 0.05)")
                                    st.markdown("This suggests that residuals have non-constant variance, which can affect standard errors.")
                                else:
                                    st.success("✅ No significant heteroscedasticity detected (p > 0.05)")
                            
                            with col2:
                                # Scale-Location plot
                                standardized_residuals = residuals / np.sqrt(np.mean(residuals**2))
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.scatter(y_pred, np.sqrt(np.abs(standardized_residuals)))
                                
                                # Add smoothed line
                                try:
                                    from scipy.stats import loess
                                    # Only attempt LOESS if supported and dataset is reasonable size
                                    if len(y_pred) < 1000:
                                        sorted_indices = np.argsort(y_pred)
                                        sorted_x = y_pred[sorted_indices]
                                        sorted_y = np.sqrt(np.abs(standardized_residuals))[sorted_indices]
                                        loess_result = loess(sorted_x, sorted_y)
                                        loess_result.fit()
                                        pred = loess_result.predict(sorted_x)
                                        ax.plot(sorted_x, pred, color='red', linewidth=2)
                                except:
                                    # Fallback if LOESS fails
                                    z = np.polyfit(y_pred, np.sqrt(np.abs(standardized_residuals)), 1)
                                    p = np.poly1d(z)
                                    ax.plot(np.sort(y_pred), p(np.sort(y_pred)), "r--", linewidth=2)
                                
                                ax.set_title('Scale-Location Plot')
                                ax.set_xlabel('Fitted Values')
                                ax.set_ylabel('Sqrt of |Standardized Residuals|')
                                st.pyplot(fig)
                                
                        except Exception as e:
                            st.warning(f"Could not perform heteroscedasticity test: {str(e)}")
                        
                        # Linearity assessment - Component + Residual plots
                        if isinstance(X_test, pd.DataFrame) and hasattr(best_model, 'coef_'):
                            st.markdown("##### Linearity Assessment: Component + Residual Plots")
                            st.markdown("These plots help identify non-linear relationships between predictors and the target:")
                            
                            # Get top features by coefficient magnitude
                            coef_importance = np.abs(best_model.coef_)
                            top_features_idx = np.argsort(-coef_importance)[:min(4, len(X_cols))]
                            top_features = [X_cols[i] for i in top_features_idx]
                            
                            # Create component-residual plots
                            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                            axes = axes.flatten()
                            
                            for i, feature in enumerate(top_features):
                                if i < 4:  # Only show up to 4 features
                                    # Component + residual calculation
                                    x = X_test[feature]
                                    coef = best_model.coef_[list(X_cols).index(feature)]
                                    component = coef * x
                                    comp_plus_resid = component + residuals
                                    
                                    # Plot
                                    axes[i].scatter(x, comp_plus_resid, alpha=0.6)
                                    
                                    # Add regression line
                                    z = np.polyfit(x, comp_plus_resid, 1)
                                    p = np.poly1d(z)
                                    axes[i].plot(np.sort(x), p(np.sort(x)), "r-", linewidth=2)
                                    
                                    # Add loess curve to detect non-linearity
                                    try:
                                        from scipy.stats import loess
                                        if len(x) < 1000:  # Only for reasonable size
                                            sorted_indices = np.argsort(x)
                                            sorted_x = x.iloc[sorted_indices] if hasattr(x, 'iloc') else x[sorted_indices]
                                            sorted_y = comp_plus_resid.iloc[sorted_indices] if hasattr(comp_plus_resid, 'iloc') else comp_plus_resid[sorted_indices]
                                            loess_result = loess(sorted_x, sorted_y)
                                            loess_result.fit()
                                            pred = loess_result.predict(sorted_x)
                                            axes[i].plot(sorted_x, pred, "g--", linewidth=1.5, label="LOESS")
                                    except:
                                        pass
                                    
                                    axes[i].set_title(f'Component + Residual Plot: {feature}')
                                    axes[i].set_xlabel(feature)
                                    axes[i].set_ylabel('Component + Residual')
                                    axes[i].grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.markdown("""
                            **How to interpret Component + Residual Plots:**
                            
                            - **Linear relationship**: Points will follow the linear regression line
                            - **Non-linear relationship**: Points will show a pattern deviating from the line
                            - **Curved pattern**: Consider adding polynomial terms or transformations for this feature
                            - **Fan shapes**: May indicate heteroscedasticity or interaction effects
                            
                            If the green LOESS curve deviates significantly from the red linear fit,
                            consider non-linear transformations of the feature.
                            """)
                
                # Tab 4: Feature Importance
                with eval_tabs[3]:
                    st.markdown("#### Feature Importance Analysis")
                    
                    # Check if model has feature_importances_ attribute
                    has_feature_importance = hasattr(best_model, 'feature_importances_')
                    has_coef = hasattr(best_model, 'coef_')
                    
                    if has_feature_importance:
                        # Get feature names
                        feature_names = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else [f"Feature {i}" for i in range(X_test.shape[1])]
                        
                        # Get feature importances
                        importances = best_model.feature_importances_
                        
                        # Create DataFrame for visualization
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importance
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), ax=ax)
                        ax.set_title('Feature Importance')
                        st.pyplot(fig)
                        
                        # Top and bottom features
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Top 10 Most Important Features**")
                            st.dataframe(importance_df.head(10))
                        with col2:
                            st.markdown("**10 Least Important Features**")
                            st.dataframe(importance_df.tail(10))
                        
                        # Cumulative importance
                        importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.lineplot(x=range(1, len(importance_df) + 1), y=importance_df['Cumulative Importance'], marker='o', ax=ax)
                        ax.set_xlabel('Number of Features')
                        ax.set_ylabel('Cumulative Importance')
                        ax.set_title('Cumulative Feature Importance')
                        ax.grid(True)
                        st.pyplot(fig)
                        
                        # Find how many features needed for 95% of importance
                        features_for_95 = len(importance_df[importance_df['Cumulative Importance'] <= 0.95])
                        st.write(f"**{features_for_95 + 1}** features explain 95% of the model's predictive power")
                        
                        # Interpretation
                        st.markdown("""
                        **Feature Importance Interpretation:**
                        
                        - Higher importance means the feature has greater influence on the model's predictions
                        - For tree-based models, importance is based on how much each feature decreases impurity
                        - A few dominant features suggest strong predictors that could be focus areas
                        - Many features with similar importance suggest complex relationships
                        
                        **Actionable Insights:**
                        
                        1. Focus on engineering and refining the most important features
                        2. Consider removing or consolidating features with very low importance
                        3. Investigate unexpected important features for domain insights
                        4. Use important features for simpler, more interpretable models
                        """)
                        
                    elif has_coef:
                        # Get feature names
                        feature_names = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else [f"Feature {i}" for i in range(X_test.shape[1])]
                        
                        # Get coefficients
                        if len(best_model.coef_.shape) == 1:
                            # Binary classification or regression
                            coef = best_model.coef_
                        else:
                            # Multi-class
                            coef = np.abs(best_model.coef_).mean(axis=0)
                            
                        # Create DataFrame for visualization
                        coef_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Coefficient': coef
                        }).sort_values('Coefficient', ascending=False, key=abs)
                        
                        # Plot coefficients
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(20), ax=ax)
                        ax.set_title('Feature Coefficients')
                        st.pyplot(fig)
                        
                        # Interpretation
                        st.markdown("""
                        **Coefficient Interpretation:**
                        
                        - Magnitude indicates influence on the prediction
                        - Positive coefficients increase the prediction, negative decrease it
                        - Coefficients are influenced by feature scale (standardization helps comparison)
                        - Linear models show direct relationship between feature and target
                        
                        **Actionable Insights:**
                        
                        1. Features with larger coefficients have stronger impact on predictions
                        2. The sign indicates the direction of impact (positive or negative)
                        3. For standardized features, coefficients directly show relative importance
                        """)
                        
                    else:
                        st.info(f"{best_model_name} doesn't provide direct feature importance metrics.")
                        
                        # Alternative: Permutation importance
                        st.markdown("#### Permutation Feature Importance")
                        st.write("Calculating permutation importance (this may take a moment)...")
                        
                        # Calculate permutation importance
                        from sklearn.inspection import permutation_importance
                        
                        with st.spinner("Calculating permutation importance..."):
                            result = permutation_importance(
                                best_model, X_test, y_test, 
                                n_repeats=5, random_state=42, n_jobs=-1
                            )
                        
                        # Get feature names
                        feature_names = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else [f"Feature {i}" for i in range(X_test.shape[1])]
                        
                        # Create DataFrame for visualization
                        perm_importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': result.importances_mean
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot permutation importance
                        fig, ax = plt.subplots(figsize=(12, 8))
                        sns.barplot(x='Importance', y='Feature', data=perm_importance_df.head(20), ax=ax)
                        ax.set_title('Permutation Feature Importance')
                        st.pyplot(fig)
                        
                        # Interpretation
                        st.markdown("""
                        **Permutation Importance Interpretation:**
                        
                        - Shows how much model performance decreases when a feature is randomly shuffled
                        - Higher importance means the feature is more critical for predictions
                        - Less affected by feature correlations than built-in importance
                        - Calculated on test data, so reflects generalization performance
                        
                        **Advantage over built-in feature importance:**
                        - Works for any model
                        - Not biased toward high-cardinality features
                        - Based on actual performance decrease, not model internals
                        """)
                
                # Tab 5: Insights & Recommendations
                with eval_tabs[4]:
                    st.markdown("### 📋 Summary Insights & Recommendations")
                    
                    # Generate comprehensive report
                    st.markdown("""
                    #### Model Performance Summary
                    """)
                    
                    # Display key metrics
                    if st.session_state.is_classification:
                        y_pred = best_model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Best Model", best_model_name)
                        col2.metric("Accuracy", f"{accuracy:.4f}")
                        
                        # Try to get AUC if possible
                        try:
                            y_prob = best_model.predict_proba(X_test)
                            if len(np.unique(y_test)) == 2:
                                from sklearn.metrics import roc_auc_score
                                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                                col3.metric("AUC", f"{roc_auc:.4f}")
                        except:
                            pass
                    else:
                        # Regression metrics
                        y_pred = best_model.predict(X_test)
                        
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Best Model", best_model_name)
                        col2.metric("R² Score", f"{r2:.4f}", 
                                   help="Proportion of variance explained by the model. Higher is better.")
                        col3.metric("RMSE", f"{rmse:.4f}",
                                  help="Root Mean Squared Error. Lower is better.")
                        
                        # Show regression equation if it's a linear model
                        if hasattr(best_model, 'coef_'):
                            feature_names = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else [f"X{i}" for i in range(X_test.shape[1])]
                            
                            # Get most significant terms
                            coeffs = best_model.coef_
                            intercept = best_model.intercept_
                            
                            # Sort by absolute coefficient value
                            coef_importance = pd.DataFrame({
                                'Feature': feature_names,
                                'Coefficient': coeffs
                            }).sort_values('Coefficient', key=abs, ascending=False)
                            
                            st.markdown("#### Model Equation (Top 5 Terms)")
                            
                            # Display simplified equation with top 5 terms
                            equation = f"y = {intercept:.4f}"
                            for i, row in coef_importance.head(5).iterrows():
                                feature = row['Feature']
                                coef = row['Coefficient']
                                sign = "+" if coef >= 0 else ""
                                equation += f" {sign} {coef:.4f} × {feature}"
                            
                            if len(coef_importance) > 5:
                                equation += " + ..."
                                
                            st.code(equation)
                    
                    # Business Insights
                    if st.session_state.is_classification:
                        st.markdown("""
                        #### Key Business Insights for Classification
                        
                        1. **Model Effectiveness**: The model achieves good predictive performance, demonstrating the feasibility of using machine learning for this type of classification problem.
                        
                        2. **Feature Importance**: The analysis identifies the most influential factors in making predictions, providing actionable intelligence for decision-makers.
                        
                        3. **Data Quality Impact**: The preprocessing steps significantly improved model performance, highlighting the importance of proper data preparation.
                        
                        4. **Model Selection**: Multiple algorithms were systematically evaluated to find the most effective classification approach, ensuring optimal performance.
                        
                        5. **Performance Trade-offs**: Different evaluation metrics show the balance between precision and recall, allowing business-appropriate threshold selection.
                        """)
                    else:
                        st.markdown("""
                        #### Key Business Insights for Linear Regression
                        
                        1. **Predictive Accuracy**: The model explains a significant portion of the variance in the target variable, allowing for reliable predictions of continuous outcomes.
                        
                        2. **Key Drivers Identified**: The coefficient analysis reveals which factors most strongly influence the target variable, providing clear direction for business decision-making.
                        
                        3. **Quantified Relationships**: The linear equation shows exactly how changes in input variables affect the predicted outcome, enabling what-if scenario analysis.
                        
                        4. **Model Robustness**: Residual analysis confirms the model's reliability and helps identify any conditions where predictions might be less accurate.
                        
                        5. **Optimization Opportunities**: Understanding the strongest predictors helps identify where small changes might yield the largest improvements in the target metric.
                        
                        6. **ROI Calculation**: With a quantified model, the business can calculate expected returns on investments that affect the key input variables.
                        """)
                    
                    # Add model quality assessment based on metrics
                    st.markdown("#### Model Quality Assessment")
                    
                    if st.session_state.is_classification:
                        # For classification, assess based on accuracy/AUC
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        if accuracy >= 0.9:
                            quality = "Excellent"
                            quality_color = "green"
                        elif accuracy >= 0.8:
                            quality = "Good"
                            quality_color = "lightgreen"
                        elif accuracy >= 0.7:
                            quality = "Moderate"
                            quality_color = "orange"
                        else:
                            quality = "Needs Improvement"
                            quality_color = "red"
                            
                        st.markdown(f"**Model Quality Rating**: :{quality_color}[{quality}] (Accuracy: {accuracy:.2%})")
                        
                    else:
                        # For regression, assess based on R²
                        if r2 >= 0.8:
                            quality = "Excellent"
                            quality_color = "green"
                        elif r2 >= 0.6:
                            quality = "Good"
                            quality_color = "lightgreen"
                        elif r2 >= 0.4:
                            quality = "Moderate"
                            quality_color = "orange"
                        else:
                            quality = "Needs Improvement"
                            quality_color = "red"
                            
                        st.markdown(f"**Model Quality Rating**: :{quality_color}[{quality}] (R² Score: {r2:.2f})")
                    
                    # Recommendations
                    if st.session_state.is_classification:
                        st.markdown("""
                        #### Recommendations for Classification Model Improvement
                        
                        1. **Data Collection**:
                           - Gather more training examples, especially for underrepresented classes
                           - Consider collecting additional features identified as potentially valuable
                           
                        2. **Feature Engineering**:
                           - Focus on improving the top predictive features identified
                           - Create interaction features between highly correlated variables
                           - Consider domain-specific transformations for key features
                           
                        3. **Model Optimization**:
                           - Fine-tune hyperparameters with more extensive grid search
                           - Consider ensemble methods combining multiple models
                           - Explore deep learning approaches for more complex patterns
                           
                        4. **Deployment Considerations**:
                           - Implement regular model retraining to adapt to changing patterns
                           - Monitor for concept drift and performance degradation
                           - Create interpretability layer for non-technical stakeholders
                        """)
                    else:
                        st.markdown("""
                        #### Recommendations for Linear Regression Model Improvement
                        
                        1. **Data Enhancement**:
                           - Collect data with greater variation in key predictors
                           - Address outliers identified in residual analysis
                           - Fill missing values with more sophisticated techniques if applicable
                           
                        2. **Model Refinement**:
                           - Consider polynomial features if component-residual plots showed non-linearity
                           - Try variable transformations (log, sqrt, etc.) for features with skewed distributions
                           - Investigate regularization strength (alpha parameter) if using Ridge/Lasso
                           - Consider weighted regression if heteroscedasticity is significant
                           
                        3. **Feature Engineering**:
                           - Create interaction terms between related variables
                           - Transform features to address non-linear relationships
                           - Feature selection to retain only significant predictors (p < 0.05)
                           - Normalize/standardize features for better coefficient interpretation
                           
                        4. **Business Applications**:
                           - Create simpler models with just the top predictors for business users
                           - Develop what-if calculators based on the regression equation
                           - Translate coefficients into actionable business recommendations
                           - Set up monitoring for variables with the largest coefficients
                        """)
                    
                    # Export model options
                    st.markdown("#### Export Options")
                    
                    # Generate report
                    if st.button("Generate Detailed Report"):
                        st.markdown("""
                        **📑 Comprehensive Analysis Report**
                        
                        A detailed report would include:
                        - Complete data profiling
                        - All preprocessing steps and their impacts
                        - Detailed model comparison across multiple metrics
                        - Feature importance analysis with business interpretations
                        - Specific recommendations for model deployment
                        - Code samples for implementation
                        
                        *In a production environment, this would generate a downloadable PDF or HTML report.*
                        """)
                    
                    # Option to save model (demo only)
                    if st.button("Save Model"):
                        st.info("In a production environment, this would save the model to disk or cloud storage.")
                        st.code("""
                        # Example code to save the model
                        import pickle
                        
                        # Save the model
                        with open('best_model.pkl', 'wb') as file:
                            pickle.dump(best_model, file)
                            
                        # Save preprocessing pipeline
                        with open('preprocessing_pipeline.pkl', 'wb') as file:
                            pickle.dump(preprocessing_pipeline, file)
                        """)
        
        # Tab 5: Make Predictions
        with ml_tabs[4]:
            st.markdown("### 🔮 Make Real-Time Predictions")
            
            if 'best_model' not in st.session_state or st.session_state.best_model is None:
                st.warning("⚠️ Please train a model first in the 'Model Selection & Training' tab")
            else:
                best_model = st.session_state.best_model
                best_model_name = st.session_state.best_model_name
                
                st.success(f"✅ Using trained model: **{best_model_name}**")
                
                # Get feature information for input
                if 'X_train_processed' in st.session_state:
                    # Check if PCA was applied during feature engineering
                    pca_applied = 'pca' in st.session_state
                    
                    if pca_applied:
                        # If PCA was applied, we need to show original features to the user
                        st.info("🔍 PCA dimensionality reduction was applied during feature engineering. You'll enter values for the original features, and we'll automatically transform them.")
                        
                        # Get original feature names from X_original
                        feature_names = st.session_state.X_original.columns.tolist()
                        
                        # Store reference to original data for statistics
                        input_reference_data = st.session_state.X_original
                    else:
                        # Normal case - use processed features directly
                        feature_names = st.session_state.X_train_processed.columns.tolist()
                        input_reference_data = st.session_state.X_train_processed
                        
                    feature_count = len(feature_names)
                    st.markdown(f"**Enter values for {feature_count} features:**")
                    
                    # Check if there's a pending example prediction
                    if 'prediction_example' in st.session_state:
                        st.info(f"Using {st.session_state.prediction_example} values as starting points. You can adjust them as needed.")
                    
                    # Create input form
                    with st.form("prediction_form"):
                        st.markdown("#### 📝 Input Feature Values")
                        
                        # Organize inputs in columns for better UX
                        cols_per_row = 3
                        cols = st.columns(cols_per_row)
                        
                        user_inputs = {}
                        
                        # We'll use the appropriate data reference for statistics
                        # (either original data for PCA case or processed data otherwise)
                        
                        for i, feature in enumerate(feature_names):
                            col_idx = i % cols_per_row
                            
                            with cols[col_idx]:
                                # Check if this is a categorical or numerical feature
                                feature_dtype = input_reference_data[feature].dtype
                                is_categorical = (
                                    feature_dtype == 'object' or 
                                    pd.api.types.is_categorical_dtype(feature_dtype) or
                                    len(input_reference_data[feature].unique()) < 10 and 
                                    not pd.api.types.is_numeric_dtype(feature_dtype)
                                )
                                
                                if is_categorical:
                                    # Handle categorical feature
                                    unique_values = sorted(input_reference_data[feature].unique())
                                    
                                    # Determine default value based on prediction_example session state
                                    if 'prediction_example' in st.session_state and len(unique_values) > 0:
                                        if st.session_state.prediction_example == "average" or st.session_state.prediction_example == "low":
                                            # Use most common value for average and low examples
                                            most_common = input_reference_data[feature].value_counts().idxmax()
                                            default_index = list(unique_values).index(most_common) if most_common in unique_values else 0
                                        elif st.session_state.prediction_example == "high" and len(unique_values) > 1:
                                            # For high examples, use the last value (may be more appropriate in some cases)
                                            default_index = len(unique_values) - 1
                                        else:
                                            # Fallback to most common
                                            most_common = input_reference_data[feature].value_counts().idxmax()
                                            default_index = list(unique_values).index(most_common) if most_common in unique_values else 0
                                    else:
                                        # Default to most common value
                                        most_common = input_reference_data[feature].value_counts().idxmax()
                                        default_index = list(unique_values).index(most_common) if most_common in unique_values else 0
                                    
                                    # Create dropdown for categorical features
                                    user_inputs[feature] = st.selectbox(
                                        f"**{feature}** (Categorical)",
                                        options=unique_values,
                                        index=default_index,
                                        help=f"Categorical feature with {len(unique_values)} unique values"
                                    )
                                    
                                    # Continue to next feature
                                    continue
                                
                                # For numerical features
                                try:
                                    feature_min = float(input_reference_data[feature].min())
                                    feature_max = float(input_reference_data[feature].max())
                                    feature_mean = float(input_reference_data[feature].mean())
                                    feature_std = float(input_reference_data[feature].std())
                                    
                                    # Create input with helpful information
                                    help_text = f"Training data range: [{feature_min:.2f}, {feature_max:.2f}]\nMean: {feature_mean:.2f} ± {feature_std:.2f}"
                                except (ValueError, TypeError):
                                    # If conversion fails, skip this feature
                                    st.error(f"Error processing feature: {feature}. Please report this issue.")
                                    continue
                                
                                # Calculate reasonable step size
                                step_size = feature_std/10 if feature_std > 0 else 0.1
                                
                                # Check if this appears to be a large-scale feature (like house square footage)
                                is_large_scale = feature_max > 1000 or abs(feature_mean) > 1000
                                
                                # Set default value based on prediction_example session state if present
                                if 'prediction_example' in st.session_state:
                                    if st.session_state.prediction_example == "average":
                                        default_value = feature_mean
                                    elif st.session_state.prediction_example == "high":
                                        default_value = input_reference_data[feature].quantile(0.9)  # Use 90th percentile
                                    elif st.session_state.prediction_example == "low":
                                        default_value = input_reference_data[feature].quantile(0.1)  # Use 10th percentile
                                    else:
                                        default_value = feature_mean
                                else:
                                    default_value = feature_mean
                                    
                                # Note: We only get here if this is a numerical feature
                                
                                # For large scale features, don't use restrictive min/max
                                if is_large_scale:
                                    user_inputs[feature] = st.number_input(
                                        f"**{feature}**",
                                        value=default_value,
                                        step=max(1.0, step_size),  # Ensure reasonable step size for large values
                                        help=help_text + "\n\nYou can enter values outside the training range.",
                                        format="%.2f" if is_large_scale else "%.4f"
                                    )
                                else:
                                    # For normal-scale features, use expanded ranges but still have some limits
                                    # Use 5x the range of the training data as the limits
                                    range_width = feature_max - feature_min
                                    user_inputs[feature] = st.number_input(
                                        f"**{feature}**",
                                        min_value=feature_min - 2*range_width if feature_min != feature_max else None,
                                        max_value=feature_max + 2*range_width if feature_min != feature_max else None,
                                        value=default_value,
                                        step=step_size,
                                        help=help_text,
                                        format="%.4f"
                                    )
                        
                        # Prediction button
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            predict_button = st.form_submit_button(
                                "🎯 Make Prediction", 
                                width="stretch",
                                type="primary"
                            )
                            
                        # Reset prediction_example after form has been rendered
                        if 'prediction_example' in st.session_state:
                            # We've used it to populate the form, now remove it
                            del st.session_state.prediction_example
                        
                        if predict_button:
                            # Prepare input data based on original features
                            input_data = np.array([[user_inputs[feature] for feature in feature_names]])
                            input_df = pd.DataFrame(input_data, columns=feature_names)
                            
                            # Check if PCA was applied during feature engineering
                            pca_applied = 'pca' in st.session_state
                            
                            if pca_applied:
                                with st.spinner("Applying PCA transformation to your input..."):
                                    # Get the stored PCA model
                                    pca = st.session_state.pca
                                    
                                    # We need to transform only the numerical features that were used for PCA
                                    # Get original numerical features that were used for PCA
                                    pca_original_features = st.session_state.pca_original_features
                                    
                                    # Extract just the numerical columns that were used for PCA
                                    pca_input = input_df[pca_original_features]
                                    
                                    # Transform the input data using PCA
                                    pca_result = pca.transform(pca_input)
                                    
                                    # Create DataFrame with PCA components
                                    n_components = pca_result.shape[1]
                                    pca_columns = [f'PCA_{i+1}' for i in range(n_components)]
                                    transformed_input = pd.DataFrame(pca_result, columns=pca_columns)
                                    
                                    # Check if we need to keep any categorical features that weren't part of PCA
                                    cat_features = [col for col in st.session_state.X_train_processed.columns 
                                                   if not col.startswith('PCA_')]
                                    
                                    # If there are categorical features in the processed data, we need to keep them
                                    if cat_features:
                                        # Get categorical features from original data
                                        cat_input_cols = [col for col in feature_names if col not in pca_original_features]
                                        
                                        if cat_input_cols:
                                            # Get categorical data from user input
                                            cat_input = input_df[cat_input_cols]
                                            
                                            # We need to apply the same preprocessing as during training
                                            # Check if we have stored encoders for categorical features
                                            if 'encoders' in st.session_state:
                                                # Apply encoding for each categorical feature
                                                for col in cat_input_cols:
                                                    if col in st.session_state.encoders:
                                                        encoder = st.session_state.encoders[col]
                                                        # Get the value from user input
                                                        value = cat_input[col].iloc[0]
                                                        
                                                        # Transform using the encoder
                                                        try:
                                                            # Transform the single value - need to reshape for sklearn
                                                            encoded_val = encoder.transform([[value]])[0]
                                                            
                                                            # Get the encoded column names
                                                            if hasattr(encoder, 'get_feature_names_out'):
                                                                encoded_cols = encoder.get_feature_names_out([col])
                                                                
                                                                # Add each encoded column to transformed input
                                                                for i, enc_col in enumerate(encoded_cols):
                                                                    transformed_input[enc_col] = encoded_val[i]
                                                            else:
                                                                # Older scikit-learn versions or different encoder
                                                                transformed_input[col] = encoded_val
                                                        except Exception as e:
                                                            st.warning(f"Could not encode {col} properly: {str(e)}")
                                                            # Use a fallback approach - copy from training data
                                                            for c in cat_features:
                                                                transformed_input[c] = st.session_state.X_train_processed[c].iloc[0]
                                            else:
                                                # Fallback if no encoders are stored - use the first row from training data
                                                st.warning("No feature encoders found. Using default values for categorical features.")
                                                for col in cat_features:
                                                    transformed_input[col] = st.session_state.X_train_processed[col].iloc[0]
                                    
                                    # Show the transformed data
                                    st.markdown("#### 🔄 PCA Transformation Applied")
                                    st.markdown("Your input has been transformed using the same PCA components from training:")
                                    st.dataframe(transformed_input)
                                    
                                    # Use the transformed input for prediction
                                    prediction_input = transformed_input
                            else:
                                # Use the original input directly for prediction
                                prediction_input = input_df
                            
                            # Make prediction
                            try:
                                prediction = best_model.predict(prediction_input)[0]
                                
                                # Display results based on task type
                                st.markdown("---")
                                st.markdown("### 🎯 Prediction Results")
                                
                                if st.session_state.is_classification:
                                    # Classification prediction
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.markdown("#### Predicted Class")
                                        st.success(f"**Class: {prediction}**")
                                    
                                    with col2:
                                        # Try to get prediction probabilities
                                        try:
                                            probabilities = best_model.predict_proba(prediction_input)[0]
                                            classes = best_model.classes_
                                            
                                            st.markdown("#### Class Probabilities")
                                            
                                            # Create probability DataFrame
                                            prob_df = pd.DataFrame({
                                                'Class': classes,
                                                'Probability': probabilities
                                            }).sort_values('Probability', ascending=False)
                                            
                                            # Display as a bar chart
                                            fig, ax = plt.subplots(figsize=(8, 4))
                                            sns.barplot(x='Probability', y='Class', data=prob_df, ax=ax)
                                            ax.set_title('Class Probabilities')
                                            ax.set_xlim(0, 1)
                                            
                                            # Add percentage labels
                                            for i, (_, row) in enumerate(prob_df.iterrows()):
                                                ax.text(row['Probability'] + 0.01, i, f'{row["Probability"]:.2%}', 
                                                       va='center', fontweight='bold')
                                            
                                            st.pyplot(fig)
                                            
                                            # Show confidence level
                                            max_prob = probabilities.max()
                                            if max_prob > 0.8:
                                                confidence = "High"
                                                confidence_color = "green"
                                            elif max_prob > 0.6:
                                                confidence = "Medium"
                                                confidence_color = "orange"
                                            else:
                                                confidence = "Low"
                                                confidence_color = "red"
                                            
                                            st.markdown(f"**Confidence Level:** :{confidence_color}[{confidence}] ({max_prob:.2%})")
                                            
                                        except:
                                            st.info("Probability estimates not available for this model")
                                            
                                else:
                                    # Regression prediction
                                    st.markdown("#### Predicted Value")
                                    st.success(f"**Prediction: {prediction:.4f}**")
                                    
                                    # Try to provide prediction interval (for some models)
                                    try:
                                        # For tree-based models, we can estimate uncertainty using individual tree predictions
                                        if hasattr(best_model, 'estimators_'):
                                            tree_predictions = np.array([tree.predict(prediction_input)[0] for tree in best_model.estimators_])
                                            std_dev = np.std(tree_predictions)
                                            
                                            lower_bound = prediction - 2*std_dev
                                            upper_bound = prediction + 2*std_dev
                                            
                                            st.markdown("#### Prediction Interval (95% confidence)")
                                            st.info(f"Range: [{lower_bound:.4f}, {upper_bound:.4f}]")
                                            
                                            # Visualize prediction distribution
                                            fig, ax = plt.subplots(figsize=(8, 4))
                                            ax.hist(tree_predictions, bins=20, alpha=0.7, edgecolor='black')
                                            ax.axvline(prediction, color='red', linestyle='--', linewidth=2, label=f'Prediction: {prediction:.4f}')
                                            ax.axvline(lower_bound, color='orange', linestyle='--', alpha=0.7, label=f'95% CI')
                                            ax.axvline(upper_bound, color='orange', linestyle='--', alpha=0.7)
                                            ax.set_xlabel('Predicted Value')
                                            ax.set_ylabel('Frequency')
                                            ax.set_title('Prediction Distribution (Individual Trees)')
                                            ax.legend()
                                            st.pyplot(fig)
                                            
                                    except:
                                        pass
                                
                                # Feature contribution analysis (for tree-based models)
                                if hasattr(best_model, 'feature_importances_'):
                                    st.markdown("#### Feature Contribution to This Prediction")
                                    
                                    # Calculate feature contributions for this specific prediction
                                    feature_importance = best_model.feature_importances_
                                    input_values = [user_inputs[feature] for feature in feature_names]
                                    
                                    # Create contribution analysis
                                    contribution_df = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Your_Value': input_values,
                                        'Importance': feature_importance,
                                        'Weighted_Contribution': np.array(input_values) * feature_importance
                                    }).sort_values('Weighted_Contribution', ascending=False, key=abs)
                                    
                                    # Display top contributing features
                                    st.markdown("**Top Contributing Features for Your Input:**")
                                    
                                    top_features = contribution_df.head(5)
                                    
                                    for _, row in top_features.iterrows():
                                        impact = "Positive" if row['Weighted_Contribution'] > 0 else "Negative"
                                        impact_color = "green" if impact == "Positive" else "red"
                                        
                                        st.markdown(f"- **{row['Feature']}**: {row['Your_Value']:.4f} → :{impact_color}[{impact} Impact] (Weight: {row['Importance']:.4f})")
                                
                                # Comparison with training data
                                st.markdown("#### How Your Input Compares to Training Data")
                                
                                comparison_data = []
                                for feature in feature_names:
                                    user_value = user_inputs[feature]
                                    feature_mean = X_train[feature].mean()
                                    feature_std = X_train[feature].std()
                                    
                                    # Calculate z-score
                                    z_score = (user_value - feature_mean) / feature_std if feature_std > 0 else 0
                                    
                                    if abs(z_score) > 2:
                                        status = "Unusual"
                                        status_color = "red"
                                    elif abs(z_score) > 1:
                                        status = "Somewhat Unusual"
                                        status_color = "orange"
                                    else:
                                        status = "Normal"
                                        status_color = "green"
                                    
                                    comparison_data.append({
                                        'Feature': feature,
                                        'Your_Value': user_value,
                                        'Training_Mean': feature_mean,
                                        'Z_Score': z_score,
                                        'Status': status
                                    })
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                
                                # Show unusual values
                                unusual_features = comparison_df[comparison_df['Status'] != 'Normal']
                                if len(unusual_features) > 0:
                                    st.warning("⚠️ Some of your input values are unusual compared to training data:")
                                    for _, row in unusual_features.iterrows():
                                        st.markdown(f"- **{row['Feature']}**: {row['Your_Value']:.4f} (Z-score: {row['Z_Score']:.2f}) - {row['Status']}")
                                else:
                                    st.success("✅ All your input values are within normal ranges")
                                
                            except Exception as e:
                                st.error(f"Error making prediction: {str(e)}")
                                st.info("This might be due to data preprocessing differences. Please ensure the input format matches the training data.")
                    
                    # Example predictions outside the form (forms can't contain other forms)
                    st.markdown("---")
                    st.markdown("#### 🚀 Quick Prediction Examples")
                    
                    # Create separate form for examples
                    with st.form("example_predictions_form"):
                        st.markdown("Select from pre-defined example values to quickly test the model:")
                        
                        example_type = st.radio(
                            "Select Example Type:",
                            options=["Average Values", "High Values (75th percentile)", "Low Values (25th percentile)", "Custom Mix"],
                            horizontal=True
                        )
                        
                        # Submit button for examples
                        example_submitted = st.form_submit_button("📊 Generate Example Prediction", width="stretch")
                    
                    # Process example prediction request
                    if example_submitted:
                        st.markdown("### 🔮 Example Prediction Result")
                        
                        try:
                            # Prepare input data based on selection
                            X_train = st.session_state.X_train_processed
                            feature_names = X_train.columns.tolist()
                            
                            if example_type == "Average Values":
                                # Use mean values
                                example_input = X_train.mean().to_frame().T
                                st.success("Using average values from training data")
                            
                            elif example_type == "High Values (75th percentile)":
                                # Use 75th percentile values
                                example_input = X_train.quantile(0.75).to_frame().T
                                st.success("Using high values (75th percentile) from training data")
                            
                            elif example_type == "Low Values (25th percentile)":
                                # Use 25th percentile values
                                example_input = X_train.quantile(0.25).to_frame().T
                                st.success("Using low values (25th percentile) from training data")
                            
                            else:  # Custom Mix
                                # Use a mix of values - mean for half, 75th for quarter, 25th for quarter
                                feature_thirds = np.array_split(feature_names, 3)
                                
                                example_input = pd.DataFrame({
                                    feature: [X_train[feature].mean()] for feature in feature_thirds[0]
                                })
                                
                                # Add high values for second third
                                for feature in feature_thirds[1]:
                                    example_input[feature] = [X_train[feature].quantile(0.75)]
                                    
                                # Add low values for last third
                                for feature in feature_thirds[2]:
                                    example_input[feature] = [X_train[feature].quantile(0.25)]
                                
                                st.success("Using mixed values (average, high, and low) from training data")
                            
                            # Show the example input values
                            with st.expander("View Example Input Values", expanded=False):
                                # Reorganize into readable format
                                readable_input = pd.DataFrame({
                                    'Feature': example_input.columns,
                                    'Value': example_input.iloc[0].values
                                })
                                st.dataframe(readable_input.style.format({'Value': '{:.4f}'}))
                            
                            # Make prediction using the model
                            best_model = st.session_state.best_model
                            prediction = best_model.predict(example_input)[0]
                            
                            # Display results based on task type
                            if st.session_state.is_classification:
                                st.markdown("#### Predicted Class")
                                st.info(f"**Prediction: {prediction}**")
                                
                                # Try to get probabilities
                                try:
                                    probabilities = best_model.predict_proba(example_input)[0]
                                    classes = best_model.classes_
                                    
                                    # Create probability DataFrame
                                    prob_df = pd.DataFrame({
                                        'Class': classes,
                                        'Probability': probabilities
                                    }).sort_values('Probability', ascending=False)
                                    
                                    # Display probabilities
                                    st.markdown("#### Class Probabilities")
                                    st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))
                                except:
                                    st.info("Probability estimates not available for this model")
                            else:
                                st.markdown("#### Predicted Value")
                                st.info(f"**Prediction: {prediction:.4f}**")
                        
                        except Exception as e:
                            st.error(f"Error generating example prediction: {str(e)}")
                            st.info("This might be due to preprocessing issues or model constraints.")
                    
                    # Quick prediction examples buttons (outside both forms)
                    st.markdown("---")
                    st.markdown("#### ⚡ Quick Form Fillers")
                    
                    example_descriptions = {
                        "average": "Uses the average (mean) value for each feature from the dataset",
                        "high": "Uses values at approximately the 90th percentile for each feature",
                        "low": "Uses values at approximately the 10th percentile for each feature"
                    }
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📊 Use Average Values"):
                            st.session_state.prediction_example = "average"
                            st.rerun()
                        st.caption(example_descriptions["average"])
                    with col2:
                        if st.button("📈 Use High Values"):
                            st.session_state.prediction_example = "high"
                            st.rerun()
                        st.caption(example_descriptions["high"])
                    with col3:
                        if st.button("📉 Use Low Values"):
                            st.session_state.prediction_example = "low"
                            st.rerun()
                        st.caption(example_descriptions["low"])
                    
                    # Help section
                    with st.expander("❓ How to Use the Prediction System", expanded=False):
                        st.markdown("""
                        **Step-by-Step Guide:**
                        
                        1. **Enter Feature Values**: Input values for each feature in the form above
                        2. **Use Helpful Hints**: Hover over each input to see the normal range and statistics
                        3. **Make Prediction**: Click the "Make Prediction" button to get results
                        4. **Interpret Results**: 
                           - For Classification: See predicted class and confidence levels
                           - For Regression: See predicted value and possible ranges
                        5. **Analyze Contributions**: Review which features most influenced the prediction
                        6. **Check Data Quality**: See if your inputs are similar to training data
                        
                        **Tips for Better Predictions:**
                        - Values closer to training data ranges will be more reliable
                        - Unusual values (high Z-scores) may lead to less accurate predictions
                        - Use the quick example buttons to see how different input ranges affect predictions
                        - Pay attention to high-importance features for the most impact
                        
                        **Understanding Confidence:**
                        - **High (>80%)**: Very confident prediction
                        - **Medium (60-80%)**: Moderately confident prediction  
                        - **Low (<60%)**: Less confident prediction, consider collecting more data
                        """)
                
                else:
                    st.error("No feature information available. Please ensure preprocessing was completed successfully.")
    
    else:
        st.info("Please upload or select a dataset to begin analysis")