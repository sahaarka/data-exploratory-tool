import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
import time
import re
from collections import defaultdict
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

class DataDiscoveryTool:
    def __init__(self):
        # Initialize session state for datasets
        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}
        
        # Initialize large file handling
        if 'large_file_chunks' not in st.session_state:
            st.session_state.large_file_chunks = {}

    def detect_advanced_data_type(self, column_data):
        """
        Detect advanced data type with more nuanced categorization based on predominant patterns.
        
        Returns:
        - 'date': For columns where majority of values match date patterns
        - 'categorical': For low-cardinality columns
        - 'high_cardinality': For columns with many unique values
        - 'numeric': For columns where majority of values are numeric
        - 'text': For text columns
        """
        # Check for completely null columns first
        if column_data.isna().all():
            return 'all_null'
        # Remove null values for analysis
        non_null_data = column_data.dropna()
        total_non_null = len(non_null_data)

        if total_non_null == 0:
            return 'all_null'
        
        pattern_counts = self.get_column_pattern_distribution(column_data)

        # Define counters for each type
        numeric_count = pattern_counts['numeric']
        date_count = pattern_counts['datetime']
        alpha_count = pattern_counts['alpha']
        alphanumeric_count = pattern_counts['alphanumeric']
        special_char_count = pattern_counts['alpha with special character']

        # Calculate percentages
        numeric_percent = numeric_count / total_non_null * 100
        date_percent = date_count / total_non_null * 100
        alpha_percent = alpha_count / total_non_null * 100
        alphanumeric_percent = alphanumeric_count / total_non_null * 100
        special_char_percent = special_char_count / total_non_null * 100

        # Determine predominant type (use a threshold, e.g., 70%)
        type_percentages = {
            'numeric': numeric_percent,
            'date': date_percent,
            'text': alpha_percent + alphanumeric_percent + special_char_percent
        }

        # Get the type with highest percentage
        predominant_type = max(type_percentages.items(), key=lambda x: x[1])

        if predominant_type[0] == 'numeric':
                return 'numeric'
        elif predominant_type[0] == 'date':
            return 'date'
        else:  # text type
            # return 'text'
            # Further categorize text based on cardinality
            unique_ratio = non_null_data.nunique() / len(non_null_data)
            if unique_ratio <= 0.1:  # Less than 10% unique values
                return 'categorical'
            elif unique_ratio <= 0.5:  # Between 10-50% unique values
                return 'semi_categorical'
            else:
                return 'high_cardinality'

        
    def get_column_pattern_distribution(self, column_data):
        """
        Analyze the distribution of data patterns in a column
        
        Returns a dictionary with counts for different data patterns
        """
        # Remove null values for analysis
        non_null_data = column_data.dropna()
        total_non_null = len(non_null_data)
        
        if total_non_null == 0:
            return {'null': len(column_data)}
        
        # Initialize counters
        pattern_counts = {
            'alpha': 0,           # Only alphabetic characters
            'numeric': 0,         # Only numeric values
            'alphanumeric': 0,    # Mix of alpha and numeric
            'datetime': 0,        # Valid datetime format
            'alpha with special character': 0     # Contains special characters
        }
        
        # Convert to string for pattern analysis
        str_data = non_null_data.astype(str)
        
        # Define patterns
        alpha_pattern = r'^[a-zA-Z\s]+$'  # Only alphabets and spaces
        numeric_pattern = r'^-?\d+(\.\d+)?$'  # Integers and decimals
        alphanumeric_pattern = r'^(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9\s]+$'  # Alphabets, numbers and spaces
        
        # Count patterns
        pattern_counts['alpha'] = str_data.str.match(alpha_pattern).sum()
        # First check for numeric values
        pattern_counts['numeric'] = str_data.str.match(numeric_pattern).sum()

        # Create a mask of numeric values to exclude them from datetime parsing
        numeric_mask = str_data.str.match(numeric_pattern)

        # Count patterns
        pattern_counts['alpha'] = str_data.str.match(alpha_pattern).sum()

        # Alphanumeric but not just alpha or numeric
        alphanumeric_total = str_data.str.match(alphanumeric_pattern).sum()
        pattern_counts['alphanumeric'] = max(0, alphanumeric_total - pattern_counts['alpha'] - pattern_counts['numeric'])

        # Try to detect dates only on non-numeric values
        non_numeric_data = non_null_data[~numeric_mask]

        # Only attempt date parsing if there are non-numeric values to check
        if len(non_numeric_data) > 0:
            # Try to detect dates
            date_formats = [
                '%d-%m-%Y', '%Y-%m-%d', '%m-%d-%Y', 
                '%d/%m/%Y', '%Y/%m/%d', '%m/%d/%Y', 
                '%d.%m.%Y', '%Y.%m.%d', 
                '%d-%b-%Y', '%Y-%b-%d',
                '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S'
            ]
            
            parsed_dates = None
            
            # Try pandas' automatic datetime detection first
            try:
                parsed_dates = pd.to_datetime(non_numeric_data, infer_datetime_format=True, errors='coerce')
            except:
                pass

            # If pandas parsing fails, try custom date formats
            if parsed_dates is None or parsed_dates.isnull().all():
                for date_format in date_formats:
                    try:
                        parsed_dates = pd.to_datetime(non_numeric_data, format=date_format, errors='coerce')
                        if not parsed_dates.isnull().all():
                            break
                    except:
                        continue
            
            # If successfully parsed as date and different from the original values
            if parsed_dates is not None and not parsed_dates.isnull().all():
                pattern_counts['datetime'] = (~parsed_dates.isna()).sum()
            else:
                pattern_counts['datetime'] = 0
        else:
            pattern_counts['datetime'] = 0
        
        # Special chars are anything that doesn't match the alphanumeric pattern
        pattern_counts['alpha with special character'] = total_non_null - (\
                    pattern_counts['alphanumeric'] \
                    + pattern_counts['alpha'] \
                    + pattern_counts['numeric']) \
                    - pattern_counts['datetime']
        
        # Add null count
        pattern_counts['null'] = len(column_data) - total_non_null
        
        return pattern_counts

    def parse_dates(self, dataframe):
        """
        Intelligently detect and convert date columns
        
        Supports multiple common date formats while preserving numeric columns
        """
        # List of common date format strings to try
        date_formats = [
            '%d-%m-%Y',   # 18-09-2023
            '%Y-%m-%d',   # 2023-09-18
            '%m-%d-%Y',   # 09-18-2023
            '%d/%m/%Y',   # 18/09/2023
            '%Y/%m/%d',   # 2023/09/18
            '%m/%d/%Y',   # 09/18/2023
            '%d.%m.%Y',   # 18.09.2023
            '%Y.%m.%d',   # 2023.09.18
            '%d-%b-%Y',   # 18-Sep-2023
            '%Y-%b-%d',   # 2023-Sep-18
            '%Y-%m-%d %H:%M:%S',  # 2023-09-18 14:30:00
            '%d-%m-%Y %H:%M:%S',   # 18-09-2023 14:30:00
            '%m-%d-%Y %H:%M:%S'
        ]
        
        for column in dataframe.columns:

            # Skip numeric columns (int or float)
            if pd.api.types.is_numeric_dtype(dataframe[column]) or \
            pd.api.types.is_datetime64_any_dtype(dataframe[column]) or \
            dataframe[column].isna().all():
                continue
            
            # Attempt parsing
            parsed_dates = None
            
            try:
                # Try manual format matching first
                for date_format in date_formats:
                    try:
                        parsed_dates = pd.to_datetime(
                            dataframe[column], 
                            format=date_format, 
                            errors='coerce'
                        )
                        
                        # If any dates were successfully parsed, break
                        if not parsed_dates.isnull().all():
                            dataframe[column] = parsed_dates
                            break  # Stop once a valid format is found
                    except Exception:
                        continue
                
            except Exception as e:
                # Log or handle parsing errors if needed
                print(f"Error parsing column {column}: {e}")
        
        return dataframe

    def smart_column_visualization(self, column_data, column_name):
        """
        Provide intelligent visualizations based on column type
        """
        # Ensure column_data is a Series
        if not isinstance(column_data, pd.Series):
            column_data = pd.Series(column_data)
        
        # Detect advanced type
        advanced_type = self.detect_advanced_data_type(column_data)
        
        # Date Column Handling
        if advanced_type == 'date' and not(column_data.isna().all()):
            # Convert to datetime if not already
            column_data = pd.to_datetime(column_data)
            
            # Time-based analysis
            plt.figure(figsize=(15, 6))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # plt.cm.plasma, plt.cm.coolwarm, plt.cm.viridis, plt.cm.Paired
            
            # Yearly distribution
            year_counts = column_data.dt.year.value_counts().sort_index()
            colors_year = plt.cm.viridis(np.linspace(0, 1, len(year_counts)))  # Generate colors
            year_counts.plot(kind='bar', ax=ax1, color=colors_year)
            ax1.set_title(f'Yearly Distribution of {column_name}')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Count')
            
            # Monthly distribution
            month_counts = column_data.dt.month.value_counts().sort_index()
            colors_month = plt.cm.plasma(np.linspace(0, 1, len(month_counts)))
            month_counts.plot(kind='bar', ax=ax2, color=colors_month)
            ax2.set_title(f'Monthly Distribution of {column_name}')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Count')

            plt.xticks(rotation=45)
            
            return fig
        
        # Numeric Column Handling
        elif advanced_type == 'numeric' and not(column_data.isna().all()):
            plt.figure(figsize=(15, 6))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram with KDE
            sns.histplot(column_data, kde=True, ax=ax1, color='mediumorchid')
            ax1.set_title(f'Histogram of {column_name}')
            
            # Box Plot
            sns.boxplot(x=column_data, ax=ax2, color='palegreen')
            ax2.set_title(f'Box Plot of {column_name}')
            
            return fig
        
        # Categorical Column Handling
        elif advanced_type in ['categorical', 'semi_categorical'] and not(column_data.isna().all()):
            # Bar chart and pie chart
            top_values = column_data.value_counts(normalize=True) * 100
            
            plt.figure(figsize=(15, 6))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar Chart
            colors = sns.color_palette("Set3", len(top_values)) # colors - Set1, Set2, coolwarm, husl
            top_values.plot(kind='bar', ax=ax1, color=colors)
            ax1.set_title(f'Distribution of {column_name}')
            ax1.set_ylabel('Percentage')
            
            # Pie Chart
            ax2.pie(top_values.values, labels=top_values.index, autopct='%1.1f%%', colors=colors, pctdistance=0.85)
            ax2.set_title(f'Percentage Distribution of {column_name}')
            
            return fig
        
        # High Cardinality Handling
        else:

            if column_data.isna().all():
                st.warning(f"Sample data from Column {column_name} does not have any data.")
                return None
            else:
                st.warning(f"Column {column_name} has too high cardinality for detailed visualization.")
                return None

    def load_large_file(self, uploaded_file, delimiter=',', has_header=True, chunk_size=100000):
        """
        Enhanced method to handle extremely large CSV files with type inference and proper missing value handling
        
        Parameters:
        - uploaded_file: Streamlit uploaded file object
        - delimiter: CSV file delimiter
        - has_header: Whether the file has a header row
        - chunk_size: Number of rows to process in each chunk
        """
        try:
            # Generate a unique identifier for this file
            file_id = uploaded_file.name + str(len(st.session_state.datasets))
            
            # Clear previous chunks if any
            if file_id in st.session_state.large_file_chunks:
                del st.session_state.large_file_chunks[file_id]
            
            # Initialize chunks list for this file
            st.session_state.large_file_chunks[file_id] = []
            
            # First pass: detect column types
            # Read a sample of the data to infer types
            type_sample = pd.read_csv(
                uploaded_file, 
                sep=delimiter, 
                header=0 if has_header else None, 
                nrows=10000,  # Sample first 10000 rows for type inference
                low_memory=False,
                na_values=['', 'NA', 'N/A', 'nan', 'NaN', 'None', 'null'],  # Handle various missing value formats
                keep_default_na=True
            )
            
            # Create type dictionary
            dtype_dict = {}
            for column in type_sample.columns:
                # Try to convert to numeric
                try:
                    pd.to_numeric(type_sample[column], errors='raise')
                    # If successful, use appropriate numeric type
                    if type_sample[column].dtype == 'int64':
                        dtype_dict[column] = 'int64'
                    else:
                        dtype_dict[column] = 'float64'
                except:
                    # If conversion fails, keep as string
                    dtype_dict[column] = 'str'
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Read chunks with inferred types
            chunk_reader = list(pd.read_csv(
                uploaded_file, 
                sep=delimiter, 
                header=0 if has_header else None, 
                chunksize=chunk_size,
                low_memory=False,
                dtype=dtype_dict,  # Use inferred types
                na_values=['', 'NA', 'N/A', 'nan', 'NaN', 'None', 'null'],  # Handle various missing value formats
                keep_default_na=True
            ))
            
            # Collect and store chunks
            total_rows = 0
            progress = 1.0
            for i, chunk in enumerate(chunk_reader):
                st.session_state.large_file_chunks[file_id].append(chunk)
                total_rows += len(chunk)
                
                # Update progress (use a fraction between 0 and 1)
                progress = min(1.0, (i + 1) / len(chunk_reader))
                
            st.sidebar.progress(progress)
            
            # Combine chunks
            df = pd.concat(st.session_state.large_file_chunks[file_id], ignore_index=True)
            
            # Parse dates
            df = self.parse_dates(df)
            
            return df, total_rows
        
        except MemoryError:
            st.sidebar.error("Memory Error: File is too large to process completely")
            return None, 0
        except Exception as e:
            st.sidebar.error(f"Error loading large file: {e}")
            return None, 0
        
    def manage_datasets(self):
        """
        Manage uploaded datasets - view and delete
        """
        st.sidebar.header('📂 Manage Datasets')
        
        if not st.session_state.datasets:
            st.sidebar.info("No datasets uploaded yet.")
            return False
        
        # Track if a dataset was deleted to trigger rerun
        deletion_occurred = False
        
        # List datasets with delete option
        for name, df in list(st.session_state.datasets.items()):  # Use list() to avoid modification during iteration
            # Use columns to place dataset name and delete button side by side
            col1, col2 = st.sidebar.columns([0.8, 0.2])
            with col1:
                st.write(f"📊 {name} ({len(df)} rows)")
            with col2:
                # Compact delete button
                if st.button(f"del", key=f"delete_{name}", help=f"Delete {name}", type="secondary"):
                    del st.session_state.datasets[name]
                    deletion_occurred = True
                    st.rerun()  # This is the key to force an immediate rerun
        
        return len(st.session_state.datasets) > 0

    def upload_dataset(self):
        """
        Enhanced dataset upload with large file support and additional configuration
        """
        st.sidebar.header('📂 Dataset Upload')
        
        # Increase file upload size limit
        st.sidebar.write("Note: For files larger than 200 MB, consider sampling or preprocessing")
        
        # File upload with no size restriction
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file (Large files supported)", 
            type="csv",
            key=f"file_uploader_{len(st.session_state.datasets)}",
            accept_multiple_files=False  # Optional: prevent multiple file upload
        )
        
        if uploaded_file is not None:
            # Get the file name and remove extension for use as dataset name
            file_name = uploaded_file.name
            default_dataset_name = file_name.rsplit('.', 1)[0]  # Remove file extension
            
            # Advanced configuration options
            with st.sidebar.expander("Large File Configuration"):
                dataset_name = st.text_input(
                    "Dataset Name", 
                    value=default_dataset_name
                )
                delimiter = st.text_input("Delimiter", value=",")
                has_header = st.checkbox("File has header", value=True)
                
                # Sampling options for very large files
                sample_data = st.checkbox("Sample data if file is very large", value=False)
                sample_size = st.number_input(
                    "Sample Size (rows)", 
                    min_value=1000, 
                    max_value=1000000, 
                    value=100000,
                    help="Number of rows to sample if file is too large"
                )
                
                chunk_size = st.number_input(
                    "Chunk Size", 
                    min_value=10000, 
                    max_value=1000000, 
                    value=100000,
                    help="Number of rows to load at a time for large files"
                )
            
            # Load file
            try:
                # Check file size
                file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
                st.sidebar.info(f"File Size: {file_size:.2f} MB")
                
                # Sampling for very large files
                if sample_data or file_size > 500:  # Adjust threshold as needed
                    st.sidebar.warning(f"Large file detected. Sampling {sample_size} rows.")
                    df = pd.read_csv(
                        uploaded_file, 
                        sep=delimiter, 
                        header=0 if has_header else None, 
                        nrows=sample_size,
                        low_memory=False,
                        na_values=['', 'NA', 'N/A', 'nan', 'NaN', 'None', 'null'],  # Handle various missing value formats
                        keep_default_na=True
                    )
                    # Parse dates for sampled data
                    df = self.parse_dates(df)
                    total_rows = sample_size
                else:
                    # Load file with chunk support
                    df, total_rows = self.load_large_file(
                        uploaded_file, 
                        delimiter=delimiter, 
                        has_header=has_header,
                        chunk_size=chunk_size
                    )
                
                if df is not None:
                    # Store the dataset
                    st.session_state.datasets[dataset_name] = df
                    st.sidebar.success(
                        f"✅ {dataset_name} uploaded successfully! "
                        f"Total Rows: {total_rows}"
                    )
            
            except Exception as e:
                st.sidebar.error(f"Error uploading file: {e}")

    def find_potential_join_keys(self):
        """
        Find potential join keys between datasets with a simplified interface
        and showing top matching values with counts from each table
        """
        st.header('🔗 Cross-Dataset Join Key Analysis')
        
        # Get list of datasets
        dataset_names = list(st.session_state.datasets.keys())
        
        # Dataset selection
        col1, col2 = st.columns(2)
        with col1:
            dataset1 = st.selectbox("Select First Dataset", dataset_names)
        with col2:
            dataset2 = st.selectbox("Select Second Dataset", 
                                [d for d in dataset_names if d != dataset1])
        
        # Add match percentage threshold with slider only
        st.subheader("Set Minimum Match Percentage")
        
        # Use a container for styling the slider
        slider_container = st.container()
        with slider_container:
            # Add custom CSS for the slider
            st.markdown("""
                <style>
                    div[data-testid="stSlider"] {
                        border-radius: 10px;
                        padding: 10px 5px;
                        background-color: #f0f2f6;
                    }
                    div[data-testid="stSlider"] > div {
                        padding-left: 10px;
                        padding-right: 10px;
                    }
                </style>
            """, unsafe_allow_html=True)
            match_threshold = st.slider(
                "Drag to set threshold",
                min_value=1.0,
                max_value=100.0,
                value=20.0,  # Default value
                step=0.5
            )
        
        # Retrieve datasets
        df1 = st.session_state.datasets[dataset1]
        df2 = st.session_state.datasets[dataset2]
        
        # Determine larger dataset for more accurate percentage calculation
        df1_rows = len(df1)
        df2_rows = len(df2)
        
        # Add information about dataset sizes
        st.info(f"Dataset sizes: {dataset1}: {df1_rows} rows, {dataset2}: {df2_rows} rows")
        
        # Find potential join keys
        potential_joins = []
        
        # Add a progress bar
        progress_bar = st.progress(0)
        total_comparisons = len(df1.columns) * len(df2.columns)
        completed = 0
        
        for col1 in df1.columns:
            for col2 in df2.columns:
                # Preprocess columns for comparison
                series1 = df1[col1].astype(str)
                series2 = df2[col2].astype(str)
                
                # Calculate bidirectional match percentages (against both datasets)
                # This accounts for differences in dataset sizes

                unique_matches1 = len(set(series1[series1.isin(series2)]))
                unique_matches2 = len(set(series2[series2.isin(series1)]))

                unique_match_percentage_df1 = (unique_matches1 / df1_rows) * 100
                unique_match_percentage_df2 = (unique_matches2 / df2_rows) * 100

                # Use the lower percentage for a more conservative estimate
                # This prevents small tables from showing artificially high percentages
                unique_match_percentage = min(unique_match_percentage_df1, unique_match_percentage_df2)
                
                # Only include matches that meet the threshold
                if unique_match_percentage >= match_threshold:

                    matches_in_df1 = (series1.isin(series2)).sum()
                    matches_in_df2 = (series2.isin(series1)).sum()
                    
                    match_percentage_df1 = (matches_in_df1 / df1_rows) * 100
                    match_percentage_df2 = (matches_in_df2 / df2_rows) * 100
                    
                    match_percentage = min(match_percentage_df1, match_percentage_df2)
                    
                    potential_joins.append({
                        f'Column in {dataset1}': col1,
                        f'Column in {dataset2}': col2,
                        'Match Percentage': round(match_percentage, 2),
                        f'{dataset1} Match %': round(match_percentage_df1, 2),
                        f'{dataset2} Match %': round(match_percentage_df2, 2),
                        f'Matched Records in {dataset1}': matches_in_df1,
                        f'Total Records in {dataset1}': df1_rows,
                        f'Matched Records in {dataset2}': matches_in_df2,
                        f'Total Records in {dataset2}': df2_rows,
                        f'Unique Matches in {dataset1}': unique_matches1,
                        f'Unique Matches in {dataset2}': unique_matches2
                    })
                
                # Update progress
                completed += 1
                progress_bar.progress(completed / total_comparisons)
        
        # Display results
        if potential_joins:
            # Sort by match percentage (descending)
            potential_joins.sort(key=lambda x: x['Match Percentage'], reverse=True)
            
            st.subheader(f"Potential Joins (Match Percentage ≥ {match_threshold}%)")
            
            # Create a more detailed dataframe for display
            display_df = pd.DataFrame(potential_joins)
            
            # Select columns to display initially (hide some of the detailed columns)
            display_columns = [
                f'Column in {dataset1}', f'Column in {dataset2}', 'Match Percentage',
                f'Unique Matches in {dataset1}', f'Unique Matches in {dataset2}'
            ]
            
            # Display the dataframe with the selected columns
            st.dataframe(display_df[display_columns])
            
            # Option to view full details
            if st.checkbox("Show detailed match metrics"):
                st.dataframe(display_df)
            
            # Allow user to inspect detailed matches for a selected pair
            if potential_joins:
                st.subheader("Detailed Match Analysis")
                selected_pair = st.selectbox(
                    "Select a column pair to inspect in detail:",
                    [f"{p[f'Column in {dataset1}']} ({dataset1}) ⟷ {p[f'Column in {dataset2}']} ({dataset2})" 
                    for p in potential_joins]
                )
                
                if selected_pair:
                    # Extract column names from selection
                    col1_name = selected_pair.split(f" ({dataset1})")[0]
                    col2_name = selected_pair.split("⟷ ")[1].split(f" ({dataset2})")[0].strip()
                    
                    # Get the matched values
                    series1 = df1[col1_name].astype(str)
                    series2 = df2[col2_name].astype(str)
                    
                    matched_values_set = set(series1[series1.isin(series2)])
                    
                    # Get the selected pair details
                    selected_details = next(
                        (p for p in potential_joins 
                        if p[f'Column in {dataset1}'] == col1_name and p[f'Column in {dataset2}'] == col2_name), 
                        None
                    )
                    
                    if selected_details:
                        # Create explanation for the match percentage
                        st.info(f"""
                        **Match Percentage Explained:**
                        - {selected_details[f'Matched Records in {dataset1}']} out of {selected_details[f'Total Records in {dataset1}']} records match in {dataset1} ({selected_details[f'{dataset1} Match %']}%)
                        - {selected_details[f'Matched Records in {dataset2}']} out of {selected_details[f'Total Records in {dataset2}']} records match in {dataset2} ({selected_details[f'{dataset2} Match %']}%)
                        - Final match percentage: {selected_details['Match Percentage']}% (minimum of both percentages)
                        - Total unique matching values: {len(matched_values_set)}
                        """)

                    if st.checkbox("Show few matching values"):
                    
                        # Show top 25 matching values with counts from each dataset
                        st.subheader(f"Top 25 Matching Values")
                        
                        # Create a dataframe to store value counts from both datasets
                        match_counts = []
                        
                        # Get value counts for each matching value in both datasets
                        for value in matched_values_set:
                            count_in_df1 = series1[series1 == value].count()
                            count_in_df2 = series2[series2 == value].count()
                            
                            match_counts.append({
                                'Value': value,
                                f'Count in {dataset1}': count_in_df1,
                                f'Count in {dataset2}': count_in_df2,
                                'Max Count': max(count_in_df1, count_in_df2)
                            })
                        
                        # Convert to dataframe and sort by total count (descending)
                        match_counts_df = pd.DataFrame(match_counts)
                        match_counts_df = match_counts_df.sort_values('Max Count', ascending=False)
                        
                        # Display top 25 values
                        st.dataframe(match_counts_df.iloc[:, :3].head(25), use_container_width=True, hide_index=True)

        else:
            st.info(f"No column pairs found with match percentage ≥ {match_threshold}%")
        
        # Clear progress bar
        progress_bar.empty()

    @st.cache_data
    def compute_statistics(_self, column_data):
        """
        Compute statistics safely for different data types
        
        Parameters:
        column_data (pd.Series): Input column data
        
        Returns:
        dict: Statistical insights
        """
        # Ensure column_data is a pandas Series
        if not isinstance(column_data, pd.Series):
            column_data = pd.Series(column_data)
        
        # Check if column is completely null
        if column_data.isna().all():
            return {
                'Error': 'Column contains only null values. No statistics available.'
            }
        
        # Detect advanced data type
        advanced_type = _self.detect_advanced_data_type(column_data)
        
        # Handle numeric data
        if advanced_type == 'numeric':
            try:
                return {
                    'Mean': np.mean(column_data).item(),
                    'Median': np.median(column_data).item(),
                    'Standard Deviation': np.std(column_data, ddof=1).item(),
                    'Min': np.min(column_data).item(),
                    'Max': np.max(column_data).item(),
                    'Quartiles': {
                        'Q1 (25%)': np.percentile(column_data, 25).item(),
                        'Q2 (50%)': np.median(column_data).item(),
                        'Q3 (75%)': np.percentile(column_data, 75).item()
                    }
                }
            except Exception as e:
                return {
                    'Error': f'Could not compute statistics: {str(e)}'
                }
        
        # Rest of the function remains the same...
        # Handle date data
        elif advanced_type == 'date':
            # Convert to datetime if not already
            column_data = pd.to_datetime(column_data, errors='coerce')
            
            return {
                'Earliest Date': column_data.min(),
                'Latest Date': column_data.max(),
                'Date Range': str(column_data.max() - column_data.min()),
                'Total Unique Dates': column_data.nunique(),
                'Most Common Year': column_data.dt.year.mode().values[0] if not column_data.dt.year.mode().empty else 'N/A',
                'Most Common Month': column_data.dt.month.mode().values[0] if not column_data.dt.month.mode().empty else 'N/A'
            }
        
        # Handle categorical data
        elif advanced_type in ['categorical', 'semi_categorical']:
            value_counts = column_data.value_counts(normalize=True)
            return {
                'Total Unique Categories': column_data.nunique(),
                'Top 5 Categories (%)': dict((value_counts * 100).head(5)),
                'Most Frequent Category': column_data.mode().values[0] if not column_data.mode().empty else 'N/A'
            }
        
        # Fallback for other types
        else:
            return {
                'Total Values': len(column_data),
                'Unique Values': column_data.nunique(),
                'Value Type': str(column_data.dtype)
            }

    @st.cache_data
    def compute_advanced_statistics(_self, column_data):
        """
        Compute advanced statistics safely for different data types
        
        Parameters:
        column_data (pd.Series): Input column data
        
        Returns:
        dict: Advanced statistical insights
        """
        # Ensure column_data is a pandas Series
        if not isinstance(column_data, pd.Series):
            column_data = pd.Series(column_data)
        
        # Check if column is completely null
        if column_data.isna().all():
            return {
                'Error': 'Column contains only null values. No advanced statistics available.'
            }
        
        # Detect advanced data type
        advanced_type = _self.detect_advanced_data_type(column_data)
        
        # Rest of the function remains the same...
        # Handle numeric data
        if advanced_type == 'numeric':
            numeric_data = pd.to_numeric(column_data, errors='coerce')
            try:
                return {
                    'Skewness': float(numeric_data.skew()),
                    'Kurtosis': float(numeric_data.kurtosis()),
                    'Variance': float(numeric_data.var()),
                    'Coefficient of Variation': float((numeric_data.std() / numeric_data.mean()) * 100)
                }
            except Exception as e:
                return {
                    'Error': f'Could not compute advanced statistics: {str(e)}'
                }
        
        # Handle date data
        elif advanced_type == 'date':
            # Convert to datetime if not already
            column_data = pd.to_datetime(column_data, errors='coerce')
            
            return {
                'Most Common Year': column_data.dt.year.mode().values[0] if not column_data.dt.year.mode().empty else 'N/A',
                'Most Common Month': column_data.dt.month.mode().values[0] if not column_data.dt.month.mode().empty else 'N/A',
                'Yearly Distribution': dict(column_data.dt.year.value_counts().head(5))
            }
        
        # Handle categorical data
        elif advanced_type in ['categorical', 'semi_categorical']:
            value_counts = column_data.value_counts(normalize=True)
            return {
                'Entropy': float(-(value_counts * np.log2(value_counts)).sum()),
                'Gini Impurity': float(1 - sum(value_counts**2)),
                'Top 5 Values (%)': dict((value_counts * 100).head(5))
            }
        
        # Fallback for other types
        else:
            return {
                'Total Values': len(column_data),
                'Unique Values': column_data.nunique(),
                'Value Type': str(column_data.dtype)
            }
        
    def generate_pattern(self, value):
        """
        Converts a string into a regex-like pattern representing its structure.
        
        - Consecutive digits → d{count}
        - Consecutive alphabets → [A-Za-z]{count}
        - Special characters remain unchanged
        
        Args:
            value (str): The input string
        
        Returns:
            str: The generalized regex pattern
        """
        if pd.isna(value):
            return "NULL"

        value = str(value)
        
        # Replace character types with regex equivalents
        pattern = []
        prev_type = None
        count = 0

        for char in value:
            if char.isdigit():
                char_type = r"\d"
            elif char.isalpha():
                char_type = r"[A-Za-z]"
            elif char.isspace():
                char_type = r"\s"
            else:
                char_type = re.escape(char)  # Escape special characters

            if char_type == prev_type:
                count += 1
            else:
                if prev_type is not None:
                    pattern.append(f"{prev_type}{{{count}}}" if count > 1 else prev_type)
                prev_type = char_type
                count = 1

        # Append the last accumulated pattern
        if prev_type:
            pattern.append(f"{prev_type}{{{count}}}" if count > 1 else prev_type)

        return "".join(pattern)

    def generalize_pattern(self, pattern):
        """
        Converts a detailed pattern into a more generic version by:
        - Replacing regex notations with human-readable terms
        - If the pattern is long, describing it based on dominant separators

        Args:
            pattern (str): The detailed regex pattern
        
        Returns:
            str: A more concise generalized pattern
        """
        # Normalize spaces: Convert multiple spaces into a single space
        pattern = re.sub(r"\s+", " ", pattern)

        # Replace explicit counts with '+'
        pattern = re.sub(r"{\d+}", "", pattern)

        # Convert regex notations to human-readable format
        pattern = pattern.replace("[A-Za-z]", "text")
        pattern = pattern.replace("\\d", "numeric")
        pattern = pattern.replace("\\s", "space")
        pattern = re.sub(r'\\([^a-zA-Z0-9])', r'\1', pattern)

        # Define threshold for long patterns
        LONG_PATTERN_THRESHOLD = 75
        special_count = 0

        if len(pattern) > LONG_PATTERN_THRESHOLD:
            # Count occurrences of space and special characters
            space_count = pattern.count("space")

            # Find all special characters (excluding spaces, text, and numeric)
            special_chars = [char for char in pattern if not char.isalnum() and char not in [" ", "+", "\\"]]
            special_freq = {char: special_chars.count(char) for char in set(special_chars)}
            special_count = sum(special_freq.values())
            
            max_special_count = max(special_freq.values(), default=0)
            dominant_specials = sorted([char for char, count in special_freq.items() if count == max_special_count])

            if special_count > space_count:
                # Identify the most frequent special character(s)
                return f"Long text separated mostly by {' and '.join(dominant_specials)}"

            elif space_count > special_count:
                # Check if special character count is at most 5 less than space count
                if special_count and (space_count - special_count) <= 5:
                    return f"Long text separated mostly with space and {' and '.join(dominant_specials)}"
                return "Long text separated mostly with space"

            else:
                # If tie, include both space and all dominant special characters
                return f"Long text separated mostly with space and {' and '.join(dominant_specials)}"
        
        else:
            # Replace "space" with actual space for shorter patterns
            return pattern.replace("space", " ")

            
    def extract_detailed_patterns(self, series):
        """
        Extracts only the detailed regex patterns without generalizing
        """
        patterns = series.astype(str).apply(lambda x: (self.generate_pattern(x), x))
        
        pattern_counts = defaultdict(lambda: {"count": 0, "examples": set()})
        
        for pat, val in patterns:
            pattern_counts[pat]["count"] += 1
            if len(pattern_counts[pat]["examples"]) < 5:
                pattern_counts[pat]["examples"].add(val)

        # Convert sets to lists for final output
        return {
            pat: {"count": data["count"], "examples": list(data["examples"])}
            for pat, data in pattern_counts.items()
        }

    def extract_generic_patterns(self, series):
        """
        Extracts detailed patterns and then generalizes them
        """
        # First get the detailed patterns
        detailed_patterns = self.extract_detailed_patterns(series)
        
        # Then generalize them
        generalized_counts = defaultdict(lambda: {"count": 0, "examples": set()})
        
        for pat, data in detailed_patterns.items():
            gen_pattern = self.generalize_pattern(pat)
            generalized_counts[gen_pattern]["count"] += data["count"]
            
            # Collect up to 5 unique examples for the generalized pattern
            for example in data["examples"]:
                if len(generalized_counts[gen_pattern]["examples"]) < 5:
                    generalized_counts[gen_pattern]["examples"].add(example)
        
        # Convert sets to lists for final output
        return {
            gp: {"count": data["count"], "examples": list(data["examples"])}
            for gp, data in generalized_counts.items()
        }

    @st.cache_data
    def get_metadata(_self, column_data):
        """
        Enhanced metadata extraction with improved type detection and missing value handling
        """
        # Create a copy to prevent issues with references
        column_data_copy = column_data.copy()
        
        # Get original index length to check for implicit missing values
        original_index_length = max(column_data_copy.index) + 1 if len(column_data_copy) > 0 else 0
        actual_length = len(column_data_copy)
        
        # Calculate implicit missing values (due to index gaps)
        implicit_missing = original_index_length - actual_length
        
        # Check for explicit missing values - ensure both NaN and empty strings are detected
        explicit_missing = column_data_copy.isna().sum()
        
        # Also check for empty strings that may not be detected as NaN
        if column_data_copy.dtype == object:
            explicit_missing += (column_data_copy == '').sum()
        
        # Total missing = explicit + implicit
        total_missing = explicit_missing + implicit_missing
        
        metadata = {
            'Column Name': column_data_copy.name,
            'Total Values': original_index_length,  # Use the full index length
            'Unique Values': column_data_copy.nunique(),
            'Missing Values': int(total_missing)
        }

        # Determine advanced data type
        advanced_type = _self.detect_advanced_data_type(column_data_copy)

        # Add data type and type-specific metadata
        if advanced_type == 'all_null':
            metadata['Data Type'] = 'All Null'
        elif advanced_type == 'date':
            metadata['Data Type'] = 'Datetime'
            try:
                metadata.update({
                    'Earliest Date': str(column_data_copy.min()),
                    'Latest Date': str(column_data_copy.max()),
                    'Date Range': str(column_data_copy.max() - column_data_copy.min())
                })
            except:
                pass
        elif advanced_type == 'numeric':
            metadata['Data Type'] = 'Numeric'
            try:
                # Filter out NaN values for min/max calculation
                non_null_data = column_data_copy.dropna()
                metadata.update({
                    'Min Value': non_null_data.min().item() if len(non_null_data) > 0 else None,
                    'Max Value': non_null_data.max().item() if len(non_null_data) > 0 else None
                })
            except:
                pass
        elif advanced_type in ['categorical', 'semi_categorical']:
            metadata['Data Type'] = 'Object/Text - Low Cardinality'
        else:
            metadata['Data Type'] = 'Object/Text - High Cardinality'
        
        return metadata
    
    # Create a function to get advanced types for all columns
    def get_all_column_types(self, df):
        column_types = {}
        for col in df.columns:
            advanced_type = self.detect_advanced_data_type(df[col])
            if df[col].isna().all():
                column_types[col] = 'all_null'
            elif advanced_type == 'numeric':
                column_types[col] = 'numeric'
            elif advanced_type == 'date':
                column_types[col] = 'datetime'
            else:
                column_types[col] = 'text'
        return pd.Series(column_types).value_counts()

    def univariate_analysis(self):
        """Optimized Univariate Analysis with improved type detection and pattern distribution"""
        st.header('📊 Univariate Analysis')

        dataset_names = list(st.session_state.datasets.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            selected_dataset = st.selectbox("Select Dataset", dataset_names)
        
        df = st.session_state.datasets[selected_dataset]
        
        with col2:
            selected_column = st.selectbox("Select Column", df.columns)
        
        column_data = df[selected_column]
        
        # Check if column is completely null
        if column_data.isna().all():
            st.warning(f"Column '{selected_column}' contains only null values (Type: All Null)")
            metadata = self.get_metadata(df[selected_column])
            st.json(metadata)
            return
        
        # Remove null values for analysis only after checking if all values are null
        non_null_data = column_data.dropna()

        # All tabs under Univariate Analysis
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Column Metadata",
            "Statistical Summary", 
            "Distribution", 
            "Advanced Insights", 
            "Datatype Distribution",
            "Pattern Analysis"  # New tab for detailed pattern analysis
        ])

        # **TAB 1: Column Metadata**
        with tab1:
            st.subheader("Column Metadata")
            # Still show metadata for null columns - this is useful information
            metadata = self.get_metadata(df[selected_column])  # Use original column with nulls
            st.json(metadata)

        # **TAB 2: Distribution**
        with tab2:
            st.subheader("Distribution")
            if len(column_data) == 0:
                st.warning(f"Column '{selected_column}' contains only null values. No distribution visualization available.")
            else:
                # Sample data for performance
                sample_data = column_data.sample(min(len(column_data), 500), random_state=42)
                
                # Smart Visualization
                fig = self.smart_column_visualization(sample_data, selected_column)
                if fig:
                    st.pyplot(fig)

        # **TAB 3: Advanced Insights**
        with tab3:
            st.subheader("Advanced Insights")
            if len(column_data) == 0:
                st.warning(f"Column '{selected_column}' contains only null values. No advanced insights available.")
            else:
                advanced_type = self.detect_advanced_data_type(column_data)
                
                if advanced_type == 'numeric':
                    insights = self.compute_advanced_statistics(column_data)
                    if 'Error' in insights:
                        st.warning(insights['Error'])
                    else:
                        st.json(insights)
                elif advanced_type == 'date':
                    # Ensure datetime conversion
                    column_data = pd.to_datetime(column_data, errors='coerce')
                    
                    # Safely handle date insights with proper datetime methods
                    try:
                        date_insights = {
                            'Most Common Year': int(column_data.dt.year.mode().values[0]) if not column_data.dt.year.mode().empty else 'N/A',
                            'Most Common Month': int(column_data.dt.month.mode().values[0]) if not column_data.dt.month.mode().empty else 'N/A',
                            'Yearly Distribution': {int(year): int(count) for year, count in column_data.dt.year.value_counts().head(10).items()}
                        }
                        st.json(date_insights)
                    except Exception as e:
                        st.warning(f"Could not generate date insights: {e}")
                elif advanced_type in ['categorical', 'semi_categorical']:
                    # Detailed category breakdown
                    category_breakdown = column_data.value_counts(normalize=True) * 100
                    st.dataframe(category_breakdown, use_container_width=True)
                else:
                    st.warning(f"Column {selected_column} has too high cardinality for advanced insights.")

        # **TAB 4: Statistical Summary**
        with tab4:
            st.subheader("Statistical Summary")
            if len(column_data) == 0:
                st.warning(f"Column '{selected_column}' contains only null values. No statistical summary available.")
            else:
                advanced_type = self.detect_advanced_data_type(column_data)
                
                if advanced_type == 'numeric':
                    stats = self.compute_statistics(column_data)
                    if 'Error' in stats:
                        st.warning(stats['Error'])
                    else:
                        st.json(stats)
                elif advanced_type == 'date':
                    # Convert to datetime if not already
                    column_data = pd.to_datetime(column_data, errors='coerce')
                    
                    # Date-specific summary
                    date_stats = {
                        'Earliest Date': column_data.min(),
                        'Latest Date': column_data.max(),
                        'Total Unique Dates': column_data.nunique(),
                        'Date Range': str(column_data.max() - column_data.min())
                    }
                    st.json(date_stats)
                else:
                    st.write("Top Categories (Value Counts):")
                    st.dataframe(column_data.value_counts().head(10), use_container_width=True)
        
        # tab 5: Pattern Distribution
        with tab5:
            st.subheader("Data Pattern Distribution")
            
            # Get pattern distribution
            patterns = self.get_column_pattern_distribution(column_data)
            
            # Create dataframe for visualization
            pattern_df = pd.DataFrame({
                'Pattern': list(patterns.keys()),
                'Count': list(patterns.values())
            })
            
            # Sort by count
            pattern_df = pattern_df.sort_values('Count', ascending=False)
            
            # Display as table
            st.dataframe(pattern_df, use_container_width=True, hide_index=True)
            
            # Visualize distribution
            fig = px.bar(
                pattern_df, 
                x='Pattern', 
                y='Count',
                title=f"Data Pattern Distribution for '{selected_column}'",
                color='Pattern',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig)

            # px.colors.qualitative.Bold
            # px.colors.qualitative.Set1
            # px.colors.qualitative.Pastel
            # px.colors.qualitative.Dark2
            
            # Provide interpretation
            st.subheader("Pattern Interpretation")
            st.write("""
            - **alpha**: Records containing only alphabetic characters and spaces
            - **numeric**: Records containing only numbers (integers or decimals)
            - **alphanumeric**: Records containing both letters and numbers
            - **datetime**: Records that match common date/time formats
            - **special_char**: Records containing special characters
            - **null**: Empty or null values
            """)
            
            # Recommended type based on distribution
            max_pattern = pattern_df.iloc[0]['Pattern']
            if max_pattern != 'null':
                recommended_type = {
                    'alpha': 'text/categorical', 
                    'numeric': 'numeric',
                    'alphanumeric': 'text',
                    'datetime': 'datetime',
                    'alpha with special character': 'text'
                }.get(max_pattern, 'text')
                
                st.info(f"Recommended data type based on pattern distribution: **{recommended_type}**")

        # **TAB 6: Pattern Analysis with improved hoverable tables**
        with tab6:
            st.subheader("Pattern Analysis")
            
            if len(non_null_data) == 0:
                st.warning(f"Column '{selected_column}' contains only null values. No pattern analysis available.")
            else:
                # Display null/NaT percentage information
                null_count = column_data.isna().sum()
                total_count = len(column_data)
                null_percentage = round((null_count / total_count * 100), 2)
                
                if null_count > 0:
                    st.info(f"This column contains {null_count} null/NaT values ({null_percentage}% of total). These are excluded from pattern analysis.")

                # Your radio button code
                pattern_display_type = st.radio(
                    "Pattern display type:",
                    ["Generic pattern", "Regex pattern"],
                    index=0,
                    horizontal=True,  # Ensure options appear in a row
                )

                # OPTION 1: Display generalized patterns using Streamlit's built-in dataframe
                if pattern_display_type == "Generic pattern":
                    st.write("### Generalized Patterns")

                    # Use non_null_data to exclude NaT and null values completely
                    patterns_data = self.extract_generic_patterns(non_null_data)
                    
                    # Sort patterns by count
                    sorted_gen_patterns = dict(sorted(patterns_data.items(), 
                                                    key=lambda x: x[1]["count"], 
                                                    reverse=True))
                    
                    # Prepare data for display, keeping top 20 patterns
                    top_gen_patterns = list(sorted_gen_patterns.items())[:20]
                    
                    # Calculate the count of "other" patterns if applicable
                    other_gen_count = sum(data["count"] for pattern, data in list(sorted_gen_patterns.items())[20:]) if len(sorted_gen_patterns) > 20 else 0
                    
                    # Create dataframe
                    gen_df_data = [
                        {
                            "Pattern": pattern,
                            "Count": data["count"],
                            "Percentage": f"{round((data['count'] / len(non_null_data) * 100), 2)}%",
                            "5 Examples": ("?|" if any("," in str(ex) for ex in data["examples"][:5]) else ", ").join(map(str, data["examples"][:5])),
                            "Example Delimiter": None if len(data["examples"]) == 1 else ("?|" if any("," in str(ex) for ex in data["examples"][:5]) else ", ")
                        } for pattern, data in top_gen_patterns
                    ]
                    
                    # Add "Other" category if needed
                    if other_gen_count > 0:
                        gen_df_data.append({
                            "Pattern": "Other",
                            "Count": other_gen_count,
                            "Percentage": f"{round((other_gen_count / len(non_null_data) * 100), 2)}%",
                            "5 Examples": "Multiple patterns"
                        })
                    
                    gen_df = pd.DataFrame(gen_df_data)
                    
                    # Use Streamlit's built-in dataframe display which is more reliable
                    st.dataframe(gen_df, use_container_width=True, hide_index=True)
                
                else:
                    # Display detailed patterns with improved dataframe
                    st.write("### Detailed Patterns")

                    # Use non_null_data to exclude NaT and null values completely
                    patterns_data = self.extract_detailed_patterns(non_null_data)
                    
                    # Sort patterns by count
                    sorted_detailed = dict(sorted(patterns_data.items(), 
                                                key=lambda x: x[1]["count"], 
                                                reverse=True))
                    
                    # Prepare data for display, keeping top 20 patterns
                    top_detailed = list(sorted_detailed.items())[:20]
                    
                    # Calculate the count of "other" patterns if applicable
                    other_detailed_count = sum(data["count"] for pattern, data in list(sorted_detailed.items())[20:]) if len(sorted_detailed) > 20 else 0
                    
                    # Create dataframe
                    detailed_df_data = [
                        {
                            "Pattern": pattern,
                            "Count": data["count"],
                            "Percentage": f"{round((data['count'] / len(non_null_data) * 100), 2)}%",
                            "5 Examples": ("?|" if any("," in str(ex) for ex in data["examples"][:5]) else ", ").join(map(str, data["examples"][:5])),
                            "Example Delimiter": None if len(data["examples"]) == 1 else ("?|" if any("," in str(ex) for ex in data["examples"][:5]) else ", ")
                        } for pattern, data in top_detailed
                    ]
                    
                    # Add "Other" category if needed
                    if other_detailed_count > 0:
                        detailed_df_data.append({
                            "Pattern": "Other",
                            "Count": other_detailed_count,
                            "Percentage": f"{round((other_detailed_count / len(non_null_data) * 100), 2)}%",
                            "5 Examples": "Multiple patterns"
                        })
                    
                    detailed_df = pd.DataFrame(detailed_df_data)
                    
                    # Use Streamlit's built-in dataframe display
                    st.dataframe(detailed_df, use_container_width=True, hide_index=True)
                    
                    # Provide interpretation
                    st.subheader("Pattern Legend")
                    st.write("""
                    - **\\d{n}**: Exactly n consecutive digits
                    - **\\d**: A single digit
                    - **[A-Za-z]{n}**: Exactly n consecutive alphabetic characters
                    - **[A-Za-z]+**: One or more alphabetic characters
                    - **\\s**: A whitespace character
                    - Other characters represent themselves (e.g., "-", ".", "@")
                    """)

    def multivariate_analysis(self):
        """
        Advanced multivariate analysis with aesthetic visualizations
        """
        st.header('🔍 Multivariate Analysis')
    
        # Dataset selection
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset", dataset_names)
        
        df = st.session_state.datasets[selected_dataset]
        
        # Select only numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Column selection
        selected_columns = st.multiselect("Select Columns for Analysis", numeric_columns)
        
        if len(selected_columns) > 1:
            # Create upper triangular matrix of column pairs
            col_pairs = []
            for i in range(len(selected_columns)):
                for j in range(i+1, len(selected_columns)):
                    col_pairs.append((selected_columns[i], selected_columns[j]))
            
            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["Correlation Analysis", "Pairwise Relationships"])
            
            with tab1:
                st.subheader("Correlation Analysis")
                
                # Create a figure with a subplot for each pair
                cols_in_grid = min(2, len(col_pairs))
                rows = math.ceil(len(col_pairs) / cols_in_grid)
                
                fig = plt.figure(figsize=(14, 5 * rows))
                
                for idx, (col1, col2) in enumerate(col_pairs):
                    # Create subplot
                    ax = fig.add_subplot(rows, cols_in_grid, idx + 1)
                    
                    # Calculate correlation
                    correlation = df[[col1, col2]].corr().iloc[0, 1]
                    
                    # Create scatter plot
                    scatter = sns.regplot(
                        x=col1, 
                        y=col2, 
                        data=df, 
                        ax=ax,
                        scatter_kws={'alpha': 0.6, 's': 50},
                        line_kws={'color': 'red'}
                    )
                    
                    # Add correlation annotation
                    ax.annotate(
                        f'Correlation: {correlation:.2f}',
                        xy=(0.05, 0.95),
                        xycoords='axes fraction',
                        fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3)
                    )
                    
                    ax.set_title(f'{col1} vs {col2}', fontsize=14)
                    
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                st.subheader("Pairwise Relationships")
                
                # Let user select number of columns in the grid
                cols_in_grid = min(3, len(col_pairs))
                
                # Create grid of scatter plots
                rows = math.ceil(len(col_pairs) / cols_in_grid)
                
                fig = make_subplots(rows=rows, cols=cols_in_grid, 
                                subplot_titles=[f"{pair[0]} vs {pair[1]}" for pair in col_pairs])
                
                # Add scatter plots to the grid
                for idx, (col1, col2) in enumerate(col_pairs):
                    row = idx // cols_in_grid + 1
                    col = idx % cols_in_grid + 1
                    
                    scatter = go.Scatter(
                        x=df[col1],
                        y=df[col2],
                        mode='markers',
                        marker=dict(
                            size=8,
                            opacity=0.7,
                            color='#636EFA',
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate=
                        f'<b>{col1}</b>: %{{x}}<br>' +
                        f'<b>{col2}</b>: %{{y}}<extra></extra>',
                        name=f"{col1} vs {col2}"
                    )
                    
                    fig.add_trace(scatter, row=row, col=col)
                    
                    # Update axes labels
                    fig.update_xaxes(title_text=col1, row=row, col=col)
                    fig.update_yaxes(title_text=col2, row=row, col=col)
                
                # Update layout for better aesthetics
                fig.update_layout(
                    title={
                        'text': 'Pairwise Relationships',
                        'y':0.98,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': dict(size=20)
                    },
                    showlegend=False,
                    height=250 * rows,
                    width=1000,
                    hovermode='closest',
                    template='plotly_white'
                )
                
                # Add hover effects and improve aesthetics
                fig.update_traces(
                    marker=dict(
                        size=8,
                        line=dict(width=1),
                        opacity=0.7
                    ),
                    selector=dict(mode='markers')
                )
                
                st.plotly_chart(fig)

    def main(self):
        """
        Main application layout with dynamic navigation
        """
        # st.title('🔬 Advanced Data Discovery Tool')

        st.markdown("""
            <style>
                .main-header {
                    background: linear-gradient(to right, #4e54c8, #8f94fb);
                    padding: 1.5rem;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .emoji-icon {
                    font-size: 2.5rem;
                    margin-bottom: 0.5rem;
                }
                .title-text {
                    font-size: 2.2rem;
                    font-weight: 600;
                }
                .subtitle {
                    font-size: 1.1rem;
                    opacity: 0.9;
                    margin-top: 0.5rem;
                }
            </style>

            <div class="main-header">
                <div class="title-text">Advanced Data Discovery Tool</div>
                <div class="subtitle">Explore, analyze, and visualize your data with powerful insights</div>
            </div>
            """, unsafe_allow_html=True)
        
        # First, upload datasets
        self.upload_dataset()
        
        # Manage datasets
        datasets_exist = self.manage_datasets()
        
        # Sidebar navigation based on dataset availability
        if datasets_exist:
            # Extend menu options to include Dataset Preview
            menu_options = ["Dataset Preview", "Univariate Analysis"]
            
            # Add Multivariate Analysis if multiple numeric columns exist
            if any(len(df.select_dtypes(include=[np.number]).columns) > 1 for df in st.session_state.datasets.values()):
                menu_options.append("Multivariate Analysis")
            
            # Add Join Key Analysis if more than one dataset
            if len(st.session_state.datasets) > 1:
                menu_options.append("Join Key Analysis")
            
            # Navigation radio
            menu = st.sidebar.radio("Navigation", menu_options)
            
            # Routing based on menu selection
            if menu == "Dataset Preview":
                # Dataset Preview functionality
                st.header("📊 Dataset Preview")
                
                # Select dataset
                selected_dataset = st.selectbox("Select Dataset", list(st.session_state.datasets.keys()))
                
                # Get the selected dataframe
                df = st.session_state.datasets[selected_dataset]
                
                # Random 10 rows preview
                st.subheader(f"Random 10 Rows from {selected_dataset}")
                st.dataframe(
                    df.sample(min(10, len(df)), random_state=int(time.time())), use_container_width=True, hide_index=True
                )
                
                # Additional dataset info
                st.subheader("Dataset Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Total Columns", len(df.columns))
                
                # Column types
                st.subheader("Column Types")
                column_types = self.get_all_column_types(df)
                # Convert Series to DataFrame
                column_df = pd.DataFrame({"DataType": column_types.index, "Count": column_types.values})

                # Custom color bar chart
                fig = px.bar(
                    column_df,
                    x="DataType",
                    y="Count",
                    title="Column Type Distribution",
                    color="DataType",
                    color_discrete_sequence=px.colors.qualitative.Set3  # Pastel color palette
                )

                st.plotly_chart(fig)
            
            elif menu == "Univariate Analysis":
                self.univariate_analysis()
            elif menu == "Multivariate Analysis":
                self.multivariate_analysis()
            elif menu == "Join Key Analysis":
                self.find_potential_join_keys()
        
        # No datasets uploaded message
        else:
            st.info("Upload a dataset to start analysis")