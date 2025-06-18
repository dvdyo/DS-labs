import streamlit as st
import pandas as pd
import os
import datetime
import urllib.request
from pathlib import Path

# Set page config
st.set_page_config(page_title="VCI/TCI/VHI Analysis", layout="wide")

# DEFINE PATHS - Creates a 'datasets' folder in the same directory as the script
script_dir = Path(__file__).parent if hasattr(Path(__file__), 'parent') else Path.cwd()
path = script_dir / 'datasets'
path.mkdir(exist_ok=True)

def list_datasets():
    """List all dataset files in the directory"""
    if path.exists():
        datasets_list = [f.name for f in path.iterdir() if f.is_file() and f.suffix == '.csv']
        datasets_list.sort(key=lambda x: int(x.split("_")[1].lstrip("ID")) if "_ID" in x else 0)
        return datasets_list
    return []

def get_datasets():
    """Download datasets from NOAA"""
    now = datetime.datetime.now()
    date_and_time = now.strftime("%d-%m-%Y-%H_%M_%S")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(1, 28):
        url = f'https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?country=UKR&provinceID={i}&year1=1981&year2=2024&type=Mean'
        
        try:
            status_text.text(f'Downloading dataset {i}/27...')
            wp = urllib.request.urlopen(url)
            text = wp.read()
            filename = f'NOAA_ID{str(i)}_{date_and_time}.csv'
            
            filepath = path / filename
            with open(filepath, 'wb') as out:
                out.write(text)
            
            progress_bar.progress(i / 27)
        except Exception as e:
            st.error(f"Error downloading dataset {i}: {str(e)}")
            return False
    
    progress_bar.progress(1.0)
    status_text.text('CSV files imported successfully!')
    return True

def clear_datasets():
    """Delete all dataset files"""
    try:
        datasets = list_datasets()
        if not datasets:
            st.warning("No datasets found to delete.")
            return True
            
        for dataset in datasets:
            dataset_path = path / dataset
            if dataset_path.is_file():
                dataset_path.unlink()
        st.success("All datasets removed successfully.")
        return True
    except Exception as e:
        st.error(f"Error removing datasets: {str(e)}")
        return False

def update_datasets():
    """Clear existing datasets and download new ones"""
    if clear_datasets():
        return get_datasets()
    return False

def get_region_mapping():
    """Get mapping of region IDs to Ukrainian names"""
    region_mapping = {
        1: "Черкаська область",
        2: "Чернігівська область", 
        3: "Чернівецька область",
        4: "Республіка Крим",
        5: "Дніпропетровська область",
        6: "Донецька область",
        7: "Івано-Франківська область",
        8: "Харківська область",
        9: "Херсонська область",
        10: "Хмельницька область",
        11: "Київська область",
        12: "Київ",
        13: "Кіровоградська область",
        14: "Луганська область",
        15: "Львівська область",
        16: "Миколаївська область",
        17: "Одеська область",
        18: "Полтавська область",
        19: "Рівненська область",
        20: "Севастополь",
        21: "Сумська область",
        22: "Тернопільська область",
        23: "Закарпатська область",
        24: "Вінницька область",
        25: "Волинська область",
        26: "Запорізька область",
        27: "Житомирська область"
    }
    return region_mapping

@st.cache_data
def load_all_data():
    """LOAD AND COMBINE ALL CSV FILES INTO A SINGLE DATAFRAME"""
    datasets = list_datasets()
    if not datasets:
        return None
    
    combined_data = []
    region_mapping = get_region_mapping()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(datasets):
        try:
            status_text.text(f'Loading file {i+1}/{len(datasets)}: {file}')
            
            # Extract region ID from filename
            region_id = int(file.split("_")[1].lstrip("ID"))
            region_name = region_mapping.get(region_id, f"Region_{region_id}")
            
            # Read CSV file with proper handling of HTML content
            headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI']
            filepath = path / file
            df = pd.read_csv(filepath, header=1, names=headers, index_col=False)
            
            # Create ID column and set as index (following Lab 2 approach)
            df['ID'] = range(1, len(df) + 1)
            df.set_index('ID', inplace=True)
            
            # Remove the last row (which contains HTML closing tags)
            if len(df) > 0:
                df.drop(len(df), inplace=True, errors='ignore')
            
            # Fix the first year entry (copying from second row as in Lab 2)
            if len(df) >= 2:
                df.at[1, 'Year'] = df.at[2, 'Year']
            
            # DATA CLEANING - Remove invalid values
            df = df[df['VHI'] != -1]
            df = df[df['VCI'] != -1]  
            df = df[df['TCI'] != -1]
            df = df.dropna()
            
            # Convert numeric columns to proper types
            numeric_cols = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            # Add region information
            df = df.reset_index(drop=True)
            df['Region_ID'] = region_id
            df['Region'] = region_name
            
            # Select only needed columns
            df = df[['Year', 'Week', 'Region_ID', 'Region', 'VCI', 'TCI', 'VHI']]
            
            combined_data.append(df)
            progress_bar.progress((i + 1) / len(datasets))
            
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
            continue
    
    if combined_data:
        # COMBINE ALL DATAFRAMES
        final_df = pd.concat(combined_data, ignore_index=True)
        final_df = final_df.sort_values(['Year', 'Week', 'Region_ID']).reset_index(drop=True)
        
        # Final data validation
        final_df = final_df[(final_df['Year'] >= 1981) & (final_df['Year'] <= 2024)]
        final_df = final_df[(final_df['Week'] >= 1) & (final_df['Week'] <= 52)]
        
        status_text.text('Data loaded successfully!')
        progress_bar.progress(1.0)
        
        return final_df
    
    return None

def reset_filters():
    """RESET ALL FILTER CONTROLS TO THEIR INITIAL STATE"""
    default_values = {
        'selected_metric': 'VHI',
        'selected_region': None,
        'year_range': None,
        'week_range': (1, 52),
        'sort_ascending': False,
        'sort_descending': False,
        'show_full_tab1': False
    }
    
    for key, value in default_values.items():
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("🔄 All filters have been reset to their initial values!")
    st.rerun()

def initialize_session_state(df=None):
    """INITIALIZE SESSION STATE WITH DEFAULT VALUES"""
    if 'filters_initialized' not in st.session_state:
        st.session_state.filters_initialized = True
        
        # Set defaults based on data availability
        if df is not None:
            if 'selected_region' not in st.session_state:
                st.session_state.selected_region = sorted(df['Region'].unique())[0]
            if 'year_range' not in st.session_state:
                st.session_state.year_range = (int(df['Year'].min()), int(df['Year'].max()))
        
        # Set other defaults
        if 'selected_metric' not in st.session_state:
            st.session_state.selected_metric = 'VHI'
        if 'week_range' not in st.session_state:
            st.session_state.week_range = (1, 52)
        if 'sort_ascending' not in st.session_state:
            st.session_state.sort_ascending = False
        if 'sort_descending' not in st.session_state:
            st.session_state.sort_descending = False
        if 'show_full_tab1' not in st.session_state:
            st.session_state.show_full_tab1 = False

# MAIN APP LAYOUT
st.title("VCI/TCI/VHI Analysis Dashboard")

# Show current working directory info for debugging
with st.expander("📁 Path Information (for debugging)"):
    st.write(f"**Script directory:** `{script_dir}`")
    st.write(f"**Datasets directory:** `{path}`")
    st.write(f"**Datasets directory exists:** {path.exists()}")
    if path.exists():
        st.write(f"**Files in datasets directory:** {len(list(path.iterdir()))} items")

# Create two columns
col1, col2 = st.columns([1, 2])

# LOAD DATA - Make it available for controls
current_datasets = list_datasets()
df = None
if current_datasets:
    with st.spinner("Loading data..."):
        df = load_all_data()

# Initialize session state
initialize_session_state(df)

with col1:
    st.header("Control Panel")
    
    # DATASET MANAGEMENT SECTION
    st.subheader("Dataset Management")
    
    current_datasets = list_datasets()
    if current_datasets:
        st.info(f"Found {len(current_datasets)} dataset files")
        with st.expander("View dataset files"):
            for dataset in current_datasets:
                st.text(dataset)
    else:
        st.warning("No datasets found")
    
    # Dataset management buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("Download Datasets", help="Download all 27 regional datasets from NOAA"):
            with st.spinner("Downloading datasets..."):
                if get_datasets():
                    st.rerun()
    
    with col_btn2:
        if st.button("Delete Datasets", help="Remove all downloaded datasets"):
            if clear_datasets():
                st.rerun()
    
    with col_btn3:
        if st.button("Update Datasets", help="Delete old and download new datasets"):
            with st.spinner("Updating datasets..."):
                if update_datasets():
                    st.rerun()
    
    st.divider()
    
    # ANALYSIS CONTROLS SECTION
    st.subheader("Analysis Controls")
    
    # Metric selection dropdown
    selected_metric = st.selectbox(
        "Select Metric",
        options=["VHI", "VCI", "TCI"],
        index=["VHI", "VCI", "TCI"].index(st.session_state.get('selected_metric', 'VHI')),
        help="Choose which index to analyze",
        key='selected_metric'
    )
    
    # Region selection dropdown
    if df is not None:
        regions_list = sorted(df['Region'].unique())
        default_region = st.session_state.get('selected_region', regions_list[0])
        if default_region not in regions_list:
            default_region = regions_list[0]
        
        selected_region = st.selectbox(
            "Select Region",
            options=regions_list,
            index=regions_list.index(default_region),
            help="Choose region for analysis",
            key='selected_region'
        )
        
        # Year range slider
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        default_year_range = st.session_state.get('year_range', (min_year, max_year))
        year_range = st.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=default_year_range,
            step=1,
            help="Choose the year range for analysis",
            key='year_range'
        )
        
        # Week range slider
        min_week = int(df['Week'].min())
        max_week = int(df['Week'].max())
        default_week_range = st.session_state.get('week_range', (min_week, max_week))
        week_range = st.slider(
            "Select Week Range",
            min_value=min_week,
            max_value=max_week,
            value=default_week_range,
            step=1,
            help="Choose the week range for analysis (1-52)",
            key='week_range'
        )
        
        # SORTING OPTIONS
        st.subheader("Sorting Options")
        st.caption(f"Sort by {selected_metric} values (affects Filtered Data and Regional Comparison tabs)")
        
        sort_ascending = st.checkbox(
            f"Sort by {selected_metric} (Ascending ↗️)",
            value=st.session_state.get('sort_ascending', False),
            help=f"Sort data from lowest to highest {selected_metric} values",
            key='sort_ascending'
        )
        
        sort_descending = st.checkbox(
            f"Sort by {selected_metric} (Descending ↘️)",
            value=st.session_state.get('sort_descending', False),
            help=f"Sort data from highest to lowest {selected_metric} values",
            key='sort_descending'
        )
        
        # Handle sorting logic
        if sort_ascending and sort_descending:
            st.warning("⚠️ Both sorting options selected! Descending sort will take priority.")
            sort_by_metric = True
            sort_ascending_flag = False
        elif sort_descending:
            sort_by_metric = True
            sort_ascending_flag = False
        elif sort_ascending:
            sort_by_metric = True
            sort_ascending_flag = True
        else:
            sort_by_metric = False
            sort_ascending_flag = True
            
        # Show current sorting status
        if sort_by_metric:
            sort_direction = "Ascending" if sort_ascending_flag else "Descending"
            st.info(f"🔄 Active sorting: {selected_metric} ({sort_direction})")
        else:
            st.info("📅 Default sorting: Chronological (Year, Week)")
        
        # RESET FILTERS BUTTON - Moved under sorting options
        st.markdown("---")
        if st.button("🔄 Reset All Filters", 
                     help="Reset all analysis controls to their initial values", 
                     type="secondary",
                     use_container_width=True):
            reset_filters()
        
    else:
        selected_region = st.selectbox(
            "Select Region",
            options=["Load data first"],
            disabled=True,
            help="Load data to see available regions"
        )
        
        # Disabled sliders when no data
        year_range = st.slider(
            "Select Year Range",
            min_value=1981,
            max_value=2024,
            value=(1981, 2024),
            disabled=True,
            help="Load data to enable year selection"
        )
        
        week_range = st.slider(
            "Select Week Range",
            min_value=1,
            max_value=52,
            value=(1, 52),
            disabled=True,
            help="Load data to enable week selection"
        )
        
        # Disabled sorting options when no data
        st.subheader("Sorting Options")
        st.caption("Load data to enable sorting")
        
        sort_ascending = st.checkbox(
            "Sort by metric (Ascending ↗️)",
            disabled=True,
            help="Load data to enable sorting"
        )
        
        sort_descending = st.checkbox(
            "Sort by metric (Descending ↘️)",
            disabled=True,
            help="Load data to enable sorting"
        )
        
        # Default values when no data
        sort_by_metric = False
        sort_ascending_flag = True

with col2:
    st.header("Analysis Area")
    
    # Display data status and load controls
    if current_datasets:
        if st.button("Reload Data", help="Reload and reprocess all dataset files"):
            load_all_data.clear()
            st.rerun()
        
        if df is not None:
            st.success(f"Data loaded successfully! Total records: {len(df)}")
            
            # DISPLAY BASIC STATISTICS
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total Records", len(df))
            with col_stat2:
                st.metric("Years Range", f"{df['Year'].min()}-{df['Year'].max()}")
            with col_stat3:
                st.metric("Regions", df['Region'].nunique())
            
            # CREATE TABS FOR DIFFERENT VIEWS
            tab1, tab2, tab3 = st.tabs(["📊 Filtered Data", "📈 Timeline Chart", "🔀 Regional Comparison"])
            
            with tab1:
                st.subheader("Filtered Dataset Table")
                
                # Show sorting status in caption
                if sort_by_metric:
                    sort_direction = "Ascending" if sort_ascending_flag else "Descending"
                    sort_info = f" | Sorted by {selected_metric} ({sort_direction})"
                else:
                    sort_info = " | Default chronological order"
                
                st.caption(f"Region: {selected_region} | Metric Focus: {selected_metric} | Years: {year_range[0]}-{year_range[1]} | Weeks: {week_range[0]}-{week_range[1]}{sort_info}")
                
                # APPLY ALL FILTERS TO THE DATA
                filtered_data = df[
                    (df['Region'] == selected_region) &
                    (df['Year'] >= year_range[0]) &
                    (df['Year'] <= year_range[1]) &
                    (df['Week'] >= week_range[0]) &
                    (df['Week'] <= week_range[1])
                ].copy()
                
                if not filtered_data.empty:
                    # Apply sorting based on user selection
                    if sort_by_metric:
                        filtered_data = filtered_data.sort_values(
                            [selected_metric, 'Year', 'Week'], 
                            ascending=[sort_ascending_flag, True, True]
                        ).reset_index(drop=True)
                    else:
                        filtered_data = filtered_data.sort_values(['Year', 'Week']).reset_index(drop=True)
                    
                    # Show filtered data statistics
                    col_fstat1, col_fstat2, col_fstat3, col_fstat4 = st.columns(4)
                    with col_fstat1:
                        st.metric("Filtered Records", len(filtered_data))
                    with col_fstat2:
                        st.metric(f"Avg {selected_metric}", f"{filtered_data[selected_metric].mean():.2f}")
                    with col_fstat3:
                        st.metric(f"Min {selected_metric}", f"{filtered_data[selected_metric].min():.2f}")
                    with col_fstat4:
                        st.metric(f"Max {selected_metric}", f"{filtered_data[selected_metric].max():.2f}")
                    
                    # Option to show full filtered dataframe or sample
                    show_full = st.checkbox("Show all filtered records (may be slow for large datasets)", 
                                          value=st.session_state.get('show_full_tab1', False), 
                                          key="show_full_tab1")
                    
                    if show_full or len(filtered_data) <= 1000:
                        st.dataframe(filtered_data, use_container_width=True)
                    else:
                        st.info(f"Showing first 1000 out of {len(filtered_data)} filtered records. Check the box above to see all records.")
                        st.dataframe(filtered_data.head(1000), use_container_width=True)
                    
                    # Additional filtered data info
                    with st.expander("Filtered Dataset Information"):
                        st.write("**Applied Filters:**")
                        st.write(f"- **Region:** {selected_region}")
                        st.write(f"- **Primary Metric:** {selected_metric}")
                        st.write(f"- **Year Range:** {year_range[0]} to {year_range[1]} ({year_range[1] - year_range[0] + 1} years)")
                        st.write(f"- **Week Range:** {week_range[0]} to {week_range[1]} ({week_range[1] - week_range[0] + 1} weeks per year)")
                        
                        st.write("**Applied Sorting:**")
                        if sort_by_metric:
                            sort_direction = "Ascending (lowest to highest)" if sort_ascending_flag else "Descending (highest to lowest)"
                            st.write(f"- **Sort by:** {selected_metric} ({sort_direction})")
                            st.write(f"- **Secondary sort:** Year and Week (ascending)")
                        else:
                            st.write(f"- **Sort by:** Default chronological order (Year, Week)")
                        
                        st.write("**Column Descriptions:**")
                        st.write("- **Year**: Year of observation (1981-2024)")
                        st.write("- **Week**: Week number (1-52)")
                        st.write("- **Region_ID**: Numeric ID of the region (1-27)")
                        st.write("- **Region**: Ukrainian name of the region")
                        st.write("- **VCI**: Vegetation Condition Index")
                        st.write("- **TCI**: Temperature Condition Index") 
                        st.write("- **VHI**: Vegetation Health Index")
                        
                        st.write("**Filtered Data Summary:**")
                        st.dataframe(filtered_data.describe())
                
                else:
                    st.warning(f"No data available for {selected_region} in the selected time range.")
                    st.info("Try adjusting the year or week range, or select a different region.")
                    
                    st.write("**Current Filters:**")
                    st.write(f"- Region: {selected_region}")
                    st.write(f"- Years: {year_range[0]} to {year_range[1]}")
                    st.write(f"- Weeks: {week_range[0]} to {week_range[1]}")
            
            with tab2:
                st.subheader(f"Timeline Chart: {selected_metric} for {selected_region}")
                st.caption(f"Years: {year_range[0]}-{year_range[1]} | Weeks: {week_range[0]}-{week_range[1]} | Timeline view (chronological order)")
                
                # Filter data for selected region, metric, and ranges
                region_data = df[
                    (df['Region'] == selected_region) &
                    (df['Year'] >= year_range[0]) &
                    (df['Year'] <= year_range[1]) &
                    (df['Week'] >= week_range[0]) &
                    (df['Week'] <= week_range[1])
                ].copy()
                
                if not region_data.empty:
                    # SORT BY YEAR AND WEEK FOR PROPER TIMELINE (NOT affected by sorting options)
                    region_data = region_data.sort_values(['Year', 'Week']).reset_index(drop=True)
                    
                    # Create the line chart
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=region_data.index,
                        y=region_data[selected_metric],
                        mode='lines+markers',
                        name=selected_metric,
                        line=dict(width=2),
                        marker=dict(size=3),
                        hovertemplate=f'<b>Year:</b> %{{customdata[0]}}<br>' +
                                    f'<b>Week:</b> %{{customdata[1]}}<br>' +
                                    f'<b>{selected_metric}:</b> %{{y:.2f}}<br>' +
                                    '<extra></extra>',
                        customdata=region_data[['Year', 'Week']].values
                    ))
                    
                    fig.update_layout(
                        title=f'{selected_metric} Timeline for {selected_region}<br><sub>Years {year_range[0]}-{year_range[1]}, Weeks {week_range[0]}-{week_range[1]}</sub>',
                        xaxis_title='Data Points (chronological)',
                        yaxis_title=f'{selected_metric} Value',
                        hovermode='x unified',
                        height=500,
                        showlegend=False
                    )
                    
                    # Add reference lines
                    if selected_metric in ['VCI', 'TCI', 'VHI']:
                        fig.add_hline(y=35, line_dash="dash", line_color="red", 
                                     annotation_text="Drought threshold (35)", annotation_position="bottom right")
                        fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                                     annotation_text="Moderate conditions (50)", annotation_position="top right")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show filtered statistics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Filtered Data Points", len(region_data))
                    with col_stat2:
                        st.metric(f"Average {selected_metric}", f"{region_data[selected_metric].mean():.2f}")
                    with col_stat3:
                        st.metric(f"Min {selected_metric}", f"{region_data[selected_metric].min():.2f}")
                    with col_stat4:
                        st.metric(f"Max {selected_metric}", f"{region_data[selected_metric].max():.2f}")
                    
                    # Additional info about the filtered data
                    with st.expander("Filtered Data Details"):
                        st.write(f"**Selected Filters:**")
                        st.write(f"- Region: {selected_region}")
                        st.write(f"- Metric: {selected_metric}")
                        st.write(f"- Years: {year_range[0]} to {year_range[1]} ({year_range[1] - year_range[0] + 1} years)")
                        st.write(f"- Weeks: {week_range[0]} to {week_range[1]} ({week_range[1] - week_range[0] + 1} weeks per year)")
                        st.write(f"- Total possible data points: {(year_range[1] - year_range[0] + 1) * (week_range[1] - week_range[0] + 1)}")
                        st.write(f"- Actual data points: {len(region_data)}")
                        
                        st.write("**Note:** Timeline chart always shows data in chronological order regardless of sorting options.")
                        
                        if len(region_data) < (year_range[1] - year_range[0] + 1) * (week_range[1] - week_range[0] + 1):
                            st.info("Some data points may be missing due to data availability or quality filters.")
                        
                else:
                    st.warning(f"No data available for {selected_region} in the selected time range.")
                    st.info("Try adjusting the year or week range, or select a different region.")
            
            with tab3:
                st.subheader(f"Regional Comparison: {selected_metric}")
                
                # Show sorting status in caption
                if sort_by_metric:
                    sort_direction = "Ascending" if sort_ascending_flag else "Descending"
                    sort_info = f" | Sorted by average {selected_metric} ({sort_direction})"
                else:
                    sort_info = " | Default descending order"
                
                st.caption(f"Average {selected_metric} values | Years: {year_range[0]}-{year_range[1]} | Weeks: {week_range[0]}-{week_range[1]}{sort_info}")
                
                # Filter data based on selected time ranges
                filtered_data = df[
                    (df['Year'] >= year_range[0]) &
                    (df['Year'] <= year_range[1]) &
                    (df['Week'] >= week_range[0]) &
                    (df['Week'] <= week_range[1])
                ].copy()
                
                if not filtered_data.empty:
                    # CALCULATE AVERAGE VALUES FOR EACH REGION
                    regional_averages = filtered_data.groupby('Region')[selected_metric].agg([
                        'mean', 'count', 'std', 'min', 'max'
                    ]).round(2)
                    regional_averages.columns = ['Average', 'Data_Points', 'Std_Dev', 'Min', 'Max']
                    regional_averages = regional_averages.reset_index()
                    
                    # Apply sorting based on user selection
                    if sort_by_metric:
                        regional_averages = regional_averages.sort_values('Average', ascending=sort_ascending_flag)
                    else:
                        regional_averages = regional_averages.sort_values('Average', ascending=False)
                    
                    # Create colors - highlight selected region
                    colors = ['#1f77b4' if region != selected_region else '#ff7f0e' 
                             for region in regional_averages['Region']]
                    
                    # Create the bar chart
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=regional_averages['Region'],
                        y=regional_averages['Average'],
                        marker_color=colors,
                        hovertemplate='<b>%{x}</b><br>' +
                                    f'<b>Average {selected_metric}:</b> %{{y:.2f}}<br>' +
                                    '<b>Data Points:</b> %{customdata[0]}<br>' +
                                    '<b>Std Dev:</b> %{customdata[1]:.2f}<br>' +
                                    '<b>Min:</b> %{customdata[2]:.2f}<br>' +
                                    '<b>Max:</b> %{customdata[3]:.2f}<br>' +
                                    '<extra></extra>',
                        customdata=regional_averages[['Data_Points', 'Std_Dev', 'Min', 'Max']].values,
                        name=f'Average {selected_metric}'
                    ))
                    
                    # Update title to reflect sorting
                    if sort_by_metric:
                        sort_direction = "Ascending" if sort_ascending_flag else "Descending"
                        title_suffix = f"<br><sub>Selected Region: {selected_region} (highlighted) | Sorted: {sort_direction}</sub>"
                    else:
                        title_suffix = f"<br><sub>Selected Region: {selected_region} (highlighted) | Default: Best to Worst</sub>"
                    
                    fig.update_layout(
                        title=f'Regional Comparison: Average {selected_metric} Values{title_suffix}',
                        xaxis_title='Regions',
                        yaxis_title=f'Average {selected_metric}',
                        height=600,
                        xaxis_tickangle=-45,
                        showlegend=False
                    )
                    
                    # Add reference lines if applicable
                    if selected_metric in ['VCI', 'TCI', 'VHI']:
                        fig.add_hline(y=35, line_dash="dash", line_color="red", 
                                     annotation_text="Drought threshold (35)", annotation_position="top right")
                        fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                                     annotation_text="Moderate conditions (50)", annotation_position="top right")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    

                    # Show ranking of selected region
                    selected_rank = regional_averages.reset_index().index[regional_averages['Region'] == selected_region].tolist()[0] + 1
                    selected_avg = regional_averages[regional_averages['Region'] == selected_region]['Average'].iloc[0]
                    
                    # Get best and worst based on current sorting
                    if sort_by_metric and sort_ascending_flag:
                        # When ascending, first is worst, last is best
                        best_region = regional_averages.iloc[-1]['Region']
                        best_value = regional_averages.iloc[-1]['Average']
                        worst_region = regional_averages.iloc[0]['Region']
                        worst_value = regional_averages.iloc[0]['Average']
                    else:
                        # When descending (default) or no sorting, first is best, last is worst
                        best_region = regional_averages.iloc[0]['Region']
                        best_value = regional_averages.iloc[0]['Average']
                        worst_region = regional_averages.iloc[-1]['Region']
                        worst_value = regional_averages.iloc[-1]['Average']
                    
                    # Display key statistics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric("Selected Region Rank", f"{selected_rank} / {len(regional_averages)}")
                    with col_stat2:
                        st.metric(f"{selected_region} Average", f"{selected_avg:.2f}")
                    with col_stat3:
                        st.metric("Best Region", f"{best_region}")
                    with col_stat4:
                        st.metric("Best Average", f"{best_value:.2f}")
                    
                    # Detailed comparison table
                    with st.expander("Detailed Regional Statistics"):
                        st.write(f"**Regional {selected_metric} Statistics**")
                        
                        if sort_by_metric:
                            sort_direction = "ascending" if sort_ascending_flag else "descending"
                            st.write(f"*Sorted by average {selected_metric} in {sort_direction} order*")
                        else:
                            st.write(f"*Sorted by average {selected_metric} in descending order (default)*")
                        
                        # Add rank column
                        display_data = regional_averages.copy()
                        display_data.insert(0, 'Rank', range(1, len(display_data) + 1))
                        
                        # Highlight selected region row
                        def highlight_selected(row):
                            if row['Region'] == selected_region:
                                return ['background-color: #fff2cc'] * len(row)
                            return [''] * len(row)
                        
                        styled_data = display_data.style.apply(highlight_selected, axis=1)
                        st.dataframe(styled_data, use_container_width=True, hide_index=True)
                        
                        # Additional insights
                        st.write("**Key Insights:**")
                        
                        # Performance comparison
                        overall_avg = filtered_data[selected_metric].mean()
                        if selected_avg > overall_avg:
                            performance = "above"
                            performance_emoji = "📈"
                        else:
                            performance = "below"
                            performance_emoji = "📉"
                        
                        st.write(f"{performance_emoji} **{selected_region}** performs {performance} the national average")
                        st.write(f"- {selected_region}: {selected_avg:.2f}")
                        st.write(f"- National average: {overall_avg:.2f}")
                        st.write(f"- Difference: {selected_avg - overall_avg:+.2f}")
                        
                        # Best and worst performers
                        st.write(f"🏆 **Best performer:** {best_region} ({best_value:.2f})")
                        st.write(f"🔻 **Lowest performer:** {worst_region} ({worst_value:.2f})")
                        st.write(f"📊 **Range:** {abs(best_value - worst_value):.2f} points difference")
                        
                        # Sorting status reminder
                        if sort_by_metric:
                            if sort_ascending and sort_descending:
                                st.info("🔄 **Sorting Note:** Both sorting options were selected, so descending order was applied.")
                            else:
                                sort_direction = "ascending (lowest to highest)" if sort_ascending_flag else "descending (highest to lowest)"
                                st.info(f"🔄 **Active Sorting:** {selected_metric} values in {sort_direction} order")
                        else:
                            st.info("📊 **Default Sorting:** Regions ordered by average performance (best first)")
                
                else:
                    st.warning("No data available for the selected time range.")
                    st.info("Try adjusting the year or week range.")
                
        else:
            st.error("Failed to load data. Please check the dataset files.")
    else:
        st.info("Download datasets first to begin analysis")
        st.write("Use the 'Download Datasets' button in the Control Panel to fetch data from NOAA.")