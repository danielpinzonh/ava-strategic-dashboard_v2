import streamlit as st
import pandas as pd
import snowflake.connector
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import textwrap
import re

# ==========================================
# 1. CONFIG & DATA LOADING
# ==========================================
st.set_page_config(page_title="AVA Strategic Dashboard", layout="wide")

# --- PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["app"]["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password.
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# --- ROBUST SNOWFLAKE CONNECTION ---
# This function bypasses the faulty st.connection initialization and uses the
# direct connection method that we've proven works.
# @st.cache_resource ensures we create the connection only once.
@st.cache_resource
def get_snowflake_connection():
    conn = snowflake.connector.connect(
        **st.secrets["snowflake"],
        client_session_keep_alive=True
    )
    return conn

# Get the connection. This will now be reliable.
conn = get_snowflake_connection()

@st.cache_data
def load_data():
    # Query data from Snowflake. Note the quotes around "Date" and "Document titles".
    # BEST PRACTICE: Explicitly list columns instead of using SELECT *.
    # This prevents errors from column reordering and ensures you only pull the data you need.
    query = """
    SELECT
        "User_Identifier",
        "Date",
        "Document titles",
        "Abstention",
        "session_number_by_user",
        "num_documents",
        "Language",
        "Query_en",
        "relevance",
        "L0_cross_cutting",
        "pillar_level_1",
        "pillar_level_2",
        "pillar_level_3",
        "L1_theme",
        "L2_sub_topic",
        "L3_standardized_topic",
        "intent",
        "region_cleaned"
    FROM AVA_TEST_RESULTS
    """
    # Since we are using a raw snowflake connection, we use pandas to execute the query.
    # This is the equivalent of conn.query()
    cursor = conn.cursor()
    cursor.execute(query)
    df = cursor.fetch_pandas_all()
    cursor.close()

    # Snowflake column names are uppercase by default. Let's make them lowercase for consistency.
    df.columns = [col.lower() for col in df.columns]
    
    # --- PRE-PROCESSING ---
    
    # 1. Date Parsing
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Ensure 'num_documents' is numeric before use
    if 'num_documents' in df.columns:
        df['num_documents'] = pd.to_numeric(df['num_documents'], errors='coerce').fillna(0)

    # Ensure 'session_number_by_user' is numeric
    if 'session_number_by_user' in df.columns:
        df['session_number_by_user'] = pd.to_numeric(df['session_number_by_user'], errors='coerce').fillna(0)

    # Ensure 'Abstention' is a proper boolean for calculations
    if 'Abstention' in df.columns:
        df['abstention'] = df['abstention'].astype(str).str.lower() == 'true'

    # 2. Coverage Score Calculation (The Supply Logic)
    def get_coverage(row):
        # Logic: 0=Gap, 1=Shallow, 2=Strong
        if row.get('abstention') or row.get('num_documents', 0) == 0:
            return 0.0
        
        flagships = ['WDR', 'Global Economic Prospects', 'Poverty Assessment', 'CEM']
        titles = str(row.get('Document titles', '')).lower()
        
        if row.get('num_documents', 0) >= 4 or any(x.lower() in titles for x in flagships):
            return 2.0
        return 1.0

    df['coverage_score'] = df.apply(get_coverage, axis=1)
    
    # 3. Fill NaNs for safe plotting
    # --- Clean and Standardize L0_cross_cutting ---
    valid_l0 = ["FCV", "Gender", "Multiple", "Vulnerable"]
    # Replace any non-valid L0 theme with 'None'
    df['l0_cross_cutting'] = df['l0_cross_cutting'].where(df['l0_cross_cutting'].isin(valid_l0), 'None')
    df['l0_cross_cutting'] = df['l0_cross_cutting'].fillna('None') # Also handle original NaNs

    df['l1_theme'] = df['l1_theme'].fillna('Unclassified')
    
    # --- Clean and Standardize pillar_level_1 ---
    valid_pillars = ["Planet", "People", "Prosperity", "Infrastructure", "Digital"]
    # Convert to title case to handle variations like 'PLANET', 'planet', etc.
    df['pillar_level_1'] = df['pillar_level_1'].str.title()
    # Replace any non-valid pillar with 'Unknown'
    df['pillar_level_1'] = df['pillar_level_1'].where(df['pillar_level_1'].isin(valid_pillars), 'Unknown')
    
    # Make region processing robust to its absence in the data file
    if 'region_cleaned' in df.columns:
        df['region_cleaned'] = df['region_cleaned'].fillna('Unknown').astype(str)
        
        # --- Standardize Region Names to WB Classification ---
        region_map = {
            'EAS': 'Africa Eastern and Southern',
            'AFR': 'Africa Western and Central',
            'Africa': 'Africa Western and Central', # Assuming general 'Africa' maps here
            'EAP': 'East Asia and Pacific',
            'ECA': 'Europe and Central Asia',
            'LCR': 'Latin America and Caribbean',
            'MNA': 'Middle East and North Africa',
            'SAR': 'South Asia'
            # Countries like DRC, IDN, UAE will remain unchanged
        }
        df['region_cleaned'] = df['region_cleaned'].replace(region_map)

    else:
        df['region_cleaned'] = 'Unknown' # Create a dummy column if it doesn't exist
    df['l2_sub_topic'] = df['l2_sub_topic'].fillna('N/A')
    df['l3_standardized_topic'] = df['l3_standardized_topic'].fillna('N/A')

    # --- Standardize Intent Labels for better readability ---
    intent_map = {
        'Learn': 'Learn about Concepts',
        'Design': 'Policy Design',
        'Drafting': 'Draft Product',
        'AVA': 'About AVA'
    }
    if 'intent' in df.columns:
        df['intent'] = df['intent'].replace(intent_map)

    # --- Standardize thematic and pillar columns to merge variations ---
    def standardize_text_column(series):
        if series.dtype == 'object':
            def clean_and_rebuild(text):
                if not isinstance(text, str):
                    return text
                
                # 1. Replace all separators ('&', ' and ') with a comma
                text = text.replace('&', ',').replace(' and ', ',')
                
                # 2. Split by comma, strip whitespace, filter out empty parts, and join back
                parts = [part.strip() for part in text.split(',') if part.strip()]
                return ', '.join(parts)

            return series.apply(clean_and_rebuild)
        return series # Return series as-is if not an object type

    for col in ['pillar_level_1', 'pillar_level_2', 'pillar_level_3', 'l1_theme', 'l2_sub_topic', 'l3_standardized_topic']:
        if col in df.columns:
            df[col] = standardize_text_column(df[col])
    
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Failed to load data from Snowflake. Please check your connection and credentials. Error: {e}")
    st.stop()

# ==========================================
# 2. SIDEBAR FILTERS (Global Scope)
# ==========================================
st.sidebar.header("🔍 Global Filters")

# A. Date Range
min_date = df_raw['date'].min()
max_date = df_raw['date'].max()
start_date, end_date = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# B. Region Filter
selected_region = 'All' # Default value
if 'region_cleaned' in df_raw.columns and df_raw['region_cleaned'].nunique() > 1:
    # Define known World Bank regions to prioritize them in the filter list
    WB_REGIONS = [
        "Africa Eastern and Southern", "Africa Western and Central", 
        "East Asia and Pacific", "Europe and Central Asia", 
        "Latin America and Caribbean", "Middle East and North Africa", 
        "South Asia", "North America"
    ]
    
    unique_regions = df_raw['region_cleaned'].unique()
    
    # Separate regions from countries
    regions_list = sorted([r for r in unique_regions if r in WB_REGIONS])
    countries_list = sorted([r for r in unique_regions if r not in WB_REGIONS and r not in ['All', 'Global', 'Unknown']])
    
    all_regions = ['All', 'Global'] + regions_list + countries_list
    
    if len(all_regions) > 1:
        selected_region = st.sidebar.selectbox("Country/Region", all_regions, help="Filter queries by a specific country or region.")

# C. Cross-Cutting Filter (L0)
l0_options = ['All'] + sorted([x for x in df_raw['l0_cross_cutting'].unique() if x != 'None'])
selected_l0 = st.sidebar.selectbox("Cross-Cutting Lens (L0)", l0_options)

# D. Abstention Filter
abstention_filter = st.sidebar.radio("Query Status", ["All Queries", "Only Unanswered (Abstentions)"])

# --- State Management for Interactive Charts ---
# This is a critical step. If any global filter changes, we must reset the drill-down
# state of the sunburst chart to avoid inconsistent views.
current_filters = (start_date, end_date, selected_region, selected_l0, abstention_filter)
if 'last_filters' not in st.session_state:
    st.session_state.last_filters = current_filters

if st.session_state.last_filters != current_filters:
    if 'sunburst_path' in st.session_state:
        st.session_state.sunburst_path = [] # Reset the path
    st.session_state.last_filters = current_filters


# --- APPLY FILTERS ---
# Convert filter dates to timezone-aware (UTC) to match the DataFrame's 'Date' column
start_date_utc = pd.to_datetime(start_date).tz_localize('UTC')
end_date_utc = pd.to_datetime(end_date).tz_localize('UTC')

mask = (df_raw['date'] >= start_date_utc) & (df_raw['date'] <= end_date_utc)

if selected_region != 'All':
    mask = mask & (df_raw['region_cleaned'] == selected_region)

if selected_l0 != 'All':
    mask = mask & (df_raw['l0_cross_cutting'] == selected_l0)

if abstention_filter == "Only Unanswered (Abstentions)":
    mask = mask & (df_raw['abstention'].astype(str) == 'True')

df = df_raw[mask].copy()
df_valid = df[df['relevance'] == 'Valid'].copy() # Most charts only use Valid queries

# ==========================================
# 3. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "🎯 Intent & Cross-Cutting",
    "🎯 Demand vs. Supply (Product)", 
    "🔎 Knowledge Explorer (Research)",
    "⚙️ Abstention",
    "📈 Trends"
])

def create_interactive_portfolio_view(data):
    """
    Creates the sunburst and interactive donut chart in an isolated function
    to ensure stable state management.
    """
    st.markdown("#### Queries by World Bank's Focus Areas")
    # Create a local dataframe for this chart.
    local_df = data[
        data['pillar_level_1'].notna() & (data['pillar_level_1'] != 'Unknown')
    ].copy()

    hierarchy_cols = ['pillar_level_1', 'pillar_level_2', 'pillar_level_3']
    # CRITICAL FIX: Remove inplace=True and assign to a new variable to prevent modifying the original df_valid.
    sunburst_df = local_df.dropna(subset=hierarchy_cols)
    
    fig_sun = px.sunburst(
        sunburst_df,
        path=hierarchy_cols,
        maxdepth=2,
        color='pillar_level_1',
        color_discrete_map={
            'People': '#E67E22', 'Planet': '#27AE60', 'Prosperity': '#2980B9', 
            'Infrastructure': '#8E44AD', 'Digital': '#C0392B'
        },
        custom_data=hierarchy_cols,
        labels={
            'pillar_level_1': 'Focus Area',
            'pillar_level_2': 'Pillar Level 2',
            'pillar_level_3': 'Pillar Level 3'
        }
    )
    # --- Readability Improvement ---
    # 1. Force horizontal text for consistency and readability.
    # 2. Add percentage of parent to give context to each slice.
    fig_sun.update_traces(
        textinfo='label', # Show labels inside slices where they fit
        insidetextorientation='horizontal',
        textfont_size=12, # Set a consistent base font size
        # Keep the rich, readable hover label for all slices
        hovertemplate='<b>%{label}</b><br>Query Count: %{value}<extra></extra>'
    )
    # Remove uniformtext to re-enable the smooth drill-down animation.
    # The animation is a key part of the user experience.
    fig_sun.update_layout(
        height=600, 
        margin=dict(t=20, l=20, r=20, b=20)
    )
    st.plotly_chart(fig_sun, use_container_width=True, key="sunburst_chart")

# --- TAB 1: STRATEGIC OVERVIEW ---
with tab1:
    st.metric("Total Valid Queries", f"{len(df_valid)}")

    st.markdown("---")
    
    # Call the function to create the Pillar-based sunburst
    create_interactive_portfolio_view(df_valid)
    
    st.markdown("---")
    st.markdown("#### Queries by Research Topic")
    
    # --- New Sunburst by Theme ---
    # Replicate the exact, working logic from the first sunburst chart.

    # --- Readability Improvement: Text Wrapping ---
    def wrap_text(text, width=20):
        """Wraps text for sunburst labels."""
        if not isinstance(text, str): return text
        return "<br>".join(textwrap.wrap(text, width=width))

    local_theme_df = df_valid.copy()
    theme_hierarchy = ['l1_theme', 'l2_sub_topic', 'l3_standardized_topic']
    
    # --- Readability Improvement: Aggregate the long tail ---
    # 1. Identify the top N themes. Let's use a threshold of 9.
    TOP_N_THEMES = 9
    top_themes = local_theme_df['l1_theme'].value_counts().nlargest(TOP_N_THEMES).index
    
    # 2. Group everything else into an 'Other Themes' category.
    local_theme_df['l1_theme'] = local_theme_df['l1_theme'].where(local_theme_df['l1_theme'].isin(top_themes), 'Other Themes')
    
    # Apply text wrapping to all levels of the hierarchy before aggregation
    # This helps more labels fit inside the slices.
    for col in theme_hierarchy:
        local_theme_df[col] = local_theme_df[col].apply(wrap_text)
    
    # 3. Do the same for l2_sub_topic, but within each l1_theme group.
    TOP_N_SUBTOPICS = 9
    # Get the list of top sub-topics for each theme
    top_subtopics = local_theme_df.groupby('l1_theme')['l2_sub_topic'].value_counts().groupby(level=0, group_keys=False).nlargest(TOP_N_SUBTOPICS).index
    # Keep only the sub-topics that are in the top list for their respective theme
    local_theme_df['l2_sub_topic'] = local_theme_df.apply(lambda row: row['l2_sub_topic'] if (row['l1_theme'], row['l2_sub_topic']) in top_subtopics else 'Other Sub-Topics', axis=1)
    
    # 4. Do the same for l3_standardized_topic, within each L1/L2 group.
    TOP_N_L3 = 9
    # Get the list of top L3 topics for each L1/L2 pair
    top_l3_topics = local_theme_df.groupby(['l1_theme', 'l2_sub_topic'])['l3_standardized_topic'].value_counts().groupby(level=[0, 1], group_keys=False).nlargest(TOP_N_L3).index
    # Keep only the L3 topics that are in the top list for their respective L1/L2 pair
    local_theme_df['l3_standardized_topic'] = local_theme_df.apply(lambda row: row['l3_standardized_topic'] if (row['l1_theme'], row['l2_sub_topic'], row['l3_standardized_topic']) in top_l3_topics else 'Other Topics', axis=1)

    # 5. Clean data for hierarchy (this logic remains the same)
    local_theme_df[theme_hierarchy] = local_theme_df[theme_hierarchy].replace(['N/A', 'Unclassified', 'Unknown'], pd.NA)
    theme_sunburst_df = local_theme_df.dropna(subset=theme_hierarchy)
    
    fig_sun_theme = px.sunburst(
        theme_sunburst_df,
        path=theme_hierarchy,
        maxdepth=2, # Display two levels by default
        color='l1_theme', # Add color to distinguish the top themes
        labels={
            'l1_theme': 'Sector',
            'l2_sub_topic': 'Sub-Topic',
            'l3_standardized_topic': 'Research Question'
        }
    )
    # --- Readability Improvement ---
    # 1. Force horizontal text for consistency and readability.
    # 2. Add percentage of parent to give context to each slice.
    fig_sun_theme.update_traces(
        textinfo='label', # Show labels inside slices where they fit
        insidetextorientation='horizontal',
        textfont_size=12, # Set a consistent base font size
        # Keep the rich, readable hover label for all slices
        hovertemplate='<b>%{label}</b><br>Query Count: %{value}<extra></extra>'
    )
    # Remove uniformtext to re-enable the smooth drill-down animation.
    # The animation is a key part of the user experience.
    fig_sun_theme.update_layout(
        height=600, 
        margin=dict(t=20, l=20, r=20, b=20)
    )
    st.plotly_chart(fig_sun_theme, use_container_width=True, key="sunburst_theme_chart")

    st.markdown("---")
    st.markdown("#### User Engagement")
    
    if 'session_number_by_user' in df_valid.columns:
        if 'user_identifier' in df_valid.columns:
            c1, c2 = st.columns([1, 2])
            with c1:
                # Calculate user-centric metrics
                total_unique_users = df_valid['user_identifier'].nunique()
                returning_user_ids = df_valid[df_valid['session_number_by_user'] > 1]['user_identifier'].unique()
                num_returning_users = len(returning_user_ids)
                
                if total_unique_users > 0:
                    percent_returning_users = (num_returning_users / total_unique_users) * 100
                    st.metric("% Returning Users", f"{percent_returning_users:.1f}%", help="Percentage of unique users who had more than one session.")

                queries_from_returning_users = df_valid[df_valid['session_number_by_user'] > 1].shape[0]
                if len(df_valid) > 0:
                    percent_queries_returning = (queries_from_returning_users / len(df_valid)) * 100
                    st.metric("% Queries from Returning Users", f"{percent_queries_returning:.1f}%", help="Percentage of all queries that came from a returning user.")

            with c2:
                # Count unique users for each session number
                users_per_session = df_valid.groupby('session_number_by_user')['user_identifier'].nunique()

                # Group sessions 10 and over into a '10+' category for cleaner visualization
                if len(users_per_session) > 10:
                    over_10_sum = users_per_session[users_per_session.index >= 10].sum()
                    users_per_session_plot = users_per_session[users_per_session.index < 10].copy()
                    users_per_session_plot.loc['10+'] = over_10_sum
                else:
                    users_per_session_plot = users_per_session.copy()
                
                users_per_session_plot = users_per_session_plot.sort_index(key=lambda x: pd.to_numeric(x.astype(str).str.replace('+', ''), errors='coerce'))
                
                fig_sessions = px.bar(
                    users_per_session_plot,
                    x=users_per_session_plot.values,
                    y=users_per_session_plot.index.astype(str), # Ensure '10+' is treated as a string
                    text_auto=True, # Add data labels to each bar
                    orientation='h',
                    title="User Count by Session Number",
                    labels={'x': 'Number of Unique Users', 'y': 'Session Number'}
                )
                # Invert y-axis and force it to be categorical to show all labels
                fig_sessions.update_yaxes(autorange="reversed", type='category')
                st.plotly_chart(fig_sessions, use_container_width=True)
        else:
            st.warning("`user_identifier` column not found. Cannot calculate user-centric engagement metrics.")
            fig_sessions = px.bar(
                title="Query Distribution by Session Number")

    st.markdown("---")
    st.markdown("#### Language Distribution")

    if 'language' in df_valid.columns and not df_valid['language'].empty:
        lang_map = {
            'en': 'English', 'fr': 'French', 'es': 'Spanish',
            'pt': 'Portuguese', 'ar': 'Arabic', 'ru': 'Russian',
            'zh': 'Chinese', 'de': 'German', 'ro': 'Romanian', 'da': 'Danish', 
            'nl': 'Dutch', 'tl': 'Tagalog', 'so': 'Somali', 'hi': 'Hindi', 
            'vi': 'Vietnamese', 'cy': 'Welsh', 'zh-tw': 'Chinese (Traditional)',
            'et': 'Estonian', 'lt': 'Lithuanian', 'it': 'Italian', 'ca': 'Catalan',
            'af': 'Afrikaans'
        }
        # Clean data, make mapping case-insensitive, and calculate percentages
        lang_counts = df_valid['language'].str.lower().str.strip().value_counts()
        lang_percentages = (lang_counts / lang_counts.sum()) * 100
        
        # Aggregate languages outside the top 10 into an 'Other' category
        if len(lang_percentages) > 10:
            plot_data = lang_percentages.head(10).copy()
            other_percentage = lang_percentages.tail(-10).sum()
            plot_data.loc['other'] = other_percentage
        else:
            plot_data = lang_percentages.copy()
        
        # Map index to full names, falling back to the original code if not in map
        plot_data.index = plot_data.index.map(lambda x: lang_map.get(x, 'Other languages' if x == 'other' else x.upper()))
        
        # Sort for horizontal bar chart
        plot_data = plot_data.sort_values(ascending=True)

        fig_lang_bar = px.bar(
            plot_data, 
            x=plot_data.values, 
            y=plot_data.index,
            orientation='h',
            labels={'x': 'Percentage of Queries (%)', 'y': 'Language'}
        )
        fig_lang_bar.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
        fig_lang_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_lang_bar, use_container_width=True)

# --- TAB 2: INTENT & CROSS-CUTTING ANALYSIS ---
with tab2:
    st.subheader("Intent and Cross-Cutting Theme Analysis")

    # --- Tab-specific filter for Pillar ---
    pillar_options = ['All'] + sorted(df_valid['pillar_level_1'].unique())
    selected_pillar = st.selectbox("Filter by Focus Area", pillar_options, key="pillar_filter_tab5")

    # --- Apply tab-specific filter ---
    tab2_data = df_valid.copy()
    if selected_pillar != 'All':
        tab2_data = tab2_data[tab2_data['pillar_level_1'] == selected_pillar]

    if tab2_data.empty:
        st.warning("No data available for the selected pillar. Please adjust the filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**User Intent Breakdown**")
            intent_counts = tab2_data['intent'].value_counts().reset_index()
            intent_counts.columns = ['Intent', 'Count']
            fig_donut = px.pie(
                intent_counts, values='Count', names='Intent', hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu,
                title=f"Intents for '{selected_pillar}' Pillar"
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

        with col2:
            st.markdown("**Top Cross-Cutting Themes**")
            l0_counts = tab2_data[tab2_data['l0_cross_cutting'] != 'None']['l0_cross_cutting'].value_counts()
            fig_bar_l0 = px.bar(
                l0_counts, title=f"Cross-Cutting Themes for '{selected_pillar}' Pillar", text_auto=True,
                labels={'index': 'Cross-Cutting Theme', 'value': 'Query Count'}
            )
            fig_bar_l0.update_layout(showlegend=False, xaxis_title=None) # Remove redundant x-axis title
            st.plotly_chart(fig_bar_l0, use_container_width=True, config={'displayModeBar': False})

# --- TAB 3: DEMAND VS SUPPLY (The GAP Matrix) ---
with tab3:
    st.subheader("Demand vs. Supply: Identify Content Gaps")
    st.markdown("""
    *Each bubble is a Research Question. Size = Volume of Queries.*
    
    **Avg Coverage**: A score from 0 (no answer) to 2 (strong answer) indicating how well AVA can answer a given question.
    """)
    
    # Aggregating Data by L3 Topic
    bubble_data = df_valid.groupby('l3_standardized_topic').agg({
        'query_en': 'count',                 # Volume (X-axis)
        'coverage_score': 'mean',            # Quality (Y-axis)
        'pillar_level_1': 'first',           # Color
        'l1_theme': 'first'                  # Hover info
    }).reset_index()
    
    bubble_data = bubble_data.rename(columns={
        'query_en': 'Demand Volume', 
        'coverage_score': 'Avg Coverage',
        'l3_standardized_topic': 'Research Question' # Rename for plotting
    })
    
    # Filter out noise (single queries) for cleaner chart
    bubble_data = bubble_data[(bubble_data['Demand Volume'] > 1) & (bubble_data['pillar_level_1'] != 'Unknown')]


    fig_bubble = px.scatter(
        bubble_data,
        x="Demand Volume",
        y="Avg Coverage",
        size="Demand Volume",
        color="pillar_level_1",
        hover_name="Research Question",
        hover_data=["l1_theme"],
        size_max=20, # Further reduce the maximum bubble size for better readability
        title="Gap Analysis Matrix (Hover for details)",
        labels={
            "pillar_level_1": "Focus Area",
            "l1_theme": "Sector"
        }
    )
    
    # Add Quadrant Lines
    fig_bubble.update_layout(coloraxis_colorbar=dict(title="Focus Area"), yaxis_range=[-0.2, 2.2])

    fig_bubble.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Adequate Coverage")
    fig_bubble.add_vline(x=bubble_data['Demand Volume'].median(), line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig_bubble, use_container_width=True, config={'displayModeBar': False})
    
    # The "Red Alert" Table
    st.subheader("🚨 Critical Gaps (High Demand, Low Coverage)")
    critical_gaps = bubble_data[
        (bubble_data['Avg Coverage'] < 0.8) & 
        (bubble_data['Demand Volume'] > bubble_data['Demand Volume'].median())
    ].sort_values('Demand Volume', ascending=False)
    
    # Rename columns for the final table display
    display_gaps = critical_gaps.rename(columns={'pillar_level_1': 'Focus Areas'})
    
    st.dataframe(
        display_gaps[['Research Question', 'Demand Volume', 'Avg Coverage', 'Focus Areas']],
        column_config={
            "Avg Coverage": st.column_config.NumberColumn(format="%.2f")
        }
    )

    st.markdown("---")
    st.subheader("Geographic Demand Distribution")
    
    # Filter out regions for the map, keeping only countries
    country_data = df_valid[~df_valid['region_cleaned'].isin(WB_REGIONS + ['Global', 'Unknown'])]
    map_data = country_data['region_cleaned'].value_counts().reset_index()
    map_data.columns = ['country', 'query_count']

    fig_map = px.choropleth(
        map_data,
        locations="country",
        locationmode='country names',
        color="query_count",
        hover_name="country",
        color_continuous_scale=px.colors.sequential.Plasma,
        title="Query Volume by Country"
    )
    # Set dark theme for the map
    fig_map.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    st.subheader("Critical Gaps by Country")
    
    # Pre-calculate which countries have gaps to show in the dropdown
    country_gap_data = country_data.groupby(['region_cleaned', 'l3_standardized_topic']).agg(
        Demand_Volume=('query_en', 'count'),
        Avg_Coverage=('coverage_score', 'mean')
    ).reset_index()

    countries_with_gaps = country_gap_data[
        (country_gap_data['Avg_Coverage'] < 0.8) & (country_gap_data['Demand_Volume'] > 1)
    ]['region_cleaned'].unique()

    if len(countries_with_gaps) > 0:
        country_options = sorted(countries_with_gaps)
        selected_country_tab3 = st.selectbox("Select a Country to see its specific gaps", country_options)

        if selected_country_tab3:
            # Filter the pre-calculated gaps for the selected country
            country_critical_gaps = country_gap_data[
                (country_gap_data['region_cleaned'] == selected_country_tab3) &
                (country_gap_data['Avg_Coverage'] < 0.8) & 
                (country_gap_data['Demand_Volume'] > 1)
            ].sort_values('Demand_Volume', ascending=False)

            st.dataframe(
                country_critical_gaps[['l3_standardized_topic', 'Demand_Volume', 'Avg_Coverage']].rename(columns={'l3_standardized_topic': 'Research Question', 'Demand_Volume': 'Demand Volume', 'Avg_Coverage': 'Avg Coverage'}),
                column_config={"Avg Coverage": st.column_config.NumberColumn(format="%.2f")}
            )
    else:
        st.info("No countries with critical gaps found for the current filter selection.")

# --- TAB 4: KNOWLEDGE EXPLORER ---
with tab4:
    st.subheader("Drill Down by Theme")
    
    # Filter by L1 Theme
    unique_l1_themes = df_valid['l1_theme'].unique()
    other_themes = sorted([t for t in unique_l1_themes if t not in ['Unclassified', 'N/A']])
    l1_options = ['All']
    if 'Unclassified' in unique_l1_themes:
        l1_options.append('Unclassified')
    l1_options.extend(other_themes)
    selected_l1 = st.selectbox("Select a Sector (L1 Theme):", l1_options)
    
    # Reset selected question if the L1 theme changes
    if 'selected_l1' not in st.session_state or st.session_state.selected_l1 != selected_l1:
        st.session_state.selected_l1 = selected_l1
        if 'selected_l3' in st.session_state:
            del st.session_state['selected_l3']
    
    # Apply the filter, or use the whole dataframe if 'All' is selected
    subset_l1 = df_valid if selected_l1 == 'All' else df_valid[df_valid['l1_theme'] == selected_l1]
    
    # Bar Chart of Top L3 Questions
    # Exclude 'N/A' from the research questions chart
    top_l3 = subset_l1[subset_l1['l3_standardized_topic'] != 'N/A']['l3_standardized_topic'].value_counts().head(15).sort_values(ascending=True)
    
    col3, col4 = st.columns([2, 1])
    
    with col3:
        fig_bar = px.bar(
            x=top_l3.values,
            y=top_l3.index,
            orientation='h',
            labels={'x': 'Query Count', 'y': 'Research Question'},
            title=f"Top Questions in {selected_l1}"
        )
        # Use Streamlit's built-in selection capability
        fig_bar.update_traces(selector=dict(type='bar'), unselected={'marker': {'opacity': 0.7}})
        fig_bar.update_layout(dragmode='select')

        selection = st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False}, on_select="rerun", key="l3_selection")

        # Check if a point was selected
        if selection.selection and selection.selection['points']:
            st.session_state.selected_l3 = selection.selection['points'][0]['y']
        
    with col4:
        st.markdown(f"**{selected_l1} Stats**")
        st.metric("Total Queries", len(subset_l1))
        st.metric("Avg Coverage", f"{subset_l1['coverage_score'].mean():.2f}/2.0")
        st.metric("Abstention Rate", f"{(subset_l1['abstention'].astype(str)=='True').mean()*100:.1f}%")

    # Detail View
    st.markdown("---")
    st.markdown("### 📄 Latest Query Details", help="Click a bar in the chart above to filter this table by a specific Research Question.")

    # Filter the detail view based on the selection from the chart
    detail_df = subset_l1
    if 'selected_l3' in st.session_state and st.session_state.selected_l3:
        st.info(f"Showing queries for: **{st.session_state.selected_l3}**")
        detail_df = subset_l1[subset_l1['l3_standardized_topic'] == st.session_state.selected_l3]
        
        # Add a button to clear the selection
        if st.button("Reset and Show All Queries for this Sector"):
            del st.session_state.selected_l3
            st.rerun()


    # Display columns that exist in the dataframe
    display_cols = ['date', 'query_en', 'l3_standardized_topic', 'intent', 'coverage_score']
    
    # Define user-friendly column names
    column_rename_map = {
        'query_en': 'Query',
        'l3_standardized_topic': 'Research Question',
        'intent': 'Intent',
        'coverage_score': 'Coverage Score',
        'region_cleaned': 'Region'
    }

    if 'region_cleaned' in detail_df.columns:
        display_cols.insert(4, 'region_cleaned') # Add region if it exists

    st.dataframe(detail_df[display_cols].rename(columns=column_rename_map).sort_values('date', ascending=False).head(50), use_container_width=True)

    st.markdown("---")
    st.subheader("Knowledge Supply Metrics")
    
    c1, c2 = st.columns(2)
    with c1:
        avg_docs = df_valid['num_documents'].mean()
        st.metric("Average Documents Cited per Query", f"{avg_docs:.2f}")
    
    with c2:
        # Use a bar chart for discrete counts instead of a histogram
        doc_counts = df_valid['num_documents'].value_counts().sort_index()
        fig_docs_hist = px.bar(
            doc_counts,
            x=doc_counts.index,
            y=doc_counts.values,
            title="Distribution of Documents Cited per Query",
            labels={'x': 'Number of Documents Cited', 'y': 'Number of Queries'}
        )
        st.plotly_chart(fig_docs_hist, use_container_width=True, config={'displayModeBar': False})


    st.markdown("---")
    st.subheader("Top Cited Documents by Sector")

    # This function will parse the 'Document titles' column
    @st.cache_data
    def get_top_documents_by_theme(data, theme):
        if theme != 'All':
            theme_df = data[data['l1_theme'] == theme].copy()
        else:
            theme_df = data.copy()
        
        theme_df = theme_df.dropna(subset=['document titles'])

        # New robust parsing logic
        def parse_titles(text):
            # Split by a pattern that looks like "1. ", "2. ", etc.
            # This handles both newline-separated and space-separated lists.
            titles = re.split(r'\s*\d+\.\s+', str(text)) # Use re.split for robustness
            
            cleaned_titles = []
            for title in titles:
                if not isinstance(title, str) or not title.strip():
                    continue
                # --- Aggressive, multi-step cleaning ---
                cleaned = title.replace('\\n', ' ').replace('\n', ' ') # Handle literal and standard newlines
                cleaned = re.sub(r'\s+', ' ', cleaned).strip() # Collapse all whitespace
                cleaned_titles.append(cleaned)
            return cleaned_titles

        # Apply the parsing function and explode the resulting lists into separate rows
        all_titles = theme_df['document titles'].apply(parse_titles).explode()

        # Count occurrences and get the top 10
        top_10_docs = all_titles.value_counts().head(10)
        return top_10_docs

    # Create a separate selector for this chart that includes 'All'
    all_themes = sorted(df_valid['l1_theme'].unique())
    # Ensure 'Unclassified' is an option if it exists
    doc_theme_options = ['All'] + ([t for t in all_themes if t == 'Unclassified']) + [t for t in all_themes if t != 'Unclassified']
    selected_doc_theme = st.selectbox("Filter by Sector", doc_theme_options, key="doc_theme_selector")

    if selected_doc_theme:
        top_docs = get_top_documents_by_theme(df_valid, selected_doc_theme)
        if not top_docs.empty:
            # --- DIAGNOSTIC STEP ---
            # Check for any remaining newlines before plotting and display a warning if found.
            problematic_titles = [title for title in top_docs.index if '\n' in title or '\\n' in title]
            if problematic_titles:
                st.warning("DEBUG: The following titles still contain newline characters after cleaning:")
                st.code("\n".join(problematic_titles))

            # Prepare data for the chart: full titles for hover, truncated for display
            plot_df = top_docs.reset_index()
            plot_df.columns = ['full_title', 'count']
            plot_df['display_title'] = plot_df['full_title'].apply(lambda x: (x[:70] + '...') if len(x) > 73 else x)
            plot_df = plot_df.sort_values('count', ascending=True)

            fig_top_docs = px.bar(
                plot_df,
                x='count',
                y='display_title',
                custom_data=['full_title'], # Pass full titles for hover
                orientation='h',
                text_auto=True,
                title=f"Top 10 Cited Documents for '{selected_doc_theme}'",
                labels={'count': 'Citation Count', 'display_title': 'Document Title'}
            )
            # Add full title to hover
            fig_top_docs.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>Citations: %{x}<extra></extra>')

            st.plotly_chart(fig_top_docs, use_container_width=True)
        else:
            st.info(f"No documents cited for the '{selected_doc_theme}' sector in the current selection.")


# --- TAB 5: SYSTEM HEALTH ---
with tab5:
    st.subheader("System Diagnostics")
    
    # --- Tab-specific filter for L1 Theme for Abstention analysis ---
    # This filter will apply to the second column and the log table below.
    theme_options = ['All'] + sorted([t for t in df_valid['l1_theme'].unique() if t not in ['Unclassified', 'N/A']])
    selected_theme_tab5 = st.selectbox("Filter Abstention Data by Sector", theme_options, key="theme_filter_tab5")

    # --- Apply tab-specific filter to a copy of the valid data ---
    tab5_data = df_valid.copy()
    if selected_theme_tab5 != 'All':
        tab5_data = tab5_data[tab5_data['l1_theme'] == selected_theme_tab5]

    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Out-of-Scope (OOS) Query Analysis**")
        rel_counts = df['relevance'].value_counts()
        
        # 1. Calculate and display the percentage of OOS vs. Valid
        valid_count = rel_counts.get('Valid', 0)
        # Filter to only include actual OOS categories for the count
        oos_count = rel_counts[rel_counts.index.str.startswith('OOS', na=False)].sum()
        
        if valid_count > 0:
            oos_percentage = (oos_count / valid_count) * 100
            st.metric("OOS vs. Valid Queries", f"{oos_percentage:.1f}%", help="Volume of out-of-scope queries as a percentage of valid queries.")
        
        # 2. Filter the graph to show only categories starting with 'OOS'
        oos_counts = rel_counts[rel_counts.index.str.startswith('OOS', na=False)]
        
        # 3. Clean up the labels for display by removing the 'OOS-' prefix
        oos_counts.index = oos_counts.index.str.replace('OOS-', '', regex=False)
        
        # 3. Create the bar chart with relevant labels
        fig_rel = px.bar(oos_counts, title="Breakdown by Rejection Reason", text_auto=True, labels={'index': 'Rejection Reason', 'value': 'Number of Queries'})
        fig_rel.update_layout(showlegend=False)
        st.plotly_chart(fig_rel, use_container_width=True, config={'displayModeBar': False})

    with c2:
        st.markdown("**Abstention Analysis (Valid Queries Only)**")
        
        # Use the filtered data (tab5_data) for this chart
        abstention_counts = tab5_data['abstention'].astype(str).value_counts()
        
        # Combine 'False' and 'nan' into 'Success'
        success_count = abstention_counts.get('False', 0) + abstention_counts.get('nan', 0)
        abstention_count = abstention_counts.get('True', 0)
        
        # Create a new dataframe for the pie chart with clear labels
        pie_data = pd.DataFrame({'Category': ['Success', 'Abstention'], 'Count': [success_count, abstention_count]})
        
        fig_abs = px.pie(pie_data, names='Category', values='Count', title="Success vs. Abstention Rate", hole=0.4)
        st.plotly_chart(fig_abs, use_container_width=True, config={'displayModeBar': False})

    st.markdown("### 📉 Log of Failed Queries (Abstentions)")
    # Use the filtered data (tab5_data) for the log
    failed_queries_df = tab5_data[tab5_data['abstention'].astype(str) == 'True']
    
    # Rename columns for display
    failed_queries_display = failed_queries_df[['date', 'query_en', 'l1_theme', 'l3_standardized_topic']].rename(columns={
        'query_en': 'Query',
        'l1_theme': 'Sector',
        'l3_standardized_topic': 'Research Question'
    })
    
    st.dataframe(failed_queries_display)

# --- TAB 6: TRENDS ---
with tab6:
    st.subheader("Query Volume Over Time")

    # Time granularity selector
    granularity = st.radio("Select time granularity", ('Daily', 'Weekly', 'Monthly'), horizontal=True)
    granularity_map = {'Daily': 'D', 'Weekly': 'W-Mon', 'Monthly': 'M'}
    
    # Breakdown selector
    breakdown_options = ['None'] + sorted([t for t in df_valid['l1_theme'].unique() if t not in ['Unclassified', 'N/A']])
    breakdown_by = st.selectbox("Breakdown by Sector", breakdown_options)

    # Resample data
    trends_df = df_valid.set_index('date')
    
    if breakdown_by == 'None':
        time_series = trends_df.resample(granularity_map[granularity])['query_en'].count().reset_index()
        fig_trends = px.line(time_series, x='date', y='query_en', title=f"{granularity} Query Volume")
        fig_trends.update_layout(yaxis_title="Number of Queries")
    else:
        # Pivot data to have themes as columns
        time_series_grouped = trends_df.groupby([pd.Grouper(freq=granularity_map[granularity]), 'l1_theme'])['query_en'].count().unstack().fillna(0)
        
        # Plot only the selected theme
        if breakdown_by in time_series_grouped.columns:
            fig_trends = px.line(time_series_grouped, x=time_series_grouped.index, y=breakdown_by, title=f"{granularity} Query Volume for {breakdown_by}", labels={'date': 'Date'})
            fig_trends.update_layout(yaxis_title="Number of Queries")

    st.plotly_chart(fig_trends, use_container_width=True)