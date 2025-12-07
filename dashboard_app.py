import streamlit as st
import pandas as pd
import snowflake.connector
from snowflake.snowpark import Session
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import textwrap
import re
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components

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

@st.cache_resource
def create_snowpark_session():
    """
    Creates a Snowpark session for local development using credentials from secrets.toml.
    This is cached to avoid creating a new session on every rerun.
    """
    return Session.builder.configs(st.secrets["snowflake"]).create()

@st.cache_data
def load_data():
    """
    Loads data from Snowflake using a hybrid approach.
    - In Snowflake: Uses the automatically provided active session.
    - Locally: Creates a new session using credentials from secrets.toml.
    """
    st.sidebar.info("📊 Loading data from Snowflake...")
    
    try:
        # This will succeed when the app is running in Snowflake
        from snowflake.snowpark.context import get_active_session
        session = get_active_session()
    except Exception:
        # This will be triggered when running locally
        session = create_snowpark_session()

    query = "SELECT * FROM AVA_TEST_RESULTS;"
    # Use the Snowpark session to directly execute SQL and fetch as a Pandas DataFrame
    df = session.sql(query).to_pandas()

    # --- Data Cleaning & Type Conversion ---
    # Snowflake column names are often uppercase. We need to standardize them.
    # Create a mapping from the SQL column names to the Python-friendly names used in the app.
    column_mapping = {
        'query_en': 'query_en',
        'relevance': 'relevance',
        'intent': 'intent',
        'l0_cross_cutting': 'l0_cross_cutting',
        'l1_theme': 'l1_theme',
        'l2_sub_topic': 'l2_sub_topic',
        'l3_standardized_topic': 'l3_standardized_topic',
        'pillar_level_1': 'pillar_level_1',
        'pillar_level_2': 'pillar_level_2',
        'pillar_level_3': 'pillar_level_3',
        'region_cleaned': 'region_cleaned',
        'date': 'date',
        'user_identifier': 'user_identifier',
        'session_number_by_user': 'session_number_by_user',
        'abstention': 'abstention',
        'num_documents': 'num_documents',
        'document titles': 'document titles',
        'language': 'language'
    }
    
    # --- Robust Column Renaming ---
    # Create a case-insensitive mapping from the expected columns to the desired new names.
    # This prevents errors if the source columns in Snowflake are uppercase, lowercase, or mixed case.
    rename_map = {col: column_mapping.get(col.lower()) for col in df.columns if col.lower() in column_mapping}
    # The above line creates a dictionary like {'DATE': 'date', 'QUERY_EN': 'query_en', ...}
    # by checking the lowercase version of each actual column name against the mapping keys.
    df = df.rename(columns=rename_map)

    # Convert data types for plotting and analysis
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    
    # --- ROBUSTNESS FIX: Ensure key columns are the correct type ---
    # Convert columns that should be numeric, handling potential errors by coercing to NaN and filling with 0.
    if 'session_number_by_user' in df.columns:
        df['session_number_by_user'] = pd.to_numeric(df['session_number_by_user'], errors='coerce').fillna(0)
    if 'num_documents' in df.columns:
        df['num_documents'] = pd.to_numeric(df['num_documents'], errors='coerce').fillna(0)
    
    # --- LOGIC MIGRATION: Calculate coverage_score ---
    # This logic was previously only in preprocess_data.py. It must be here
    # to ensure the column exists when data is loaded directly from Snowflake.
    if 'abstention' in df.columns:
        df['abstention'] = df['abstention'].astype(str).str.lower() == 'true'

    def get_coverage(row):
        if row.get('abstention') or row.get('num_documents', 0) == 0: return 0.0
        flagships = ['WDR', 'Global Economic Prospects', 'Poverty Assessment', 'CEM']
        titles = str(row.get('document titles', '')).lower() # Use the renamed column
        if row.get('num_documents', 0) >= 4 or any(x.lower() in titles for x in flagships): return 2.0
        return 1.0
    
    df['coverage_score'] = df.apply(get_coverage, axis=1)

    df.dropna(subset=['date'], inplace=True)
    st.sidebar.success("✅ Data loaded successfully!")
    return df

try:
    df_raw = load_data()
except snowflake.connector.errors.DatabaseError as e:
    # This provides a more specific and helpful message for the most common connection issue.
    st.error(
        "**A Snowflake database error occurred.** This could be a permissions issue "
        "or a problem with the table/view you are trying to access. "
        f"Snowflake error: {e}"
    )
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading data. Error: {e}")
    st.stop()

# ==========================================
# 2. SIDEBAR FILTERS (Global Scope)
# ==========================================
st.sidebar.header("🔍 Global Filters")

# Handle case where the dataframe might be empty after cleaning
if df_raw.empty:
    st.warning("No valid data available for the selected time period after cleaning. Please check the data source.")
    st.stop()

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
    # --- ROBUSTNESS FIX: Filter out None values before sorting ---
    # The 'unique_regions' list can contain None if the source data has NULLs.
    # We must explicitly check `if r` to ensure we only process valid strings.
    countries_list = sorted([r for r in unique_regions if r and r not in WB_REGIONS and r not in ['All', 'Global', 'Unknown']])
    
    all_regions = ['All', 'Global'] + regions_list + countries_list
    
    if len(all_regions) > 1:
        selected_region = st.sidebar.selectbox("Country/Region", all_regions, help="Filter queries by a specific country or region.")

# C. Cross-Cutting Filter (L0)
# Convert all unique values to strings before sorting to prevent TypeError
# when the column contains mixed types (e.g., strings and NaN/floats).
unique_l0_values = [str(x) for x in df_raw['l0_cross_cutting'].unique() if pd.notna(x) and str(x) != 'None']
l0_options = ['All'] + sorted(list(set(unique_l0_values))) # Use set to ensure uniqueness after str conversion
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", 
    "🕸️ Knowledge Graph",
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
    theme_hierarchy = ['l1_theme', 'l2_sub_topic'] # Removed l3_standardized_topic for a higher-level view
        
    # Apply text wrapping to all levels of the hierarchy before aggregation
    # This helps more labels fit inside the slices.
    for col in theme_hierarchy:
        local_theme_df[col] = local_theme_df[col].apply(wrap_text)
    
    # Clean data for hierarchy
    local_theme_df[theme_hierarchy] = local_theme_df[theme_hierarchy].replace(['N/A', 'Unclassified', 'Unknown'], pd.NA)
    theme_sunburst_df = local_theme_df.dropna(subset=theme_hierarchy)
    
    fig_sun_theme = px.sunburst(
        theme_sunburst_df,
        path=theme_hierarchy,
        maxdepth=2, # Display two levels by default
        color='l1_theme', # Add color to distinguish the top themes
        labels={
            'l1_theme': 'Sector', 
            'l2_sub_topic': 'Sub-Topic'
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
with tab3:
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

# --- TAB 7: KNOWLEDGE GRAPH ---
with tab2:
    st.subheader("🕸️ Interactive Knowledge Graph")
    st.markdown("Explore the connections between Strategic Pillars. Node size represents query volume.")

    # --- Graph-specific Filters ---
    # Filter by Pillar
    graph_pillar_options = ['All'] + sorted([p for p in df_valid['pillar_level_1'].unique() if p != 'Unknown'])
    selected_graph_pillar = st.selectbox("Focus on a Pillar", graph_pillar_options, key="graph_pillar_filter")

    # Apply filters to a local copy of the data
    graph_df = df_valid.copy()
    if selected_graph_pillar != 'All':
        graph_df = graph_df[graph_df['pillar_level_1'] == selected_graph_pillar]

    # --- On-Demand Graph Generation ---
    if st.button("🚀 Generate Graph", key="generate_graph_button"):
        if not graph_df.empty:
            with st.spinner("Building Knowledge Graph... This may take a moment."):
                # Initialize pyvis network
                net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=True, cdn_resources='in_line')
                
                # Set physics options for a better layout
                net.set_options("""
                var options = {
                  "physics": {
                    "forceAtlas2Based": {
                      "gravitationalConstant": -50,
                      "centralGravity": 0.01,
                      "springLength": 230,
                      "springConstant": 0.08
                    },
                    "minVelocity": 0.75,
                    "solver": "forceAtlas2Based"
                  },
                  "interaction": {
                    "hover": true,
                    "barnesHut": {
                      "gravitationalConstant": -80000,
                      "springConstant": 0.001,
                      "springLength": 200
                    }
                  }
                }
                """)

                # --- UNIFIED DATA HANDLING (CORRECTED) ---
                # All calculations (nodes, sizes, edges, weights) are now based on the same clean,
                # filtered dataframe `clean_graph_df`. This resolves the inconsistency that was
                # preventing nodes and edges from being drawn correctly.
                clean_graph_df = graph_df.dropna(subset=['pillar_level_1', 'pillar_level_2']).copy()
                clean_graph_df = clean_graph_df[clean_graph_df['pillar_level_1'] != 'Unknown']

                p1_counts = clean_graph_df['pillar_level_1'].value_counts()
                p2_counts = clean_graph_df['pillar_level_2'].value_counts()
                p3_counts = clean_graph_df.dropna(subset=['pillar_level_3'])['pillar_level_3'].value_counts()
                edge_p1_p2_counts = clean_graph_df.groupby(['pillar_level_1', 'pillar_level_2']).size()
                edge_p2_p3_counts = clean_graph_df.dropna(subset=['pillar_level_3']).groupby(['pillar_level_2', 'pillar_level_3']).size()

                # --- Add Nodes ---
                for p1, count in p1_counts.items():
                    net.add_node(p1, label=p1, title=f"Pillar L1 | Queries: {count}", value=count, color='#007bff', group=1)
                for p2, count in p2_counts.items():
                    net.add_node(p2, label=p2, title=f"Pillar L2 | Queries: {count}", value=count, color='#28a745', group=2)
                for p3, count in p3_counts.items():
                    net.add_node(p3, label=textwrap.shorten(p3, width=30, placeholder="..."), title=f"Pillar L3: {p3} | Queries: {count}", value=count, color='#ffc107', group=3)

                # --- Add Edges ---
                # Get unique edges between L2 and L1
                for (p1, p2), weight in edge_p1_p2_counts.items(): net.add_edge(p2, p1, value=weight, title=f"Connection Strength: {weight} queries")

                # Get unique edges between L3 and L2
                for (p2, p3), weight in edge_p2_p3_counts.items():
                    net.add_edge(p3, p2, value=weight, title=f"Connection Strength: {weight} queries")


            # --- Generate and Display the Graph ---
            try:
                # Generate the HTML source directly to a string in memory.
                source_code = net.generate_html()
                
                components.html(source_code, height=800, scrolling=True)

            except Exception as e:
                st.error(f"Could not generate the graph. Error: {e}")

        else:
            st.warning("No data available for the selected filters to build a knowledge graph.")

# --- TAB 3: DEMAND VS SUPPLY (The GAP Matrix) ---
with tab4:
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
with tab5:
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
with tab6:
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
with tab7:
    st.header("📈 Trends & Momentum")

    # --- 1. EXECUTIVE SUMMARY (KPIs with Period-over-Period Change) ---
    st.markdown("#### Executive Summary")
    
    # --- New Logic: Ignore global date filters and use rolling periods from the latest data point ---
    latest_data_date = df_raw['date'].max()

    # --- Comparison Period Filter ---
    comparison_options = {
        "Last 7 Days": pd.DateOffset(days=7),
        "Last 30 Days": pd.DateOffset(days=30),
        "Last 90 Days": pd.DateOffset(days=90)
    }
    selected_comparison = st.selectbox(
        "Show KPIs for:",
        options=list(comparison_options.keys()),
        index=1, # Default to 'Last 30 Days'
        help="Compares the selected rolling period to the one immediately preceding it (e.g., Last 30 Days vs. the 30 days prior)."
    )

    # Define current and previous periods based on the selection
    offset = comparison_options[selected_comparison]
    
    # Current period (e.g., last 30 days)
    current_end_date = latest_data_date
    current_start_date = latest_data_date - offset
    
    # Previous period (e.g., the 30 days before that)
    prev_end_date = current_start_date
    prev_start_date = current_start_date - offset
    
    # Filter data for the current and previous periods from the raw, unfiltered dataframe
    df_current_period = df_raw[
        (df_raw['date'] >= current_start_date) & (df_raw['date'] < current_end_date) &
        (df_raw['relevance'] == 'Valid')
    ].copy()
    
    df_prev_period = df_raw[
        (df_raw['date'] >= prev_start_date) & 
        (df_raw['date'] < prev_end_date) &
        (df_raw['relevance'] == 'Valid')
    ].copy()
    
    # Calculate KPIs for both periods
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # KPI: Total Valid Queries
    queries_current = len(df_current_period)
    queries_prev = len(df_prev_period)
    kpi1.metric("Total Valid Queries", f"{queries_current}", f"{queries_current - queries_prev:+d}")

    # KPI: Unique Users
    users_current = df_current_period['user_identifier'].nunique() if 'user_identifier' in df_current_period else 0
    users_prev = df_prev_period['user_identifier'].nunique() if 'user_identifier' in df_prev_period else 0
    kpi2.metric("Unique Users", f"{users_current}", f"{users_current - users_prev:+d}")

    # KPI: Abstention Rate
    abstention_current = (df_current_period['abstention'].astype(str) == 'True').mean() * 100 if not df_current_period.empty else 0
    abstention_prev = (df_prev_period['abstention'].astype(str) == 'True').mean() * 100 if not df_prev_period.empty else 0
    kpi3.metric("Abstention Rate", f"{abstention_current:.1f}%", f"{abstention_current - abstention_prev:.1f} pts", delta_color="inverse")

    # KPI: Average Coverage Score
    coverage_current = df_current_period['coverage_score'].mean() if not df_current_period.empty else 0
    coverage_prev = df_prev_period['coverage_score'].mean() if not df_prev_period.empty else 0
    kpi4.metric("Avg. Coverage Score", f"{coverage_current:.2f}", f"{coverage_current - coverage_prev:+.2f}")

    st.markdown("---")

    # --- Time Granularity Selector (Applies to all charts below) ---
    granularity = st.radio("Select time granularity for charts", ('Weekly', 'Monthly'), index=0, horizontal=True)
    granularity_map = {'Weekly': 'W-Mon', 'Monthly': 'M'}
    resample_rule = granularity_map[granularity]

    # --- 2. USER GROWTH & ENGAGEMENT ---
    st.subheader("User Growth & Engagement Over Time")
    trends_df = df_valid.set_index('date')
    
    # Resample for both queries and unique users
    user_trends = trends_df.resample(resample_rule)['user_identifier'].nunique()
    query_trends = trends_df.resample(resample_rule).size()
    
    fig_user_growth = go.Figure()
    fig_user_growth.add_trace(go.Scatter(x=query_trends.index, y=query_trends.values, name='Total Queries', mode='lines+markers'))
    fig_user_growth.add_trace(go.Scatter(x=user_trends.index, y=user_trends.values, name='Unique Users', yaxis='y2', mode='lines+markers'))
    
    fig_user_growth.update_layout(
        title="Query Volume vs. Unique Users",
        yaxis=dict(title='Total Queries'),
        yaxis2=dict(title='Unique Users', overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_user_growth, use_container_width=True)

    st.markdown("---")

    # --- 3. THE SHIFTING LANDSCAPE OF DEMAND ---
    st.subheader("The Shifting Landscape of Demand")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Demand by Strategic Pillar**")
        # Pivot data to have pillars as columns
        pillar_trends = trends_df.groupby([pd.Grouper(freq=resample_rule), 'pillar_level_1'])['query_en'].count().unstack().fillna(0)
        
        # Create a stacked area chart
        fig_pillar_trends = px.area(
            pillar_trends,
            title="Proportion of Queries by Pillar Over Time",
            labels={'value': 'Number of Queries', 'pillar_level_1': 'Pillar'}
        )
        st.plotly_chart(fig_pillar_trends, use_container_width=True)

    with c2:
        st.markdown("**Top Trending Research Questions**")
        # --- Corrected Logic: Use the rolling window from the KPI filter ---
        # The 'offset' variable is already defined from the KPI section (e.g., DateOffset(days=30))
        # We calculate the duration as a Timedelta, which can be divided.
        period_duration = current_end_date - current_start_date
        mid_point = current_start_date + (period_duration / 2)
        
        # We must filter from df_raw to get the correct rolling window, ignoring global filters.
        # The period is [current_start_date, current_end_date).
        first_half_df = df_raw[(df_raw['date'] >= current_start_date) & (df_raw['date'] < mid_point) & (df_raw['relevance'] == 'Valid')]
        second_half_df = df_raw[(df_raw['date'] >= mid_point) & (df_raw['date'] < current_end_date) & (df_raw['relevance'] == 'Valid')]

        if not first_half_df.empty and not second_half_df.empty:
            first_half_counts = first_half_df['l3_standardized_topic'].value_counts()
            second_half_counts = second_half_df['l3_standardized_topic'].value_counts()
            
            # Combine into a dataframe
            trends_data = pd.DataFrame({'first_half': first_half_counts, 'second_half': second_half_counts}).fillna(0)
            trends_data = trends_data[trends_data.index != 'N/A'] # Exclude N/A
            
            # Calculate percentage change, handling division by zero
            trends_data['change'] = ((trends_data['second_half'] - trends_data['first_half']) / trends_data['first_half'].replace(0, 1)) * 100
            
            # --- NEW LOGIC as per user request ---
            # 1. Calculate the total queries for the entire period.
            trends_data['total_queries'] = trends_data['first_half'] + trends_data['second_half']
            
            # 2. Sort by the total number of queries to find the most popular topics.
            # 3. Then, show the growth (positive or negative) for these top topics.
            top_topics_by_volume = trends_data.sort_values('total_queries', ascending=False).head(10)
            
            if not top_topics_by_volume.empty:
                st.dataframe(
                    top_topics_by_volume[['total_queries', 'change']].rename(columns={
                        'total_queries': 'Total Queries', 
                        'change': '% Growth (vs prior half)'
                    }),
                    column_config={"% Growth (vs prior half)": st.column_config.NumberColumn(format="%+.0f%%")}
                )
            else:
                st.info("No significant trending topics found for the selected period.")
        else:
            st.info("Not enough data to calculate trending topics.")

    st.markdown("---")

    # --- 4. SYSTEM PERFORMANCE OVER TIME ---
    st.subheader("System Performance Over Time")
    
    # Resample for abstention rate and coverage score
    abstention_trend = trends_df.groupby(pd.Grouper(freq=resample_rule))['abstention'].apply(lambda x: (x.astype(str) == 'True').mean() * 100)
    query_trend_perf = trends_df.resample(resample_rule).size() # Re-using query_trends from above

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=abstention_trend.index, y=abstention_trend.values, name='Abstention Rate (%)', mode='lines+markers'))
    fig_perf.add_trace(go.Scatter(x=query_trend_perf.index, y=query_trend_perf.values, name='Total Queries', yaxis='y2', mode='lines+markers'))

    fig_perf.update_layout(
        title="Abstention Rate vs. Total Queries",
        yaxis=dict(title='Abstention Rate (%)'),
        yaxis2=dict(title='Total Queries', overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_perf, use_container_width=True)