import streamlit as st
import pandas as pd
import plotly.express as px
import ast

# Set page configuration
st.set_page_config(
    page_title="Topic Exploration",
    layout="wide",
    page_icon="🔍"
)

@st.cache_data
def load_data(file):
    """Load CSV file and return a DataFrame."""
    df = pd.read_csv(file)
    # Ensure 'multiple_topics' column is parsed as a list
    if "multiple_topics" in df.columns:
        df["multiple_topics"] = df["multiple_topics"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    return df

# Default file path
default_file_path = "./src/outputs/topics/df_topic.csv"


# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Use the uploaded file
    df = load_data(uploaded_file)
else:
    # Use the default file
    st.info(f"No file uploaded. Using the default file: '{default_file_path}'")
    df = load_data(default_file_path)

# Fixed topic column
selected_topic_column = "one_topic_name"

# Prepare topic counts
topic_counts = df[selected_topic_column].value_counts().reset_index()
topic_counts.columns = ["Topic", "Count"]
available_topics = topic_counts["Topic"].tolist()

# Unique values for filters
unique_experiments = df["Experiment"].unique()
unique_conditions = df["Condition"].unique()
unique_ids = df["Id"].unique()
unique_files = df["File Name"].unique()

st.title("🔍 Topic Exploration")

st.markdown("""
This app allows you to:
- Explore the frequency of each topic with a pie chart.
- View topic distribution by filtering for a specific `Experiment`, `Condition`, `Id`, or `File Name`.
- Combine filters for `Experiment` and `Condition`.
- View detailed documents for each topic.
""")

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Overview", "Filtered Distribution", "Topic Detail"])

# ---------------------------
# Tab 1: Overview
# ---------------------------
with tab1:
    st.subheader("Topic Overview")

    # Pie chart for overall topic distribution
    st.markdown("**Overall Topic Distribution (Pie Chart)**")
    fig = px.pie(
        topic_counts,
        names="Topic",
        values="Count",
        title="Overall Topic Distribution",
        hole=0.3
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Tab 2: Filtered Distribution
# ---------------------------
with tab2:
    st.subheader("Filtered Topic Distribution")

    # Filter logic
    st.markdown("### Select Filters")
    selected_experiment = st.selectbox("Filter by Experiment:", options=["All"] + list(unique_experiments))
    selected_condition = st.selectbox("Filter by Condition:", options=["All"] + list(unique_conditions))
    filter_type = st.radio("Additional Filter by:", ["None", "ID", "File Name"], horizontal=True)

    # Apply Experiment and Condition filters
    filtered_data = df.copy()
    if selected_experiment != "All":
        filtered_data = filtered_data[filtered_data["Experiment"] == selected_experiment]
    if selected_condition != "All":
        filtered_data = filtered_data[filtered_data["Condition"] == selected_condition]

    # Apply additional filters
    if filter_type == "ID":
        selected_id = st.selectbox("Select an ID:", options=unique_ids)
        filtered_data = filtered_data[filtered_data["Id"] == selected_id]
    elif filter_type == "File Name":
        selected_file = st.selectbox("Select a File Name:", options=unique_files)
        filtered_data = filtered_data[filtered_data["File Name"] == selected_file]

    # Topic distribution for filtered data
    filtered_topic_counts = filtered_data[selected_topic_column].value_counts().reset_index()
    filtered_topic_counts.columns = ["Topic", "Count"]

    # Show pie chart for filtered data
    st.markdown("**Filtered Topic Distribution (Pie Chart)**")
    if not filtered_topic_counts.empty:
        fig_filtered = px.pie(
            filtered_topic_counts,
            names="Topic",
            values="Count",
            title="Filtered Topic Distribution",
            hole=0.3
        )
        st.plotly_chart(fig_filtered, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

# ---------------------------
# Tab 3: Topic Detail
# ---------------------------
with tab3:
    st.subheader("View Documents by Topic")

    # 1. The user selects a topic by name
    selected_topic = st.selectbox("Select a Topic:", options=available_topics)

    # 2. Filter where the name matches
    single_topic_rows = df[df["one_topic_name"] == selected_topic]

    # 3. Filter where multiple_topics_name includes that topic name
    multiple_topic_rows = df[df["multiple_topics_name"].apply(lambda x: selected_topic in x)]
    extra_rows = multiple_topic_rows[~multiple_topic_rows.index.isin(single_topic_rows.index)]

    st.markdown(f"**Documents associated with '{selected_topic}' in 'one_topic':**")
    st.dataframe(
        single_topic_rows[["Experiment", "Id", "Condition", "Speaker", "Index", "Content"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown(f"**Extra rows associated with '{selected_topic}' in 'multiple_topics':**")
    if not extra_rows.empty:
        st.dataframe(
            extra_rows[["Experiment", "Id", "Condition", "Speaker", "Index", "Content", "multiple_topics_name"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No extra rows found in 'multiple_topics' for the selected topic.")
