import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Topic Exploration",
    layout="wide",
    page_icon="🔍"
)

@st.cache_data
def load_data(file_path):
    """Load CSV file and return a DataFrame."""
    return pd.read_csv(file_path)

# Load data
df = load_data("OBE1_topics_minsize13.csv")

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

    selected_topic = st.selectbox("Select a Topic:", options=available_topics)

    # Filter documents for the selected topic
    topic_docs = df[df[selected_topic_column] == selected_topic]
    if selected_experiment != "All":
        topic_docs = topic_docs[topic_docs["Experiment"] == selected_experiment]
    if selected_condition != "All":
        topic_docs = topic_docs[topic_docs["Condition"] == selected_condition]

    st.markdown(f"**Documents associated with '{selected_topic}':**")
    st.dataframe(topic_docs[["File Name", "Id", "Experiment", "Condition", "Content", "preprocessed_content"]], use_container_width=True)
