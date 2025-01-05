import streamlit as st
import pandas as pd
import plotly.express as px
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode

# Set page configuration
st.set_page_config(
    page_title="Topic Exploration with Tagging",
    layout="wide",
    page_icon="🔍"
)

@st.cache_data
def load_data(file_path):
    """Load CSV file and return a DataFrame."""
    return pd.read_csv(file_path)

# Load data
df = load_data("df_topic.csv")

# Add a 'Tag' column if not already present
if 'Tag' not in df.columns:
    df['Tag'] = None

# Define topic columns
single_topic_column = "one_topic_name"
multiple_topic_column = "multiple_topics_name"

# Prepare data for topic counts
df["multiple_topics_list"] = df[multiple_topic_column].apply(
    lambda x: eval(x) if isinstance(x, str) else []
)
all_topics = (
    pd.concat([df[single_topic_column], df.explode("multiple_topics_list")["multiple_topics_list"]])
    .dropna()
    .unique()
)

# Unique values for filters
unique_experiments = df["Experiment"].unique()
unique_conditions = df["Condition"].unique()
unique_ids = df["Id"].unique()
unique_files = df["File Name"].unique()

st.title("🔍 Topic Exploration with Tagging")

st.markdown("""
This app allows you to:
- Explore the frequency of topics with visualizations.
- Tag rows of interest interactively for further analysis.
- Save tagged rows or the full dataset for later use.
""")

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Overview", "Filtered Distribution", "Topic Detail with Tagging"])

# ---------------------------
# Tab 1: Overview
# ---------------------------
with tab1:
    st.subheader("Topic Overview")

    # Pie chart for overall topic distribution
    topic_counts = df[single_topic_column].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
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
    filtered_topic_counts = filtered_data[single_topic_column].value_counts().reset_index()
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
# Tab 3: Topic Detail with Tagging
# ---------------------------
with tab3:
    st.subheader("Tag Rows of Interest")

    # Select a topic
    selected_topic = st.selectbox("Select a Topic:", options=all_topics)

    # Toggle to include rows with multiple topics
    include_multiple_topics = st.checkbox("Include rows where the topic appears in `multiple_topics`", value=False)

    # Filter rows based on the selected topic and toggle
    topic_docs = df[df[single_topic_column] == selected_topic]
    if include_multiple_topics:
        multiple_topic_docs = df[df["multiple_topics_list"].apply(lambda x: selected_topic in x)]
        new_extra_rows = multiple_topic_docs[~multiple_topic_docs.index.isin(topic_docs.index)]
        topic_docs = pd.concat([topic_docs, new_extra_rows])

    # Select relevant columns to display
    columns_to_display = ["Experiment", "Id", "Condition", "Speaker", "Content", "Tag"]
    filtered_topic_docs = topic_docs[columns_to_display]

    # Interactive Ag-Grid Table
    st.markdown(f"**Interactive Table for Topic '{selected_topic}':**")
    gb = GridOptionsBuilder.from_dataframe(filtered_topic_docs)
    gb.configure_pagination(enabled=True)
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_column("Tag", editable=False)  # Tags are added dynamically via input
    grid_options = gb.build()

    # Enable autosizing for all columns
    grid_options["defaultColDef"] = {
        "flex": 1,  # Allow columns to take up available space proportionally
        "autoSizeColumns": True  # Automatically adjust the size of columns to fit their content
    }

    response = AgGrid(
        filtered_topic_docs,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=True,
        height=500,
        width="100%",
        reload_data=True,
    )

    # Extract selected rows
    selected_rows = pd.DataFrame(response["selected_rows"])

    # Input field for adding a tag
    tag_input = st.text_input("Enter a tag to assign to the selected rows:", value="")

    # Add the tag to selected rows
    if st.button("Apply Tag to Selected Rows"):
        if not selected_rows.empty and tag_input.strip():
            selected_rows["Tag"] = tag_input.strip()  # Apply the entered tag to selected rows
            df.update(selected_rows)  # Update the main DataFrame with the tagged rows
            st.success(f"Tag '{tag_input.strip()}' has been applied to {len(selected_rows)} rows.")
        else:
            st.warning("Please select rows and enter a valid tag.")

    # Display tagged rows
    if not selected_rows.empty:
        st.markdown("### Selected Rows with Tags")
        st.dataframe(selected_rows, use_container_width=True)

    # # Save selected rows to a new DataFrame
    # if st.button("Save Selected Rows"):
    #     output_path = "selected_rows.csv"
    #     selected_rows.to_csv(output_path, index=False, encoding="utf-8-sig")
    #     st.success(f"Selected rows saved to '{output_path}'.")

    # Save the entire DataFrame with tags
    if st.button("Save Full Tagged DataFrame"):
        output_path = f"tagged_df_{tag_input.strip()}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        st.success(f"Full tagged DataFrame saved to '{output_path}'.")
