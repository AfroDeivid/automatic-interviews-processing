import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Simple Topic Management",
    layout="wide",
    page_icon="🔍"
)

@st.cache_data
def load_data(file_path):
    """Load CSV file and return a DataFrame with a BiggerTopic column."""
    df = pd.read_csv(file_path)
    if "BiggerTopic" not in df.columns:
        df["BiggerTopic"] = ""
    return df

def save_data(df, file_path="updated_topics.csv"):
    """Save updated DataFrame to CSV."""
    df.to_csv(file_path, index=False)

# Load data
df = load_data("df_topic.csv")

# Sidebar options
st.sidebar.header("Configuration")
selected_topic_column = st.sidebar.selectbox(
    "Select Topic Column:",
    options=["one_topic_name", "multiple_topics_name"]
)

output_file = st.sidebar.text_input("Output file name:", value="updated_topics.csv")

# If multiple topics, explode to separate rows
if selected_topic_column == "multiple_topics_name":
    df = df.explode(selected_topic_column)

# Prepare topic counts
topic_counts = df[selected_topic_column].value_counts().reset_index()
topic_counts.columns = ["Topic", "Count"]
available_topics = topic_counts["Topic"].tolist()

st.title("🔍 Simple Topic Management")

st.markdown("""
This app helps you:
- Explore the frequency of each topic.
- Assign topics to broader categories (BiggerTopics).
- View and modify assignments at both topic and document level.
""")

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Topic Detail & Assignment", "Bulk Assignment", "Save & Export"])

# ---------------------------
# Tab 1: Overview
# ---------------------------
with tab1:
    st.subheader("Topic Overview")
    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown("**Topic Counts**")
        st.dataframe(topic_counts, use_container_width=True)

    with col2:
        st.markdown("**Topic Distribution (Bar Chart)**")
        fig = px.bar(
            topic_counts,
            x="Topic",
            y="Count",
            title="Count of Each Topic",
            labels={"Count": "Count", "Topic": "Topic"},
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("Use the tabs above to manage assignments and explore details.")


# ---------------------------
# Tab 2: Topic Detail & Assignment
# ---------------------------
with tab2:
    st.subheader("View and Assign a Single Topic")

    selected_topic = st.selectbox("Select a Topic:", options=available_topics)

    # Show documents for the selected topic
    topic_docs = df[df[selected_topic_column] == selected_topic]
    st.markdown(f"**Documents associated with '{selected_topic}':**")
    st.dataframe(topic_docs[["Content", "BiggerTopic"]], use_container_width=True)

    # Assign or update BiggerTopic for this selected topic
    st.markdown("**Assign Topic to a BiggerTopic**")
    existing_bigger_topics = [bt for bt in df["BiggerTopic"].unique() if isinstance(bt, str) and bt.strip()]

    bigger_topic_mode = st.radio("Choose how to assign:", ["Use existing BiggerTopic", "Create new BiggerTopic"])

    if bigger_topic_mode == "Use existing BiggerTopic" and existing_bigger_topics:
        chosen_bigger_topic = st.selectbox("Select existing BiggerTopic:", existing_bigger_topics)
    else:
        chosen_bigger_topic = st.text_input("Enter new BiggerTopic name:", placeholder="E.g., Customer Experience")

    if st.button("Assign Topic"):
        if chosen_bigger_topic.strip():
            df.loc[df[selected_topic_column] == selected_topic, "BiggerTopic"] = chosen_bigger_topic.strip()
            st.success(f"Assigned topic '{selected_topic}' to BiggerTopic '{chosen_bigger_topic}'.")
        else:
            st.warning("Please provide a valid BiggerTopic name.")


# ---------------------------
# Tab 3: Bulk Assignment of Multiple Topics
# ---------------------------
with tab3:
    st.subheader("Bulk Assign Multiple Topics")

    # Select multiple topics
    selected_bulk_topics = st.multiselect(
        "Select topics to assign:",
        options=available_topics
    )

    # Assign to BiggerTopic
    st.markdown("**Assign to BiggerTopic**")
    bigger_topic_mode = st.radio("How to assign?", ["Use existing BiggerTopic", "Create new BiggerTopic"], key="bulk")

    existing_bigger_topics = [bt for bt in df["BiggerTopic"].unique() if isinstance(bt, str) and bt.strip()]

    if bigger_topic_mode == "Use existing BiggerTopic" and existing_bigger_topics:
        bulk_bigger_topic = st.selectbox("Select BiggerTopic:", existing_bigger_topics, key="bulk_select")
    else:
        bulk_bigger_topic = st.text_input("Enter new BiggerTopic name:", placeholder="E.g., Key Product Feedback", key="bulk_text")

    if st.button("Bulk Assign"):
        if selected_bulk_topics and bulk_bigger_topic.strip():
            df.loc[df[selected_topic_column].isin(selected_bulk_topics), "BiggerTopic"] = bulk_bigger_topic.strip()
            st.success(f"Assigned topics {selected_bulk_topics} to BiggerTopic '{bulk_bigger_topic}'.")
        else:
            st.warning("Please select topics and provide a valid BiggerTopic name.")

    # Show summary of BiggerTopics
    st.markdown("### BiggerTopic Assignments Summary")
    bigger_topic_summary = df.groupby("BiggerTopic")[selected_topic_column].unique().reset_index()
    bigger_topic_summary["Topics"] = bigger_topic_summary[selected_topic_column].apply(lambda x: ", ".join(x))
    bigger_topic_summary = bigger_topic_summary[["BiggerTopic", "Topics"]]
    st.dataframe(bigger_topic_summary, use_container_width=True)


# ---------------------------
# Tab 4: Save & Export
# ---------------------------
with tab4:
    st.subheader("Save Changes")
    st.write("Click the button below to save your changes to a new CSV file.")

    if st.button("Save to New File"):
        save_data(df, file_path=output_file)
        st.success(f"Changes saved to '{output_file}'.")

    st.write("You can now download or use this updated CSV elsewhere.")
