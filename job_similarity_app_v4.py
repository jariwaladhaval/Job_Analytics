# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 14:53:46 2026

@author: Dhaval.Jariwala
"""

import streamlit as st
import pandas as pd
from job_similarity_engine import search_by_natural_language
# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="RoleGraph AI – Job Similarity Engine",
    layout="wide"
)

# ----------------------------------
# LOAD DATA (CACHED)
# ----------------------------------
@st.cache_data
def load_data():
    results = pd.read_excel("job_similarity_output_v1.xlsx")
    matrix = pd.read_excel("job_similarity_matrix.xlsx", index_col=0)

    # Clean Job IDs
    results["Job ID"] = results["Job ID"].astype(str).str.replace(",", "")
    results["Compared Job ID"] = results["Compared Job ID"].astype(str).str.replace(",", "")
    matrix.index = matrix.index.astype(str).str.replace(",", "")
    matrix.columns = matrix.columns.astype(str).str.replace(",", "")

    return results, matrix

results_df, similarity_matrix = load_data()







# ----------------------------------
# HEADER
# ----------------------------------
st.title("🧠 RoleGraph AI – Intelligent Job Similarity Engine")

st.markdown(
    """
    **How it works:**  
    This engine uses **Deep NLP embeddings (Sentence-BERT)** to understand job responsibilities, deliverables, 
    and outcomes, combined with **competency-level semantic matching**, to compute explainable, 
    percentage-based similarity between enterprise job roles.
    """
)

st.markdown("---")

# ----------------------------------
# SIDEBAR CONTROLS
# ----------------------------------
st.sidebar.header("🔎 Explore Similar Roles")

search_mode = st.sidebar.radio(
    "Search Mode",
    [
        "Search by Job ID",
        "Filter by Similarity Threshold",
        "NLP Search"
    ]
)

# ----------------------------------
# MODE 1 — SEARCH BY JOB ID
# ----------------------------------
if search_mode == "Search by Job ID":

    job_ids = sorted(results_df["Job ID"].unique())

    selected_job = st.sidebar.selectbox(
        "Select Job ID",
        job_ids
    )

    min_sim = st.sidebar.slider(
        "Minimum Similarity %",
        min_value=0,
        max_value=100,
        value=50
    )

    filtered = (
        results_df[
            (results_df["Job ID"] == selected_job) &
            (results_df["Similarity %"] >= min_sim)
        ]
        .sort_values("Similarity %", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader(f"📌 Similar roles for Job ID: {selected_job}")
    st.caption(f"🔢 {len(filtered)} matching roles found")

    st.dataframe(filtered, use_container_width=True)

# ----------------------------------
# MODE 2 — FILTER BY SIMILARITY %
# ----------------------------------
elif search_mode == "Filter by Similarity Threshold":
    
    threshold = st.sidebar.slider(
        "Show Job Pairs with Similarity ≥",
        min_value=0,
        max_value=100,
        value=70
    )

    # Filter pairs based on threshold
    filtered = (
        results_df[
            results_df["Similarity %"] >= threshold
        ]
        .sort_values("Similarity %", ascending=False)
        .reset_index(drop=True)        
    )

    st.subheader(f"📈 Job pairs with similarity ≥ {threshold}%")
    st.caption(f"🔢 {len(filtered)} job pairs found")

    # ----------------------------------
    # NEW: SIMILARITY SUMMARY DASHBOARD
    # ----------------------------------

    # Count frequency of each Job ID in filtered pairs
    job_counts = {}

    for _, row in filtered.iterrows():
        job_a = str(row["Job ID"])
        job_b = str(row["Compared Job ID"])

        job_counts[job_a] = job_counts.get(job_a, 0) + 1
        job_counts[job_b] = job_counts.get(job_b, 0) + 1

    unique_jobs = len(job_counts)


    st.markdown("---")

    # Show filtered dataframe
    st.dataframe(filtered, 
                 use_container_width=True,
                 hide_index=True)



# ----------------------------------
# MODE 3 — NLP Search
# ----------------------------------

elif search_mode == "NLP Search":

    st.subheader("🧠 Natural Language Job Search")

    query = st.text_input(
        "Describe the role you are looking for",
        placeholder="e.g. Find jobs similar to a data architect role"
    )

    if query:
        results = search_by_natural_language(query)

        st.caption(f"🔎 Top {len(results)} roles matching your query")
        st.dataframe(results, use_container_width=True)

# ----------------------------------
# SIDEBAR SIMILARITY SUMMARY (EXACT MATCH COUNT TABLE)
# ----------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Similarity Summary")

# Basic Metrics
st.sidebar.markdown(f"""
**Total Matching Pairs:** {len(filtered)}  
**Unique Job IDs:** {unique_jobs}
""")

# ----------------------------------
# EXACT DISTRIBUTION (NO RANGES)
# ----------------------------------

if job_counts:

    # Convert dictionary to DataFrame
    job_count_df = pd.DataFrame(
        list(job_counts.values()),
        columns=["Match Count"]
    )

    # Count how many Job IDs per match count
    distribution = (
        job_count_df
        .value_counts()
        .reset_index(name="Number of Job IDs")
        .sort_values("Match Count")
        .reset_index(drop=True)
    )

    st.sidebar.markdown("### Distribution of Job Match Counts")
    st.sidebar.dataframe(distribution, 
                         use_container_width=True,
                         hide_index=True)

else:
    st.sidebar.info("No matching job pairs at selected threshold.")

    


# ----------------------------------
# MATRIX VIEW (JOB-SPECIFIC)
# ----------------------------------
with st.expander("🧮 Job-Specific Similarity Matrix View"):

    matrix_job = st.selectbox(
        "Select Job ID to view its similarity matrix",
        similarity_matrix.index.tolist()
    )

    matrix_view = (
        similarity_matrix
        .loc[[matrix_job]]
        .T
        .sort_values(by=matrix_job, ascending=False)
        .rename(columns={matrix_job: "Similarity %"})
    )

    st.caption(f"Showing similarity scores for Job ID: {matrix_job}")
    st.dataframe(matrix_view, use_container_width=True)

#st.markdown("### 📥 Download Outputs")

import io

#st.markdown("### 📥 Download Outputs")

# Create in-memory Excel file
excel_buffer = io.BytesIO()
similarity_matrix.to_excel(excel_buffer, engine="openpyxl")
excel_buffer.seek(0)

st.download_button(
    label="⬇️ Download Full Job Similarity Matrix (Excel)",
    data=excel_buffer,
    file_name="job_similarity_matrix.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)



# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown("---")
st.caption(
    "Powered by Sentence-BERT, cosine similarity, and competency-level semantic matching • Built for Workforce Intelligence"
)
