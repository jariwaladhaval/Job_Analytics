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
    jobs_master = pd.read_csv("jobs_dataset.csv", encoding="latin1")
    
    
    
    # Clean column names
    results.columns = results.columns.str.strip()
    matrix.columns = matrix.columns.str.strip()
    jobs_master.columns = jobs_master.columns.str.strip()

    # Clean Job IDs
    results["Job ID"] = results["Job ID"].astype(str).str.replace(",", "").str.strip()
    results["Compared Job ID"] = results["Compared Job ID"].astype(str).str.replace(",", "").str.strip()
    matrix.index = matrix.index.astype(str).str.replace(",", "")
    matrix.columns = matrix.columns.astype(str).str.replace(",", "")

    jobs_master["Job ID"] = jobs_master["Job ID"].astype(str).str.replace(",", "").str.strip()

    return results, matrix, jobs_master

results_df, similarity_matrix, jobs_master = load_data()

# ----------------------------------
# CREATE MASTER LOOKUP TABLE
# ----------------------------------

job_lookup = (
    jobs_master[
        ["Job ID", "Job", "work steam", "Domain"]
    ]
    .drop_duplicates(subset=["Job ID"])
    .rename(columns={
        "Job": "Job Name"
    })
)



# ----------------------------------
# CREATE JOB ID → JOB NAME MAPPING
# ----------------------------------


job_id_to_name = (
    jobs_master
    .drop_duplicates(subset=["Job ID"])
    .set_index("Job ID")["Job"]
    .to_dict()
)



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

    job_display_options = {
        job_id: f"{job_id} – {job_id_to_name.get(job_id, '')}"
        for job_id in job_ids
    }
    
    selected_job = st.sidebar.selectbox(
        "Select Job",
        job_ids,
        format_func=lambda x: job_display_options[x]
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

    filtered_display = filtered.copy()

    # Merge for main Job ID
    filtered_display = filtered_display.merge(
        job_lookup,
        on="Job ID",
        how="left"
    )
    
    # Merge for Compared Job ID
    filtered_display = filtered_display.merge(
        job_lookup.rename(columns={
            "Job ID": "Compared Job ID",
            "Job Name": "Compared Job Name",
            "Work Stream": "Compared Work Stream",
            "Domain": "Compared Domain"
        }),
        on="Compared Job ID",
        how="left"
    )

    
    # Reorder columns (clean UI)
    priority_cols = [
    "Job ID",
    "Job Name",
    "Work Stream",
    "Domain",
    "Compared Job ID",
    "Compared Job Name",
    "Compared Work Stream",
    "Compared Domain",
    "Similarity %",
    "Text Similarity %",
    "Competency Similarity %",
    "Similarity Reason"
    ]

    
    existing_priority_cols = [c for c in priority_cols if c in filtered_display.columns]
    remaining_cols = [c for c in filtered_display.columns if c not in existing_priority_cols]
    
    filtered_display = filtered_display[existing_priority_cols + remaining_cols]

    
    st.dataframe(filtered_display, width="stretch", hide_index=True)




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

    filtered = (
        results_df[
            results_df["Similarity %"] >= threshold
        ]
        .sort_values("Similarity %", ascending=False)
        .reset_index(drop=True)
    )

    # Compute job counts
    job_counts = {}

    for _, row in filtered.iterrows():
        job_a = str(row["Job ID"])
        job_b = str(row["Compared Job ID"])

        job_counts[job_a] = job_counts.get(job_a, 0) + 1
        job_counts[job_b] = job_counts.get(job_b, 0) + 1

    unique_jobs = len(job_counts)

    # ✅ Sidebar Summary (MUST stay inside this block)
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Similarity Summary")

    st.sidebar.markdown(f"""
    **Total Matching Pairs:** {len(filtered)}  
    **Unique Job IDs:** {unique_jobs}
    """)

    if job_counts:

        job_count_df = pd.DataFrame(
            list(job_counts.values()),
            columns=["Match Count"]
        )

        distribution = (
            job_count_df
            .value_counts()
            .reset_index(name="Number of Job IDs")
            .sort_values("Match Count")
            .reset_index(drop=True)
        )

        st.sidebar.markdown("### Distribution of Job Match Counts")
        st.sidebar.dataframe(
            distribution,
            width="stretch",   # updated from use_container_width
            hide_index=True
        )

    else:
        st.sidebar.info("No matching job pairs at selected threshold.")

    # Main page table
    st.subheader(f"📈 Job pairs with similarity ≥ {threshold}%")
    st.caption(f"🔢 {len(filtered)} job pairs found")
    filtered_display = filtered.copy()

    filtered_display = filtered.copy()

    # Merge for main Job ID
    filtered_display = filtered_display.merge(
        job_lookup,
        on="Job ID",
        how="left"
    )
    
    # Merge for Compared Job ID
    filtered_display = filtered_display.merge(
        job_lookup.rename(columns={
            "Job ID": "Compared Job ID",
            "Job Name": "Compared Job Name",
            "Work Stream": "Compared Work Stream",
            "Domain": "Compared Domain"
        }),
        on="Compared Job ID",
        how="left"
    )

    
    priority_cols = [
    "Job ID",
    "Job Name",
    "Compared Job ID",
    "Compared Job Name",
    "Similarity %",
    "Text Similarity %",
    "Competency Similarity %",
    "Similarity Reason"
    ]
    
    existing_priority_cols = [c for c in priority_cols if c in filtered_display.columns]
    remaining_cols = [c for c in filtered_display.columns if c not in existing_priority_cols]
    
    filtered_display = filtered_display[existing_priority_cols + remaining_cols]

    
    st.dataframe(filtered_display, width="stretch", hide_index=True)



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
        results_display = results.copy()

    if "Job ID" in results_display.columns:
        results_display = results_display.merge(
        job_lookup,
        on="Job ID",
        how="left"
    )


    st.dataframe(results_display, width="stretch", hide_index=True)




    


# ----------------------------------
# MATRIX VIEW (JOB-SPECIFIC)
# ----------------------------------
with st.expander("🧮 Job-Specific Similarity Matrix View"):

    matrix_job = st.selectbox(
    "Select Job",
    similarity_matrix.index.tolist(),
    format_func=lambda x: f"{x} – {job_id_to_name.get(x, '')}"
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
