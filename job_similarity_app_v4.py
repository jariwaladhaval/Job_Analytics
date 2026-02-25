

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
# STANDARDIZE COLUMN NAMES
# ----------------------------------

jobs_master.columns = jobs_master.columns.str.strip().str.lower()

# Fix spelling issue once
jobs_master = jobs_master.rename(columns={
    "work steam": "work stream"
})

# ----------------------------------
# CREATE MASTER LOOKUP TABLE
# ----------------------------------

job_lookup = (
    jobs_master[
        ["job id", "job", "work stream", "domain"]
    ]
    .drop_duplicates(subset=["job id"])
    .rename(columns={
        "job id": "Job ID",
        "job": "Job Name",
        "work stream": "Work Stream",
        "domain": "Domain"
    })
)

# For sidebar formatting
job_id_to_name = (
    job_lookup
    .set_index("Job ID")["Job Name"]
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
    
    # Merge main job
    filtered_display = filtered_display.merge(
        job_lookup,
        on="Job ID",
        how="left"
    )
    
    # Merge compared job
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
    
    # Remove symmetric duplicates
    filtered["pair_key"] = filtered.apply(
        lambda x: tuple(sorted([x["Job ID"], x["Compared Job ID"]])),
        axis=1
    )
    
    filtered = filtered.drop_duplicates(subset=["pair_key"]).drop(columns=["pair_key"])
    
    # VERY IMPORTANT: build display dataset from THIS cleaned filtered
    filtered_clean = filtered.copy()

    # ----------------------------------
    # Compute UNIQUE job match counts
    # ----------------------------------
    
    # Build mapping of Job ID → set of matched Job IDs
    job_match_map = {}
    
    for _, row in filtered.iterrows():
        job_a = str(row["Job ID"])
        job_b = str(row["Compared Job ID"])
    
        if job_a != job_b:  # safety check
            job_match_map.setdefault(job_a, set()).add(job_b)
            job_match_map.setdefault(job_b, set()).add(job_a)
    
    # Convert to counts
    job_counts = {
        job_id: len(matches)
        for job_id, matches in job_match_map.items()
    }
    
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
            width="stretch",
            hide_index=True
        )
        
        # 🔹 ADD SELECTBOX HERE (Immediately After Table)
        
        selected_match_count = st.sidebar.selectbox(
            "Select Match Count to View Job IDs",
            distribution["Match Count"].tolist()
        )
        
        

    else:
        st.sidebar.info("No matching job pairs at selected threshold.")

    # Main page table
    st.subheader(f"📈 Job pairs with similarity ≥ {threshold}%")
    st.caption(f"🔢 {len(filtered)} job pairs found")
    filtered_display = filtered_clean.copy()



    # Merge main job
    filtered_display = filtered_display.merge(
        job_lookup,
        on="Job ID",
        how="left"
    )
    
    # Merge compared job
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

    
    
    # ----------------------------------
    # MAIN LAYOUT: 2 COLUMNS
    # ----------------------------------
    
    col1, col2 = st.columns([2, 1])  # Left wider than right
    
    # LEFT SIDE → Full Matching Pairs
    with col1:
        st.subheader(f"📈 Job pairs with similarity ≥ {threshold}%")
        st.caption(f"🔢 {len(filtered)} job pairs found")
        st.dataframe(filtered_display, width="stretch", hide_index=True)
    
    # RIGHT SIDE → Drilldown
    with col2:
    
        if job_counts:

            st.subheader("📌 Drilldown View")
        
            # Get Job IDs having selected match count
            job_ids_with_count = sorted([
                job_id for job_id, count in job_counts.items()
                if count == selected_match_count
            ])
        
            # Build full drilldown list (undirected logic)
            drill_rows = []
        
            for job_id in job_ids_with_count:
        
                temp = filtered_clean[
                    (filtered_clean["Job ID"] == job_id) |
                    (filtered_clean["Compared Job ID"] == job_id)
                ].copy()
        
                temp["Primary Job ID"] = job_id
                drill_rows.append(temp)
        
            if drill_rows:
                drilldown_df = pd.concat(drill_rows, ignore_index=True)
            else:
                drilldown_df = pd.DataFrame()
        
            # Sort ascending
            drilldown_df = drilldown_df.sort_values(
                by=["Primary Job ID", "Job ID", "Compared Job ID"]
            ).reset_index(drop=True)
        
            st.caption(f"{len(job_ids_with_count)} Job IDs found")
        
            st.dataframe(drilldown_df, width="stretch", hide_index=True)
        
            csv = drilldown_df.to_csv(index=False).encode("utf-8")
        
            st.download_button(
                label="⬇️ Download Drilldown",
                data=csv,
                file_name=f"job_match_count_{selected_match_count}.csv",
                mime="text/csv"
            )


    
        


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
