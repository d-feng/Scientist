import streamlit as st
import scanpy as sc
import openai
import re
import pandas as pd
import json
import gseapy as gp
import matplotlib.pyplot as plt
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")
#adata = sc.datasets.pbmc68k_reduced()
#sc.tl.pca(adata, svd_solver="arpack")
#st.title("🔬 Gene-Based PCA Plot Generator")
API_KEY = ''
from openai import OpenAI
client = OpenAI(
    api_key = API_KEY
    #api_key = os.environ.get("OPENAI_API_KEY"),
    )
datasets = [
    {"name": "pbmc3k_raw.h5ad", "description": "Peripheral blood mononuclear cells from healthy donor"},
    {"name": "cancer.h5ad", "description": "Single-cell data from lung tumor and adjacent normal tissue"},
    {"name": "brain_dev.h5ad", "description": "Mouse brain developmental atlas"},
]
system_prompt = """
You are a bioinformatics assistant. Given a user’s request and a list of datasets (with name and description),
choose the most relevant dataset file.

Only return the `name` of the chosen dataset from the list.
"""

# User input
user_prompt = st.text_input("What would you like to explore?")

if st.button("Choose Dataset"):
    dataset_text = "\n".join([f"{d['name']}: {d['description']}" for d in datasets])

    llm_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Available datasets:\n{dataset_text}\n\nUser prompt: {user_prompt}"}
        ]
    )

    selected_dataset = llm_response.choices[0].message.content.strip()
    st.success(f"📂 Selected dataset: {selected_dataset}")

    # Load and store in session state
    adata = sc.read_h5ad(f"./data/{selected_dataset}")
    st.session_state["adata"] = adata



#adata = sc.datasets.pbmc3k()

# Set your OpenAI API key
#openai.api_key = "YOUR_OPENAI_API_KEY"

st.title("🧬 Smart Gene Visualization & Pathway Analysis")

# User input
user_input = st.text_input("Enter a sentence (e.g., 'umap for CD3D', 'Pathway analysis for cluster 1')")
# Initialize session state for context
# Initialize session state
for key in ["context_summary", "fig", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ----- Redisplay previous result (if exists) -----
if st.session_state["fig"]:
    st.pyplot(st.session_state["fig"])

if st.session_state["results"] is not None and not st.session_state["results"].empty:
    st.subheader("\U0001F9EA Previous Pathway Results")
    st.dataframe(st.session_state["results"][["Term", "Adjusted P-value", "Overlap", "Combined Score"]].head(10))


#if "context_summary" not in st.session_state:
#    st.session_state["context_summary"] = None
#plot_type = gene = cluster = None
#results = None  # pathway result placeholder
#context_summary = None  # for explanation prompt

if st.button("Submit"):
    with st.spinner("Asking the Agent..."):
        adata = st.session_state.get("adata", None)
        if adata is None:
            st.warning("No dataset loaded. Please select a dataset first.")
            st.stop()
        # LLM prompt
        system_prompt = """
        You are a bioinformatics assistant. From the user's sentence, extract:
        - plot_type: One of "pca", "umap", "violin", or "pathway_analysis", if user ask for visulize the gene or data, output "umap" as default
        - gene: Gene name to visualize (can be null for pathway analysis)
        - cluster: Cluster number as a string (can be null)

        Return a valid JSON object with keys: plot_type, gene, cluster.
        Example:
        {
          "plot_type": "pca",
          "gene": "CD3D",
          "cluster": null
        }
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )

        raw_reply = response.choices[0].message.content.strip()

        # Strip markdown code block if needed
        if raw_reply.startswith("```"):
            raw_reply = raw_reply.split("```")[1].strip()
            if raw_reply.startswith("json"):
                raw_reply = "\n".join(raw_reply.split("\n")[1:]).strip()

        # Parse JSON safely
        try:
            info = json.loads(raw_reply)
            plot_type = info.get("plot_type", "").lower()
            gene = info.get("gene")
            if gene:
                gene = gene.upper()
            cluster = info.get("cluster")
            st.success(f"Detected Task: `{plot_type}` | Gene: `{gene}` | Cluster: `{cluster}`")
        except Exception as e:
            st.error(f"JSON parsing failed: {e}")
            st.code(raw_reply)
            st.stop()

    # ----- Plot handling -----
    try:
        if plot_type in ["pca", "umap", "violin"]:
            #if gene not in adata.var_names:
            #    st.error(f"Gene '{gene}' not found.")
            #else:
                if plot_type == "pca":
                    sc.tl.pca(adata)
                    fig = sc.pl.pca(adata, color=gene, return_fig=True)
                    st.session_state["context_summary"] = f"A PCA plot colored by expression of {gene}."
                    #context_summary = f"A PCA plot was generated colored by expression of {gene}."
                elif plot_type == "umap":
                    sc.pp.neighbors(adata)
                    sc.tl.umap(adata)
                    fig = sc.pl.umap(adata, color=gene, return_fig=True)
                    #context_summary = f"A UMAP plot was generated colored by expression of {gene}."
                    st.session_state["context_summary"] = f"A UMAP plot colored by expression of {gene}."
                elif plot_type == "violin":
                    if "louvain" not in adata.obs:
                        sc.tl.louvain(adata)
                    fig = sc.pl.violin(adata, gene, groupby="louvain", return_fig=True)
                    #context_summary = f"A violin plot shows expression of {gene} across clusters."
                    st.session_state["context_summary"] = f"A violin plot showing {gene} expression across clusters."
                st.session_state["fig"] = fig
                st.pyplot(fig)

        elif plot_type == "pathway_analysis":
            st.subheader("🧪 Pathway Enrichment")

            if "louvain" not in adata.obs:
                sc.tl.louvain(adata)

            sc.tl.rank_genes_groups(adata, groupby="louvain", method="t-test")
            cluster_id = cluster if cluster else "0"
            df = sc.get.rank_genes_groups_df(adata, group=cluster_id)
            top_genes = df.query("pvals_adj < 0.05").head(100)["names"].tolist()

            if not top_genes:
                st.warning("No significant genes found.")
            else:
                enr = gp.enrichr(gene_list=top_genes,
                                 gene_sets=['KEGG_2019_Human', 'GO_Biological_Process_2018'],
                                 organism='Human',
                                 outdir=None,
                                 cutoff=0.05)
                results = enr.results
                if results.empty:
                    st.warning("No enriched pathways found.")
                else:
                    st.session_state["results"] = results
                    st.dataframe(results[["Term", "Adjusted P-value", "Overlap", "Combined Score"]].head(10))
                    #context_summary = f"Pathway analysis of cluster {cluster_id} shows top terms like {', '.join(results['Term'].head(3))}."
                    st.session_state["context_summary"] = (
                            f"Pathway analysis of cluster {cluster_id} identified key terms like "
                            f"{', '.join(results['Term'].head(3))}."
                        )
                    fig, ax = plt.subplots()
                    top = results.head(10)
                    ax.barh(top["Term"], top["Combined Score"])
                    ax.set_xlabel("Combined Score")
                    ax.set_title("Top Enriched Pathways")
                    st.pyplot(fig)

        else:
            st.error(f"Unsupported plot type: {plot_type}")
    except Exception as e:
        st.error(f"Plotting error: {e}")

# ---------- Explanation ----------
context_summary = st.session_state.get("context_summary", None)
st.write("📋 Explanation context:", context_summary)

# ----- Explain Button -----
if context_summary:
    if st.button("🧠 Explain This Result"):
        with st.spinner("Generating explanation..."):
            st.write("📋 Explanation context:", context_summary)
            explanation_prompt = f"""
            Explain the following result to a graduate-level biology student:

            {context_summary}
            """

            explanation_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful biology professor."},
                    {"role": "user", "content": explanation_prompt}
                ]
            )

            explanation = explanation_response.choices[0].message.content.strip()
            st.subheader("🔍 LLM Explanation")
            st.markdown(explanation)