import scanpy as sc
import openai
import pandas as pd
import json
import argparse
import os
import anndata
import scipy.sparse
import numpy as np
from collections import Counter
import gseapy as gp
import gseapy.plot as gseaplot
import matplotlib.pyplot as plt

from dotenv import load_dotenv


def create_gene_id_to_symbol_map(mouse_file_path, human_file_path):
    """
    Creates dictionaries to map Ensembl gene IDs to gene symbols for mouse and human.

    Args:
        mouse_file_path (str): Path to the mouse Ensembl gene mapping file (MS_Ensl_Gene_mart.txt).
        human_file_path (str): Path to the human Ensembl gene mapping file (Hu_Ensbl_GeneSymbol.txt).

    Returns:
        tuple: A tuple containing two dictionaries:
               - mouse_gene_map (dict): Maps mouse Ensembl IDs to gene symbols.
               - human_gene_map (dict): Maps human Ensembl IDs to gene symbols.
    """
    mouse_gene_map = {}
    human_gene_map = {}

    print(f"Processing mouse gene mapping file: {mouse_file_path}")
    with open(mouse_file_path, 'r') as f:
        # Skip header
        header = f.readline().strip().split('\t') # Changed back to split by tab
        gene_id_col_mouse = header.index('Gene stable ID')
        gene_name_col_mouse = header.index('Gene name')

        for line in f:
            parts = line.strip().split('\t') # Changed back to split by tab
            if len(parts) > max(gene_id_col_mouse, gene_name_col_mouse):
                gene_id = parts[gene_id_col_mouse]
                gene_symbol = parts[gene_name_col_mouse]
                
                # For mouse, use the ENSMUSG ID without version as the key
                if gene_id.startswith('ENSMUSG'):
                    key_id = gene_id.split('.')[0]
                else:
                    key_id = gene_id # Fallback for unexpected formats

                if key_id and gene_symbol:
                    mouse_gene_map[key_id] = gene_symbol
    print(f"Loaded {len(mouse_gene_map)} mouse gene mappings.")

    print(f"Processing human gene mapping file: {human_file_path}")
    with open(human_file_path, 'r') as f:
        # Skip header
        header = f.readline().strip().split('\t')
        gene_id_col_human = header.index('Gene stable ID')
        hgnc_symbol_col_human = header.index('HGNC symbol')

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > max(gene_id_col_human, hgnc_symbol_col_human):
                gene_id = parts[gene_id_col_human]
                hgnc_symbol = parts[hgnc_symbol_col_human]
                if gene_id and hgnc_symbol:
                    human_gene_map[gene_id] = hgnc_symbol
    print(f"Loaded {len(human_gene_map)} human gene mappings.")

    return mouse_gene_map, human_gene_map

def load_marker_genes_from_csv(file_path):
    """
    Loads marker genes from a CSV file, mapping gene symbols to a list of cell names.
    """
    marker_genes_map = {}
    try:
        # Added encoding='latin-1' to handle potential encoding issues in the CSV file.
        df = pd.read_csv(file_path, encoding='latin-1')
        for index, row in df.iterrows():
            symbol = row['Symbol']
            cell_name = row['cell_name']
            if pd.notna(symbol) and pd.notna(cell_name):
                if symbol not in marker_genes_map:
                    marker_genes_map[symbol] = []
                marker_genes_map[symbol].append(cell_name)
    except Exception as e:
        print(f"Error loading marker genes from CSV: {e}")
    return marker_genes_map

load_dotenv()

try:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    print(f"Attempting to use API Key: {os.getenv('OPENAI_API_KEY')[:5]}...{os.getenv('OPENAI_API_KEY')[-5:]}")
except KeyError:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running the agent.")

# Load gene mappings once
script_dir = os.path.dirname(__file__)
ref_dir = os.path.join(script_dir, "ref")

MOUSE_GENE_MAP, HUMAN_GENE_MAP = create_gene_id_to_symbol_map(
    mouse_file_path=os.path.join(ref_dir, "MS_Ensl_GeneSymbol.txt"),
    human_file_path=os.path.join(ref_dir, "Hu_Ensbl_GeneSymbol.txt")
)

# Load marker genes from CSV
MARKER_GENES_FROM_CSV = load_marker_genes_from_csv(
    file_path=os.path.join(ref_dir, "Cell_marker_table_short.csv")
)
print(f"Loaded {len(MARKER_GENES_FROM_CSV)} marker genes from CSV.")


def normalize_gene_symbols(adata):
    """
    Normalizes gene symbols to uppercase human gene symbols.
    Handles mouse gene symbols and Ensembl IDs.
    """
    print("\n--- Normalizing Gene Symbols ---")
    
    current_var_names = adata.var_names.tolist()
    new_var_names = []

    for name in current_var_names:
        original_name = name
        origin = identify_gene_origin(original_name)

        if name.startswith('ENSG'):
            # Human Ensembl ID
            ensembl_id_base = name.split('.')[0]
            mapped_name = HUMAN_GENE_MAP.get(ensembl_id_base, name)
            new_var_names.append(mapped_name)
            print(f"Original: {original_name} ({origin}) -> Mapped: {mapped_name}")
        elif name.startswith('ENSMUSG'):
            # Mouse Ensembl ID - use full ID without version
            ensembl_id_base = name.split('.')[0]
            mapped_name = MOUSE_GENE_MAP.get(ensembl_id_base, name)
            new_var_names.append(mapped_name)
            print(f"Original: {original_name} ({origin}) -> Mapped: {mapped_name}")
        elif name.islower() or (name[0].isupper() and name[1:].islower()):
            # Convert potential mouse genes to uppercase (assuming human genes are already uppercase)
            mapped_name = name.upper()
            new_var_names.append(mapped_name)
            print(f"Original: {original_name} ({origin}) -> Mapped: {mapped_name}")
        else:
            new_var_names.append(name)
            print(f"Original: {original_name} ({origin}) -> No change")

    adata.var_names = new_var_names
    # Make gene names unique, which is important after conversion
    adata.var_names_make_unique()
    print("Gene symbols normalized and made unique.")
    return adata

def identify_gene_origin(gene_id_or_symbol):
    """
    Identifies if a gene ID or symbol belongs to human or mouse.
    """
    gene_str = str(gene_id_or_symbol)
    if gene_str.startswith('ENSG'):
        return "Human (Ensembl ID)"
    elif gene_str.startswith('ENSMUSG'):
        return "Mouse (Ensembl ID)"
    elif gene_str.isupper():
        return "Human (Symbol)"
    elif gene_str[0].isupper() and gene_str[1:].islower():
        return "Mouse (Symbol)"
    else:
        return "Unknown"

def suggest_analyses(adata):
    """
    Uses an LLM to suggest analyses based on the AnnData object's metadata.
    """
    system_prompt = """
    You are a single-cell bioinformatics expert. Based on the provided metadata
    (obs columns and a sample of the data), suggest a list of potential
    analyses that can be performed. For each analysis, provide a brief
    description and the columns that would be involved.

    In addition to general analyses, specifically consider suggesting a "Cell Type Annotation Validation" analysis.
    This analysis should involve checking if the identified major cell types are correctly annotated by examining the expression of known cell marker genes.
    Mention that this would require a list of relevant marker genes.

    Respond with a JSON object containing a list of suggested analyses.
    Each item in the list should be an object with the keys 'analysis_name',
    'description', and 'relevant_columns'.

    Example:
    {
      "suggested_analyses": [
        {
          "analysis_name": "Differential Expression Analysis",
          "description": "Identify genes that are differentially expressed between different cell types.",
          "relevant_columns": ["major_celltype"]
        },
        {
          "analysis_name": "Trajectory Inference",
          "description": "Infer developmental trajectories of cells based on their gene expression profiles.",
          "relevant_columns": ["cell_subtype", "pseudotime"]
        },
        {
          "analysis_name": "Cell Type Annotation Validation using Marker Genes",
          "description": "Validate the major cell type annotations by checking the expression of known marker genes within each cell type. This analysis requires a predefined list of cell-type specific marker genes.",
          "relevant_columns": ["major_celltype"]
        }
      ]
    }
    """
    obs_df = adata.obs
    metadata_summary = {
        "columns": obs_df.columns.tolist(),
        "data_sample": obs_df.head().to_dict()
    }

    user_prompt = f"AnnData metadata summary: {json.dumps(metadata_summary, indent=2)}"

    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred while communicating with the OpenAI API: {e}")
        return {}

def get_cell_type_columns(columns):
    """
    Uses an LLM to identify cell type columns from a list of column names.
    """
    system_prompt = """
    You are a single-cell bioinformatics expert. Your task is to identify which of the
    following column names from a single-cell AnnData object are likely to contain
    cell type annotations. Please classify them as either 'major_celltype' or 'cell_subtype'.
    A 'major_celltype' column contains broad cell categories, while a 'cell_subtype'
    column contains more specific cell types. If a column is likely a cell type but you
    are unsure if it is a major or subtype, classify it as 'major_celltype'.

    Respond with a JSON object where the keys are the identified cell type column names
    and the values are either 'major_celltype' or 'cell_subtype'.
    If no columns appear to be cell type annotations, return an empty JSON object.

    Example:
    {
      "louvain": "major_celltype",
      "leiden_subclusters": "cell_subtype",
      "cell_type_coarse": "major_celltype"
    }
    """
    user_prompt = f"Column names: {', '.join(columns)}"

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred while communicating with the OpenAI API: {e}")
        return {}

def refine_cell_type_columns(adata, potential_columns):
    """
    Uses an LLM to refine the cell type column identification by looking at a
    sample of the data.
    """
    system_prompt = """
    You are a single-cell bioinformatics expert. Based on the column names and a
    sample of the data, confirm which columns are cell type annotations and
    classify them as 'major_celltype' or 'cell_subtype'.

    Respond with a JSON object where the keys are the confirmed cell type column
    names and the values are 'major_celltype' or 'cell_subtype'.
    """
    data_samples = {}
    for col in potential_columns:
        data_samples[col] = adata.obs[col].unique()[:5].tolist()

    user_prompt = f"Column names and data samples: {json.dumps(data_samples, indent=2)}"

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred while communicating with the OpenAI API: {e}")
        return {}

def convert_ids_to_symbols(gene_list):
    """
    Converts a list of gene identifiers (Ensembl IDs or symbols) to gene symbols.
    """
    converted_genes = []
    for gene in gene_list:
        if str(gene).startswith('ENSG'):
            # Human Ensembl ID
            ensembl_id_base = str(gene).split('.')[0]
            converted_genes.append(HUMAN_GENE_MAP.get(ensembl_id_base, gene))
        elif str(gene).startswith('ENSMUSG'):
            # Mouse Ensembl ID
            ensembl_id_base = str(gene).split('.')[0]
            converted_genes.append(MOUSE_GENE_MAP.get(ensembl_id_base, gene))
        else:
            converted_genes.append(gene)
    return converted_genes



def match_cell_type_with_llm(query_cell_type, available_cell_types):
    """
    Uses an LLM to find the best matching cell type from a list of available cell types.
    """
    system_prompt = """
    You are an expert in single-cell biology and cell type nomenclature.
    Your task is to find the best matching cell type from a provided list
    that semantically corresponds to a given query cell type name.

    Respond with a JSON object containing a single key 'matched_cell_type'.
    The value should be the exact string from the 'available_cell_types' list
    that best matches the 'query_cell_type'. If no suitable match is found,
    return "None".
    """
    user_prompt = f"Query cell type: \"{query_cell_type}\". Available cell types: {json.dumps(available_cell_types)}."

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred during LLM cell type matching: {e}")
        return "None"

def llm_validate_unmatched_cell_type(cell_type_name, computed_markers):
    """
    Uses an LLM to validate a cell type based on its computed marker genes when no direct match is found.
    """
    system_prompt = """
    You are a single-cell bioinformatics expert. You have been provided with a cell type name
    and a list of its top computed marker genes from a single-cell RNA-seq experiment.
    Your task to assess if the provided cell type name is consistent with the given marker genes.

    If you are confident that the cell type name is well-supported by the marker genes, respond with "Validated".
    If the cell type name is plausible but requires further investigation or is not strongly supported by the markers, respond with "Needs Review".
    If the cell type name is inconsistent with the markers or you are unsure, respond with "Unvalidated".

    Respond with a JSON object containing a single key 'judgment'.
    """
    user_prompt = f"Cell Type: \"{cell_type_name}\". Top Computed Markers: {json.dumps(computed_markers)}."

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        judgment_result = json.loads(response.choices[0].message.content)
        return judgment_result.get('judgment', 'Unvalidated')
    except Exception as e:
        print(f"An error occurred during LLM knowledge-based validation: {e}")
        return "Unvalidated"

def validate_with_computed_markers(adata, marker_genes_dict, n_top_genes=25, overlap_threshold=0.5):
    """
    Validates cell type annotations by comparing computed marker genes with provided marker genes.
    Assumes data is normalized and log-transformed.
    """
    print("\n--- Starting Validation with Computed Marker Genes ---")
    computed_validation_results = {}

    if 'cell_type' not in adata.obs.columns:
        print("'cell_type' column not found. Cannot compute marker genes for validation.")
        return computed_validation_results

    adata_for_computation = adata.copy()

    try:
        print("Running rank_genes_groups on pre-normalized data...")
        sc.tl.rank_genes_groups(adata_for_computation, 'cell_type', method='wilcoxon')
    except Exception as e:
        print(f"Error computing ranked genes groups: {e}")
        return computed_validation_results

    available_marker_types = list(marker_genes_dict.keys())

    for cell_type_in_adata in adata_for_computation.obs['cell_type'].unique():
        matched_cell_type_dict = match_cell_type_with_llm(str(cell_type_in_adata), available_marker_types)
        matched_cell_type = matched_cell_type_dict.get('matched_cell_type', 'None')

        if matched_cell_type == "None" or matched_cell_type not in marker_genes_dict:
            print(f"No direct reference match for '{cell_type_in_adata}'. Attempting LLM knowledge-based validation...")
            try:
                if 'rank_genes_groups' not in adata_for_computation.uns or 'names' not in adata_for_computation.uns['rank_genes_groups']:
                    computed_validation_results[cell_type_in_adata] = {
                        "status": "Failed",
                        "reason": "rank_genes_groups results not found or incomplete for LLM knowledge validation."
                    }
                    continue

                computed_markers_df = pd.DataFrame(adata_for_computation.uns['rank_genes_groups']['names'])
                if str(cell_type_in_adata) not in computed_markers_df.columns:
                    computed_validation_results[cell_type_in_adata] = {
                        "status": "Failed",
                        "reason": f"Computed markers for '{cell_type_in_adata}' not found in rank_genes_groups results for LLM knowledge validation."
                    }
                    continue

                top_computed_markers_ensembl = computed_markers_df[str(cell_type_in_adata)].head(n_top_genes).tolist()
                top_computed_markers = convert_ids_to_symbols(top_computed_markers_ensembl)
                llm_judgment = llm_validate_unmatched_cell_type(str(cell_type_in_adata), top_computed_markers)
                computed_validation_results[cell_type_in_adata] = {
                    "status": llm_judgment,
                    "reason": "Validated by LLM knowledge based on computed markers.",
                    "top_computed_markers": top_computed_markers
                }
            except Exception as e:
                print(f"Error during LLM knowledge-based validation for {cell_type_in_adata}: {e}")
                computed_validation_results[cell_type_in_adata] = {
                    "status": "Error",
                    "reason": f"Error during LLM knowledge-based validation: {e}"
                }
            continue

        # This function expects marker_genes_dict to be {cell_type: [marker_genes]}
        # However, MARKER_GENES_FROM_CSV is {marker_symbol: [cell_names]}
        # We need to adapt the logic here to use MARKER_GENES_FROM_CSV.
        # Instead of comparing with provided_markers, we'll collect associated cell names.

        try:
            if 'rank_genes_groups' not in adata_for_computation.uns or 'names' not in adata_for_computation.uns['rank_genes_groups']:
                computed_validation_results[cell_type_in_adata] = {
                    "status": "Failed",
                    "reason": "rank_genes_groups results not found or incomplete."
                }
                continue

            computed_markers_df = pd.DataFrame(adata_for_computation.uns['rank_genes_groups']['names'])
            if str(cell_type_in_adata) not in computed_markers_df.columns:
                computed_validation_results[cell_type_in_adata] = {
                    "status": "Failed",
                    "reason": f"Computed markers for '{cell_type_in_adata}' not found in rank_genes_groups results."
                }
                continue

            top_computed_markers_ensembl = computed_markers_df[str(cell_type_in_adata)].head(n_top_genes).tolist()
            top_computed_markers = convert_ids_to_symbols(top_computed_markers_ensembl)

            # Collect all associated cell names from the CSV for the top computed markers
            associated_cell_names = []
            for marker_symbol in top_computed_markers:
                if marker_symbol in MARKER_GENES_FROM_CSV:
                    associated_cell_names.extend(MARKER_GENES_FROM_CSV[marker_symbol])
            
            # Remove duplicates and sort for consistency
            associated_cell_names = sorted(list(set(associated_cell_names)))

            # Determine cell type by majority vote from associated_cell_names_from_csv
            voted_cell_type = "Unvalidated"
            if associated_cell_names:
                # Count occurrences of each cell name
                cell_name_counts = Counter(associated_cell_names)
                # Get the most common cell name(s)
                most_common = cell_name_counts.most_common()

                if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                    # Clear majority
                    voted_cell_type = most_common[0][0]
                    status = "Validated by Majority Vote"
                    reason = f"Majority vote from associated cell names: {voted_cell_type}"
                else:
                    # Tie or no clear majority, use LLM for judgment
                    llm_judgment = llm_validate_unmatched_cell_type(str(cell_type_in_adata), top_computed_markers)
                    voted_cell_type = llm_judgment # LLM returns a judgment string
                    status = f"Validated by LLM ({llm_judgment})"
                    reason = "LLM judgment due to no clear majority in associated cell names."
            else:
                # No associated cell names from CSV, use LLM knowledge-based validation
                llm_judgment = llm_validate_unmatched_cell_type(str(cell_type_in_adata), top_computed_markers)
                voted_cell_type = llm_judgment # LLM returns a judgment string
                status = f"Validated by LLM ({llm_judgment})"
                reason = "LLM knowledge-based validation (no associated cell names from CSV)."

            computed_validation_results[cell_type_in_adata] = {
                "status": status,
                "matched_reference_type": voted_cell_type,
                "reason": reason,
                "top_computed_markers": top_computed_markers,
                "associated_cell_names_from_csv": associated_cell_names
            }

        except Exception as e:
            print(f"Error extracting computed markers for {cell_type_in_adata}: {e}")
            computed_validation_results[cell_type_in_adata] = {
                "status": "Error",
                "reason": f"Error extracting computed markers: {e}"
            }
            continue

    print("--- Validation with Computed Marker Genes Complete ---")
    return computed_validation_results

def run_differential_expression_analysis(adata, condition_column, output_dir, pvalue_threshold=0.05, logfc_threshold=0.5):
    """
    Performs differential expression analysis for each cell type based on a given condition column.
    Filters for significant genes and records conditions.
    """
    print(f"\n--- Starting Differential Expression Analysis for condition: {condition_column} ---")
    deg_results = {}

    if 'cell_type' not in adata.obs.columns:
        print("'cell_type' column not found. Cannot perform DEG analysis.")
        return deg_results

    unique_cell_types = adata.obs['cell_type'].unique()

    for cell_type in unique_cell_types:
        print(f"\nProcessing cell type: {cell_type}")
        cell_type_adata = adata[adata.obs['cell_type'] == cell_type].copy()

        if condition_column not in cell_type_adata.obs.columns:
            print(f"Condition column '{condition_column}' not found for cell type '{cell_type}'. Skipping.")
            continue

        unique_conditions = cell_type_adata.obs[condition_column].unique()

        if len(unique_conditions) < 2:
            print(f"Not enough unique conditions in '{condition_column}' for cell type '{cell_type}'. Skipping.")
            continue

        try:
            # Determine reference if possible, otherwise default to 'rest'
            # For simplicity, we'll assume the first unique condition is the reference if not specified
            # Scanpy's rank_genes_groups by default compares each group against the rest.
            # If a specific reference is needed, it should be passed to `reference` parameter.
            # For now, we'll just note the comparison is against 'rest'.
            sc.tl.rank_genes_groups(cell_type_adata, condition_column, method='wilcoxon')
            result = cell_type_adata.uns['rank_genes_groups']
            groups = result['names'].dtype.names

            cell_type_deg = {}
            for group in groups:
                group_data = pd.DataFrame({
                    'gene_names': result['names'][group],
                    'scores': result['scores'][group],
                    'pvals': result['pvals'][group],
                    'pvals_adj': result['pvals_adj'][group],
                    'logfoldchanges': result['logfoldchanges'][group]
                })

                # Filter for significant genes
                significant_genes = group_data[
                    (group_data['pvals_adj'] < pvalue_threshold)
                ]

                increased_genes = significant_genes[significant_genes['logfoldchanges'] > logfc_threshold]
                decreased_genes = significant_genes[significant_genes['logfoldchanges'] < -logfc_threshold]

                cell_type_deg[group] = {
                    "comparison": f"{group} vs. rest", # Explicitly state comparison
                    "all_genes": group_data.to_dict(orient='list'),
                    "significant_increased_genes": increased_genes.to_dict(orient='list'),
                    "significant_decreased_genes": decreased_genes.to_dict(orient='list')
                }

                # Prepare ranked gene list for GSEApy
                # GSEApy prerank expects a Series with gene names as index and scores (logFC) as values
                ranked_genes = group_data.set_index('gene_names')['logfoldchanges']
                
                # Convert ranked_genes index from Ensembl IDs to gene symbols
                ranked_genes.index = convert_ids_to_symbols(ranked_genes.index)

                # Run GSEA analysis
                gsea_results = run_gsea_analysis(ranked_genes, "KEGG_2019_Human", output_dir, f"{cell_type}_{group}")
                cell_type_deg[group]["gsea_results"] = gsea_results

                # Run Enrichment analysis
                up_gene_symbols = convert_ids_to_symbols(increased_genes['gene_names'].tolist())
                down_gene_symbols = convert_ids_to_symbols(decreased_genes['gene_names'].tolist())
                
                enrichment_results = run_enrichment_analysis(
                    up_genes=up_gene_symbols,
                    down_genes=down_gene_symbols,
                    gene_set_name='GO_Biological_Process_2021',
                    output_dir=output_dir,
                    comparison_name=f"{cell_type}_{group}"
                )
                cell_type_deg[group]["enrichment_results"] = enrichment_results

            deg_results[cell_type] = cell_type_deg

        except Exception as e:
            print(f"Error performing DEG for {cell_type} with condition {condition_column}: {e}")

    print("--- Differential Expression Analysis Complete ---")
    return deg_results





def run_gsea_analysis(ranked_genes, gene_set_name, output_dir, comparison_name):
    """
    Performs GSEApy enrichment analysis.
    """
    print(f"\n--- Starting GSEA for {comparison_name} using gene set: {gene_set_name} ---")
    try:
        # Ensure output directory exists
        gsea_output_dir = os.path.join(output_dir, "gsea_results", comparison_name.replace(" ", "_").replace("/", "_"))
        os.makedirs(gsea_output_dir, exist_ok=True)

        # Run GSEA
        # The example uses `gseapy.prerank`, which takes a pandas Series with gene_name as index and scores as values.
        # We will use logfoldchanges as scores.
        enr = gp.prerank(rnk=ranked_genes,
                         gene_sets=gene_set_name,
                         outdir=gsea_output_dir,
                         no_plot=False, # Generate plots
                         verbose=True)

        # Save results
        results_df = enr.res2d
        results_file = os.path.join(gsea_output_dir, f"{comparison_name.replace(' ', '_').replace('/', '_')}_gsea_results.csv")
        results_df.to_csv(results_file)
        print(f"Saved GSEA results to {results_file}")

        # Generate plots for enriched terms
        if not results_df.empty:
            for i, term in enumerate(results_df.index):
                if i >= 5: # Limit to top 5 plots to avoid too many files
                    break
                try:
                    plot_file = os.path.join(gsea_output_dir, f"{comparison_name.replace(' ', '_').replace('/', '_')}_{term.replace(' ', '_').replace('/', '_')}_gsea_plot.pdf")
                    gseaplot.gseaplot(rank_metric=enr.ranking,
                                      term=term,
                                      ofname=plot_file,
                                      **enr.results[term])
                    print(f"Generated GSEA plot for term '{term}': {plot_file}")
                except Exception as plot_e:
                    print(f"Error generating plot for term '{term}': {plot_e}")
        else:
            print("No enriched terms found to plot.")

        return results_df.to_dict(orient='records') # Return as list of dicts for JSON serialization

    except Exception as e:
        print(f"Error performing GSEA for {comparison_name}: {e}")
        return {}

def run_enrichment_analysis(up_genes, down_genes, gene_set_name, output_dir, comparison_name):
    """
    Performs enrichment analysis on up and down regulated gene lists.
    """
    print(f"\n--- Starting Enrichment Analysis for {comparison_name} using gene set: {gene_set_name} ---")
    try:
        # Ensure output directory exists
        enrichment_output_dir = os.path.join(output_dir, "enrichment_results", comparison_name.replace(" ", "_").replace("/", "_"))
        os.makedirs(enrichment_output_dir, exist_ok=True)

        enr_up_results = None
        enr_dw_results = None
        enr_up = None
        enr_dw = None

        # Enrichment for up-regulated genes
        if up_genes:
            enr_up = gp.enrichr(gene_list=up_genes,
                                gene_sets=[gene_set_name],
                                outdir=None,
                                )
            if enr_up.results is not None and not enr_up.results.empty:
                enr_up_results = enr_up.results
                up_results_file = os.path.join(enrichment_output_dir, f"{comparison_name}_up_enrichment.csv")
                enr_up_results.to_csv(up_results_file)
                print(f"Saved up-regulated enrichment results to {up_results_file}")
                gp.dotplot(enr_up.res2d, figsize=(3,5), title="Up-regulated", cmap = plt.cm.autumn_r, ofname=os.path.join(enrichment_output_dir, f"{comparison_name}_up_dotplot.png"))

        # Enrichment for down-regulated genes
        if down_genes:
            enr_dw = gp.enrichr(gene_list=down_genes,
                                gene_sets=[gene_set_name],
                                outdir=None,
                                )
            if enr_dw.results is not None and not enr_dw.results.empty:
                enr_dw_results = enr_dw.results
                dw_results_file = os.path.join(enrichment_output_dir, f"{comparison_name}_down_enrichment.csv")
                enr_dw_results.to_csv(dw_results_file)
                print(f"Saved down-regulated enrichment results to {dw_results_file}")
                gp.dotplot(enr_dw.res2d, figsize=(3,5), title="Down-regulated", cmap = plt.cm.winter_r, ofname=os.path.join(enrichment_output_dir, f"{comparison_name}_down_dotplot.png"))

        # Combined plot
        if enr_up is not None and enr_dw is not None and enr_up_results is not None and not enr_up_results.empty and enr_dw_results is not None and not enr_dw_results.empty:
            enr_up.res2d['UP_DW'] = "UP"
            enr_dw.res2d['UP_DW'] = "DOWN"
            enr_res = pd.concat([enr_up.res2d.head(), enr_dw.res2d.head()])
            gp.dotplot(enr_res,
                       column='Adjusted P-value',
                       x='UP_DW',
                       size=5,
                       cmap = plt.cm.coolwarm,
                       figsize=(3,5),
                       ofname=os.path.join(enrichment_output_dir, f"{comparison_name}_combined_dotplot.png"))
            print(f"Saved combined enrichment plot to {enrichment_output_dir}")
            return {"up_regulated": enr_up_results.to_dict(orient='records'),
                    "down_regulated": enr_dw_results.to_dict(orient='records')}
        elif enr_up_results is not None and not enr_up_results.empty:
             return {"up_regulated": enr_up_results.to_dict(orient='records'),
                    "down_regulated": {}}
        elif enr_dw_results is not None and not enr_dw_results.empty:
            return {"up_regulated": {},
                    "down_regulated": enr_dw_results.to_dict(orient='records')}

    except Exception as e:
        print(f"Error performing enrichment analysis for {comparison_name}: {e}")
    
    return {}

def get_condition_columns_from_llm(columns):
    """
    Uses an LLM to identify potential condition columns from a list of column names.
    """
    system_prompt = """
    You are a single-cell bioinformatics expert. Your task is to identify which of the
    following column names from a single-cell AnnData object are likely to represent
    experimental conditions, treatments, disease, or disease statuses relevant for differential
    expression analysis. Prioritize columns that indicate a clear experimental variable
    or biological state difference.

    Respond with a JSON object containing a single key 'condition_columns'.
    The value should be a list of strings, where each string is a column name
    identified as a potential condition column. If no suitable columns are found,
    return an empty list.

    Example:
    {
      "condition_columns": [
        "Sample Characteristic[stimulus]",
        "Sample Characteristic[disease]",
        "Factor Value[treatment]"
      ]
    }
    {
      "condition_columns": [
        "disease",
        "disease_state",
        "treatment",
       
    }


    """
    user_prompt = f"Column names: {', '.join(columns)}"

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content).get('condition_columns', [])
    except Exception as e:
        print(f"An error occurred while communicating with the OpenAI API: {e}")
        return []



def main():
    parser = argparse.ArgumentParser(
        description="Annotate cell types and suggest analyses for a single-cell dataset."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input h5ad file.")
    parser.add_argument(
        "--marker_genes_json",
        type=str,
        help="JSON string of marker genes for validation. E.g., '{\"T cells\": [\"CD3D\", \"CD8A\"]}'"
    )
    args = parser.parse_args()

    # Create output directory based on input file name
    input_filename = os.path.basename(args.input_file)
    output_dir_name = os.path.splitext(input_filename)[0]
    output_dir = os.path.join(os.path.dirname(args.input_file), output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load the data
    adata = sc.read_h5ad(args.input_file)
    print(f"Loaded dataset: {args.input_file}")

    # Normalize gene symbols
    adata = normalize_gene_symbols(adata)
    print(f"\nSample of adata.var_names after normalization: {adata.var_names.tolist()[:10]}")

    # Normalize expression if not already log-normalized
    if 'log1p' not in adata.uns:
        print("Data is not log-normalized. Performing normalization...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print("Normalization complete.")
    else:
        print("Data is already log-normalized.")

    print("\n--- Starting Cell Type Annotation ---")
    columns = adata.obs.columns.tolist()
    print(f"Columns found: {columns}")

    cell_type_map = get_cell_type_columns(columns)
    if not cell_type_map:
        print("LLM did not identify any potential cell type columns based on names.")
        print("Refining analysis with data samples...")
        cell_type_map = refine_cell_type_columns(adata, columns)

    if not cell_type_map or 'error' in cell_type_map:
        print("No cell type columns identified after refining with data samples.")
    else:
        print(f"Identified cell type columns: {cell_type_map}")
        for old_name, new_name_llm in cell_type_map.items():
            if new_name_llm == "major_celltype":
                target_name = "cell_type"
            elif new_name_llm == "cell_subtype":
                target_name = "cell_subtype"
            else:
                target_name = new_name_llm

            if target_name and target_name not in adata.obs.columns:
                adata.obs.rename(columns={old_name: target_name}, inplace=True)
                print(f"Renamed column '{old_name}' to '{target_name}'")
            elif target_name:
                print(f"Column '{target_name}' already exists or is a duplicate. Skipping renaming of '{old_name}'.")

    # Save the annotated file
    output_annotated_file = os.path.join(output_dir, f"{output_dir_name}_annotated.h5ad")
    try:
        adata.write_h5ad(output_annotated_file)
        print(f"\nSaved annotated and normalized file to {output_annotated_file}")
    except Exception as e:
        print(f"Error saving annotated file: {e}")

    print("\n--- Starting Analysis Suggestion ---")
    suggestions = suggest_analyses(adata)
    if suggestions:
        print(json.dumps(suggestions, indent=2))
        output_suggestions_file = os.path.join(output_dir, f"{output_dir_name}_analysis_suggestions.json")
        with open(output_suggestions_file, 'w') as f:
            json.dump(suggestions, f, indent=2)
        print(f"Saved analysis suggestions to {output_suggestions_file}")
    else:
        print("No analysis suggestions generated.")

    # Marker gene validation
    computed_validation_results = validate_with_computed_markers(adata, MARKER_GENES_FROM_CSV)
    if computed_validation_results:
        print("\nComputed Marker Gene Validation Results:")
        print(json.dumps(computed_validation_results, indent=2))
        output_computed_validation_file = os.path.join(output_dir, f"{output_dir_name}_computed_validation_results.json")
        with open(output_computed_validation_file, 'w') as f:
            json.dump(computed_validation_results, f, indent=2)
        print(f"Saved computed validation results to {output_computed_validation_file}")
    else:
        print("No computed validation results generated.")

    print("\n--- Identifying condition columns for DEG analysis using LLM ---")
    condition_columns = get_condition_columns_from_llm(adata.obs.columns.tolist())
    print(f"LLM identified condition columns: {condition_columns}")

    for col in condition_columns:
        if col in adata.obs.columns:
            print(f"\nUnique values in identified condition column '{col}': {adata.obs[col].unique().tolist()}")
            deg_results = run_differential_expression_analysis(
                adata,
                col,
                output_dir,
                pvalue_threshold=0.05,
                logfc_threshold=0.5
            )
            if deg_results:
                print(f"\nDifferential Expression Analysis Results for {col}:")
                # print(json.dumps(deg_results, indent=2)) # This can be very long
                output_deg_file = os.path.join(
                    output_dir,
                    f"{output_dir_name}_deg_results_{col.replace('[', '_').replace(']', '_').replace(' ', '_')}.json"
                )
                with open(output_deg_file, 'w') as f:
                    json.dump(deg_results, f, indent=2)
                print(f"Saved DEG results to {output_deg_file}")
            else:
                print(f"No DEG results generated for {col}.")
        else:
            print(f"Condition column '{col}' not found in metadata. Skipping DEG for this column.")




if __name__ == "__main__":
    main()
