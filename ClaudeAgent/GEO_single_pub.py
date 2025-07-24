############### Test 01 #########################
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import io
import base64
# Standard Library
import os
import time
import re
import json
import gzip
import shutil
import tarfile
import subprocess
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from urllib.parse import urljoin, urlparse, parse_qs, unquote
from collections import defaultdict

# Third-Party Data Science Libraries
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# OpenAI/LLM Libraries
from openai import OpenAI
import openai

# Jupyter/IPython
from IPython.display import display, Markdown

# Bioinformatics Libraries
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
import gseapy as gp
from geofetch import Geofetcher
import scanpy as sc
from sanbomics.plots import volcano

# LangChain/LangGraph
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda

# Environment Configuration
from dotenv import load_dotenv


API_KEY= ''
os.environ['OPENAI_API_KEY']=API_KEY
client = OpenAI()



def get_series_matrix_url(gse_id: str) -> str:
    """
    Find the final HTTP URL of the actual .series_matrix.txt.gz file
    for a given GEO study by first finding the FTP matrix directory,
    then scraping the listing page for the correct file.

    Args:
        gse_id (str): e.g. "GSE94840"

    Returns:
        str: Final download URL, or None if not found.
    """
    base_geo_url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
    response = requests.get(base_geo_url + gse_id)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    ftp_matrix_url = None

    # Step 1: find ftp:// link for Series Matrix File(s)
    for a in soup.find_all("a", href=True):
        href = a['href']
        if href.startswith("ftp://") and "matrix" in href and gse_id in href:
            ftp_matrix_url = href
            break

    if not ftp_matrix_url:
        return None

    # Step 2: Convert FTP to HTTPS to scrape the file listing page
    https_matrix_dir = ftp_matrix_url.replace("ftp://", "https://")

    response = requests.get(https_matrix_dir)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for a in soup.find_all("a", href=True):
        if a['href'].endswith("series_matrix.txt.gz"):
            # Return full HTTPS URL to the file
            return https_matrix_dir + a['href']

    return None

def download_series_matrix(gse_id, output_dir="geofetch_metadata"):
    url = get_series_matrix_url(gse_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{gse_id}_series_matrix.txt.gz"
    if output_path.exists():
        return str(output_path)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return str(output_path)
    else:
        raise FileNotFoundError(f"Could not download series matrix for {gse_id}")
def find_and_merge_count_files(gse_id: str, base_dir="rna_seq_analysis") -> pd.DataFrame:
    gse_dir = Path(base_dir) / gse_id
    if not gse_dir.exists():
        print(f"❌ GSE folder not found: {gse_dir}")
        return None

    patterns = ["*.csv", "*.tsv", "*.txt", "*.xlsx", "*.gz"]
    files = []
    seen_uncompressed = set()
    for pattern in patterns:
        for f in gse_dir.glob(pattern):
            if f.suffix == ".gz":
                uncompressed_name = f.with_suffix("")
                if uncompressed_name.name in seen_uncompressed:
                    print(f"⚠️ Skipping compressed duplicate: {f.name}")
                    continue
            else:
                seen_uncompressed.add(f.name)
            files.append(f)

    if not files:
        print(f"⚠️ No count files found in {gse_dir}")
        return None

    print(f"🔍 Found {len(files)} count files for {gse_id}")

    # Handle single uncompressed file
    uncompressed_files = [f for f in files if f.suffix != ".gz"]
    if len(uncompressed_files) == 1:
        file = uncompressed_files[0]
        print(f"📄 Only one uncompressed count file found: {file.name} — attempting to load without merging.")
        try:
            if file.suffix == ".csv":
                df = pd.read_csv(file)
            elif file.suffix == ".xlsx":
                df = pd.read_excel(file)
            else:  # .txt or .tsv
                df = pd.read_csv(file, sep="\t")

            #df.columns = df.columns.astype(str)

            # Set first column as gene index
            #gene_col = df.columns[0]
            #df.set_index(gene_col, inplace=True)

            # Keep only numeric sample columns
            #numeric_df = df.apply(pd.to_numeric, errors='coerce')
            #numeric_df = numeric_df.dropna(axis=1, how='all')  # Drop fully non-numeric
            #numeric_df = numeric_df.fillna(0).astype(int)
  
            df.columns = df.columns.astype(str)

            # Set the first column (likely gene names) as index
            gene_col = df.columns[0]
            df.set_index(gene_col, inplace=True)

                        
            annotation_keywords = {"gene_length", "length", "start", "end", "strand", "chromosome", "chr"}
            
            # Identify numeric sample columns and exclude known annotation fields
            numeric_cols = []
            for col in df.columns:
                col_lower = col.strip().lower()
                if any(key in col_lower for key in annotation_keywords):
                    continue
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_cols.append(col)
                except Exception:
                    continue
            
            # Filter to only good sample columns
            df = df[numeric_cols]
            
            # Final numeric conversion and zero fill
            numeric_df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)          
             
            
            
            # Conditionally apply +1 normalization if any zeros are present
            if (numeric_df == 0).any().any():
                print("⚠️ Zero counts detected — applying +1 normalization.")
                numeric_df += 1
            else:
                print("✅ No zero counts — no normalization applied.")

            print(f"✅ Single count file processed. Shape: {numeric_df.shape}")
            return numeric_df

        except Exception as e:
            print(f"❌ Failed to load single count file: {e}")
            return None

    # Proceed with merging multiple files
    merged_df = None
    used_column_names = set()

    for file in files:
        try:
            sample_base = file.stem.split("_")[0] if "_" in file.stem else file.stem

            if file.suffix == ".gz":
                with gzip.open(file, 'rt') as f:
                    df = pd.read_csv(f, sep="\t", header=0)
            elif file.suffix == ".csv":
                df = pd.read_csv(file, header=0)
            elif file.suffix == ".xlsx":
                df = pd.read_excel(file, header=0)
            else:
                df = pd.read_csv(file, sep="\t", header=0)

            if df.shape[1] < 2:
                raise ValueError(f"{file.name} has fewer than 2 columns, cannot extract count data.")

            df.columns = ['Gene'] + [f"V{i}" for i in range(1, df.shape[1])]
            df.set_index('Gene', inplace=True)
            df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
            df = df.iloc[:, [0]]  # First data column only

            col_name = sample_base
            counter = 1
            while col_name in used_column_names:
                col_name = f"{sample_base}_{counter}"
                counter += 1
            used_column_names.add(col_name)
            df.columns = [col_name]

            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.join(df, how="outer")

        except Exception as e:
            print(f"❌ Failed to read {file.name}: {e}")

    if merged_df is not None:
        merged_df = merged_df.fillna(0).astype(int)

        if (merged_df == 0).any().any():
            print("⚠️ Zero counts detected in merged matrix — applying +1 normalization.")
            merged_df += 1
        else:
            print("✅ No zero counts — no normalization applied.")

        print(f"✅ Final merged count matrix shape: {merged_df.shape}")
        return merged_df
    else:
        print("❌ No valid count data could be merged.")
        return None
def scrape_geo_supplementary_downloads(gse_id: str):
    """
    Scrape the GEO page for a given GSE ID and return a list of supplementary
    files that are .tar, .gz, or .tar.gz with valid HTTP download links.

    Returns:
        List of tuples: (file_name, http_download_url)
    """
    base_url = "https://www.ncbi.nlm.nih.gov"
    url = f"{base_url}/geo/query/acc.cgi?acc={gse_id}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table")

        for table in tables:
            if "Supplementary file" in table.text:
                rows = table.find_all("tr")
                files = []

                for row in rows[1:]:  # skip header
                    cols = row.find_all("td")
                    if len(cols) >= 3:
                        file_name = cols[0].text.strip()
                        if any(ext in file_name for ext in [".tar", ".gz","XLSX"]):
                            all_links = cols[2].find_all("a", href=True)
                            for a in all_links:
                                href = a["href"]
                                if href.startswith("/geo/download"):
                                    files.append((file_name, urljoin(base_url, href)))
                                    break
                                elif href.startswith("http"):
                                    files.append((file_name, href))
                                    break

                return files

        return []

    except Exception:
        return []



def download_and_extract_gse(gse_id, base_dir="rna_seq_analysis"):
    """
    Downloads and extracts supplementary files for a given GSE.
    """
    gse_dir = Path(base_dir) / gse_id
    gse_dir.mkdir(parents=True, exist_ok=True)

    files = scrape_geo_supplementary_downloads(gse_id)
    if not files:
        print(f"⚠️ No supplementary files found for {gse_id}")
        return str(gse_dir)

    file_name, url = files[0]
    if not file_name:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        file_name = unquote(qs.get("file", ["unnamed_file"])[0])

    file_path = gse_dir / file_name
    print(f"📥 Downloading {file_name} → {file_path.as_posix()}")

    try:
        subprocess.run(["curl", "-L", "-o", file_path.as_posix(), url], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed for {gse_id}: {e}")
        return str(gse_dir)

    try:
        if tarfile.is_tarfile(file_path):
            print("🗂️ Extracting TAR...")
            with tarfile.open(file_path, "r:*") as tar:
                tar.extractall(path=gse_dir)
        elif file_path.suffix == ".gz" and not file_path.name.endswith(".tar.gz"):
            unzipped_path = file_path.with_suffix("")
            print("🗂️ Extracting GZ...")
            with gzip.open(file_path, 'rb') as f_in:
                with open(unzipped_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        print(f"⚠️ Extraction failed for {gse_id}: {e}")
def send_mini_table_to_llm(client, gse_id, series_df):
    mini_df = series_df.sample(n=min(6, len(series_df)), random_state=42)
    records = mini_df.reset_index().to_dict(orient="records")

    prompt = f"""
You are a biomedical data curator assisting with RNA-seq DESeq2 analysis.

Below is a compact metadata table for GEO study {gse_id}.
Each row represents a sample (geo_accession). The columns represent metadata fields.

Your task:
- Identify the columns (metadata fields) that contain key information useful for defining groups for differential expression analysis.
- Focus on metadata about: *treatment*, *time*, *gender*, *dose*, *genotype*, *WT/KO*.
- "control" often labelled as control, mock, media, DMSO, untreated, blank etc. No extra text, no explanation.
- Return ONLY a JSON list of column names that are useful. No extra text, no explanation.

Here is the table:
{json.dumps(records, indent=2)}
"""
    print(f"📝 Sending prompt (first 500 chars): {prompt[:500]}...")
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```json", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"```$", "", content).strip()

    try:
        col_list = json.loads(content)
        print(f"✅ LLM returned {len(col_list)} useful columns from original metadata")
        return col_list
    except json.JSONDecodeError:
        print("❌ Failed to parse LLM output. Raw content preview:")
        print(content[:500])
        raise

def parse_series_matrix_to_dataframe(series_matrix_path):
    data = {}
    sample_ids = None
    field_counter = defaultdict(int)

    open_func = gzip.open if series_matrix_path.endswith(".gz") else open
    with open_func(series_matrix_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.startswith("!Sample_"):
                parts = line.strip().split("\t")
                raw_field = parts[0].lstrip("!")
                count = field_counter[raw_field]
                field_name = raw_field if count == 0 else f"{raw_field}_{count}"
                field_counter[raw_field] += 1
                values = [p.strip('"') for p in parts[1:]]
                data[field_name] = values
                if raw_field == "Sample_geo_accession":
                    sample_ids = values

    if sample_ids is None:
        raise ValueError("❌ No Sample_geo_accession found.")

    df = pd.DataFrame(data)
    df.columns = df.columns.str.replace(r'\\', '', regex=True).str.strip()
    df.index = sample_ids
    df.index.name = "geo_accession"
    print(f"✅ Series matrix parsed: {df.shape}")
    return df
def decompress_gse_in_dir(gse_id: str, root_dir: str = "rna_seq_analysis"):
    """
    Decompress .gz and .tar files only if no uncompressed files exist
    in the subdirectory for the specific GSE.
    """
    gse_path = os.path.join(root_dir, gse_id)
    print(f"🔍 Scanning for compressed files in: {gse_path}")
    if not os.path.exists(gse_path):
        print(f"⚠️ Directory does not exist: {gse_path}")
        return

    filenames = os.listdir(gse_path)
    if any(f.lower().endswith(('.csv', '.tsv', '.txt', '.xlsx')) for f in filenames):
        return

    for fname in filenames:
        full_path = os.path.join(gse_path, fname)
        if tarfile.is_tarfile(full_path):
            try:
                print(f"📦 Extracting TAR: {full_path}")
                with tarfile.open(full_path, "r:*") as tar:
                    tar.extractall(path=gse_path)
            except Exception as e:
                print(f"❌ Failed to extract TAR {fname}: {e}")
        elif fname.endswith(".gz") and not fname.endswith(".tar.gz"):
            out_path = os.path.join(gse_path, fname[:-3])
            if os.path.exists(out_path):
                print(f"⚠️ Skipping (already exists): {out_path}")
                continue
            try:
                print(f"🗜️ Decompressing GZ: {full_path} → {out_path}")
                with gzip.open(full_path, 'rb') as f_in:
                    with open(out_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                print(f"❌ Failed to decompress GZ {fname}: {e}")


def derive_condition_with_llm(client, gse_id, metadata_df, useful_cols):
    df_for_llm = metadata_df[useful_cols].copy()
    df_for_llm["geo_accession"] = metadata_df.index
    records = df_for_llm.to_dict(orient="records")

    prompt = f"""
You are a biomedical data curator assisting with RNA-seq DESeq2 analysis.

Below is metadata for GEO study {gse_id}. Each row represents a sample.

Your task:
- Determine if each sample represents a "treated" or "control" condition.
- Use the provided metadata fields (treatment, time, genotype, etc).
- "control" often labelled as control, mock, media, DMSO, untreated, blank etc. No extra text, no explanation.
- alternatively, we can compare two condition: etc primary infection,secondary infection, primary infection as control, secondary infection as treatment
- Return ONLY a JSON mapping geo_accession → "treated with X" or "control" or "treated with Y, or other based on key metadata field. No explanation.

Here is the metadata:
{json.dumps(records, indent=2)}
"""
    print(f"📝 Sending LLM condition prompt (first 500 chars): {prompt[:500]}...")
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```json", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"```$", "", content).strip()

    try:
        condition_map = json.loads(content)
        print(f"✅ LLM returned conditions for {len(condition_map)} samples")
    except json.JSONDecodeError:
        print("❌ Failed to parse LLM output. Raw content preview:")
        print(content[:500])
        raise

    # Apply condition map correctly (do not use dict as indexer)
    metadata_df["condition"] = metadata_df.index.to_series().map(lambda x: condition_map.get(x, None))
    print(f"✅ derived condition LLM construct design {metadata_df} from minitable metadata")
    return metadata_df

def remove_uniform_columns(df):
    nunique = df.nunique(dropna=False)
    non_uniform_cols = nunique[nunique > 1].index
    cleaned_df = df[non_uniform_cols].copy()
    print(f"✅ Removed uniform columns: {df.shape[1] - cleaned_df.shape[1]} dropped, {cleaned_df.shape[1]} kept")
    return cleaned_df

def run_deg_for_single_gse_run_1(state, client):
    gse_id = state.get("selected_gse")
    if not gse_id:
        print("❌ No selected GSE found.")
        return state

    deg_paths = {}
    series_matrix_path = f"geofetch_metadata/{gse_id}_series_matrix.txt.gz"
    decompress_gse_in_dir(gse_id)

    try:
        series_df = parse_series_matrix_to_dataframe(series_matrix_path)
        series_df = remove_uniform_columns(series_df)

        useful_cols = send_mini_table_to_llm(client, gse_id, series_df)
        metadata_df = series_df[useful_cols].copy()
        metadata_df.index.name = "sample_geo_accession"

        metadata_df = derive_condition_with_llm(client, gse_id, metadata_df, useful_cols)
        print(f"llm processde metadata: {metadata_df}")
        download_dir = download_and_extract_gse(gse_id)
        counts_df = find_and_merge_count_files(gse_id)
        if counts_df is None:
            print(f"⚠️ No count matrix found for {gse_id}")
            return state

        valid_samples = metadata_df.index.intersection(counts_df.columns)
        counts_df = counts_df[valid_samples].T
        metadata_df = metadata_df.loc[valid_samples]
        
        #metadata_df = metadata_df[metadata_df["condition"].isin(["treated", "control"])]

        if len(metadata_df["condition"].unique()) < 2:
            raise ValueError(f"❌ Need both treated and control groups for DESeq2, found: {metadata_df['condition'].unique()}")

        # Dynamically determine contrast
        cond_values = metadata_df["condition"].dropna().unique()
        treated_group, control_group = cond_values[0], cond_values[1]
        print(f"✅ Using contrast: {treated_group} vs {control_group}")

        print(f"✅ Counts shape: {counts_df.shape}")
        print(f"✅ Metadata shape: {metadata_df.shape}")
        print(f"✅ Condition counts:\n{metadata_df['condition'].value_counts()}")

        inference = DefaultInference(n_cpus=4)
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata_df,
            design_factors="condition",
            refit_cooks=True,
            inference=inference
        )
        dds.fit_size_factors()
        dds.fit_genewise_dispersions()
        dds.fit_dispersion_trend()
        dds.fit_dispersion_prior()
        dds.fit_MAP_dispersions()
        dds.fit_LFC()
        dds.calculate_cooks()
        if dds.refit_cooks:
            dds.refit()

        stat_res = DeseqStats(
            dds,
            alpha=0.05,
            contrast=["condition", treated_group, control_group],
            cooks_filter=True,
            independent_filter=True
        )
        stat_res.summary()

        res = stat_res.results_df
        output_csv = f"rna_seq_analysis/{gse_id}_deseq2_results.csv"
        res.to_csv(output_csv, index=False)
        deg_paths[gse_id] = output_csv
        print(f"✅ DESeq2 results saved: {output_csv}")

        sig_genes = res[(res["padj"] < 0.05) & (abs(res["log2FoldChange"]) > 1)]
        print(f"🧬 {sig_genes.shape[0]} significant genes found")
        # Additional visualization + analysis
        # Add log1p layer
        dds.layers["log1p"] = np.log1p(dds.layers["normed_counts"])
        
        # Select significant genes dynamically
        sigs = res[(res["padj"] < 0.05) & (res["log2FoldChange"].abs() > 1.5)]
        if sigs.empty:
            print("⚠️ No significant genes found for heatmap/volcano plot.")
        else:
            # Build grapher matrix
            dds_sigs = dds[:, sigs.index]
            grapher = pd.DataFrame(
                dds_sigs.layers["log1p"],
                index=dds_sigs.obs_names,
                columns=dds_sigs.var_names
            ).T
            grapher.index.name = "gene"
        
            # Dynamically generate column labels
            labeled_columns = [
                f"{sample}_{metadata_df.loc[sample, 'condition']}"
                for sample in grapher.columns
            ]
            grapher.columns = labeled_columns
        
            # Volcano plot
            res["symbol"] = res.index
            volcano(
                res,
                symbol="symbol",
                log2fc="log2FoldChange",
                pval_thresh=0.05,
                to_label=10,
                log2fc_thresh=0.75,
                colors=["dimgrey", "lightgrey", "green"]
            )
            plt.show()
        
            # Heatmap
            sns.clustermap(grapher, z_score=0, cmap="RdBu_r", col_cluster=False)
            plt.show()
        
            # PCA
            sc.tl.pca(dds)
            sc.pl.pca(dds, color="condition", size=200)

    except Exception as e:
        print(f"❌ Error: {e}")

    return {**state, "deg_results": deg_paths}

#gse_id="GSE242272"



def run_deg(gse_id, client=client):
    chat_log = []  # must be list of dicts with 'role' and 'content'
    try:
        chat_log.append({"role": "system", "content": f"▶️ Analyzing {gse_id}"})
        series_matrix_path = f"geofetch_metadata/{gse_id}_series_matrix.txt.gz"
        decompress_gse_in_dir(gse_id)

        chat_log.append({"role": "system", "content": f"📥 Parsing metadata..."})
        series_df = parse_series_matrix_to_dataframe(series_matrix_path)
        series_df = remove_uniform_columns(series_df)
        useful_cols = send_mini_table_to_llm(client, gse_id, series_df)
        metadata_df = series_df[useful_cols].copy()
        metadata_df.index.name = "sample_geo_accession"
        metadata_df = derive_condition_with_llm(client, gse_id, metadata_df, useful_cols)

        chat_log.append({"role": "system", "content": f"📥 Finding and merging count files..."})
        counts_df = find_and_merge_count_files(gse_id)
        if counts_df is None:
            raise ValueError("No count matrix found.")

        valid_samples = metadata_df.index.intersection(counts_df.columns)
        counts_df = counts_df[valid_samples].T
        metadata_df = metadata_df.loc[valid_samples]

        conditions = metadata_df["condition"].unique()
        if len(conditions) < 2:
            raise ValueError("At least two condition groups required.")

        treated, control = conditions[:2]
        chat_log.append({"role": "system", "content": f"📊 Running DESeq2: {treated} vs {control}"})

        inference = DefaultInference(n_cpus=4)
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata_df,
            design_factors="condition",
            refit_cooks=True,
            inference=inference
        )
        dds.fit_size_factors()
        dds.fit_genewise_dispersions()
        dds.fit_dispersion_trend()
        dds.fit_dispersion_prior()
        dds.fit_MAP_dispersions()
        dds.fit_LFC()
        dds.calculate_cooks()
        if dds.refit_cooks:
            dds.refit()

        stat_res = DeseqStats(
            dds,
            alpha=0.05,
            contrast=["condition", treated, control],
            cooks_filter=True,
            independent_filter=True
        )
        stat_res.summary()
        res = stat_res.results_df

        dds.layers["log1p"] = np.log1p(dds.layers["normed_counts"])
        sigs = res[(res["padj"] < 0.05) & (res["log2FoldChange"].abs() > 1.5)]
        if sigs.empty:
            raise ValueError("No significant genes for plotting.")

        dds_sigs = dds[:, sigs.index]
        grapher = pd.DataFrame(
            dds_sigs.layers["log1p"],
            index=dds_sigs.obs_names,
            columns=dds_sigs.var_names
        ).T

        labeled_cols = [f"{s}_{metadata_df.loc[s, 'condition']}" for s in grapher.columns]
        grapher.columns = labeled_cols
        grapher.index.name = "gene"

        chat_log.append({"role": "system", "content": f"✅ DESeq2 complete. {sigs.shape[0]} significant genes found."})
        return res, grapher, dds, chat_log

    except Exception as e:
        chat_log.append({"role": "system", "content": f"❌ Error: {str(e)}"})
        return None, None, None, chat_log
def create_volcano_plot(res):
    res["symbol"] = res.index
    fig, ax = plt.subplots(figsize=(6, 5))
    volcano(
        res,
        symbol="symbol",
        log2fc="log2FoldChange",
        pval_thresh=0.05,
        to_label=10,
        log2fc_thresh=0.75,
        colors=["dimgrey", "lightgrey", "green"],
        ax=ax
    )
    return fig

import numpy as np
import matplotlib.pyplot as plt

def volcano(
    df,
    symbol="symbol",
    log2fc="log2FoldChange",
    pval_thresh=0.05,
    log2fc_thresh=1.0,
    to_label=10,
    colors=("blue", "lightgrey", "red"),  # (down, neutral, up)
    ax=None
):
    """
    Plot a volcano plot with significance coloring and optional gene labels.

    Parameters:
        df (pd.DataFrame): DataFrame with at least `log2fc`, `padj`, and `symbol` columns.
        symbol (str): Column name for gene symbols.
        log2fc (str): Column name for log2 fold change.
        pval_thresh (float): Threshold for adjusted p-value significance.
        log2fc_thresh (float): Threshold for fold change significance.
        to_label (int): Number of top genes to annotate based on p-value.
        colors (tuple): Colors for (downregulated, neutral, upregulated).
        ax (matplotlib.axes.Axes, optional): Axis to plot into. If None, creates a new one.

    Returns:
        matplotlib.axes.Axes: The axis with the plot drawn on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["padj", log2fc])
    df["-log10_padj"] = -np.log10(df["padj"])

    # Assign color based on thresholds
    df["color"] = colors[1]  # default: neutral
    df.loc[(df[log2fc] < -log2fc_thresh) & (df["padj"] < pval_thresh), "color"] = colors[0]
    df.loc[(df[log2fc] > log2fc_thresh) & (df["padj"] < pval_thresh), "color"] = colors[2]

    # Scatter plot
    ax.scatter(df[log2fc], df["-log10_padj"], c=df["color"], s=10, alpha=0.7, edgecolor='none')

    # Label top genes
    top_genes = df.nsmallest(to_label, "padj")
    for _, row in top_genes.iterrows():
        ax.text(row[log2fc], row["-log10_padj"], row[symbol], fontsize=6, ha='right')

    # Threshold lines
    ax.axhline(-np.log10(pval_thresh), color="grey", linestyle="--", linewidth=1)
    ax.axvline(log2fc_thresh, color="grey", linestyle="--", linewidth=1)
    ax.axvline(-log2fc_thresh, color="grey", linestyle="--", linewidth=1)

    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10 Adjusted P-value")
    ax.set_title("Volcano Plot")

    return ax

def create_volcano_plot(res):
    res["symbol"] = res.index
    fig, ax = plt.subplots()
    volcano(res, symbol="symbol", log2fc="log2FoldChange", ax=ax)
    return fig


def create_heatmap(grapher):
    fig = plt.figure()
    clust = sns.clustermap(grapher, z_score=0, cmap="RdBu_r", col_cluster=False)
    plt.close()
    return clust.fig


def create_pca_plot(dds):
    sc.tl.pca(dds)
    sc.pl.pca(dds, color="condition", size=200, show=False)
    fig = plt.gcf()
    return fig


def pipeline(gse_id):
    res, grapher, dds, chat_log = run_deg(gse_id)
    if res is None:
        return None, None, None, chat_log
    fig1 = create_volcano_plot(res)
    fig2 = create_heatmap(grapher)
    fig3 = create_pca_plot(dds)
    return fig1, fig2, fig3, chat_log


demo = gr.Interface(
    fn=pipeline,
    inputs=gr.Textbox(label="GSE ID (e.g., GSE242272)"),
    outputs=[
        gr.Plot(label="Volcano Plot"),
        gr.Plot(label="Heatmap"),
        gr.Plot(label="PCA Plot"),
        gr.Chatbot(label="Log", type="messages")
    ],
    title="GEO DEG Plot Viewer",
    description="Enter a GEO GSE ID to run DESeq2 and visualize volcano plot, heatmap, and PCA."
)


if __name__ == "__main__":
    demo.launch(server_port=7861)
