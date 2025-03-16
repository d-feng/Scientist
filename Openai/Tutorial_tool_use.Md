# **Automated Deep Search for scRNA-seq Tools Using OpenAI Agents**

## **Overview**
This project utilizes OpenAI’s `GPT-4o` model with **web search capabilities** to find the latest bioinformatics tools for **single-cell RNA sequencing (scRNA-seq)**. It systematically searches for software related to:
- **Preprocessing & Quality Control**
- **Normalization & Batch Correction**
- **Clustering & Dimensionality Reduction**
- **Differential Expression Analysis**
- **Trajectory Inference & Pseudotime Analysis**
- **Gene Regulatory Networks & Cell-Cell Interactions**
- **Visualization & Interpretation**

The script **automates information retrieval** from **GitHub repositories, official documentation, and academic sources**, reducing the manual effort required for tool discovery.

---

## **Installation & Setup**
### **1. Install Dependencies**
Ensure you have Python installed, then install the required libraries:
```bash
pip install openai ipython nest_asyncio

##2. Set Up OpenAI API Key
Add your OpenAI API key in your environment:
```bash
import os
os.environ['OPENAI_API_KEY'] = "your_openai_api_key"

##3. Enable Async Execution in Jupyter
If running in a Jupyter Notebook:
```bash
import nest_asyncio
nest_asyncio.apply()
##Implementation
1. Import Required Modules
```bash
from agents import Agent, Runner
from IPython.display import display, Markdown
import os
from openai import OpenAI
```
# Initialize OpenAI Client
client = OpenAI()

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = "your_openai_api_key"

# Enable nested async execution in Jupyter
```bash
import nest_asyncio
nest_asyncio.apply()

##2. Define Category-Specific Search Prompts
The script uses structured prompts to find tools in various categories of scRNA-seq analysis:
```bash
prompts = """Category-Specific Deep Search Prompts
1. Preprocessing & Quality Control
"Find at least 10 tools for preprocessing and quality control of single-cell RNA-seq (scRNA-seq) data. Include tools for filtering low-quality cells, doublet detection, mitochondrial read filtering, and dataset quality assessment. Provide links to GitHub or official documentation."

2. Normalization & Batch Correction
"List at least 10 bioinformatics tools for normalization and batch effect correction in single-cell RNA sequencing (scRNA-seq) data. Include methods like SCTransform, Harmony, and MNN. Provide links to documentation and GitHub repositories."

3. Clustering & Dimensionality Reduction
"Search for at least 10 scRNA-seq tools specialized in clustering and dimensionality reduction. Include methods using PCA, t-SNE, UMAP, and graph-based clustering approaches. Provide official sources and GitHub repositories."

4. Differential Expression Analysis
"Identify 10+ software packages for performing differential expression analysis in single-cell RNA sequencing (scRNA-seq). Compare statistical models used in each tool. Include links to their GitHub repositories or official documentation."

5. Trajectory Inference & Pseudotime Analysis
"List at least 10 tools for trajectory inference and pseudotime analysis in single-cell RNA sequencing (scRNA-seq) datasets. Include methods such as Monocle, Slingshot, and SCORPIUS. Provide links to each tool’s documentation and GitHub."

6. Gene Regulatory Networks & Cell-Cell Interactions
"Find at least 10 computational methods for constructing gene regulatory networks and inferring cell-cell communication from single-cell RNA sequencing (scRNA-seq) data. Include approaches like SCENIC, CellChat, and NicheNet. Provide documentation links."

7. Visualization & Interpretation
"Gather at least 10 tools designed for visualization and exploration of single-cell RNA sequencing (scRNA-seq) datasets. Include interactive visualization software for gene expression, clustering, and trajectory analysis. Provide GitHub or official tool documentation."
"""
3. Execute OpenAI Web Search
The responses.create() function is used to call OpenAI’s GPT-4o with the web search tool enabled:
```bash
response = client.responses.create(
    model="gpt-4o",
    input=prompts,
    tools=[{"type": "web_search"}]
)
##4. Extract and Display Search Results
Once the search is complete, the results are extracted and displayed in a structured markdown format:
```bash

# Extract text content from response output
output_texts = [content.text for message in response.output for content in message.content]

# Combine into a single formatted string
formatted_text = "\n\n".join(output_texts)

# Display formatted markdown output in Jupyter Notebook
display(Markdown(formatted_text))
##Results
The script retrieves categorized lists of scRNA-seq tools, providing:

Tool Names
Features
GitHub Repositories
Official Documentation Links

✔️ Example Output (Rendered in Jupyter)
```bash
### Preprocessing & Quality Control
1. **Cell Ranger**  
   - Features: Filtering low-quality cells and mitochondrial content.  
   - Documentation: [10x Genomics](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/what-is-cell-ranger)  
   - GitHub: [Cell Ranger](https://github.com/10XGenomics/cellranger)  

2. **Scrublet**  
   - Features: Doublet detection.  
   - Documentation: [Scrublet GitHub](https://github.com/swolock/scrublet)  
   - GitHub: [Scrublet](https://github.com/swolock/scrublet)  

3. **scater**  
   - Features: Quality control and normalization.  
   - Documentation: [Bioconductor](https://bioconductor.org/packages/release/bioc/html/scater.html)  
   - GitHub: [scater](https://github.com/dufkot/scater)  

Advantages of This Approach
✅ Automates Literature Review – Reduces manual searching effort.
✅ Structured & Targeted Search – Uses category-specific queries.
✅ Fetches Latest Tools – Retrieves real-time GitHub repositories & documentation.
✅ Markdown-Formatted Output – Perfect for Jupyter Notebooks.

Future Enhancements
🔹 Expand Search Scope – Extend queries to cover other omics technologies.
🔹 Automate API Calls – Fetch real-time data from bioinformatics APIs.
🔹 Create Interactive Dashboards – Convert markdown results into interactive tables.

Conclusion
This project demonstrates how OpenAI’s GPT-4o can be used for automating tool discovery in bioinformatics. By leveraging structured prompts and OpenAI’s web search, researchers can quickly identify and access the latest scRNA-seq analysis tools, reducing the time spent on manual literature reviews.
