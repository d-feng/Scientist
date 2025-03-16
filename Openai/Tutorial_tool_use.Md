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

