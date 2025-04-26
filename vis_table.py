import streamlit as st
import pandas as pd
from transformers import pipeline
import os
import openai
import os
from dotenv import load_dotenv
load_dotenv(".env")
my_api_key = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=my_api_key)

def process_with_llm(combined_answers):
    """
    Process the combined answers with GPT-4 to generate a summary.
    """
    input="".join(combined_answers)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Based on the information of '{input}', generate a summary."}
            ]
        )
        # Extract the generated summary from the response
        generated_summary = response.choices[0].message.content.strip()
        return generated_summary
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Load LLM for text processing
#llm = pipeline("text-generation", model="gpt-3.5-turbo", max_length=150)

# Title of the app
st.title("Interactive Q&A App")

# File upload or default handling
uploaded_file = st.file_uploader("Upload your input file (Excel)", type=["xlsx"])

try:
    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name="Output")
        st.write("Data Preview (Uploaded File - Sheet 'Output'):")
    else:
        default_file = "input_qa.xlsx"  # Ensure this file exists in the same directory
        if os.path.exists(default_file):
            df = pd.read_excel(default_file, sheet_name="Output")
            st.write("Data Preview (Default File - Sheet 'Output'):")
        else:
            st.error(f"No file uploaded and default file '{default_file}' is missing!")
            st.stop()
    
    st.dataframe(df)

    # Selection options for Summary_
    summary_options = [col for col in df.columns if col.startswith("Summary_")]
    selected_summary = st.selectbox("Select a Summary column to work with:", summary_options)

    if selected_summary:
        # Display Evidence
        if st.button("Evidence"):
            st.write("Selected Summary Column and Questions:")
            st.dataframe(df[[selected_summary, "question"]])

        
        # Generate single paragraph answer
        if st.button("Generation"):
            selected_data = df[selected_summary].dropna().tolist()
            combined_answers = " ".join(selected_data)
            st.write("Combined Answers:")
            st.write(combined_answers)

            # Process with LLM
            llm_result = process_with_llm(combined_answers)
            st.write("LLM Processed Result:")
            st.write(llm_result)



except ValueError as e:
    st.error(f"An error occurred: {str(e)}")
