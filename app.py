import csv
import streamlit as st
import os
import tempfile
from helper import get_pdf_sections, crop_section_as_image, save_to_csv

# Placeholder functions
def process_pdf(file_path):
    print(file_path)
    sections = get_pdf_sections(file_path)
    st.write("PDF processed")
    return sections

def search_section(section_name, sections, temp_file_path):
    image = crop_section_as_image(temp_file_path, section_name, sections=sections, resolution=300)
    if 'not found' in image:
        st.write(image)
    else:
        st.image(image, caption="Search Result")

def save_table(section_name, sections, temp_file_path, openai_api_key):
    msg = save_to_csv(section_name, sections, temp_file_path, openai_api_key)
    st.write(msg)

# Streamlit app
def main():
    st.title("PDF Processing Streamlit App")
    sections = {}
    temp_file_path = None
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Upload PDF section
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Call the process function
        sections = process_pdf(temp_file_path)
        st.session_state.pdf_processed = True

    # Search section
    if st.session_state.get('pdf_processed'):
        search_str = st.text_input("Search Section")
        if st.button("Search"):
            search_section(search_str, sections, temp_file_path)

    # Save table section
    if st.session_state.get('pdf_processed'):
        section_name = st.text_input("Enter Section Name")
        if not openai_api_key:
            openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

        if st.button("Save Table"):
            if openai_api_key:
                save_table(section_name, sections, temp_file_path, openai_api_key)
            else:
                st.warning("Please enter OpenAI API Key first")

if __name__ == "__main__":
    main()

