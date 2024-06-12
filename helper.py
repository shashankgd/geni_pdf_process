import csv
import os
import tempfile

import pdfplumber
from PIL import Image
import re
import PyPDF2
import openai
from openai import OpenAI
import pandas as pd
from io import StringIO


# Extract all headings from the given PDF file
def extract_all_headings(file_path):
    headings = []
    pdf = open(file_path, "rb")
    reader = PyPDF2.PdfReader(pdf)

    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()  # Extract text from the page
        lines = text.split("\n")  # Split text into lines
        for line in lines:
            line = line.strip()  # Remove leading and trailing whitespace
            # Check if the line is likely a heading
            if line and (line[0].isdigit() and len(line) > 2) and (line[1].isspace() or line[1] == '.'):
                headings.append(line.strip())  # Append line to headings if it matches the pattern

    pdf.close()

    # Filter out invalid headings
    valid_headings = [h for h in headings if is_valid_heading(h)]

    return valid_headings


# Check if a line is a valid heading
def is_valid_heading(line):
    return re.match(r'^\d+(\.\d+)* ', line) and ',' not in line


# Create key-value pairs of sections and their respective next headings
def create_key_value_pairs(valid_headings):
    # Get the hierarchical level of the heading
    def get_level(heading):
        level = len(list(map(int, re.match(r'^\d+(\.\d+)*', heading).group(0).split('.'))))
        return level

    sections = {}

    for i, heading in enumerate(valid_headings):
        current_level = get_level(heading)
        sections[heading] = None
        k = i + 1
        if k >= len(valid_headings):
            break
        next_heading = valid_headings[k]
        next_level = get_level(next_heading)

        # Determine the relationship between current heading and next heading
        if current_level == next_level:
            sections[heading] = next_heading

        elif current_level > next_level:
            sections[heading] = next_heading

        elif next_level > current_level:
            n = next_level
            c = current_level
            j_int = i + 1
            while n > c and j_int < len(valid_headings):
                next_heading = valid_headings[j_int]
                next_level = get_level(next_heading)
                if c >= next_level:
                    sections[heading] = next_heading
                j_int = j_int + 1
                n = next_level

    return sections


# Find the start and end boundaries of a section
def find_section_boundaries(section_heading, sections):
    section_start, section_end = None, None
    for sec in sections.keys():
        if section_heading in sec:
            section_start = sec
            section_end = sections[sec]
            break

    return section_start, section_end


def get_pdf_sections(pdf_path):
    # Extract all headings from the PDF
    headings = extract_all_headings(pdf_path)
    # Create key-value pairs of sections and their respective next headings
    sections = create_key_value_pairs(headings)
    return sections


# Crop the specified section of the PDF and save it as an image
def crop_section_as_image(pdf_path, section_heading, sections, resolution=300):
    # Find the boundaries of the specified section
    section_start, section_end = find_section_boundaries(section_heading, sections)

    if section_start is None:
        msg = f"Section '{section_heading}' not found in the document."
        return msg
    print(section_start, section_end)
    output_image_path = f"{section_start}.png"

    start_found = False
    end_found = False
    crop_boxes = []
    images = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True, x_tolerance=1)
            words = page.extract_words(x_tolerance=1)
            if len(words) < 2:
                continue
            if len(images) > 0 and section_end not in text:
                x0 = 0
                top = words[0]['top']
                bottom = words[-1]['bottom']
                crop_boxes[-1] = (crop_boxes[-1][0], crop_boxes[-1][1], crop_boxes[-1][2], bottom)
                crop_boxes.append((x0, top, page.width, bottom))
                cropped_page = page.within_bbox(crop_boxes[-1]).to_image(resolution=resolution).original.convert("RGBA")
                images.append(cropped_page)
            else:
                for j in range(len(words)):
                    if not start_found:
                        word_list = [word['text'] for word in words[j:j + len(section_start.split())]]
                        if word_list == section_start.split():
                            # Section start found, set the coordinates
                            x0 = 0
                            top = words[j]['top']
                            x1 = page.width
                            bottom = words[-1]['bottom']
                            start_found = True

                    if section_end and section_end in text:
                        word_list = [word['text'] for word in words[j:j + len(section_end.split())]]
                        if word_list == section_end.split():
                            # Section end found, adjust the bottom coordinate
                            bottom = words[j]['bottom'] - 1

                            if section_start not in text:
                                x0 = 0
                                top = words[0]['top']
                                x1 = page.width
                            end_found = True

                if start_found or end_found:
                    crop_boxes.append((x0, top, x1, bottom))
                    cropped_page = page.within_bbox(crop_boxes[-1]).to_image(resolution=resolution).original.convert(
                        "RGBA")
                    images.append(cropped_page)
                if start_found and end_found:
                    break

        if images:
            # Combine all cropped images into one
            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)
            combined_image = Image.new('RGBA', (max_width, total_height))

            y_offset = 0
            for img in images:
                combined_image.paste(img, (0, y_offset))
                y_offset += img.height

            combined_image.save(output_image_path, format='PNG')
            print(f"Section '{section_heading}' saved as high-quality image to {output_image_path}")
            return output_image_path
        else:
            msg = f"Section '{section_heading}' not found in the entire document."
            return msg


# Function to send text to OpenAI GPT model and get a response
def get_table_from_text(text, api_key):
    # print(text)
    openai.api_key = api_key
    client = OpenAI(api_key=api_key)

    prompt = (
        "Extract the table from the following text. If there is no table, respond with 'No table found'. "
        "Provide the table in CSV format:\n\n"
        f"{text}"
    )
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.3,
    )

    return response.choices[0].text.strip()


def is_valid_csv(csv_string):
    try:
        df = pd.read_csv(StringIO(csv_string))
        # Check if the DataFrame has more than one row and one column
        return not df.empty and df.shape[1] > 1
    except pd.errors.ParserError:
        return False


# Function to save CSV string to a CSV file
def save_csv(csv_string, output_path):
    with open(output_path, 'w') as file:
        file.write(csv_string)


def get_section_text(section_name, sections, temp_file_path):
    section_start, section_end = find_section_boundaries(section_name, sections)

    capture = False
    captured_lines = []
    with pdfplumber.open(temp_file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True, x_tolerance=1)
            if not text:
                continue

            for line in text.split('\n'):
                if section_start in line and not capture:
                    capture = True
                    captured_lines.append(line)
                    continue
                if section_end in line:
                    capture = False
                    break  # Stop capturing once the end of the section is found
                elif capture:
                    captured_lines.append(line)

        section_text = '\n'.join(captured_lines)
        return section_text


def save_to_csv(section_name, sections, temp_file_path, openai_api_key):
    # Function to extract text from a single page of a PDF
    section_text = get_section_text(section_name, sections, temp_file_path)
    if section_text is None:
        msg = 'Not able to locate section'
        return msg
    csv_string = get_table_from_text(section_text, openai_api_key)
    output_csv_path = "output.csv"
    if csv_string.lower() == "no table found" or not is_valid_csv(csv_string):
        msg = "No table found in the PDF."
    else:
        save_csv(csv_string, output_csv_path)
        msg = f"CSV saved to {output_csv_path}"
    return msg
