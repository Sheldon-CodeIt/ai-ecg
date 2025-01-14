import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from fpdf import FPDF  # Importing fpdf for PDF generation
import tempfile
import pymupdf as fitz

# Load environment variables
load_dotenv()

os.getenv("GEMINI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# System prompt to guide Gemini in summarizing ECG reports
ecg_summary_system_prompt = """
You are given ECG report data. Your task is to summarize it in a structured format based on the following template:
- Key findings such as Heart Rate, QRS Duration, QT Interval, Corrected QT Interval, PR Interval, and P-R-T Angles should be clearly extracted.
- Each finding should be explained with its range and an interpretation of what it might indicate about the health condition.
- Finish with an overall summary of the findings in 3 lines.

Example:
Based on the ECG report for Michael P Mascarenhas, here are the key findings and what they might indicate:

Heart Rate (VR): 71 bpm
This is within the normal resting heart rate range for adults (60-100 bpm).

QRS Duration (QRSD): 88 ms
This is within the normal range (less than 120 ms), indicating normal ventricular depolarization.

QT Interval (QT): 384 ms
This is within the normal range for men (less than 450 ms), indicating normal ventricular repolarization.

Corrected QT Interval (QTcB): 416 ms
This is also within the normal range, which is important for assessing the risk of arrhythmias.

PR Interval (PRI): 220 ms
This is slightly prolonged (normal is 120-200 ms), which might indicate first-degree heart block. This condition is usually benign but should be monitored.

P-R-T Angles: 51° NA 53°
These angles provide information about the electrical axis of the heart. The values here are within normal limits.

Summary:
The ECG report shows mostly normal findings with a slightly prolonged PR interval, which might indicate a first-degree heart block. This condition is generally not serious but should be monitored by a healthcare provider.
"""

# Initialize the Gemini model with the system prompt
ecg_summary_model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest',
                                          system_instruction=ecg_summary_system_prompt)

# Helper function to format extracted text into the desired format
def format_output(pdf_name, extracted_text):
    print(extracted_text)
    lines = extracted_text.splitlines()
    summary = f"Based on the extracted text from {pdf_name}, here are the key findings:\n\n"

    # Initialize variables
    id = ""
    name = ""
    age_gender = ""
    date = ""

    # Variables to handle multi-line issues
    temp_name = ""
    temp_date = ""
    temp_age_gender = ""

    # Extract name, age, gender, and date
    for i, line in enumerate(lines):
        if "Patient ID" in line:
            temp_id = line.split(":")[-1].strip()  # Initial id split
            # Only add the next line if it contains additional information for the id
            if i + 1 < len(lines) and lines[i + 1].strip():
                temp_id += " " + lines[i + 1].strip()  # Add the next line to the id
            id = temp_id.strip()  # Set the id

        if "Patient Name" in line:
            temp_name = line.split(":")[-1].strip()  # Initial name split
            # Check if the name is split across multiple lines or not
            if i + 1 < len(lines) and lines[i + 1].strip() and not ":" in lines[i + 1]:
                temp_name += " " + lines[i + 1].strip()  # Add the next line to the name only if it's a continuation
            name = temp_name.strip()  # Set the name

        if "Age / Gender" in line:
            temp_age_gender = line.split(":")[-1].strip()  # Initial name split
            # Check if the name is split across multiple lines or not
            if i + 1 < len(lines) and lines[i + 1].strip() and not ":" in lines[i + 1]:
                temp_age_gender += " " + lines[i + 1].strip()  # Add the next line to the name only if it's a continuation
            age_gender = temp_age_gender.strip()  # Set the name
        if "Date and Time" in line:
            # For Date and Time, handle it carefully to capture the full date and time
            temp_date = line.split(":", 1)[-1].strip()  # Split only at the first colon
            date = temp_date.strip()  # Set the date

        # Extract Heart Rate
        elif "Heart Rate" in line or "HR" in line:
            summary += f"Heart Rate: {line.split(':')[-1].strip()} bpm\n"

        # Extract PR Interval
        elif "PR Interval" in line:
            summary += f"PR Interval: {line.split(':')[-1].strip()}\n"

        # Extract QT Interval
        elif "QT Interval" in line:
            summary += f"QT Interval: {line.split(':')[-1].strip()}\n"

        # Extract Summary
        elif "Summary" in line:
            summary += f"\nSummary:\n{line.split(':', 1)[-1].strip()}\n"

    # If summary is empty, include the extracted text in the output
    if summary == f"Based on the extracted text from {pdf_name}, here are the key findings:\n\n":
        summary += extracted_text

    # Return the summary with extracted details included
    return summary.strip(), id, name, age_gender, date




# Function to call the Gemini model for summary generation
def generate_summary_from_gemini(extracted_text):
    # Prepare the prompt for the model
    prompt = f"Generate a summary for the following ECG report content:\n\n{extracted_text}"
    
    # Generate the summary from Gemini model
    response = ecg_summary_model.generate_content(prompt)
    return response.text  # Ensure the model response contains the 'text' key

def add_text_to_pdf(input_pdf_path, output_pdf_path, text_array, left_margin=18, bottom_margin=80, fontsize=8):
    # Open the uploaded PDF
    doc = fitz.open(input_pdf_path)
    
    # Loop through all pages to add text
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        
        # Define starting position for text (adjusted by margins)
        bottom_left_position = fitz.Point(left_margin, page.rect.height - bottom_margin)
        
        # Insert each line of text in the array
        for line in text_array:
            # Insert the line of text at the current position
            page.insert_text(bottom_left_position, line, fontsize=fontsize, color=(50/255, 50/255, 50/255))
            
            # Move the position down by the font size for the next line
            bottom_left_position.y += fontsize + 2  # Adjust spacing between lines (2 is the line spacing adjustment)
    
    # Save the modified PDF
    doc.save(output_pdf_path)


# Function to merge PDFs
def merge_pdfs(uploaded_pdf_path, generated_pdf_path, output_pdf_path):
    result = fitz.open()

    # Add the uploaded PDF
    with fitz.open(uploaded_pdf_path) as uploaded_pdf:
        result.insert_pdf(uploaded_pdf)

    # Add the generated summary PDF
    with fitz.open(generated_pdf_path) as generated_pdf:
        result.insert_pdf(generated_pdf)
    
    # Save the merged document
    result.save(output_pdf_path)
    result.close()


# Function to call the Gemini model for summary generation
def generate_pdf(pdf_name, summary, id, name, age_gender, date):
    # Create a PDF document
    pdf = FPDF()
    pdf.add_page()

    # Set Healthspring logo at the top (make sure the logo file exists in the project directory)
    logo_path = "assets/healthspring_logo.png"  # Replace with the actual path to the logo image

    # Set the logo width and calculate the x position for centering
    logo_width = 50  # Adjust this to the width you want for the logo
    x_position = (210 - logo_width) / 2  # 210 is the default width of an A4 page in mm

    # Set logo size and position (centered at the top of the page)
    pdf.image(logo_path, x=x_position, y=8, w=logo_width)  # Adjust the logo position (x, y) and size (w)

    # Set font for title and body
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(35)  # Move the cursor down after the logo

    # Add patient details (Name, Age, Gender, Date)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(20, 10, txt="Patient ID:", ln=False)  # Title cell (adjust width as needed)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, txt=f"{id}", ln=True)  # Value cell

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(25, 10, txt="Patient Name:", ln=False)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, txt=f"{name}", ln=True)


    pdf.set_font("Arial", 'B', 10)
    pdf.cell(25, 10, txt="Age / Gender:", ln=False)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, txt=f"{age_gender}", ln=True)

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(10, 10, txt="Date:", ln=False)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, txt=f"{date}", ln=True)


    pdf.ln(10)  # Line break after patient details

    # Set font for summary title
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Summary of ECG Report", ln=True, align='C')

    pdf.ln(10)  # Line break
    pdf.set_font("Arial", size=10)

    # Add the summary content
    pdf.multi_cell(0, 5, txt=summary)

    # Save the PDF to a file
    pdf_output_path = os.path.join("output", f"{pdf_name}_summary.pdf")
    pdf.output(pdf_output_path)
    return pdf_output_path


# Extract summary from the text and split it into an array of lines
def extract_summary(text, max_line_length=100):
    # Split the text at 'Summary:' and take the content after it
    summary_start = text.split("Summary:")[-1].strip()
    
    # Split the summary into lines based on max_line_length
    lines = []
    while len(summary_start) > max_line_length:
        # Find the last space before the max_line_length to avoid cutting words
        break_point = summary_start.rfind(' ', 0, max_line_length)
        
        # If no space is found, just cut at max_line_length
        if break_point == -1:
            break_point = max_line_length
        
        # Append the line to the list and update summary_start
        lines.append(summary_start[:break_point])
        summary_start = summary_start[break_point:].strip()
    
    # Append any remaining part of the summary
    if summary_start:
        lines.append(summary_start)
    
    return lines


# Streamlit App

st.title("AI ECG Report Summary")
st.write("Upload your ECG PDFs for AI-driven extraction and summarization")

uploaded_files = st.file_uploader(
    "Upload multiple PDFs", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    pdf_text_data = {}

# Show spinner while processing the files
    with st.spinner("Processing files and generating summaries..."):
        for uploaded_file in uploaded_files:
            try:
                pdf_name = uploaded_file.name
                pdf_content = uploaded_file.read()

                # Create a temporary file to write the PDF content
                with tempfile.NamedTemporaryFile(delete=False) as temp_pdf_file:
                    temp_pdf_file.write(pdf_content)
                    temp_pdf_file_path = temp_pdf_file.name  # Save the file path

                # Extract text using PyMuPDF (fitz)
                doc = fitz.open(temp_pdf_file_path)

                # Extract text from all pages
                text = ""
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text += page.get_text("text")  # Extract text from the page

                # Format extracted text
                formatted_output, id, name, age_gender, date = format_output(pdf_name, text)

                # Generate summary using Gemini
                summary = generate_summary_from_gemini(formatted_output)

                # Extract the actual summary (content after "Summary:")
                summary_text = extract_summary(summary)

                pdf_text_data[pdf_name] = {"formatted_output": formatted_output, "summary": summary}

                # Generate and save the PDF containing the summary
                pdf_file_path = generate_pdf(pdf_name, summary, id, name, age_gender, date)

                # Add text to the bottom-left of the uploaded PDF
                modified_pdf_path = os.path.join("output", f"{pdf_name}_modified.pdf")
                add_text_to_pdf(temp_pdf_file_path, modified_pdf_path, summary_text)

                # Merge the uploaded PDF with the new generated PDF
                output_pdf_path = os.path.join("output", f"{pdf_name}_merged.pdf")
                merge_pdfs(modified_pdf_path, pdf_file_path, output_pdf_path)

                st.subheader(f"Generated Summary and Recommendation for {pdf_name}")
                st.text(summary)
                st.download_button(
                    label=f"Download {pdf_name} Summary as PDF",
                    data=open(output_pdf_path, "rb").read(),
                    file_name=f"{pdf_name}_summary.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")


else:
    st.info("Upload a PDF to extract and summarize ECG data with AI.")
