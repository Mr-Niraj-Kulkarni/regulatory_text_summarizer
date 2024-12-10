# Regulations Summarizer with Mock LLM Integration

This project processes a regulations document, extracts sections, generates summaries using a simulated Large Language Model (LLM), and outputs the results into a CSV file.

---

## Table of Contents
1. [Instructions](#instructions)
2. [Assumptions and Design Decisions](#assumptions-and-design-decisions)
3. [Simulated LLM Integration](#simulated-llm-integration)
4. [Demonstration of Prompt Usage for LLM Integration](#Demonstration-of-Prompt-Usage-for-LLM-Integration)

---

## Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/Mr-Niraj-Kulkarni/regulatory_text_summarizer.git
   cd regulatory_text_summarizer

2. Ensure you have Python 3.8+ installed.

3. Install dependencies using the command: 
   ```bash
   pip install -r requirements.txt

4. Run the extract_requirements.py script using the command:
   ```bash
   python extract_requirements.py --input_file regulations.txt --prompt_template prompt_template.txt

5. Check the output in the generated `extracted_requirements.csv` file 

## Assumptions and Design Decisions

### Assumptions:
The regulations file is structured with sections and headings, where:
- The title is the first line of the document.
- Headings are followed by content divided into numbered sections.
- The prompt template file contains placeholders like `{section_text}` to format prompts. 

### Design Decisions:
#### 1. Mock LLM Integration:
- Simulated using TF-IDF-based sentence ranking for section summaries.
- This mock integration demonstrates functionality without relying on an actual LLM.
#### 2. Modular Design:
- Encapsulates functionality in classes for text processing and LLM simulation.
### 3. Output Format:
- Results are saved as a CSV with columns for original text, section headings, and summarized requirements
    1. file_path	
    2. file_title	
    3. section_number	
    4. original_text	
    5. section_heading	
    6. summarized_requirements

## Simulated LLM Integration

The script simulates LLM behavior by:

1. Tokenizing the input text into sentences using simple string processing.
2. Calculating the importance of sentences using Term Frequency-Inverse Document Frequency (TF-IDF).
3. Selecting the most important sentence(s) as the "summary" for each section.

## Demonstration of Prompt Usage for LLM Integration

The `prompt_template` and prompt functionality, implemented using LangChain, are designed to demonstrate how prompts would be utilized in a real-world scenario with an actual LLM. If integrating with LLM APIs (e.g., OpenAI or other cloud-based models), the constructed prompt would be passed directly to the API endpoint. For open-source LLMs, the prompt would first be tokenized and then passed as input to a text-generation pipeline, such as those provided by Hugging Face.

This mock integration provides a blueprint for handling prompts and generating outputs, making it easier to adapt the implementation to different LLM platforms as needed. It also ensures modularity, allowing seamless replacement of the mock LLM with a production-grade LLM in the future.
