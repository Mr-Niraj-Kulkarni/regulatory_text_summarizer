import re
import pandas as pd
import argparse
import spacy 
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFileLoader:
    """
    A class to handle loading text content from files.
    """

    @staticmethod
    def load_text_file(file_path: str) -> str:
        """
        Reads the contents of a text file and returns it as a string.

        Args:
            file_path (str): The path to the text file.

        Returns:
            str: The contents of the file as a string.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_contents = file.read()
            return file_contents
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {file_path} does not exist.")
        except IOError as e:
            raise IOError(f"An error occurred while reading the file: {e}")

class TextSegmentGenerator:
    """
    A class to generate text segments from a given text document.
    """

    @staticmethod
    def generate_segments(text: str):
        """
        Parses the text to extract a title and content segmented by headings.

        Args:
            text (str): The input text to parse.

        Returns:
            tuple: A tuple containing the title (str) and a dictionary of segmented content (dict).

        Raises:
            ValueError: If the input text is empty or invalid.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")

        try:
            # Regular expressions for headings and their sections
            heading_pattern = r"^(.*?):"  # Matches headings ending with a colon
            section_pattern = r"((?:\d\.\s.*?(?:\n|$))+)"  # Matches numbered sections

            # Extract the title (first line)
            lines = text.strip().split("\n", 1)
            title = lines[0].strip()  # The first line is considered the title
            remaining_text = lines[1] if len(lines) > 1 else ""

            # Match headings and their sections
            matches = re.findall(
                rf"{heading_pattern}\s*{section_pattern}",
                remaining_text,
                re.DOTALL | re.MULTILINE,
            )

            # Process matches into a dictionary with combined sections
            segmented_content = {}
            for match in matches:
                # Remove the colon from the heading
                heading = match[0].strip().rstrip(":")
                # Split sections into paragraphs, removing extra newlines
                sections = [
                    section.strip()
                    for section in match[1].strip().split("\n\n")
                    if section.strip()
                ]
                # Remove digits and periods from the beginning of sections
                sections = [
                    re.sub(r"\d+\.\s*", "", section) for section in sections
                ]
                # Combine all sections into a single string
                combined_sections = " ".join(sections)
                segmented_content[heading] = combined_sections

            return title, segmented_content

        except re.error as e:
            raise RuntimeError(f"Regex parsing error: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while generating segments: {e}") 





class MockLLMWithLangChain:
    """
    A class to simulate the functionality of a language model (LLM) using LangChain.
    """

    def __init__(self, model=None, tokenizer=None):
        """
        Initializes the mock LLM with optional model and tokenizer.

        Args:
            model (object, optional): Placeholder for a language model. Defaults to None.
            tokenizer (object, optional): Placeholder for a tokenizer. Defaults to None.
        """
        self.model = model
        self.tokenizer = tokenizer
        # Load the SpaCy model for NER
        self.nlp =spacy.load('en_core_web_sm')

    import re

    def extract_section_text(self, input_text):
        """
        Extracts the last section of text between "Section_text:" and 
        "Key_Business_Requirements:" from the input text.

        Args:
            input_text (str): The text to be processed.

        Returns:
            str: The extracted section text, stripped of leading/trailing whitespace.
            None: If no match is found or an error occurs.
        """
        try:
            # Define a regular expression to match the desired section of text.
            # `re.DOTALL` allows the '.' to match newline characters.
            pattern = r"Section_text:\s*(.*?)\s*Key_Business_Requirements:"
            matches = re.findall(pattern, input_text, re.DOTALL)

            if matches:
                # Return the last matched section after stripping leading/trailing whitespace.
                return matches[-1].strip()
            else:
                # Raise an exception if no matches are found.
                raise ValueError("No match found for the given pattern.")
        
        except re.error as e:
            # Handle exceptions related to regular expressions.
            print(f"Regex error: {e}")
            return None
        
        except Exception as e:
            # Handle any other exceptions that might occur.
            print(f"An error occurred: {e}")
            return None
 

    @staticmethod
    def load_and_format_prompt(text_section: str, prompt_template_path: str) -> str:
        """
        Loads a prompt template using LangChain and formats it with the given text section.

        Args:
            text_section (str): The text section to include in the prompt.
            prompt_template_path (str): Path to the prompt template file.

        Returns:
            str: The formatted prompt.

        Raises:
            FileNotFoundError: If the prompt template file does not exist.
        """
        try:
            # Load the prompt template from file
            with open(prompt_template_path, "r", encoding="utf-8") as file:
                template_content = file.read()
            # Use LangChain's PromptTemplate to format the template
            prompt_template = PromptTemplate(template=template_content, input_variables=["section_text"])
            formatted_prompt = prompt_template.format(section_text=text_section)
        
            return formatted_prompt
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template file not found at: {prompt_template_path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while formatting the prompt: {e}")

    def mock_llm_implementation(self,prompt: str) -> str:
        """
        Simulates a mock LLM output using TF-IDF-based sentence ranking.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The summarized output.
        """
        # For demonstration purposes, the section_text is extracted again 
        # from the actual prompt.
        mock_prompt = self.extract_section_text(prompt)

        # Process the extracted text using the NLP pipeline.
        sentences = self.nlp(mock_prompt)


        # Tokenize into sentences
        sentences = [sent.text for sent in sentences.sents]

        # Compute TF-IDF scores
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        # Rank sentences by their importance (highest TF-IDF scores)
        sentence_scores = tfidf_matrix.sum(axis=1).flatten()
        ranked_sentences = [sentences[i] for i in sentence_scores[0].argsort()[::-1].tolist()[0]]

        # Select the top n sentences for the summary (e.g., top 1)
        top_sentences = ranked_sentences[:1]

        summary = " ".join(top_sentences)
        # print(summary)
        return summary

    def simulate_llm_summary(self, text_section: str, prompt_template_path: str) -> str:
        """
        Simulates an LLM summary generation process.

        Args:
            text_section (str): The text section to summarize.
            prompt_template_path (str): Path to the prompt template file.

        Returns:
            str: The simulated LLM summary.
        """
        prompt = self.load_and_format_prompt(text_section, prompt_template_path)
        result = self.mock_llm_implementation(prompt)
        return result


def main(file_path: str, prompt_template_path: str):
    """
    Main function to process the regulations file, extract sections, 
    summarize using a mock LLM, and save the summarized requirements to a CSV.

    Args:
        file_path (str): Path to the regulations file.
        prompt_template_path (str): Path to the prompt template file.
    """
    # Initialize the mock LLM with LangChain
    llm_processor = MockLLMWithLangChain()

    # Step 1: Load the regulations file
    regulations_file_text = TextFileLoader.load_text_file(file_path)

    # Step 2: Extract title and text sections from the regulations file
    title, text_sections = TextSegmentGenerator.generate_segments(regulations_file_text)
    # Step 3: Process each section and summarize using the LLM
    extracted_data = []
    section_count = 1
    for heading, section_text in text_sections.items():
        try:
            # Generate LLM summary for the section
            summarized_requirements = llm_processor.simulate_llm_summary(
                text_section=section_text, 
                prompt_template_path=prompt_template_path
            )

            # Append the processed data as a dictionary
            extracted_data.append({
                "file_path": file_path,
                "file_title": title,
                "section_number": section_count,
                "original_text": section_text,
                "section_heading": heading,
                "summarized_requirements": summarized_requirements.strip()
            })

            section_count += 1

        except Exception as e:
            print(f"Error processing section {section_count} - {heading}: {e}")
            continue

    # Step 4: Convert the results to a DataFrame
    output_df = pd.DataFrame(extracted_data)

    # Step 5: Save the extracted requirements to a CSV file
    output_csv_path = "extracted_requirements.csv"
    output_df.to_csv(output_csv_path, index=False)
    print(f"Summarized requirements saved to: {output_csv_path}")


if __name__ == "__main__":    
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Generate business requirement summaries using an LLM.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the regulations file.")
    parser.add_argument("--prompt_template", type=str, required=True, help="Path to the prompt template file.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.input_file, args.prompt_template)