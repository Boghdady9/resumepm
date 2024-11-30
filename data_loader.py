from glob import glob
import pymupdf
import re


class PDFTextProcessor:
    def __init__(self, pdf_path):
        """
        Initialize the processor with the path to a PDF file.
        """
        self.pdf_path = pdf_path
        self.raw_text = ""
        self.processed_text = ""
        self.formatted_text = {}

    def extract_text(self):
        """
        Extracts text from the PDF and stores it in `self.raw_text`.
        """
        doc = pymupdf.open(self.pdf_path)
        self.raw_text = ""
        for page in doc:
            self.raw_text += page.get_text()
        return self.raw_text

    @staticmethod
    def preprocess_text(text):
        """
        Preprocess the given text:
        - Replace line breaks followed by bullets with spaces.
        - Remove leading/trailing spaces.
        - Replace multiple spaces with a single space.
        """
        processed_text = re.sub(r'\n•\n', ' • ', text)
        processed_text = processed_text.strip()
        processed_text = re.sub(r'\s+', ' ', processed_text)
        return processed_text

    @staticmethod
    def split_sections(text):
        """
        Splits text into sections based on headers and organizes them.
        """
        sections = re.split(r'\n([A-Za-z &]+)\n', text)
        sections = [section.strip() for section in sections if section.strip()]

        formatted_text = {}
        for i in range(0, len(sections), 2):
            section_name = sections[i]
            section_content = sections[i + 1] if i + 1 < len(sections) else ""
            formatted_text[section_name] = section_content.split(' • ')

        return formatted_text

    @staticmethod
    def format_text(formatted_text):
        """
        Formats the sections into a markdown-like output.
        """
        output = ""
        for section, items in formatted_text.items():
            output += f"### {section}\n\n"
            for item in items:
                if item.strip():
                    output += f"- {item.strip()}\n"
            output += "\n"
        return output

    def process(self):
        """
        Processes the PDF text through all steps and returns formatted output.
        """
        if not self.raw_text:
            self.extract_text()
        self.processed_text = self.preprocess_text(self.raw_text)
        self.formatted_text = self.split_sections(self.processed_text)
        return self.format_text(self.formatted_text)


