from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from data_loader import PDFTextProcessor
from Agent import agent as create_agent
import os
from IPython.display import display, HTML

# Create FastAPI instance

        # Process the PDF
pdf_processor = PDFTextProcessor('Maha Jonas.pdf')
resume_text = pdf_processor.process()

# Pass the extracted text to the agent for further processing
processed_output = create_agent(input="Can you enhance Professional Summary?"+resume_text)
print(processed_output['output'])

