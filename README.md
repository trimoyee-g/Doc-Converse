"DocConverse," is a versatile and interactive web-based platform designed to facilitate seamless interactions with various types of documents including PDFs, CSVs, and handwritten notes.
It has a landing page which serves as the entry point, providing users with options to choose between interacting with PDFs, CSVs, or converting handwritten notes.

Features:

PDF Interaction:
-Users can upload multiple PDF documents.
-Text from the uploaded PDFs is extracted, segmented into smaller chunks, and stored in a vectorized format for further analysis.
-The application integrates AI-driven chat functionalities, enabling interactive discussions and queries within the PDF content.

CSV Interaction:
-Users can upload CSV files containing data.
-Utilizes an AI-powered conversational model integrated with Pandas to analyze and engage in conversations based on the uploaded data.
-Potential visualization capabilities to explore and interpret data from CSV files using Matplotlib.

Handwritten Text Recognition:
-Allows users to upload images of handwritten notes.
-Employs OCR (Optical Character Recognition) technology through Tesseract to convert handwritten content into editable text, bridging the gap between analog and digital formats.

Technologies Used:
Streamlit: Provides the web interface and user interaction capabilities.
OpenAI: Powers AI-driven chat functionalities.
PyPDF2: Extracts text from PDF documents.
Pandas: Handles data processing and interaction with CSV files.
Matplotlib: Offers potential visualization of data insights.
Pytesseract: Integrates Optical Character Recognition for handwritten text recognition.
