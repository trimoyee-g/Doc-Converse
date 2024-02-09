"Doc Converse" is a versatile and interactive web-based platform designed to facilitate seamless interactions with various types of documents including PDFs, CSVs, websites, and handwritten notes. Users can engage in interactive discussions, query analysis, and data exploration directly within their documents. The landing page serves as the entry point, providing users with options to choose between interacting with PDFs, CSVs, websites, or converting handwritten notes.

Features:

- PDF Interaction:Users can upload multiple PDF documents. The application integrates AI-driven chat functionalities, enabling interactive discussions and queries within the PDF content.

- CSV Interaction: Users can upload CSV files containing data. Utilizes an AI-powered conversational model integrated with Pandas to analyze and engage in conversations based on the uploaded data. Potential visualization capabilities to explore and interpret data from CSV files using Matplotlib.

- Website Interaction: Allows users to enter website URLs and engage in interactive discussions directly within the websites, leveraging AI-driven chat functionalities.

- Handwritten Text Recognition: Allows users to upload images of handwritten notes. Employs OCR (Optical Character Recognition) technology through Tesseract to convert handwritten content into editable text, bridging the gap between analog and digital formats.

Technologies Used:
- Streamlit: Provides the web interface and user interaction capabilities.
- OpenAI: Powers AI-driven chat functionalities.
- PyPDF2: Extracts text from PDF documents.
- Pandas: Handles data processing and interaction with CSV files.
- Matplotlib: Offers potential visualization of data insights.
- Pytesseract: Integrates Optical Character Recognition for handwritten text recognition.
