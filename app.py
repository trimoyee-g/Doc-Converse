from langchain.llms import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import pytesseract
from PIL import Image
import pandas as pd
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def pdf_page():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDF")
    st.write("Empower conversations with documents. Engage in interactive discussions directly within PDFs using AI-driven chat functionalities!")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        if st.button("<"):
            st.session_state.page = "Landing Page"

        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
           
                
openai_api_key = os.getenv("OPENAI_API_KEY")
def chat_with_csv(df,prompt):
    llm = OpenAI(api_token=openai_api_key)
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result = pandas_ai.chat(prompt)
    return result


def csv_page():

    st.header("Chat with multiple CSV")
    st.write("Empower conversations with documents. Engage in interactive discussions directly within CSVs using AI-driven chat and visualization functionalities!")

    with st.sidebar:
        if st.button("<"):
            st.session_state.page = "Landing Page"

        st.subheader("Your documents")
        
        input_csvs = st.file_uploader("Upload a CSV file", type="csv", accept_multiple_files=True)
        if input_csvs:
        # Select a CSV file from the uploaded files using a dropdown menu
            selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
            selected_index = [file.name for file in input_csvs].index(selected_file)

        #load and display the selected csv file 
            st.info("CSV uploaded successfully")
            data = pd.read_csv(input_csvs[selected_index])

        #Enter the query for analysis
    st.info("Chat Below")
    input_text = st.text_area("Enter the query")

    #Perform analysis
    if input_text and st.button("Chat with csv"):
        st.info("Your Query: "+ input_text)
        result = chat_with_csv(data, input_text)
            
        fig_number = plt.get_fignums()
        if fig_number:
            st.pyplot(plt.gcf())
        else:
            st.success(result)
            

# Path to Tesseract executable (change this based on your installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def handwritten_to_text():
    st.header("Convert Handwritten Note to Text File")
    st.write("Unlock the power of handwritten communication. Seamlessly convert handwritten notes into editable text, fostering a bridge between analog and digital worlds through AI-driven Handwritten Text Recognition!")

    with st.sidebar:
        if st.button("<"):
            st.session_state.page = "Landing Page"

        uploaded_file = st.file_uploader("Upload a handwritten image", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Recognize Text'):
            text = perform_ocr(image)
            st.write('**Extracted Text:**')
            st.write(text)

def perform_ocr(image):
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"An error occurred during text recognition: {e}")
        return None



def landing_page():
    st.title("Welcome to DocConverse")
    st.markdown("")
    st.subheader("Seamlessly Interact with PDFs, CSVs, and Handwritten Notes!")
    st.markdown("")
    st.markdown("")
    st.write("Hello human! How may I help you today?")
    st.markdown("")
    if st.button("Chat with multiple PDF"):
        st.session_state.page = "PDF Page"
    elif st.button("Chat with multiple CSV"):
        st.session_state.page = "CSV Page"
    elif st.button("Convert Handwritten Note to Text File"):
        st.session_state.page = "HANDWRITTEN Page"
def main():
    load_dotenv()
    
    st.set_page_config(page_title="DocConverse",
                       page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Landing Page"

    if st.session_state.page == "Landing Page":
        landing_page()
    elif st.session_state.page == "PDF Page":
        pdf_page()
    elif st.session_state.page == "HANDWRITTEN Page":
        handwritten_to_text()
    elif st.session_state.page == "CSV Page":
        csv_page()

if __name__ == '__main__':
    main()

