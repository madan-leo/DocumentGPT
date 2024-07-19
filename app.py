import streamlit as st
import callLLM as callLLM
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv('DOCGPT_OPENAI_API_KEY')

# Initialize Session State variables
if "msgLoop" not in st.session_state:
    st.session_state.msgLoop = []
if "docsearch" not in st.session_state:
    st.session_state.docsearch = ''

# For browser bar - title, logo etc. & to keep side bar expanded
st.set_page_config(
    page_title="getGPT",
    page_icon="icon.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Set sidebar logo and color
st.sidebar.image("logo.jpg", width=250)
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #1A2129;
    }
</style>
""", unsafe_allow_html=True)


def call_llm(myprompt, document_search):
    if document_search:
        retriever = document_search.as_retriever()
        template = """You are a cheerful assistant in answering questions related to the uploaded document.
        Answer the question based only on the following context:
        {context}
        Question: {question}
        If no answer found from context, then answer from your knowledge from internet.
        """
        retrieverprompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | retrieverprompt
                | model
                | StrOutputParser()
        )
        output = chain.invoke(myprompt)
        return output
    else:
        systemprompt = """Answer the question based on your knowledge from internet        
                        """
        output = callLLM.callopenai(myprompt, systemprompt)
        return output

# Read Uploaded File
# Answer Questions from a doc involves 4 steps:
# 1) Splitting the document into smaller chunks.
# 2) Convert text chunks into embeddings.
# 3) Perform a similarity search on the embeddings.
# 4) Generate answers to questions using an LLM.

@st.cache_data
def processFile(myFile):
    pdfreader = PdfReader(myFile)

    # Read text from Pdf
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        extraction = page.extract_text()
        if extraction:
            raw_text += extraction

    # Split the text so it doesn't exceed token size
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # FAISS creates a vector store in the RAM
    st.session_state.docsearch = FAISS.from_texts(texts, embeddings)

    # Display file upload message
    with st.chat_message("ai", avatar="icon.jpg"):
        st.write(myFile.name, ".........uploaded")

myFile = st.sidebar.file_uploader(label=":red[Upload a file]")
if myFile:
    processFile(myFile)

st.sidebar.markdown("<h2 style='text-align: left; color: white;'>Chat History</h2>", unsafe_allow_html=True)

# Display Clear Chat button on the sidebar. Clear chat conversations when clicked.
with st.sidebar:
    if st.sidebar.button("Clear Chat"):
        st.session_state.msgLoop = []

# LLM Call
# Enter Prompt: Read prompt and append to msg array, also append response to array.
prompt = st.chat_input("Say Something!")
if prompt:
    st.session_state.msgLoop.append({"role": "user", "content": prompt})
    resp = call_llm(prompt, st.session_state.docsearch)
    st.session_state.msgLoop.append({"role": "ai", "content": resp})

# Display Chat: Loop through all appended msgs and display as needed
# Using 'message' from 'streamlit_chat' to display user input on right side, is_user = True makes it happen
# avatar_style gets icons from https://www.dicebear.com/styles/; Key sets unique identifier
for i, msg in enumerate(st.session_state.msgLoop):
    if msg["role"] == "user":
        content = msg["content"]
        message(content, is_user=True, avatar_style="icons", seed="Mimi", key=str(i))
        st.sidebar.markdown(f"<h5 style='text-align: left; color: white;'>{content}</h5>", unsafe_allow_html=True)
    elif msg["role"] == "ai":
        temp1 = "<h6 style='text-align: left; color: None;'>" + str(msg["content"]) + "</h1>"
        with st.chat_message(msg["role"], avatar="icon.jpg"):
            st.markdown(temp1, unsafe_allow_html=True)

# Code to display chat in streamlit default method
# st.markdown(
#       """
#       <style>
#         .stChatMessage {
#             text-align: right;
#            }
#        </style>
#        """,
#            unsafe_allow_html=True,
#        )
#        #temp = "<h6 style='text-align: right; color: black;'>"+messages["content"]+"</h1>"
#        with st.chat_message(messages["role"]):
#            st.write(messages["content"])
#            #st.markdown(temp, unsafe_allow_html=True)
#    if messages["role"]=="ai":
#        temp1 = "<h6 style='text-align: left; color: None;'>" + str(messages["content"]) + "</h1>"
#        with st.chat_message(messages["role"]):
#            st.markdown(temp1, unsafe_allow_html=True)
