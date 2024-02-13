import streamlit as st
import getGPTLLM as lc
import time
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# For browser bar - title, logo etc. & to keep side bar expanded
st.set_page_config(
    page_title="getGPT",
    page_icon="icon.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State variables
if "msgLoop" not in st.session_state:
    st.session_state.msgLoop = []
if "docsearch" not in st.session_state:
    st.session_state.docsearch = []

# LLM Call
def call_LLM(prompt, document_search):
    chain = load_qa_chain(OpenAI(),chain_type="stuff")
    docs = document_search.similarity_search(prompt)
    output = chain.run(input_documents=docs, question=prompt)
    #output = lc.callopenai(prompt)
    return output

# Set sidebar logo and color
st.sidebar.image("logo.jpg", width=250)
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #19274A;
    }
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='text-align: left; color: white;'>Prompt History</h2>", unsafe_allow_html=True)

# Read Uploaded File
myFile = st.sidebar.file_uploader(label="Upload a file to Analyze")
@st.cache_data
def processFile(myFile):
    pdfreader = PdfReader(myFile)
    # Read text from Pdf
    from typing_extensions import Concatenate
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    # Split the text so it doesn't exceed token size
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(raw_text)
    # Download embeddings from OpenAOI
    embeddings = OpenAIEmbeddings()
    st.session_state.docsearch = FAISS.from_texts(texts,embeddings)
    # Display file upload message
    with st.chat_message("ai", avatar="icon.jpg"):
        st.write(myFile.name, ".........uploaded")
if myFile:
    processFile(myFile)

# Enter Prompt: Read prompt and append to msg array, also append response to array.
prompt = st.chat_input("Say Something!")
if prompt:
   st.session_state.msgLoop.append({"role":"user","content":prompt})
   resp = call_LLM(prompt, st.session_state.docsearch)
   st.session_state.msgLoop.append({"role": "ai", "content": resp})

# Display Chat: Loop thru all appended msgs and display as needed
# Using 'message' from 'streamlit_chat' to display user input on right side, is_user = True makes it happen
# avatar_style gets icons from https://www.dicebear.com/styles/; Key sets unique identifier
for i, msg in enumerate(st.session_state.msgLoop):
    if msg["role"]=="user":
        message(msg["content"],is_user=True, avatar_style="icons", seed="Mimi", key=str(i))
    elif msg["role"]=="ai":
        temp1 = "<h6 style='text-align: left; color: None;'>" + str(msg["content"]) + "</h1>"
        with st.chat_message(msg["role"],avatar="icon.jpg"):
            st.markdown(temp1, unsafe_allow_html=True)
 # Code to display chat in streamlit default method
 #st.markdown(
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



