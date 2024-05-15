from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import  ConversationBufferWindowMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from utils import *
from data_helper import Data_Generator, initiator
import streamlit as st
from streamlit_chat import message
import os
load_dotenv()


# setting up groq api key
os.environ["GROQ_API_KEY"] = st.secrets.GROQ_API_KEY

# chat set up
GROQ_LLM = ChatGroq(temperature=0.5, model_name="llama3-8b-8192")

st.set_page_config(page_title='🤖 Data Generator Assistant', layout='centered', page_icon='🤖')
st.title("🤖 Chat with Data Generator")

# initial message
INIT_MESSAGE = {"role": "assistant",
                "content": "Hello! I am a Data Science Consultant. I will help you create the right data for your project. "}


def init_conversationchain() -> ConversationChain:
    chat_executor = DataScienceConsultant()

    # Store LLM generated responses
    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]

    return chat_executor

def generate_response(conversation: ConversationChain, input_text: str) -> str:
    try:
        response = conversation.predict(input_text)
    except ValueError as e:
        print("################",str(e))
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            response = "There were some error in answering this question. "
        else:
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    return response


# Re-initialize the chat
def new_chat() -> None:
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []
    conv_chain = init_conversationchain()


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type='primary')


# Initialize the conversation chain
conversation = init_conversationchain()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
user_input = st.chat_input(placeholder="Your message ....", key="input")

# display user input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    user_message = st.chat_message("user")
    user_message.write(user_input)

# Generate response
if st.session_state.messages[-1]["role"] != "assistant":
    response = generate_response(conversation, user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    assistant_message = st.chat_message("assistant")
    assistant_message.write(response)

    # checking the chat history
    CHAT_HISTORY = st.session_state.messages
    # checking back with initiator
    initiate_ = initiator(CHAT_HISTORY)
    if initiate_ =='START':
        data_generation_ = Data_Generator(CHAT_HISTORY)
        if data_generation_ == "Data generated successfully!":
            st.balloons()
            st.balloons()