from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import  ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
import streamlit as st
import os

load_dotenv()

# setting up groq api key
os.environ["GROQ_API_KEY"] = st.secrets.GROQ_API_KEY

# chat set up
class DataScienceConsultant:
    def __init__(self, temperature=0.5, model_name="llama3-8b-8192"):
        self.chat = ChatGroq(temperature=temperature, model_name=model_name)
        self.chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a Data Science Consultant. You have over 20 years of experience in the field. 
        You are currently working with a client to create a synthetic data for his project. 
        The data is a real world based daa exhibits the same characteristics as the real data.

        Initially you don't know anything about the project, that's why you want to ask the client some questions to understand the data requirements.
        This is the work flow you have been following:
            1. Initiate a besutiful and interactive sessons with the user.
            2. Ask questions about the project the client is working on.
            3. Ask for the purpose of this data in relation to the project.
            4. Ask for possible columns and the data types. You can also suggest based on the project.
            5. After for the intended data size. You can also suggest based on the project. The client can also specify the size.
            6. Collate all the information and present back to the client for review.
            7. Review thw generated code with the client requirements.
            8. After User has verified that all the requirements are met, respond with - ALL REQUIREMENTS RECORDED.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Current conversation:
        {history}
        Human: {input}
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        AI Assistant:"""

        PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.chat_template)
        chat_conversation = ConversationChain(
            prompt=PROMPT,
            llm=self.chat,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=10, ai_prefix="AI Assistant"),
            output_parser=StrOutputParser(),
        )
    def predict(self, input_text):
        return self.chat_conversation.predict(input=input_text)
    

