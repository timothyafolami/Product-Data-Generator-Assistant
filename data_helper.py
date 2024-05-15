from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.chains import ConversationChain
from langchain.memory import  ConversationBufferWindowMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
import json
import streamlit as st
import traceback
load_dotenv()

# setting up groq api key
os.environ["GROQ_API_KEY"] = st.secrets.GROQ_API_KEY

# chat set up
GROQ_LLM = ChatGroq(temperature=0, model_name="llama3-8b-8192")


### Chains #####
# Initiator
def initiator(chat_history: list) -> str:
    initiator_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert at evaluating the conversation between a USER and an AI. \n
        Your focus is on the AI response. You're looking for a place where the AI response has "ALL REQUIREMENTS RECORDED" in it. \n
        
        If you see "ALL REQUIREMENTS RECORDED" in thre AI repsonse:
            Return 'START' as your response. \
        Else if it's not there:
            Return 'WAIT' as your response
        
        You can only return "START" or "WAIT", no other explanation is needed.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        CHAT_HISTORY: {chat_history} \n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["chat_history"],
    )

    initiator_router = initiator_prompt | GROQ_LLM | StrOutputParser()
    output = initiator_router.invoke({"chat_history":chat_history})
    return output

## Rewrite Router
def chat_reviewer(chat_history: list) -> str:
    chat_reviewer = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an expert at evaluating the conversation between a USER and an AI. \n
        You are checking to see if the AI has already understood the user requirements and if the user approves of it too. \n

        Ask yourself the following questions: \n\n

        Has the user mentioned all of his requirements?
        Has the AI understood the requirements correctly?
        Has the AI presented the requirements back to the user for verification?
        Has the user approved of the requirements?

        Give a binary choice 'understood' (if the requirements has been met)) or 'not_understood' (if the ai still needs more information) based on the chat conversation.
        Return only the binary choice and no premable or explaination. \
        Be Sure of your decision before you submit. \n
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        CHAT_HISTORY: {chat_history} \n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["chat_history"],
    )

    chat_review_router = chat_reviewer | GROQ_LLM | StrOutputParser()
    output = chat_review_router.invoke({"chat_history":chat_history})
    return output

## Requirement Extractor
def requirement_extractor(chat_history: list, chat_review: str) -> dict:
    requirement_extractor_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are the Coding Requirement Extraction Agent take the CHAT_HISTORY of a conversation between the USER and AI, the CHAT_REVIEW  \
                which tells if the ai understands the user requirment \
                If the CHAT_REVIEW is 'understood' then extract the code requirement from the chat_history. \n
                
                Extract a code requirement in this format:
                    1. Problem statement: The problem that the user is trying to solve.
                    2. Column names: The columns that the user wants in the data.
                    3. Data types: The data types for each column.
                    4. Data size: The intended size of the data. This should be the number of rows.
                    5. Data Variability: The possible variability in the data as per the user's requirement. This is optional.

                
                You never make up information that hasn't been provided by the CHAT_HISTORY. \n
                Return the code requirement as a JSON with the keys 'problem_statement', 'column_names', 'data_types', 'data_size', and 'data_variability' if provided. \n
                If the CHAT_REVIEW is 'not_understood' then return 'Nothing-To-Extract' as JSON with a single key 'no_extraction' and no premable or explaination. \n

                
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        CHAT_HISTORY: {chat_history} \n
        CHAT_REVIEW: {chat_review} \n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["chat_history","chat_review"],
    )

    requirement_extraction_chain = requirement_extractor_prompt | GROQ_LLM | JsonOutputParser()
    output = requirement_extraction_chain.invoke({"chat_history":chat_history, "chat_review":chat_review})
    return output

## Generate Code
def code_generator(chat_history: list, chat_review: str, extracted_requirements: dict) -> str:
    code_generator_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are the Data Science Coding Agent. You read the CHAT_HISTORY below, the conversation between the USER and AI. \
        , the CHAT_REVIEW that reviews the chat and tells if the AI understands the USER requirements, \
        the Extracted Requirements from REQUIREMENT_EXTRACTOR.

        Here is your task: \n

            1. You write excellent codes for the client based on the Extracted Requirements. \n
            2. Analyze the Extracted Requirements and make sure the code meets the requirements. \n
            3. You can use Pandas, Numpy, or any other library you think is necessary. \n
            4. You are to always ensure that the code is well-documented and easy to understand. \n
            5. Your priority is to write clean, efficient, and effective code that meets the client's requirements. \n
            6. Always check the datatypes compatibility to avoid errors. \n Use the libraries effectively to generate the data. \n
            7. You are to return the code as a string and no premable or explanation. \n

        Other Informations: \n
            You never make up information that hasn't been provided by the CHAT_HISTORY, CHAT_REVIEW, or REQUIREMENT_EXTRACTOR. \n
            The CHAT_REVIEW is either 'understood' or 'not_understood'. If CHAT_REVIEW is 'understood', \
            the Extracted Requirements will be in the format of a JSON with the keys 'problem_statement', 'column_names', 'data_types', 'data_size', and 'data_variability' if provided. \

            If the CHAT_REVIEW is 'not_understood' then the Extracted Requirements will be 'Nothing-To-Extract' or 'True' as JSON with a single key 'no_extraction'. \
            So when we have 'Nothing-To-Extract' then we don't have any requirements to work with. \
            Else you return 'no-code generated' as a string and no premable or explanation. \


        <|eot_id|><|start_header_id|>user<|end_header_id|>
        CHAT_HISTORY: {chat_history} \n\n
        CHAT_REVIEW: {chat_review} \n\n
        REQUIREMENT_EXTRACTOR: {extracted_requirements} \n\n

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["chat_history","chat_review","extracted_requirements"],
    )

    code_generator_chain = code_generator_prompt | GROQ_LLM | StrOutputParser()
    output = code_generator_chain.invoke({"chat_history":chat_history, "chat_review":chat_review, "extracted_requirements":extracted_requirements})
    return output

## Rewrite code
def code_rewriter(generated_code: str, extracted_requirements: dict) -> str:
    code_rewriter_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are the Data Science Coding Agent. You read the GENERATED_CODE, and the Extracted Requirements from REQUIREMENT_EXTRACTOR. \n

        Here is your task: \n

            1. Extract the coding part in the GENERATED_CODE. \n
            2. You compare this with the Extracted Requirements, to ensure that the code meets the requirements. \n
            3. You are to check the code for any errors, inconsistencies, or missing requirements. \n
            4. You make a deep search and simulate the code to ensure that it works as expected. For every method used you check to make sure it is correct\n
            5. You are rewriting the code with a 100 percent correctness. \n
            6. You are to return the code as a string and no premable or explanation. \n
            7. You are to ONLY RETURN THE CORRECTED/REWRITTEN CODE. \n 


        Make sure you are 100 percent sure of your corrections before you submit. \n

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        GENERATED_CODE: {generated_code} \n\n
        REQUIREMENT_EXTRACTOR: {extracted_requirements} \n\n

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generated_code","extracted_requirements"],
    )

    code_rewriter_chain = code_rewriter_prompt | GROQ_LLM | StrOutputParser()
    output = code_rewriter_chain.invoke({"generated_code":generated_code, "extracted_requirements":extracted_requirements})
    return output

## Extract code
def code_extractor(generated_code:str) -> str:
    code_extractor_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a Code Extractor Agent. \n
        You take in codes that might have some texts added to it. \n

        Only extract the code and return it as the only output. \n

        Return only the code and nothing added to it.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        CODE: {generated_code} \n\n

        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generated_code"],
    )

    code_extractor_chain = code_extractor_prompt | GROQ_LLM | StrOutputParser()
    output = code_extractor_chain.invoke({"generated_code":generated_code})
    return output

## Modify codes to classes
def code_modifier(generated_code: str) -> str:
  code_modifier_prompt = PromptTemplate(
      template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
      You are a Coding Agent. You receive from CODES texts that contain a full code that creates a synthetic data for a client. \n
      Your job is to convert this code into a high level Class that has a crate function whereby when called the data is created. \n
      You are returning a high level class that has a create function that creates the data.\n

      This is how you should structure the class:\n
        import the necessary libraries\n
        classname:  SyntheticDataGenerator\n
          method:   create - creates the synthetic data\n
          method:   save_to_csv - saves the data to a csv file. This takes in the data name as variable.\n
          method: generate_and_save_data - generates and saves the data to a csv file. The data is saved as data.csv\n

          
      Return just the code, no extra text.
      <|eot_id|><|start_header_id|>user<|end_header_id|>
      CODEs: {generated_code} \n\n

      <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
      input_variables=["generated_code"],
  )

  code_modifier_chain = code_modifier_prompt | GROQ_LLM | StrOutputParser()
  output = code_modifier_chain.invoke({"generated_code":generated_code})
  return output

# code cleaner
def remove_triple_quotes(text):
    if (text.startswith("'''") and text.endswith("'''")) or (text.startswith("```") and text.endswith("```")):
        return text[3:-3]  # Remove triple quotes from both ends
    elif text.startswith("'''") or text.startswith("```"):
        return text[3:]  # Remove triple quotes from the beginning
    elif text.endswith("'''") or text.endswith("```"):
        return text[:-3]  # Remove triple quotes from the end
    else:
        return text

# Code saver
def code_saver(code, filepath):
    with open(filepath, 'w') as f:
        f.write(code)
        

# code_excecutor

file_path = 'synthetic_data_generator.py'
def code_executor():
    # Check if the file exists
    if os.path.exists(file_path):
        try:
            # importing the class from the file
            from synthetic_data_generator import SyntheticDataGenerator

            # creating an instance of the class
            data = SyntheticDataGenerator()

            # calling the create function
            data.generate_and_save_data()

            return "Data generated successfully!"
        except Exception as e:
        # Capture the complete error, including the traceback
            error = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
                # saving the error in a json file
            with open('error.json', 'w') as f:
                json.dump(error, f)
            
            return 'Error'

def Data_Generator(CHAT_HISTORY: list) -> str:
    # Initiating the flow
    initiation_ = initiator(CHAT_HISTORY)
    print(initiation_)
    if initiation_ == "START":
        # Now proceeding with the chat flow
        try:
            # chat_reviewer section to review chats
            chat_review_ = chat_reviewer(CHAT_HISTORY)
            print("Review Completed")
            # Project Requirement Extractor
            extracted_req = requirement_extractor(CHAT_HISTORY, chat_review_)
            print("Extraction Completed")
            # Python Code Generator
            code_output = code_generator(CHAT_HISTORY, chat_review_, extracted_req)
            print("Code generated")
            # Extracting only the code
            extracted_code = code_extractor(code_output)
            print("Code Extracted!!")
            # Modifying the code into classes
            modified_code = code_modifier(extracted_code)
            print("Code Modified")
            # Removing backstrings from the code
            cleaned_code = remove_triple_quotes(modified_code)
            print("Code Cleaned")
            # saving the cleaned code
            code_saver(cleaned_code, "synthetic_data_generator.py")
            print("Code Saved")
            # Exxecuting the codes
            execute = code_executor()
            print(execute)
            return execute
        except Exception as e:
        # Capture the complete error, including the traceback
            error = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
                # saving the error in a json file
            with open('error.json', 'w') as f:
                json.dump(error, f)

CHAT_HISTORY =  {
    "messages": [
        {"user": "USER", "text": "Hi, I'm trying to create a dataset for sales in a city. I'm looking to analyze the sales of fashion products in New York City. I want to track the sales of specific products to understand their market performance.", "timestamp": "2023-02-10T14:25:00"},
        {"ai": "AI", "text": "Sounds like an interesting project! Can you please tell me more about what you're trying to achieve? What specific problem are you trying to solve?", "timestamp": "2023-02-10T14:26:00"},
        {"user": "USER", "text": "The problem I'm trying to solve is to analyze the sales trends of blouses, tops, and dresses in New York City. I want to understand which products are most popular and identify any seasonal patterns.", "timestamp": "2023-02-10T14:27:00"},
        {"ai": "AI", "text": "Got it! So, you're looking to analyze sales data for blouses, tops, and dresses in New York City. Can you please list the columns you'd like to include in the dataset?", "timestamp": "2023-02-10T14:28:00"},
        {"user": "USER", "text": "I'd like to include columns for date, product category (blouses, tops, dresses), product sub-category (e.g. long-sleeve blouses), and sales amount.", "timestamp": "2023-02-10T14:30:00"},
        {"ai": "AI", "text": "That makes sense! What data types do you expect for each column? For example, would the sales amount be a numerical value?", "timestamp": "2023-02-10T14:31:00"},
        {"user": "USER", "text": "Yes, the sales amount would be a numerical value. And the date would be a date type. The product categories would be categorical.", "timestamp": "2023-02-10T14:33:00"},
        {"ai": "AI", "text": "Got it! For the sales amount, I assume you want a numerical value, such as a float or integer. For the date, I assume it would be a date data type. And for the product categories, I assume it would be a categorical value, such as strings or categorical data types like enum or categorical.", "timestamp": "2023-02-10T14:34:00"},
        {"user": "USER", "text": "Yes, that's correct! I'd also like to know how many rows of data you'd like to generate.", "timestamp": "2023-02-10T14:35:00"},
        {"ai": "AI", "text": "I can generate a dataset with a specified number of rows. How many rows would you like the dataset to have?", "timestamp": "2023-02-10T14:36:00"},
        {"user": "USER", "text": "I'd like the dataset to have 10,000 rows.", "timestamp": "2023-02-10T14:37:00"},
        {"ai": "AI", "text": "Okay, I'll generate a dataset with 10,000 rows. Would you like to specify any variability in the data?", "timestamp": "2023-02-10T14:38:00"},
        {"user": "USER", "text": "No, I don't need any variability in the data.", "timestamp": "2023-02-10T14:39:00"},
        {"ai": "AI", "text": "Alright, I've generated the dataset with the specified columns, data types, and number of rows. I've excluded variability to ensure consistent data. ALL REQUIREMENTS RECORDED", "timestamp": "2023-02-10T14:40:00"}
    ]
}

# print(Data_Generator(CHAT_HISTORY))