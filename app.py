# import os 
# import streamlit as st 
# from dotenv import load_dotenv
# from google.generativeai import configure
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()

# ## Langsmith Tracking 
# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_TRACING_V2'] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "End_to_End_Q&A_Chatbot"

# ## Prompt Template 
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system","You are a helpful assistant. Please response to the user queries."),
#         ("user","Question:{question}")
#     ]
# )

# def get_response(question,api_key,llm,temperature,max_tokens):
#     os.environ['GOOGLE_API_KEY'] = os.getenv(str(api_key))
#     llm = ChatGoogleGenerativeAI(model=llm)
#     output_parser = StrOutputParser()
#     chain = prompt | llm | output_parser
#     answer = chain.invoke({"question":question})
#     return answer 

# ## Title of the Streamlit app
# st.title("End to End Q&A Chatbot")

# ## Slidebar for setting 
# st.sidebar.title("Setting")
# api_key = st.sidebar.text_input("Enter your Open AI API Key:",type="password")

# ## Drop down Select various Open ai Model 
# llm = st.sidebar.selectbox("Select an Open Ai Model",["gemini-pro","gemini-1.5-pro"])

# ## Adjust response Parameter 
# temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
# max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)


# ## main Interface for user input 
# st.write("Go ahead and ask any Question")
# user_input = st.text_input("You:")

# if user_input:
#     response = get_response(user_input,api_key,llm,temperature,max_tokens)
#     st.write(response)
# else:
#     st.write("Please enter a question")



import os
import streamlit as st
from dotenv import load_dotenv
from google.generativeai import configure
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["LANGCHAIN_PROJECT"] = "End_to_End_Q&A_Chatbot"

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question:{question}")
    ]
)

# Function to get a response from the AI model
def get_response(question, api_key, llm, temperature, max_tokens):
    try:
        # Configure Google Generative AI
        configure(api_key=api_key)
        
        # Initialize LLM model
        llm = ChatGoogleGenerativeAI(model=llm)
        output_parser = StrOutputParser()
        
        # Chain prompt, model, and output parser
        chain = prompt | llm | output_parser
        answer = chain.invoke({"question": question})
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Title of the Streamlit app
st.title("End to End Q&A Chatbot")

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")

# Dropdown to select the model
llm = st.sidebar.selectbox("Select a Model", ["gemini-1.5-pro","gemini-pro"])

# Sliders to adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask a question:")
user_input = st.text_input("You:")

if user_input:
    if not api_key:
        st.error("Please provide your Google API Key in the sidebar.")
    else:
        # Fetch the response
        response = get_response(user_input, api_key, llm, temperature, max_tokens)
        st.write(response)
else:
    st.write("Please enter a question.")
