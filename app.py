from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser

import streamlit as st

# Streamlit app configuration
st.set_page_config(page_title="AI Text Assistant", page_icon="🤖")

# Title and initial description
st.title('AI Chatbot')
st.write("This chatbot is created by TALHA KHAN")
st.markdown("Hello! I'm your AI assistant. How can I assist you today?")

# Load the API key from Streamlit secrets
api_key = st.secrets["gimni"]["api_key"]

# Initialize the chat history
chat_history = StreamlitChatMessageHistory()

# Initialize the language model with API key and model
llm = ChatGoogleGenerativeAI(api_key=api_key, model="chat-bison-001")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Define the get_session_history function
def get_session_history():
    return chat_history.messages

# Define the chain manually using prompt and LLM
def chat_with_memory(input_text):
    # Retrieve session history
    session_history = get_session_history()
    # Build the prompt with history
    full_prompt = prompt.format(history=session_history, input=input_text)
    # Get response from LLM
    response = llm.predict(full_prompt)
    # Update chat history with user input and AI response
    chat_history.add_user_message(input_text)
    chat_history.add_ai_message(response)
    return response

# Input box for user queries
user_input = st.text_input("Ask something:", "")

# Process user input
if user_input:
    try:
        response = chat_with_memory(user_input)
        st.write("AI Assistant:", response)
    except Exception as e:
        st.error(f"Error generating response: {e}")


