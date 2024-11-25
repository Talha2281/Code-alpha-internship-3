from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
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

# Error handling for RunnableWithMessageHistory initialization
try:
    chat_with_memory = RunnableWithMessageHistory(
        prompt=prompt,
        llm=llm,
        history=chat_history,
        output_parser=StrOutputParser()
    )
except Exception as e:
    st.error(f"Error initializing chat_with_memory: {e}")

# Input box for user queries
user_input = st.text_input("Ask something:", "")

# Process user input
if user_input:
    try:
        response = chat_with_memory.invoke({"input": user_input})
        st.write("AI Assistant:", response)
    except Exception as e:
        st.error(f"Error generating response: {e}")


