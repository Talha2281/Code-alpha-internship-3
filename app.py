from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.output_parser import StrOutputParser
import streamlit as st
import requests
import json

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

# Define a custom LLM class to interact with GIMNI API
class GIMNIChatLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://gimni-api-endpoint"  # Replace with GIMNI's API endpoint

    def call_api(self, input_text: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "input": input_text
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get("response", "No response from API")
        else:
            return f"Error: {response.status_code}, {response.text}"

    def generate_response(self, input_text: str) -> str:
        # Call the API and return the response
        return self.call_api(input_text)

# Initialize the custom LLM
llm = GIMNIChatLLM(api_key)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Create the runnable chain with message history
chat_with_memory = RunnableWithMessageHistory(
    prompt=prompt,
    llm=llm,
    chat_history=chat_history,
    output_parser=StrOutputParser()
)

# Input box for user queries
user_input = st.text_input("Ask something:", "")

# Process user input
if user_input:
    try:
        # Pass the user input to the chat chain for processing
        response = chat_with_memory.invoke({"input": user_input})
        st.write("AI Assistant:", response)
    except Exception as e:
        st.error(f"Error generating response: {e}")

