import streamlit as st
from langchain.llms import Gemini
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores 1  import FAISS

# Function to retrieve the API key from Streamlit secrets
def get_gemini_api_key():
    """Retrieves the Gemini API key from Streamlit secrets.

    Returns:
        str: The Gemini API key, or None if not found.
    """
    try:
        return st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("Please set the 'GEMINI_API_KEY' in your Streamlit secrets.")
        return None

# Get the API key (handle potential missing secret gracefully)
api_key = get_gemini_api_key()

if api_key is not None:
    # Create the Gemini LLM with the retrieved API key
    llm = Gemini(api_key=api_key)

    # Rest of your code using the llm object
    # ... (Your code for loading data, creating embeddings, vector store, and QA chain)

    def main():
        st.title("FAQ Chatbot")

        user_query = st.text_input("Ask your question:")

        if user_query:
            response = qa_chain.run(user_query)
            st.write(response)

    if __name__ == "__main__":
        main()
else:
    st.stop()  # Halt the app execution if the API key is missing
