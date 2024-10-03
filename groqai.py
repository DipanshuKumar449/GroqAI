# Importing necessary libraries
import streamlit as st 
import os  
from groq import Groq 
import random  

from langchain.chains import ConversationChain  
from langchain.chains.conversation.memory import ConversationBufferWindowMemory  
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 
import os 

# Load environment variables from the .env file
load_dotenv()

# Use a try-except block to catch issues with loading the API key
try:
    groq_api_key = os.environ['GROQ_API_KEY']  # Retrieve Groq API key from environment variables
except KeyError:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()  # Stop the app if the API key is not found

def main():
    # Add a try-except block for the main function
    try:
        st.title("Groq Chat App")
    except Exception as e:
        st.error(f"Error displaying app title: {e}")

    # Sidebar for model selection and customization options
    try:
        st.sidebar.title('Select an LLM')  # Add a title to the sidebar
        model = st.sidebar.selectbox(
            'Choose a model',  # Drop-down to select the model
            ['mixtral-8x7b-32768', 'gemma2-9b-it', 'llama3-8b-8192', 'llama-3.1-8b-instant']  # Model choices
        )

        # Slider for controlling the conversational memory length
        conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)
    except Exception as e:
        st.error(f"Error in sidebar customization: {e}")

    # Set up memory buffer for the conversation
    try:
        memory = ConversationBufferWindowMemory(k=conversational_memory_length)
    except Exception as e:
        st.error(f"Error initializing conversation memory: {e}")

    # Text area for the user to input their question
    try:
        user_question = st.text_area("Ask a question:")
    except Exception as e:
        st.error(f"Error displaying text area: {e}")

    # Check if 'chat_history' is already present in session_state (persistent across user sessions)
    try:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []  # Initialize chat history if it doesn't exist
        else:
            # If there is existing chat history, restore it in memory
            for message in st.session_state.chat_history:
                memory.save_context({'input': message['human']}, {'output': message['AI']})  # Save previous context into memory
    except Exception as e:
        st.error(f"Error managing session state or restoring chat history: {e}")

    # Initialize the Groq Langchain chat object with selected model and API key
    try:
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,  # Use the API key from the environment
            model_name=model  # Use the model selected in the sidebar
        )
    except Exception as e:
        st.error(f"Error initializing Groq Chat object: {e}")
        st.stop()  # Stop the app if Groq chat object cannot be initialized

    # Create a conversation chain object with the Groq chat model and memory
    try:
        conversation = ConversationChain(
            llm=groq_chat,  # Set the selected LLM model
            memory=memory  # Use the memory object to maintain conversational context
        )
    except Exception as e:
        st.error(f"Error initializing conversation chain: {e}")
        st.stop()  # Stop the app if conversation chain cannot be initialized

    # If the user has submitted a question, process it
    if user_question:
        try:
            response = conversation(user_question)  # Get the response from the conversation chain
            message = {'human': user_question, 'AI': response['response']}  # Store the interaction as a dict
            st.session_state.chat_history.append(message)  # Append the message to the chat history
            st.write("Chatbot:", response['response'])  # Display the chatbot's response
        except Exception as e:
            st.error(f"Error processing user question or conversation: {e}")

# Call the main function to run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error running the app: {e}")
