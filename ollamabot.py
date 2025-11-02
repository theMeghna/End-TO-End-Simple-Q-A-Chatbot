import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A CHATBOT WITH OLLAMA"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Function to generate response
def generate_user(question, engine, temperature):
    llm = Ollama(model=engine, temperature=temperature)  # max_tokens removed
    output_parser = StrOutputParser()
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)
    answer = chain.run({"question": question})
    return answer

# Streamlit App
st.title("Simple Q&A Chatbot")
st.sidebar.title("Settings")

# Sidebar options
engine = st.sidebar.selectbox(
    "Select an Ollama Model",
    ["gemma3:1b", "nomic-embed-text", "llama2"]
)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# Main interface
st.write("Go ahead and ask any question")
user_input = st.text_input("You: ")

if user_input:
    response = generate_user(user_input, engine, temperature)
    st.write(response)
else:
    st.write("Please provide the query")
