## works completely fine
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize tools
arxiv = ArxivQueryRun(top_k_results=1)
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search = DuckDuckGoSearchRun()

st.title("Langchain - Chat with Search")
st.write("In this example, we are using 'StreamlitCallbackHandler' to display thoughts and actions.")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key", type="password")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

prompt = st.chat_input(placeholder="What is machine learning?")

if prompt:
    st.session_state.messages.append({'role': "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-it", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(prompt, callbacks=[st_cb])
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)