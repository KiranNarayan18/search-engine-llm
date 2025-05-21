import os

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

os.environ['HF_TOKEN'] = st.secrets.HF_TOKEN
os.environ['GROQ_API_KEY'] = st.secrets.GROQ_API_KEY 

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, 
doc_content_chars_max=500)
tool_arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)


api_wikki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
tool_wrapper = WikipediaQueryRun(api_wrapper=api_wikki_wrapper)

tool_search = DuckDuckGoSearchRun()


st.title('Langchain')
st.sidebar.title('Settings')

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': "Hi,I'm a chatbot who can search the web. How can I help you?"}]



for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])


if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(model_name="Llama3-8b-8192",streaming=True)
    tools=[tool_arxiv, tool_wrapper, tool_search]

    search_agent=initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])

        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)
