import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage

# ------------------------------------- CONFIGURATION -------------------------------------

CONFIG = {'configurable':{'thread_id':1}}

# ------------------------------------- INITIALIZE CHAT HISTORY -------------------------------------

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# ------------------------------------- DISPLAY PREVIOUS CHAT HISTORY -------------------------------------

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# ------------------------------------- USER CHAT INPUT -------------------------------------

user_input = st.chat_input('Type here...')

if user_input:

    # ------------------------------------- STORE USER MESSAGE -------------------------------------

    st.session_state['message_history'].append({'role':'user','content':user_input})

    # ------------------------------------- DISPLAY USER MESSAGE -------------------------------------

    with st.chat_message('user'):
        st.text(user_input)

    # ------------------------------------- GENERATE AI RESPONSE -------------------------------------

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {'message':[HumanMessage(content=user_input)]},
                    config={'configurable':{'thread_id':1}},
                    stream_mode='messages'
            )
        )

        # ------------------------------------- STORE AI RESPONSE -------------------------------------

        st.session_state['message_history'].append({'role':'assistant','content':ai_message})