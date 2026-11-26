
import streamlit as st
import uuid

from backend import (
    chatbot,
    retreive_all_thread,
    update_user_memory,
    save_thread_title
)

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk
)

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv


load_dotenv()


# ==========================================
# LLM CONFIGURATION
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def generate_thread_id():

    return str(uuid.uuid4())


def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    st.session_state["message_history"] = []

    add_thread(thread_id)


def add_thread(thread_id):

    chat_threads = st.session_state["chat_threads"]

    if not any(
        thread_id in thread
        for thread in chat_threads
    ):

        chat_threads.append({
            thread_id: "New conversation"
        })


def generate_con_name(info):

    prompt = f"""
Generate a short conversation title.

Maximum 4 words.

Conversation:
{info}

Return only the title.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


def update_thread_name(thread_id, name):

    for thread in st.session_state["chat_threads"]:

        if thread_id in thread:

            thread[thread_id] = name

            break

    # Save title permanently
    save_thread_title(thread_id, name)


# ==========================================
# LOAD CONVERSATION FROM LANGGRAPH
# ==========================================

def load_con(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    if not state or not state.values:

        return []

    return state.values.get("message", [])


def load_conversation(thread_id):

    messages = load_con(thread_id)

    temp_msg = []

    for msg in messages:

        if isinstance(msg, HumanMessage):

            role = "user"

        elif isinstance(msg, AIMessage):

            role = "assistant"

        else:

            continue

        # Handle text messages
        content = msg.content

        if isinstance(content, str):

            temp_msg.append({
                "role": role,
                "content": content
            })

    st.session_state["thread_id"] = thread_id

    st.session_state["message_history"] = temp_msg

    add_thread(thread_id)


# ==========================================
# INITIALIZE SESSION STATE
# ==========================================

if "message_history" not in st.session_state:

    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = retreive_all_thread()


# Normalize old session data
st.session_state["chat_threads"] = [

    thread if isinstance(thread, dict)

    else {
        thread: "Previous conversation"
    }

    for thread in st.session_state["chat_threads"]

]


# Add current thread
add_thread(st.session_state["thread_id"])


# ==========================================
# SIDEBAR UI
# ==========================================

st.sidebar.title("LangGraph Chatbot")


if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):

    reset_chat()

    st.rerun()


st.sidebar.header("My Conversations")


for thread_dict in reversed(
    st.session_state["chat_threads"]
):

    thread_id = next(iter(thread_dict.keys()))

    con_name = thread_dict[thread_id]

    if st.sidebar.button(
        con_name,
        key=thread_id,
        use_container_width=True
    ):

        load_conversation(thread_id)

        st.rerun()


# ==========================================
# MAIN CHAT UI
# ==========================================

st.title("LangGraph Chatbot")


# Display previous messages
for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# USER CHAT INPUT
# ==========================================

user_input = st.chat_input("Type here...")


if user_input:

    thread_id = st.session_state["thread_id"]

    CONFIG = {
        "configurable": {
            "thread_id": thread_id
        }
    }


    # --------------------------------------
    # STORE USER MESSAGE
    # --------------------------------------

    st.session_state["message_history"].append({

        "role": "user",

        "content": user_input

    })


    with st.chat_message("user"):

        st.markdown(user_input)


    # --------------------------------------
    # GENERATE AI RESPONSE
    # --------------------------------------

    with st.chat_message("assistant"):

        def generate_response():

            for message_chunk, metadata in chatbot.stream(

                {
                    "message": [
                        HumanMessage(content=user_input)
                    ]
                },

                config=CONFIG,

                stream_mode="messages"

            ):

                if isinstance(
                    message_chunk,
                    (AIMessageChunk, AIMessage)
                ):

                    content = message_chunk.content

                    if isinstance(content, str) and content:

                        yield content


        ai_message = st.write_stream(
            generate_response()
        )


    # --------------------------------------
    # STORE AI RESPONSE
    # --------------------------------------

    st.session_state["message_history"].append({

        "role": "assistant",

        "content": ai_message

    })


    # --------------------------------------
    # UPDATE LONG-TERM MEMORY
    # --------------------------------------

    try:

        update_user_memory(user_input)

    except Exception as e:

        st.warning(
            f"Could not update long-term memory: {e}"
        )


    # --------------------------------------
    # GENERATE CONVERSATION TITLE
    # --------------------------------------

    if len(
        st.session_state["message_history"]
    ) == 2:

        try:

            con_name = generate_con_name(user_input)

            update_thread_name(
                thread_id,
                con_name
            )

        except Exception as e:

            st.warning(
                f"Could not generate conversation title: {e}"
            )
