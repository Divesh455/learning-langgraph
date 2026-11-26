
import sqlite3

from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage
)

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

# Long-term memory table
conn.execute("""
CREATE TABLE IF NOT EXISTS user_memory (
    user_id TEXT PRIMARY KEY,
    memory TEXT NOT NULL
)
""")

# Conversation title table
conn.execute("""
CREATE TABLE IF NOT EXISTS conversation_titles (
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL
)
""")

conn.commit()


# Single-user application
# For multi-user applications, use authenticated user IDs.
USER_ID = "default_user"


# ==========================================
# LLM CONFIGURATION
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# ==========================================
# LONG-TERM MEMORY
# ==========================================

def get_user_memory():

    cursor = conn.execute(
        """
        SELECT memory
        FROM user_memory
        WHERE user_id = ?
        """,
        (USER_ID,)
    )

    result = cursor.fetchone()

    return result[0] if result else ""


def save_user_memory(memory):

    conn.execute(
        """
        INSERT INTO user_memory (user_id, memory)
        VALUES (?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET memory = excluded.memory
        """,
        (USER_ID, memory)
    )

    conn.commit()


def update_user_memory(user_message):

    old_memory = get_user_memory()

    prompt = f"""
You are a long-term memory manager.

Existing user memory:
{old_memory}

New user message:
{user_message}

Your task:
1. Extract useful personal facts explicitly shared by the user.
2. Preserve existing facts.
3. Update facts if the user corrects them.
4. Never invent information.
5. Keep the memory short and organized.
6. Return the complete updated memory.
7. If no new personal information is provided,
   return the existing memory unchanged.

Return only the memory text.
"""

    response = llm.invoke(prompt)

    new_memory = response.content

    if isinstance(new_memory, str) and new_memory.strip():
        save_user_memory(new_memory.strip())


# ==========================================
# CONVERSATION TITLE MANAGEMENT
# ==========================================

def save_thread_title(thread_id, title):

    conn.execute(
        """
        INSERT INTO conversation_titles (thread_id, title)
        VALUES (?, ?)
        ON CONFLICT(thread_id)
        DO UPDATE SET title = excluded.title
        """,
        (thread_id, title)
    )

    conn.commit()


def get_thread_title(thread_id):

    cursor = conn.execute(
        """
        SELECT title
        FROM conversation_titles
        WHERE thread_id = ?
        """,
        (thread_id,)
    )

    result = cursor.fetchone()

    return result[0] if result else "Previous conversation"


# ==========================================
# LANGGRAPH STATE
# ==========================================

class ChatState(TypedDict):

    message: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ==========================================
# CHATBOT NODE
# ==========================================

def chat_node(state: ChatState):

    # Get complete conversation history
    messages = state["message"]

    # Retrieve long-term user memory
    memory = get_user_memory()

    system_prompt = f"""
You are a helpful AI assistant.

LONG-TERM USER MEMORY:
{memory if memory else "No personal information is stored yet."}

INSTRUCTIONS:
- Remember the user's stored personal information.
- Use previous conversation messages as context.
- Answer questions using both conversation history
  and long-term memory.
- Never invent personal information.
- If you don't know something, say so honestly.
"""

    # Pass complete conversation history to Gemini
    conversation = [
        SystemMessage(content=system_prompt)
    ] + messages

    response = llm.invoke(conversation)

    return {
        "message": [response]
    }


# ==========================================
# LANGGRAPH CONFIGURATION
# ==========================================

checkpointer = SqliteSaver(conn=conn)

# Initialize checkpointer tables
checkpointer.setup()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ==========================================
# RETRIEVE ALL CONVERSATIONS
# ==========================================

def retreive_all_thread():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        configurable = checkpoint.config.get(
            "configurable", {}
        )

        thread_id = configurable.get("thread_id")

        if thread_id:
            all_threads.add(thread_id)

    # Return dictionaries as expected by frontend
    return [
        {
            thread_id: get_thread_title(thread_id)
        }
        for thread_id in sorted(all_threads)
    ]
