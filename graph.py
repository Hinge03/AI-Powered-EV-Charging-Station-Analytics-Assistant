from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from tools import get_database_schema, execute_sql

    

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    reasoning_effort="none",
    max_tokens=600
)


trimmer = trim_messages(
    max_tokens=5000,
    strategy="last",
    token_counter=llm,
    include_system=False,
    start_on="human",
    allow_partial=False
)


class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

    sql_retry_count: int


prompt = """
        You are an EV charging database assistant.

        IMPORTANT:
        You are NOT a general-purpose question-answering assistant.

        The SQLite database is the ONLY source of truth for
        database-related questions.

        DATABASE RULES:

        1. Every question that asks for information stored in the
        database MUST be answered using the database tools.

        2. NEVER answer a database question from your internal
        knowledge, memory, assumptions, or general knowledge.

        3. You MUST call execute_sql before providing the final answer
        to any database-related question.

        4. Do not guess, estimate, or invent database values.

        5. For counting questions, use COUNT(*) whenever appropriate.

        6. Only provide the final answer after receiving the result
        from execute_sql.

        7. If execute_sql returns "No results found",
        respond exactly:

            No results found

        8. If SQL execution fails, analyze the SQL error and generate
        a corrected SQL query. Execute the corrected query before
        answering the user.

        9. Use get_database_schema when you need additional schema
            information.

        10. Do not return a database answer without an execute_sql
            tool call.

        CONVERSATION CONTEXT:

        11. Use previous conversation to understand references such as:
            - "them"
            - "their"
            - "those customers"
            - "that station"
            - "the previous result"

        12. Maintain context across multiple user questions.

        SQL RULES:

        1. Generate valid SQLite SQL.

        2. Never invent records that are not present in the database.

        3. Keep queries efficient and avoid unnecessarily large result sets.

        4. Use window functions like rank(), dense_rank(), row_number(), lead(), lag() whenever necessary.

        5. Use only tables and columns present in the schema.

        6. Generate only SELECT statements.

        7. Generate only one SQL statement at a time.

        8. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, REPLACE, TRUNCATE, or any other data-modification statement.
        
        9. Never assume tables or columns that are not present in the schema.
"""



tools = [get_database_schema,execute_sql]

llm_with_tools = llm.bind_tools(tools)


# Node1
def chat_node(state: ChatState):

    conversation = state["messages"]

    trimmed_messages = trimmer.invoke(conversation)

    messages = [SystemMessage(content=prompt),*trimmed_messages]

    try:

        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    except Exception as e:

        print("\n========== GROQ ERROR ==========")
        print("ERROR TYPE:", type(e))
        print("ERROR:", e)

        if hasattr(e, "body"):
            print("ERROR BODY:", e.body)

        print("================================\n")

        raise



# Node 2
def update_retry_count(state: ChatState):

    messages = state["messages"]

    retry_count = state.get("sql_retry_count", 0)

    # Look at the most recent tool message
    if messages:

        last_message = messages[-1]

        if isinstance(last_message, ToolMessage):

            tool_name = getattr(last_message, "name", "")

            # Count only execute_sql calls
            if tool_name == "execute_sql":

                retry_count += 1
    
    return {"sql_retry_count": retry_count}


# helper function
def check_retry_limit(state: ChatState):

    retry_count = state.get("sql_retry_count", 0)

    messages = state["messages"]

    # Find the most recent execute_sql result
    last_tool_message = None

    for message in reversed(messages):

        if isinstance(message, ToolMessage):

            if getattr(message, "name", "") == "execute_sql":

                last_tool_message = message
                break

    # No execute_sql result
    if last_tool_message is None:

        return "continue"

    result = str(last_tool_message.content)

    # Successful execution
    if not result.startswith("SQL Error:"):

        return "continue"

    # SQL failed
    if retry_count >= 3:

        return "stop"

    # SQL failed but retry is available
    return "retry"



tool_node = ToolNode(tools)

check_pointer = InMemorySaver()

graph = StateGraph(ChatState)


graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("update_retry_count", update_retry_count)


graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition,{"tools": "tools", END: END})
graph.add_edge("tools","update_retry_count")
graph.add_conditional_edges("update_retry_count", check_retry_limit,{"retry": "chat_node","continue": "chat_node","stop": END})


workflow = graph.compile(checkpointer=check_pointer)

