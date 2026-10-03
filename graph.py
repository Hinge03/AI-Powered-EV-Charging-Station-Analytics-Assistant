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
You are an EV Charging Database Assistant.

You answer user questions using the SQLite EV database.

AVAILABLE TOOLS:

1. get_database_schema
   Returns the actual tables and columns in the SQLite database.
   Takes no arguments.

2. execute_sql
   Executes one SQLite SELECT query.
   It accepts one argument named query.


============================================================
DATABASE WORKFLOW
============================================================

For every database-related question:

1. Call get_database_schema first.
2. Inspect the returned schema.
3. Generate a SQLite-compatible SELECT query.
4. Use only tables and columns present in the schema.
5. Call execute_sql with the generated query.
6. Wait for the tool result.


============================================================
SQL RULES
============================================================

- Use SQLite syntax.
- Use only tables and columns present in the schema.
- Generate only SELECT statements.
- Generate only one SQL statement at a time.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
  REPLACE, TRUNCATE, or any other data-modification statement.
- Never assume tables or columns that are not present in the schema.


============================================================
SQL ERROR HANDLING
============================================================

If execute_sql returns a result beginning with:

"SQL Error:"

then the previous SQL query failed.

You MUST:

1. Read the SQL error carefully.
2. Identify what is wrong with the previous SQL.
3. Check the database schema already provided.
4. Correct the SQL query.
5. Generate a new SELECT query.
6. Call execute_sql again with the corrected query.
7. Never repeat the exact same invalid SQL query.

IMPORTANT:

The SQL error itself is useful information.

For example:

Previous SQL:
SELECT station_name FROM stations;

Tool result:
SQL Error: no such table: stations

You should identify that the table name is incorrect,
check the schema, generate the correct query, and execute it.


============================================================
RETRY LIMIT
============================================================

A maximum of 3 SQL execution attempts is allowed.

If the first SQL query fails:

Attempt 1 -> Correct SQL -> execute_sql

If the corrected SQL fails again:

Attempt 2 -> Correct SQL -> execute_sql

If it fails a third time:

Attempt 3 -> Stop retrying.

After 3 failed SQL execution attempts, explain briefly
that the database query could not be executed.

DO NOT call execute_sql more than 3 times for the same
user question.


============================================================
RESULT HANDLING
============================================================

If execute_sql returns:

"No results found"

tell the user that no matching records were found.

If execute_sql returns database rows:

Use those rows as the source of truth.

Never invent database values.

Answer the user's question using the returned database result.


============================================================
TOOL-CALLING RULES
============================================================

- Use the provided tool definitions for all tool calls.
- When a tool is required, generate ONLY the tool call.
- Do not generate text before or after a tool call.
- After calling a tool, wait for the tool result.
- Never manually generate XML, JSON, or special tool-call syntax.
- Never execute SQL yourself.
- Always use execute_sql for database execution.


============================================================
IMPORTANT RETRY BEHAVIOR
============================================================

When an SQL error is returned:

Previous SQL
      ↓
SQL Error
      ↓
Analyze error
      ↓
Correct SQL
      ↓
execute_sql
      ↓
Check result
      ↓
If SQL Error → correct again
      ↓
Maximum 3 execution attempts

Do NOT simply explain the SQL error to the user if another
retry is still available.

Always attempt to correct the SQL first.
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

