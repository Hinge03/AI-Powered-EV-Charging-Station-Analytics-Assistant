import sqlite3

from langchain_core.tools import tool

DB_PATH = 'EV.db'

@tool
def get_database_schema() -> str:
    """
    Get the tables and columns available in the SQLite database.
    Use this tool before generating SQL queries.
    """

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    tables = cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """)

    result = tables.fetchall()

    schema = []

    for (table_name, ) in result:

        columns = cursor.execute(f"PRAGMA table_info({table_name})")

        result1 = columns.fetchall()

        column_names = []

        for column in result1:
            column_name = column[1]
            data_type = column[2]
            primary_key = column[5]

            pk = " PRIMARY KEY" if primary_key else ""

            column_names.append(f"{column_name} {data_type}{pk}")
            
        schema.append(f"{table_name}("+", ".join(column_names)+")")

    conn.close()

    return "\n".join(schema)

@tool
def execute_sql(query: str) -> str:
    """
    Execute a SELECT SQL query on the SQLite database.

    Rules:
    - Only execute SELECT queries.
    - Do not execute INSERT, UPDATE, DELETE, DROP, ALTER, etc.
    - Do not include a trailing semicolon.
    - Use SQLite SQL syntax.
    """
    query = query.strip()

    if not query.lower().startswith("select"):
        return "SQL Error: Only SELECT queries are allowed."

    try:

        conn = sqlite3.connect(DB_PATH)

        cursor = conn.cursor()

        cursor.execute(query)

        rows = cursor.fetchall()

        columns = [description[0] for description in cursor.description]

        MAX_ROWS = 50

        limited_rows = rows[:MAX_ROWS]

        result = []

        result.append(" | ".join(columns))
        result.append("-" * 80)

        for row in limited_rows:

            result.append(
                " | ".join(
                    str(value)
                    for value in row
                )
            )

        if len(rows) > MAX_ROWS:

            result.append(
                f"\nShowing first {MAX_ROWS} rows "
                f"out of {len(rows)} rows."
            )

        return "\n".join(result)

    except Exception as e:

        return f"SQL Error: {str(e)}"

    finally:

        if conn:
            conn.close()

