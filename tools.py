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
    Execute a read-only SELECT query on the EV database.
    """

    query = query.strip()

    if not query.lower().startswith("select"):
        return "SQL Error: Only SELECT queries are allowed."

    conn = None

    try:
        conn = sqlite3.connect(DB_PATH)

        cursor = conn.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()

        if not rows:
            return "No results found. The query returned zero rows."

        columns = [description[0] for description in cursor.description]

        result = []
        result.append(" | ".join(columns))
        result.append("-" * 80)

        for row in rows[:50]:
            result.append(
                " | ".join(str(value) for value in row)
            )

        if len(rows) > 50:
            result.append(
                f"Showing first 50 rows out of {len(rows)} rows."
            )

        return "\n".join(result)

    except Exception as e:
        return f"SQL Error: {str(e)}"

    finally:
        if conn is not None:
            conn.close()
