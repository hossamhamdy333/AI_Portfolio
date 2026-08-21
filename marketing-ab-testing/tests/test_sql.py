import duckdb

with open("../sql/01_campaign_summary.sql") as f:
    sql = f.read()

# strip full-line comments before splitting on ';', so a statement that
# happens to start with a comment block doesn't get thrown away whole
lines = [line for line in sql.splitlines() if not line.strip().startswith("--")]
sql_no_comments = "\n".join(lines)
statements = [s.strip() for s in sql_no_comments.split(";") if s.strip()]

con = duckdb.connect()
for i, stmt in enumerate(statements, 1):
    print(f"--- statement {i} ---")
    print(con.sql(stmt).df())
    print()
