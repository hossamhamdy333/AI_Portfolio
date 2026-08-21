import duckdb

with open("01_campaign_summary.sql") as f:
    sql = f.read()

sql = sql.replace("marketing_AB_fake.csv", "marketing_AB.csv")

lines = [line for line in sql.splitlines() if not line.strip().startswith("--")]
sql_no_comments = "\n".join(lines)
statements = [s.strip() for s in sql_no_comments.split(";") if s.strip()]

con = duckdb.connect()
for i, stmt in enumerate(statements, 1):
    print(f"--- statement {i} ---")
    print(con.sql(stmt).df())
    print()