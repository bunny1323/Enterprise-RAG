import psycopg
import sys
import os

try:
    conn = psycopg.connect("postgresql://postgres:postgres@localhost/enterprise_rag")
    cur = conn.cursor()
    
    cur.execute("DELETE FROM ingestion_jobs")
    cur.execute("DELETE FROM documents")
    conn.commit()
    print("Deleted all docs!")
except Exception as e:
    import traceback
    traceback.print_exc()

