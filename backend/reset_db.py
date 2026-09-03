import asyncio
import os
from pathlib import Path

async def reset_db():
    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                key, _, value = line.partition("=")
                if key:
                    os.environ[key.strip()] = value.strip()
                    
    import asyncpg
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("Error: DATABASE_URL not found in .env")
        return
        
    print(f"Connecting to Postgres...")
    conn = await asyncpg.connect(db_url)
    try:
        print("Clearing 'documents' table (this will cascade delete chunks, jobs, and states)...")
        await conn.execute("TRUNCATE TABLE documents CASCADE;")
        print("Database has been reset successfully! You can now re-upload your files.")
    except Exception as e:
        print(f"Error resetting database: {e}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(reset_db())
