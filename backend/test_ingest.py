import asyncio
import json
import os
from pathlib import Path

# Adjust paths if needed
os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/enterprise_rag"
os.environ["WEAVIATE_URL"] = "http://localhost:8080"
# Add other necessary config or let dotenv handle it

async def run_ingestion():
    from app.workflows.ingestion import IngestionWorkflow
    from app.models.document import Document
    from app.database.connection import get_db
    
    # We might need to mock or setup properly, 
    # but maybe just running the workflow directly is too hard without app setup.
    pass

if __name__ == "__main__":
    asyncio.run(run_ingestion())

