import asyncio
import os
from uuid import uuid4
from datetime import datetime

# Configure env
os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost/enterprise_rag"
os.environ["WEAVIATE_URL"] = "http://localhost:8080"
# Add any other required config

async def run_direct():
    from app.workflows.ingestion import IngestionWorkflow
    from app.agents.supervisor.state import IngestionState
    
    doc_id = uuid4()
    job_id = uuid4()
    
    file_path = "backend/data/raw/f16a061fe5ed4ab887f12e76ec161b03_hyundai-r215l-smart-component-mounting-torque-manual.pdf"
    
    # Check if we should initialize app state if needed
    state = IngestionState(
        document_id=doc_id,
        job_id=job_id,
        tenant_id="default",
        filename="hyundai-r215l-smart-component-mounting-torque-manual.pdf",
        storage_path=file_path,
        stage_checkpoints={},
    )
    
    print(f"Starting ingestion for {doc_id}")
    workflow = IngestionWorkflow()
    # Assuming .run() or similar exists, wait, let's check what methods it has.
    try:
        final_state = await workflow.run(state)
        print("Final State:", final_state.status)
        print(final_state.model_dump_json(indent=2))
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_direct())

