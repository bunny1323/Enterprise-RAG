import asyncio
import httpx

async def run():
    base_url = "http://127.0.0.1:8001"
    for port in [8001, 8000]:
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                res = await client.get("/health/live", timeout=2.0)
                if res.status_code == 200:
                    base_url = f"http://127.0.0.1:{port}"
                    print(f"Connected to server on {base_url}!")
                    break
        except Exception:
            continue

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            
        print("Uploading document...")
        pdf_path = "backend/data/raw/f16a061fe5ed4ab887f12e76ec161b03_hyundai-r215l-smart-component-mounting-torque-manual.pdf"
        with open(pdf_path, "rb") as f:
            files = {"file": ("hyundai-r215l-smart-component-mounting-torque-manual.pdf", f, "application/pdf")}
            res = await client.post("/api/v1/documents", files=files)
            print("Upload response:", res.status_code, res.text)
            
        if res.status_code != 202:
            return
            
        data = res.json()
        job_id = data.get("job_id")
        doc_id = data.get("document_id")
        
        print(f"Waiting for job {job_id} to complete...")
        while True:
            res = await client.get(f"/api/v1/jobs/{job_id}")
            if res.status_code != 200:
                print("Job check failed:", res.status_code, res.text)
                break
            job_data = res.json()
            status = job_data.get("status")
            print(f"Status: {status}")
            if status in ["COMPLETED", "FAILED"]:
                print("Job finished:", job_data)
                break
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(run())

