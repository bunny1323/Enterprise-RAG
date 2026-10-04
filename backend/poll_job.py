import asyncio
import httpx
import sys

async def run():
    job_id = "9e53d4bd-9e08-459c-9487-bedaa4214891"
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        while True:
            res = await client.get(f"/api/v1/jobs/{job_id}", timeout=60.0)
            if res.status_code != 200:
                print("Error:", res.status_code)
                break
            job_data = res.json()
            status = job_data["status"]
            print(f"Status: {status} - Progress: {job_data.get('progress_percent')}%")
            if status in ["COMPLETED", "FAILED", "TIMEOUT"]:
                print("Job finished:", job_data)
                break
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(run())

