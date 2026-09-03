import asyncio
import httpx

async def test():
    print("Testing /health/live ...")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get("http://127.0.0.1:8001/api/v1/health/live")
            print("Live:", res.status_code, res.text)
    except Exception as e:
        print("Live failed:", e)

    print("Testing /health ...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get("http://127.0.0.1:8001/api/v1/health")
            print("Health:", res.status_code, res.text[:200])
    except Exception as e:
        print("Health failed:", e)

if __name__ == "__main__":
    asyncio.run(test())
