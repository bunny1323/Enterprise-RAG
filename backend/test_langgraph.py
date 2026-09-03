import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def main():
    print("Methods:", [m for m in dir(AsyncPostgresSaver) if not m.startswith('_')])

asyncio.run(main())
