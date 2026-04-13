import asyncio, os, sys
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

engine = create_async_engine(
    os.environ['DATABASE_URL'],
    connect_args={'ssl': True}
)
async def test():
    async with engine.begin() as conn:
        res = await conn.execute(text('SELECT 1'))
        print("Success:", res.scalar())

asyncio.run(test())
