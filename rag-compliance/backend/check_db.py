import asyncio
import asyncpg

async def test():
    conn = await asyncpg.connect("postgresql://raguser:ragpass@localhost:5433/ragdb")
    v = await conn.fetchval("SELECT version()")
    print("DB connected:", v[:60])
    rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname='public'")
    print("Tables:", [r["tablename"] for r in rows])
    await conn.close()
    print("DB check PASSED")

asyncio.run(test())
