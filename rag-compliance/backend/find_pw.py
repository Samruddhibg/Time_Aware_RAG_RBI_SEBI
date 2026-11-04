import asyncio
import asyncpg

async def try_connect(pw):
    try:
        conn = await asyncpg.connect(f"postgresql://raguser:{pw}@localhost:5433/ragdb")
        user = await conn.fetchval("SELECT current_user")
        await conn.close()
        print(f"Password [{pw}] -> WORKS, user={user}")
        return True
    except Exception as e:
        print(f"Password [{pw}] -> FAILED: {type(e).__name__}")
        return False

async def main():
    for pw in ["ragpass", "samruddhi", "raguser", "postgres", "password", "secret"]:
        if await try_connect(pw):
            break

asyncio.run(main())
