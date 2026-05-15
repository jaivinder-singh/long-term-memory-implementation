import asyncio
from graph import chatbot

config = {"configurable": {"thread_id": "1", "user_id": "u1"}}


async def main():
    async for chunk in chatbot.astream(
        {"messages": [{"role": "user", "content": "Hello, what is the sum of 1 and 2? also do you remember my name?"}]},
        config=config,
    ):
        print(chunk)


asyncio.run(main())