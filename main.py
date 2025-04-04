import asyncio

from src.app import run_chat


if __name__ == "__main__":
    # Streamlit application is constantly looping itself.
    # Use the ProactorEventLoop (better for subprocess handling) to isolate the execution of playwright for all async operations.
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)

    # wWits for the group of pending tasks
    loop.run_until_complete(run_chat())

    # Shutdown the loop
    loop.close()
