import asyncio

from src.app import run_chat

if __name__ == "__main__":
    try:
        # Get currently running event loop to avoid destroying pending tasks
        loop = asyncio.get_event_loop()

    except RuntimeError:
        # Use the ProactorEventLoop (better for subprocess handling) to isolate the execution of playwright for all async operations.
        loop = asyncio.ProactorEventLoop()

        # Set event loop on entry or when previous coroutines have been completed and loop was closed
        asyncio.set_event_loop(loop)
    
    finally:
        # Create a task to run chat
        task = loop.create_task(run_chat())
        
        # Wait for the of pending tasks to complete
        pending = asyncio.all_tasks(loop)
        group = asyncio.gather(*pending)
        loop.run_until_complete(group)

        # Shutdown the loop
        loop.close()
