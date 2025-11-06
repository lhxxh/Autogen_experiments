import asyncio
import json
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from agdebugger.backend import BackendRuntimeManager
import logging
import aiofiles
import pickle
import os
import time

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.agents import AssistantAgent

async def write_cache_and_history_result(history, cache, result):
    hist_path = f"output/1_complete_history/history.pickle"
    cache_path = f"output/1_complete_history/cache.pickle"
    result_path = f"output/1_complete_history/log.pickle"

    await write_file_async(hist_path, history)
    await write_file_async(cache_path, cache)
    await write_file_async(result_path, result)

    print("Saved AgDebugger cache files to: ", [hist_path, cache_path, result_path])


async def write_file_async(path, data):
    async with aiofiles.open(path, "wb") as f:
        buffer = pickle.dumps(data)
        await f.write(buffer)

async def get_agent_team():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    assistant_agent = AssistantAgent(name="assistant_agent",system_message="You are a helpful assistant",model_client=model_client)
    surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model_client,
    )
    termination = MaxMessageTermination(10)
    team = RoundRobinGroupChat([assistant_agent, surfer], termination_condition=termination)

    return team

async def main():
    team = await get_agent_team()

    # Create backend to track history
    logger = logging.getLogger()
    backend = BackendRuntimeManager(team, logger)
    await backend.async_initialize()

    print("=" * 60)
    print("PHASE 1: Initial Run")
    print("=" * 60)

    # Run to completion
    #await Console(team.run_stream(task="What is the weather in Seattle?"))
    result = await team.run(task="What is the weather in Seattle?")
    print(result)

    # Show history
    detailed_history = backend.get_current_history()
    print(f"\n📊 History length after initial run: {len(detailed_history)}")

    os.makedirs("output/1_complete_history", exist_ok=True)

    # Write history and cache
    await write_cache_and_history_result(backend.get_current_history_raw_type(), backend.agent_checkpoints, result)

if __name__ == "__main__":
    asyncio.run(main())