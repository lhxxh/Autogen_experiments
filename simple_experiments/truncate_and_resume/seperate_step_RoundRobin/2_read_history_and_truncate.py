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
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.agents import AssistantAgent

async def write_cache_and_history(history, cache):
    hist_path = f"output/2_truncated_history/history.pickle"
    cache_path = f"output/2_truncated_history/cache.pickle"

    await write_file_async(hist_path, history)
    await write_file_async(cache_path, cache)

    print("Saved AgDebugger cache files to: ", [hist_path, cache_path])

async def write_file_async(path, data):
    async with aiofiles.open(path, "wb") as f:
        buffer = pickle.dumps(data)
        await f.write(buffer)


def get_content(obj):
    """Recursively search for 'content' attribute in nested objects"""
    # Handle strings directly
    if isinstance(obj, str):
        return obj
    
    # Handle lists
    if isinstance(obj, list):
        for item in obj:
            result = get_content(item)
            if result:
                return result
        return ''
    
    # Handle objects with attributes
    if hasattr(obj, 'content'):
        content = obj.content
        if isinstance(content, list):
            return ' '.join(str(c) for c in content)
        return str(content) if content else ''
    
    # Recursively check common message attributes
    for attr in ['message', 'chat_message', 'agent_response']:
        if hasattr(obj, attr):
            result = get_content(getattr(obj, attr))
            if result:
                return result
    
    return ''

def read_history_and_cache_result(history, cache, result):
    with open(history, "rb") as f:
        loaded_history = pickle.load(f)

    with open(cache, "rb") as f:
        loaded_cache = pickle.load(f)

    with open(result, "rb") as f:
        loaded_result = pickle.load(f)

    return loaded_history, loaded_cache, loaded_result


async def get_agent_team():
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

    assistant_agent = AssistantAgent(name="assistant_agent",system_message="You are a helpful assistant",model_client=model_client)
    surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model_client,
    )
    termination = MaxMessageTermination(4)
    team = RoundRobinGroupChat([assistant_agent, surfer], termination_condition=termination)

    return team

async def main():
    history = "output/1_complete_history/history.pickle"
    cache = "output/1_complete_history/cache.pickle"
    result = "output/1_complete_history/log.pickle"

    # detailed history and cache
    loaded_history, loaded_cache, loaded_result = read_history_and_cache_result(history, cache, result)

    # Extract team_id from cached agent IDs
    sample_key = next(iter(loaded_cache[1]))
    team_id = sample_key.split('/')[-1]
    print(f"Team ID: {team_id}")

    last_message_idx = 18

    team = await get_agent_team()
    team._team_id = team_id

    # Create backend to track history
    logger = logging.getLogger()
    backend = BackendRuntimeManager(team, logger, message_history=loaded_history, state_cache=loaded_cache) 
    await backend.async_initialize()

    print("=" * 60)
    print("PHASE 2: Truncate History")
    print("=" * 60)

    # Truncate history after the last message
    await backend.revert_message(last_message_idx)

    os.makedirs("output/2_truncated_history", exist_ok=True)

    # save the truncated history and cache
    await write_cache_and_history(backend.get_current_history_raw_type(), backend.agent_checkpoints)

    print(backend.get_current_history())
    print(backend.agent_checkpoints)

if __name__ == "__main__":
    result = asyncio.run(main())