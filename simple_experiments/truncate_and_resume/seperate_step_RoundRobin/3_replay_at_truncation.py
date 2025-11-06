'''
This script is used to replay the conversation at the truncation point.
'''

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
from autogen_agentchat.agents import AssistantAgent

from autogen_core import DefaultTopicId, AgentId
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

from autogen_agentchat.teams._group_chat._events import GroupChatStart


async def write_cache_and_history(history, cache):
    hist_path = f"output/3_replay_at_truncation/history.pickle"
    cache_path = f"output/3_replay_at_truncation/cache.pickle"

    await write_file_async(hist_path, history)
    await write_file_async(cache_path, cache)

    print("Saved AgDebugger cache files to: ", [hist_path, cache_path])

async def write_file_async(path, data):
    async with aiofiles.open(path, "wb") as f:
        buffer = pickle.dumps(data)
        await f.write(buffer)

def read_history_and_cache(history, cache):
    with open(history, "rb") as f:
        loaded_history = pickle.load(f)

    with open(cache, "rb") as f:
        loaded_cache = pickle.load(f)

    return loaded_history, loaded_cache

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
    history = "output/2_truncated_history/history.pickle"
    cache = "output/2_truncated_history/cache.pickle"

    loaded_history, loaded_cache = read_history_and_cache(history, cache)

    # Extract team_id from cached agent IDs
    sample_key = next(iter(loaded_cache[1]))
    team_id = sample_key.split('/')[-1]
    print(f"Team ID: {team_id}")

    team = await get_agent_team()
    team._team_id = team_id

    # Create backend to track history
    logger = logging.getLogger()
    backend = BackendRuntimeManager(team, logger, message_history=loaded_history, state_cache=loaded_cache) 
    await backend.async_initialize()

    print("=" * 60)
    print("PHASE 3: Replay at Truncation")
    print("=" * 60)

    await Console(team.run_stream(task="Please summarize everything you have discovered so far in our conversation. DO NOT ASK TO WebSurfer to SEARCH ANYTHING"))

    os.makedirs("output/3_replay_at_truncation", exist_ok=True)
    await write_cache_and_history(backend.get_current_history_raw_type(), backend.agent_checkpoints)



if __name__ == "__main__":
    asyncio.run(main())