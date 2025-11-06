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

from autogen_core import DefaultTopicId, AgentId
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import MagenticOneGroupChat
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

    surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model_client,
    )

    team = MagenticOneGroupChat([surfer], model_client=model_client, max_turns = 10)

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
    backend.start_processing()  # Start runtime BEFORE publishing

    print("=" * 60)
    print("PHASE 3: Replay at Truncation")
    print("=" * 60)

    # # check if the team has successfully loaded the state
    team_state = await team.save_state()
    print(team_state['agent_states'].keys())
    print(team_state['agent_states']['group_chat_manager/64a24a5b-b88f-4f6e-9b56-a50f6de1bd5b'])
    print(team_state['agent_states']['collect_output_messages/64a24a5b-b88f-4f6e-9b56-a50f6de1bd5b'])
    print(team_state['agent_states']['WebSurfer/64a24a5b-b88f-4f6e-9b56-a50f6de1bd5b'])

    ppppppppppppppppppppp

    # print("=" * 60)
    # print("Current history:")
    # print(backend.get_current_history())
    # print("=" * 60)

    # NOTE: New message after the truncation point.
    # result = await team.run(task="Please summarize everything you have discovered about Seattle so far in our conversation. What tasks have you completed? DO NOT ASK TO WebSurfer to SEARCH ANYTHING")
    # print(result)
    # await Console(team.run_stream(task="Please summarize everything you have discovered so far in our conversation. What tasks have you completed? DO NOT ASK TO WebSurfer to SEARCH ANYTHING"))
    
    msg = TextMessage(content="Please summarize everything you have discovered so far in our conversation. DO NOT ASK TO WebSurfer to SEARCH ANYTHING", source="LLM feedback")
    # topic = DefaultTopicId(
    #     type=backend.groupchat._group_topic_type,
    #     source=team_id  # Use team_id as topic source!
    # )
    # backend.publish_message(msg, topic)

    # Send GroupChatStart with the message
    group_chat_start = GroupChatStart(messages=[msg])

    # Send to group chat manager, not group topic
    await backend.runtime.send_message(
        group_chat_start,
        recipient=AgentId(
            type=backend.groupchat._group_chat_manager_topic_type,
            key=team_id
        )
    )

    # for i in range(10):
    #     await backend.process_next()

    #     newest = backend.get_current_history()[-1]
    #     print(f"Step {i+1}: {newest['content'] if 'content' in newest else newest}")

    # await backend.stop_processing()
    
    await backend.runtime.stop_when_idle()

    os.makedirs("output/3_replay_at_truncation", exist_ok=True)

    # save the truncated history and cache
    await write_cache_and_history(backend.get_current_history_raw_type(), backend.agent_checkpoints)

    #print(backend.get_current_history())
    #print(backend.agent_checkpoints)

if __name__ == "__main__":
    asyncio.run(main())