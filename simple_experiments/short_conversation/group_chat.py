import asyncio, json, os
from pathlib import Path
from dotenv import load_dotenv

from autogen_agentchat.teams import RoundRobinGroupChat, MagenticOneGroupChat, SelectorGroupChat, Swarm
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import CodeExecutorAgent


BASE_DIR = Path(__file__).resolve().parent
STATE1 = BASE_DIR / "team1_state.json"
STATE2 = BASE_DIR / "team2_state.json"

# def build_team(client: OpenAIChatCompletionClient) -> RoundRobinGroupChat:
#     coder = MagenticOneCoderAgent("Assistant", model_client=client)
#     web_surfer = MultimodalWebSurfer("WebSurfer", model_client=client)
#     return RoundRobinGroupChat(
#         [coder, web_surfer],
#         max_turns=2,
#     )

# def build_team(client: OpenAIChatCompletionClient) -> MagenticOneGroupChat:
#     coder = MagenticOneCoderAgent("Assistant", model_client=client)
#     executor = CodeExecutorAgent("ComputerTerminal", code_executor=LocalCommandLineCodeExecutor())
#     file_surfer = FileSurfer("FileSurfer", model_client=client)
#     web_surfer = MultimodalWebSurfer("WebSurfer", model_client=client)
#     return MagenticOneGroupChat(
#         [coder, executor, file_surfer, web_surfer],
#         model_client=client,
#         max_turns=2,
#     )

# def build_team(client: OpenAIChatCompletionClient) -> SelectorGroupChat:
#     coder = MagenticOneCoderAgent("Assistant", model_client=client)
#     web_surfer = MultimodalWebSurfer("WebSurfer", model_client=client)
#     return SelectorGroupChat(
#         [coder, web_surfer],
#         model_client=client,
#         max_turns=2,
#     )

def build_team(client: OpenAIChatCompletionClient) -> Swarm:
    coder = MagenticOneCoderAgent("Assistant", model_client=client)
    web_surfer = MultimodalWebSurfer("WebSurfer", model_client=client)
    return Swarm(
        [coder, web_surfer],
        max_turns=2,
    )


async def main():
    load_dotenv()
    client = OpenAIChatCompletionClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )

    team1 = build_team(client)
    await Console(team1.run_stream(task="Search one poem written by Li Bai."), output_stats=False)
    state1 = await team1.save_state()
    STATE1.write_text(json.dumps(state1, indent=2), encoding="utf-8")

    team2 = build_team(client)
    await team2.load_state(json.loads(STATE1.read_text(encoding="utf-8")))
    await Console(team2.run_stream(task="What poem you mentioned before? Answer briefly."), output_stats=False)
    state2 = await team2.save_state()
    STATE2.write_text(json.dumps(state2, indent=2), encoding="utf-8")

if __name__ == "__main__":
    asyncio.run(main())