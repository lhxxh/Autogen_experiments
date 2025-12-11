import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator

from dotenv import load_dotenv

from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import MagenticOneGroupChat, SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
TRANSCRIPTS_DIR = ARTIFACT_DIR / "transcripts"
SNAPSHOTS_DIR = ARTIFACT_DIR / "snapshots"
TREE_FILE = ARTIFACT_DIR / "tree.json"
LOG_FILE = BASE_DIR / "log.log"

ROUND_LIMIT = 4            # main path length
BRANCH_FROM = 2            # zero-based index to branch from main
DEFAULT_TASK = "On the BBC Earth YouTube video of the Top 5 Silliest Animal Moments, what species of bird is featured?"


def ensure_output_dirs():
    for path in (ARTIFACT_DIR, TRANSCRIPTS_DIR, SNAPSHOTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def jsonable(obj: Any):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def save_transcript(round_id: str, transcript) -> Path:
    path = TRANSCRIPTS_DIR / f"{round_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump([jsonable(m) for m in transcript], f, indent=2, default=str)
    return path


def save_snapshot(round_id: str, snapshot) -> Path:
    path = SNAPSHOTS_DIR / f"{round_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    return path


# def build_team(client: OpenAIChatCompletionClient) -> SelectorGroupChat:
#     coder = MagenticOneCoderAgent("Assistant", model_client=client)
#     web_surfer = MultimodalWebSurfer("WebSurfer", model_client=client)
#     return SelectorGroupChat(
#         [coder, web_surfer],
#         model_client=client,
#         max_turns=2,
#     )

def build_team(client: OpenAIChatCompletionClient) -> RoundRobinGroupChat:
    assistant = AssistantAgent("Assistant", model_client=client)
    web_surfer = MultimodalWebSurfer("WebSurfer", model_client=client)
    return RoundRobinGroupChat(
        [assistant, web_surfer],
        max_turns=2,
    )


async def run_one_round(
    team: MagenticOneGroupChat,
    task: str | None,
    label: str,
    round_id: str,
    parent_id: str | None,
    nodes: list[dict],
):
    task_text = task or "Continue the conversation to solve the task."
    print(f"\n=== {label} ===")
    print(f"Round {round_id} started, parent_id={parent_id}, task='{task_text}'")

    transcript = []

    async def tap(stream: AsyncGenerator):
        async for msg in stream:
            transcript.append(msg)
            yield msg

    stream = tap(team.run_stream(task=task_text))
    await Console(stream, output_stats=False)

    snapshot = await team.save_state()

    tx_path = save_transcript(round_id, transcript)
    snap_path = save_snapshot(round_id, snapshot)

    nodes.append(
        {
            "id": round_id,
            "label": label,
            "parent": parent_id,
            "transcript_path": str(tx_path),
            "snapshot_path": str(snap_path),
        }
    )

    print(f"Saved {label}: transcript->{tx_path.name}, snapshot->{snap_path.name}")
    return transcript, snapshot


async def main():
    load_dotenv()
    ensure_output_dirs()

    client = OpenAIChatCompletionClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )

    nodes: list[dict] = []

    # Root snapshot (empty state)
    root_team = build_team(client)
    root_snapshot = await root_team.save_state()
    root_snap_path = save_snapshot("root", root_snapshot)
    nodes.append(
        {
            "id": "root",
            "label": "Root (initial state)",
            "parent": None,
            "transcript_path": None,
            "snapshot_path": str(root_snap_path),
        }
    )

    snapshots: dict[str, Any] = {"root": root_snapshot}

    # Main path: each node loads parent snapshot before running
    user_task = os.getenv("INITIAL_TASK", DEFAULT_TASK)
    next_task = user_task
    parent_id = "root"
    parent_snapshot = snapshots[parent_id]

    for i in range(ROUND_LIMIT):
        round_id = f"main_r{i+1}"
        team = build_team(client)
        await team.load_state(parent_snapshot)  # load parent state
        _, snap = await run_one_round(
            team,
            task=next_task,
            label=f"Main round {i+1}",
            round_id=round_id,
            parent_id=parent_id,
            nodes=nodes,
        )
        snapshots[round_id] = snap
        parent_id = round_id
        parent_snapshot = snap
        next_task = None  # subsequent rounds continue same conversation

    # Branch: load from chosen main node snapshot, then continue
    branch_from_id = f"main_r{BRANCH_FROM+1}"
    branch_parent_snapshot = snapshots[branch_from_id]
    branch_parent = branch_from_id
    branch_task = "Branch: change course to focus on remote-friendly sessions and async collaboration."

    for j in range(BRANCH_FROM, ROUND_LIMIT):
        round_id = f"branch_r{j+1}"
        team = build_team(client)
        await team.load_state(branch_parent_snapshot)
        await run_one_round(
            team,
            task=branch_task if j == BRANCH_FROM else None,
            label=f"Branch round {j+1}",
            round_id=round_id,
            parent_id=branch_parent,
            nodes=nodes,
        )
        branch_parent = round_id
        branch_parent_snapshot = snapshots.get(branch_from_id, branch_parent_snapshot)
        branch_task = None  # inject branch instruction only once

    # Persist tree overview
    with TREE_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "root": "root",
                "branch_from": branch_from_id,
                "nodes": nodes,
            },
            f,
            indent=2,
        )

    print(f"\nDone. Artifacts saved under {ARTIFACT_DIR}")
    print(f"- Tree: {TREE_FILE}")
    print(f"- Transcripts: {TRANSCRIPTS_DIR}")
    print(f"- Snapshots: {SNAPSHOTS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())