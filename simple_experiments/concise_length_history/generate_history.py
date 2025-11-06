# generate_both_histories.py
import asyncio
import json
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# For detailed history
from agdebugger.backend import BackendRuntimeManager
import logging

def extract_concise_from_detailed(detailed_history):
    """
    Convert detailed runtime history back to concise format
    """
    concise = []
    user_added = False  # Track if user message already added
    
    for entry in detailed_history:
        msg = entry.get('message', {})
        msg_type = msg.get('type')
        
        # Extract initial user message from GroupChatStart (only once)
        if msg_type == 'GroupChatStart' and not user_added:
            messages = msg.get('messages', [])
            for m in messages:
                if m.get('source') == 'user':
                    concise.append({
                        "role": "user",
                        "name": "user",
                        "content": m.get('content', '')
                    })
                    user_added = True
        
        # Only include GroupChatMessage (not GroupChatAgentResponse to avoid duplicates)
        elif msg_type == 'GroupChatMessage':
            inner_msg = msg.get('message', {})
            content = inner_msg.get('content', '')
            source = inner_msg.get('source', 'assistant')
            
            # Skip termination messages
            if inner_msg.get('type') != 'StopMessage':
                concise.append({
                    "role": source,
                    "name": source,
                    "content": content
                })
    
    return concise


async def compare_histories():
    """
    Compare both formats side-by-side - USE SAME TEAM RUN
    """
    print("\n" + "=" * 60)
    print("COMPARISON: CONCISE vs DETAILED")
    print("=" * 60)
    
    # Setup ONE team with backend
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-pro", api_key="sk-xGE2JSXAbA4XjxJ6DT1vtQrUADE80Kdjb2iSjIJdtT7bKOm9", base_url="https://www.chataiapi.com/v1"
    )
    
    from local_agent import LocalAgent
    agent1 = LocalAgent("Agent1", model_client=model_client)
    agent2 = LocalAgent("Agent2", model_client=model_client)
    
    termination = MaxMessageTermination(5)
    team = RoundRobinGroupChat([agent1, agent2], termination_condition=termination)
    
    # Create backend BEFORE running team
    logger = logging.getLogger()
    backend = BackendRuntimeManager(team, logger)
    await backend.async_initialize()
    
    # Run the team (this will capture in backend's intervention handler)
    result = await team.run(task="Count from 1 to 5")
    
    # Extract CONCISE from result
    concise_history = []
    for msg in result.messages:
        concise_history.append({
            "role": getattr(msg, 'source', 'assistant'),
            "name": getattr(msg, 'source', None),
            "content": msg.content if hasattr(msg, 'content') else str(msg)
        })
    
    # Extract DETAILED from backend
    detailed_history = backend.get_current_history()
    
    # Save both
    with open('concise_history.json', 'w') as f:
        json.dump({"question": "Count from 1 to 5", "history": concise_history}, f, indent=2)
    
    with open('detailed_history.json', 'w') as f:
        json.dump(detailed_history, f, indent=2, default=str)
    
    # Print comparison
    print(f"\n📄 CONCISE: {len(concise_history)} messages")
    print(f"📋 DETAILED: {len(detailed_history)} messages")
    print(f"📊 RATIO: ~{len(detailed_history) / max(len(concise_history), 1):.1f}x")

    extracted_concise_history = extract_concise_from_detailed(detailed_history)
    print("Extracted concise history: ", extracted_concise_history)


if __name__ == "__main__":
    asyncio.run(compare_histories())