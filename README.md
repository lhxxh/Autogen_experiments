# Setup

1. autogen推荐使用以下版本，否则代码有许多兼容问题，已经在pyproject.toml更新

```sh
autogen-agentchat  0.4.8
autogen-core       0.4.8
autogen-ext        0.4.8
```

2. 为了支持 Autogen web_surfer， 需要以下package：

```sh
pip install playwright
playwright install
```


# Test

1. 测试代码在tests文件夹下面， 测试serialization 

```sh
python tests/test_serialization.py
```

2. 测试backend, 需要 pytest-asyncio,

```sh
pip install pytest-asyncio
```

安装pytest之后，测试命令

```sh
pytest tests/test_backend.py # run all tests
pytest tests/test_backend.py::test_edit_message_queue # Run specific test
pytest tests/test_backend.py -v # With verbose output
```

# Examples

1. examples文件夹下提供使用例子， 拿local-agents举例

```sh
cd examples/local-agents 
```

2. 打开frontend

```sh
agdebugger scenario:get_agent_team
```

3. 在左上角messages启动聊天，先设置direct message 去 group_chat_manager, message 如下

```sh
0 
```

点击右侧绿色arrow，会被放在下方的message queue中处于待处理

4. 按红色按钮开始处理message queue

5. 处理过的message会被放在message history，overview会在右侧显示

6. 如果想修改聊天记录继续生成，直接在history上改再点击save & revert，但只能改Response Message才会继续生成，。


# Simple experiments

1. 做了俩部分实验， 都放在 simple_experiment下面

2. concise_length history 是对比 chat-agent level 和 runtime level 有什么区别

3. truncate_and_resume 是模拟检测到trace出现error，在error截断并重新生成

4. 结论：
    （1）Autogen 有俩层操作，关系如下

        ```sh
        ┌──────────────────┐
        │  team.run()      │  ← USER CALLS THIS
        └────────┬─────────┘
                │
                ▼
        ┌──────────────────────────────────────────────┐
        │      AgentChat Layer (High-Level)            │
        │  • Orchestrates conversation                 │
        │  • Returns TaskResult with .messages         │
        │  • GENERATES: Concise history (1.json)       │
        └────────┬─────────────────────────────────────┘
                │
                ▼
        ┌──────────────────────────────────────────────┐
        │      Runtime Layer (Low-Level)               │
        │  • Handles message routing                   │
        │  • Manages agent state                       │
        │  • InterventionHandler captures ALL          │
        │  • GENERATES: Detailed history (agdebugger)  │
        └──────────────────────────────────────────────┘

        ```  

    （2）相比较于 chat-agent level， runtime level 的 message 更加冗长。 
        AGdebugger在每次message传递时会触发handler去截取message，导致同一个information会连续出现在不同类型message里
        比如

        ```sh
        Timestamps 0-4: INITIALIZATION
        ├─ T0: GroupChatStart (user: "Count from 1 to 5")
        ├─ T1-2: Manager processes start
        ├─ T3: GroupChatRequestPublish ← TRIGGER for Agent1
        └─ T4: Manager's internal response

        Timestamps 5-6: AGENT1 TURN
        ├─ T5: GroupChatMessage (Agent1: "1") ← BROADCAST (informational)
        └─ T6: GroupChatAgentResponse (Agent1: "1") ← TO MANAGER (actionable)

        Timestamps 7-9: AGENT2 TURN  
        ├─ T7: GroupChatRequestPublish ← TRIGGER for Agent2
        ├─ T8: GroupChatMessage (Agent2: "1") ← BROADCAST
        └─ T9: GroupChatAgentResponse (Agent2: "1") ← TO MANAGER

        Timestamps 10-12: AGENT1 TURN
        ├─ T10: GroupChatRequestPublish ← TRIGGER for Agent1
        ├─ T11: GroupChatMessage (Agent1: "2") ← BROADCAST
        └─ T12: GroupChatAgentResponse (Agent1: "2") ← TO MANAGER
        ```

        除此之外，runtime level history 的记录更加raw data
        比如runtime level history 会记录image 的 Base64-encoded PNG image data，
        但对应的chat level history只简单用<image>来表示


    （3）AGdebugger 在每次截取message，设立一个timestamp，每个timestamp都出储存当下的runtime history和对应的state。
        所以相比较于chat level的 team.save_state/load_state只能load save最后一个message的state， AGdegbugger会记录整个conversation里每个message的state。
        
        
    （4）注意这里的runtime history只有AGdebugger自己会用，其底层的Autogen不需要这个runtime history。
        Autogen只依赖state去恢复状态。


    （4）并不是每个agent都能maintain state，只有那些依赖AssistantAgent才会maintain state通过记录llm_context.
        他们只有在group manager让他们回应时才会得到context，出现在message_buffer，回应结束后message_buffer自动清空。

     
    （5）Group manger的message_thread会记录context.他只会把记录的context传给AssistantAgent类的agent。
        如果这个team没有AssistantAgent类的agent，只有其他agent（e.g. websurfer， coder），那load state没有意义。

    (6) Round-robin不会出现load state跑team.run()会覆盖之前的context的情况。 
        Magentic-One load state没有意义因为新的任务team.run(task='...')会覆盖message_thread



# AGDebugger

AGDebugger is an interactive system to help you debug your agent teams. It offers interactions to:

1. Send and step through agent messages
2. Edit previously sent agent messages and revert to earlier points in a conversation
3. Navigate agent conversations with an interactive visualization

![screenshot of AGDebugger interface](.github/screenshots/agdebugger_sc.png)

## Local Install

You can install AGDebugger locally by cloning the repo and installing the python package.

```sh
# Install & build frontend
cd frontend
npm install
npm run build
# Install & build agdebugger python package
cd ..
pip install .
```

## Usage

AGDebugger is built on top of [AutoGen](https://microsoft.github.io/autogen/stable/). To use AGDebugger, you provide a python file that exposes a function that creates an [AutoGen AgentChat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html) team for debugging. You can then launch AgDebugger with this agent team.

For example, the script below creates a simple agent team with a single WebSurfer agent.

```python
# scenario.py
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def get_agent_team():
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model_client,
    )
    team = MagenticOneGroupChat([surfer], model_client=model_client)

    return team
```

We can then launch the interface with:

```sh
 agdebugger scenario:get_agent_team
```

Once in the interface, you can send a GroupChatStart message to the start the agent conversation and begin debugging!

## Citation

See our [CHI 2025 paper](https://arxiv.org/abs/2503.02068) for more details on the design and evaluation of AGDebugger.

```bibtex
@inproceedings{epperson25agdebugger,
    title={Interactive Debugging and Steering of Multi-Agent AI Systems},
    author={Will Epperson and Gagan Bansal and Victor Dibia and Adam Fourney and Jack Gerrits and Erkang Zhu and Saleema Amershi},
    year={2025},
    publisher = {Association for Computing Machinery},
    booktitle = {Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems},
    series = {CHI '25}
}
```
