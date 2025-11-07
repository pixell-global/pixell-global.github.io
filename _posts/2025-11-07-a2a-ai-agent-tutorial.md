---
layout: post
title: "A2A SDK로 AI Agent 구축하고 자연어로 연결하기 (Python)"
date: 2025-11-07 00:00:00 +0900
categories: a2a ai-agent python tutorial
tags: [a2a-sdk, langgraph, langchain, starlette, jsonrpc]
---

### 개요
이 글에서는 로컬에서 AI Agent 서버를 실행하고, A2A SDK 클라이언트로 자연어 메시지를 보내는 전체 과정을 안내합니다. 예제는 다음 저장소 구조를 기반으로 합니다.

```
b:\Workspace\a2a-ddd-sample\
  - main.py
  - executor.py
  - graph.py
  - state.py
  - tool.py
  - test_client.py
  - pyproject.toml
```

### 준비물
- Python 3.13+
- 가상환경(venv) 권장
- uv 또는 pip

설치 및 환경 구성:

```bash
# (선택) 가상환경 생성 및 활성화
python -m venv .venv
.\.venv\Scripts\activate

# uv가 있다면 (권장)
uv sync

# 또는 pip 사용 시
pip install -U pip
pip install -e .
```

`pyproject.toml`의 주요 의존성에는 `a2a-sdk`, `langgraph`, `langchain-openai`, `uvicorn` 등이 포함됩니다.

### 1) Tool 정의: 주사위/별 그리기
에이전트가 호출할 수 있는 툴을 `langchain_core.tools`로 정의합니다. 아래는 현재 프로젝트의 `tool.py` 일부입니다.

```1:40:b:\Workspace\a2a-ddd-sample\tool.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List
import os
import dotenv
import random
import math

dotenv.load_dotenv()

#region args_schema
class DrawStarArgs(BaseModel):
	string: str = Field(..., description="The string to draw the star from")

class RollDiceArgs(BaseModel):
	pass
#endregion

@tool(
	name_or_callable="draw_star",
	description="Draw a star from a string",
	args_schema=DrawStarArgs
)
def draw_star(string: str) -> str:
	if not isinstance(string, str):
		raise TypeError("string must be a str")
	if string == "":
		return ""

	# 토큰 길이에 따라 별의 격자 크기(셀 개수)를 자동 조정 (항상 홀수)
	base_cells = 21  # 기본 격자 폭/높이 (셀 단위)
	cells = base_cells - max(0, (len(string) - 1)) * 4
	cells = max( nine := 9, cells)  # 최소 9
	if cells % 2 == 0:
		cells += 1

	size = cells
	radius = size // 2
	cx = cy = radius
```

```40:114:b:\Workspace\a2a-ddd-sample\tool.py
	# ... (중략) ...
	return "\n".join(lines)

@tool(
	name_or_callable="roll_dice",
	description="Roll a dice",
	args_schema=RollDiceArgs
)
def roll_dice() -> int:
	return random.randint(1, 6)
```

### 2) 그래프 구성: LLM과 툴 바인딩
`graph.py`에서 LLM에 툴을 바인딩하고 간단한 상태 그래프를 구성합니다.

```12:22:b:\Workspace\a2a-ddd-sample\graph.py
from tool import draw_star, roll_dice

llm_tool = ChatOpenAI(model="gpt-4o-mini").bind_tools([draw_star, roll_dice])
```

```22:61:b:\Workspace\a2a-ddd-sample\graph.py
def agent_node(state: GraphState, llm) -> GraphState:
	"""LLM 에이전트: 툴을 선택하고 실행"""
	# ... (중략) ...
	#region agent_system_message
	system_message = SystemMessage(content="""You are an AI agent.
Select and execute the tools necessary to answer the user's question.

Tool information is in English. Translate the user's input into English, then choose the appropriate tools.

Important:
1. Precisely extract the required parameters from the query.
2. Provide a clear answer to the user based on the results.""")
	#endregion
```

그래프는 `should_continue`에서 답변이 생성되면 종료(END)로 흐름을 넘깁니다.

### 3) Executor: A2A 요청을 그래프로 연결
`executor.py`에서 A2A 서버가 수신한 메시지를 그래프로 전달하고, 응답 메시지를 생성합니다.

```1:36:b:\Workspace\a2a-ddd-sample\executor.py
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentSkill
from a2a.utils.message import new_agent_text_message
from graph import run_graph

class AIAgentExecutor(AgentExecutor):
	def __init__(self):
		self.agent_skills: list[AgentSkill] = [
			AgentSkill(
				id="ai_agent",
				name="ai_agent",
				description="AI Agent. It receives natural language and executes the function. So please pass it on to the parameters in natural language.\n [Parameters] query: user input natural language query",
				tags=["ai", "agent", "function"],
				examples=[
					"[query] hello? \n [parameters] query: hello?",
					"[query] please roll dice \n [parameters] query: please roll dice"
				]
			)
		]

	async def execute(
		self, context: RequestContext, event_queue: EventQueue
	) -> None:
		result = run_graph(context.message.metadata["params"]["query"])
		a2a_message = new_agent_text_message(result["answer"])
		await event_queue.enqueue_event(a2a_message)
```

### 4) 서버: A2A Starlette 앱 구동
`main.py`는 `A2AStarletteApplication`으로 JSON-RPC 서버를 띄웁니다. 기본 RPC 경로는 `/`이며, AgentCard의 `url`도 해당 경로를 가리킵니다.

```8:32:b:\Workspace\a2a-ddd-sample\main.py
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from executor import AIAgentExecutor

if __name__ == '__main__':

	ai_agent_executor = AIAgentExecutor()

	agent_card = AgentCard(
		name='AI Agent',
		description='AI Agent',
		url='http://localhost:9999/',
		version='1.0.0',
		defaultInputModes=['text'],
		defaultOutputModes=['text'],
		capabilities=AgentCapabilities(streaming=True),
		skills=ai_agent_executor.agent_skills,
	)

	request_handler = DefaultRequestHandler(
		agent_executor=ai_agent_executor,
		task_store=InMemoryTaskStore(),
	)

	server = A2AStarletteApplication(
		agent_card=agent_card, http_handler=request_handler
	)

	uvicorn.run(server.build(), host='0.0.0.0', port=9999)
```

실행:

```bash
python main.py
# 서버는 9999 포트에서 /.well-known/agent-card.json, / (JSON-RPC) 엔드포인트를 제공합니다.
```

### 5) 클라이언트: A2A SDK로 자연어 전송
입력 루프를 통해 자연어를 전송하는 간단한 클라이언트는 다음과 같습니다.

```1:55:b:\Workspace\a2a-ddd-sample\test_client.py
import asyncio
import uuid

from a2a.client.client_factory import ClientFactory
from a2a.types import Message, Part, Role, TextPart
from a2a.utils.message import get_message_text


AGENT_URL = "http://localhost:9999/"


async def send_once(client, user_text: str) -> None:
    msg = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=user_text))],
        message_id=str(uuid.uuid4()),
        # 서버의 Executor가 metadata.params.query를 사용하므로 함께 전달
        metadata={
            "params": {"query": user_text},
        },
    )

    async for event in client.send_message(msg):
        if isinstance(event, tuple):
            task, update = event
            # 상태 업데이트에 메시지가 포함되면 출력
            try:
                if update and getattr(update, "status", None) and update.status.message:
                    print(get_message_text(update.status.message))
            except Exception:
                pass
        else:
            # 최종 Message 응답
            print(get_message_text(event))


async def amain() -> None:
    client = await ClientFactory.connect(AGENT_URL)
    try:
        while True:
            user_text = await asyncio.to_thread(input, "> ")
            if not user_text:
                continue
            if user_text.strip().lower() in {"exit", "quit"}:
                break
            await send_once(client, user_text)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(amain())
```

실행:

```bash
python test_client.py
> please roll dice
> draw a star with string "*"
```

### 6) 동작 예시 프롬프트
- "please roll dice" → 주사위 숫자 응답
- "draw a star with string '*'" → `*`로 그린 별 ASCII 아트 응답
- "주사위를 굴려서 나온 숫자로 별을 그려줘" → 시스템 프롬프트에 따라 영어로 해석되어 툴 조합을 시도

### 7) 문제 해결
- 타임아웃/네트워크 오류(`httpx.ReadTimeout`, `A2AClientHTTPError`):
  - 서버가 실행 중인지(9999 포트), 방화벽 문제 없는지 확인
  - `AGENT_URL`이 서버의 AgentCard URL(`http://localhost:9999/`)과 일치하는지 확인
- 응답이 비어있음:
  - 프롬프트를 명확히(roll dice, draw star 등)
  - OpenAI 키 등 LLM 설정(.env) 확인

### 마무리
이 튜토리얼로 기본 AI Agent를 로컬에서 구동하고, A2A SDK를 통해 자연어 입력을 전송하는 엔드투엔드 흐름을 구성했습니다. 필요에 따라 툴을 추가하고, 권한/보안(AgentCard.security)이나 전송 프로토콜을 확장해보세요.


