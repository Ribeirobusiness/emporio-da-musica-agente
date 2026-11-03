"""Agent loop tests with a fake client (no network)."""

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emporio_agent.agent import StoreAgent, trim_history  # noqa: E402
from emporio_agent.data import load_data  # noqa: E402
from emporio_agent.policies import load_sections  # noqa: E402
from emporio_agent.tools import STORE_TZ, StoreTools  # noqa: E402


class Block(NS):
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None}


def response(blocks, stop):
    return NS(content=blocks, stop_reason=stop, usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    """Scripted model: first asks for a tool, then answers with text."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        return self.script.pop(0)


def make_agent(client):
    tools = StoreTools(load_data(), load_sections(),
                       now_fn=lambda: datetime(2026, 3, 24, 10, 0, tzinfo=STORE_TZ))
    return StoreAgent(tools, client=client, model="fake")


def test_tool_call_then_answer():
    client = FakeClient([
        response([Block(type="tool_use", id="t1", name="detalhes_produto", input={"produto": "Takamine GD20"})], "tool_use"),
        response([Block(type="text", text="O Takamine GD20 sai por R$ 2.199,00.")], "end_turn"),
    ])
    agent = make_agent(client)
    turn = agent.reply("Quanto custa o Takamine GD20?")
    assert turn.reply == "O Takamine GD20 sai por R$ 2.199,00."
    assert [c.name for c in turn.tool_calls] == ["detalhes_produto"]
    assert json.loads(turn.tool_calls[0].output)["preco_tabela"] == "R$ 2.199,00"
    # second API call must carry the tool_result paired with the tool_use id
    last_user = client.calls[1][-1]
    assert last_user["role"] == "user" and last_user["content"][0]["tool_use_id"] == "t1"
    # history: user, assistant(tool_use), user(tool_result), assistant(text)
    assert [m["role"] for m in agent.history] == ["user", "assistant", "user", "assistant"]


def test_step_limit_stops_runaway_loops():
    loop = [response([Block(type="tool_use", id=f"t{i}", name="status_da_loja", input={})], "tool_use")
            for i in range(20)]
    agent = make_agent(FakeClient(loop))
    turn = agent.reply("oi")
    assert "encaminhe" in turn.reply
    assert len(turn.tool_calls) == 8


def test_trim_history_keeps_tool_pairs():
    history = []
    for i in range(20):
        history += [
            {"role": "user", "content": f"msg {i}"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
    trimmed = trim_history(history, max_user_turns=3)
    assert trimmed[0] == {"role": "user", "content": "msg 17"}
    assert len(trimmed) == 12
