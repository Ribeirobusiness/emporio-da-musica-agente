"""Agent loop tests with a fake OpenAI client (no network, no API key)."""

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


def tool_call(call_id, name, args):
    return NS(id=call_id, function=NS(name=name, arguments=json.dumps(args)))


def response(content=None, tool_calls=None):
    usage = NS(prompt_tokens=100, completion_tokens=20, prompt_tokens_details=NS(cached_tokens=0))
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))], usage=usage)


class FakeClient:
    """Scripted model: returns the queued responses in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        return self.script.pop(0)


def make_agent(client):
    tools = StoreTools(load_data(), load_sections(),
                       now_fn=lambda: datetime(2026, 3, 24, 10, 0, tzinfo=STORE_TZ))
    return StoreAgent(tools, client=client, model="fake")


def test_tool_call_then_answer():
    client = FakeClient([
        response(tool_calls=[tool_call("c1", "detalhes_produto", {"produto": "Takamine GD20"})]),
        response(content="O Takamine GD20 sai por R$ 2.199,00."),
    ])
    agent = make_agent(client)
    turn = agent.reply("Quanto custa o Takamine GD20?")
    assert turn.reply == "O Takamine GD20 sai por R$ 2.199,00."
    assert [c.name for c in turn.tool_calls] == ["detalhes_produto"]
    assert json.loads(turn.tool_calls[0].output)["preco_tabela"] == "R$ 2.199,00"
    # the second API call must carry the tool result paired with the call id
    last = client.calls[1][-1]
    assert last["role"] == "tool" and last["tool_call_id"] == "c1"
    assert [m["role"] for m in agent.history] == ["user", "assistant", "tool", "assistant"]
    assert turn.model_calls == 2 and turn.input_tokens == 200


def test_markdown_bold_becomes_whatsapp_bold():
    agent = make_agent(FakeClient([response(content="Seu protocolo é **EM-20261104-1234**.")]))
    turn = agent.reply("Registra pra mim?")
    assert turn.reply == "Seu protocolo é *EM-20261104-1234*."
    assert agent.history[-1]["content"] == turn.reply


def test_malformed_arguments_do_not_crash():
    bad = NS(id="c1", function=NS(name="detalhes_produto", arguments="{not json"))
    agent = make_agent(FakeClient([response(tool_calls=[bad]), response(content="ok")]))
    turn = agent.reply("oi")
    assert "erro" in json.loads(turn.tool_calls[0].output)
    assert turn.reply == "ok"


def test_step_limit_stops_runaway_loops():
    loop = [response(tool_calls=[tool_call(f"c{i}", "status_da_loja", {})]) for i in range(20)]
    agent = make_agent(FakeClient(loop))
    turn = agent.reply("oi")
    assert "encaminhe" in turn.reply
    assert len(turn.tool_calls) == 8


def test_trim_history_keeps_tool_pairs():
    history = []
    for i in range(20):
        history += [
            {"role": "user", "content": f"msg {i}"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "{}"},
            {"role": "assistant", "content": "ok"},
        ]
    trimmed = trim_history(history, max_user_turns=3)
    assert trimmed[0] == {"role": "user", "content": "msg 17"}
    assert len(trimmed) == 12
