"""Agent loop: native tool use (function calling) on the Anthropic Messages API.

The loop is the classic ReAct cycle without a framework:
model -> tool_use blocks -> run tools in Python -> tool_result -> model ...
until the model answers in plain text or the step limit is hit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import anthropic

from .prompts import build_system_prompt
from .tools import STORE_TZ, TOOL_SCHEMAS, StoreTools

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOOL_STEPS = 8
MAX_TURNS_IN_CONTEXT = 12  # user turns kept in the prompt; older ones are dropped
FALLBACK_REPLY = (
    "Desculpa, tive um problema técnico agora e não consegui consultar o sistema. "
    "Pode tentar de novo em instantes? Se preferir, fale com a equipe pelo (67) 3341-4444."
)


@dataclass
class ToolCall:
    name: str
    input: dict
    output: str


@dataclass
class Turn:
    reply: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def _to_plain(content: Any) -> Any:
    """SDK content blocks -> plain dicts, so history can be saved as JSON."""
    if isinstance(content, list):
        return [b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else b for b in content]
    return content


def trim_history(messages: list[dict], max_user_turns: int = MAX_TURNS_IN_CONTEXT) -> list[dict]:
    """Keep the last N user turns. Cuts only at plain-text user messages, so a
    tool_use is never separated from its tool_result."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], str)]
    if len(starts) <= max_user_turns:
        return messages
    return messages[starts[-max_user_turns]:]


class StoreAgent:
    def __init__(
        self,
        tools: StoreTools,
        client: anthropic.Anthropic | None = None,
        model: str | None = None,
        history: list[dict] | None = None,
    ):
        self.tools = tools
        self.client = client or anthropic.Anthropic()
        self.model = model or os.getenv("EMPORIO_MODEL", DEFAULT_MODEL)
        self.history: list[dict] = history or []

    def _system(self) -> list[dict]:
        # cache_control lets the API reuse the long, stable prefix (tools + system) across calls
        now = self.tools.now_fn().astimezone(STORE_TZ)
        return [{"type": "text", "text": build_system_prompt(now), "cache_control": {"type": "ephemeral"}}]

    def reply(self, user_text: str) -> Turn:
        self.history.append({"role": "user", "content": user_text})
        messages = list(trim_history(self.history))  # copy: this list grows inside the loop
        turn = Turn(reply="")
        tools = [dict(t) for t in TOOL_SCHEMAS]
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

        for _ in range(MAX_TOOL_STEPS):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=self._system(),
                    tools=tools,
                    messages=messages,
                )
            except anthropic.APIError as exc:
                turn.reply = FALLBACK_REPLY
                turn.tool_calls.append(ToolCall("api_error", {}, str(exc)))
                self.history.append({"role": "assistant", "content": FALLBACK_REPLY})
                return turn

            turn.input_tokens += response.usage.input_tokens
            turn.output_tokens += response.usage.output_tokens
            assistant_msg = {"role": "assistant", "content": _to_plain(response.content)}
            messages.append(assistant_msg)
            self.history.append(assistant_msg)

            if response.stop_reason != "tool_use":
                turn.reply = "".join(b.text for b in response.content if b.type == "text").strip()
                return turn

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                output = self.tools.call(block.name, block.input)
                turn.tool_calls.append(ToolCall(block.name, dict(block.input), output))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            tool_msg = {"role": "user", "content": results}
            messages.append(tool_msg)
            self.history.append(tool_msg)

        # Step limit reached: answer safely instead of looping forever
        turn.reply = ("Essa ficou mais complexa do que consigo resolver por aqui. "
                      "Quer que eu encaminhe para alguém da equipe?")
        self.history.append({"role": "assistant", "content": turn.reply})
        return turn
