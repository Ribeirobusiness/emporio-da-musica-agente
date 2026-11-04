"""Agent loop: function calling on the OpenAI Chat Completions API.

The loop is the classic ReAct cycle without a framework:
model -> tool_calls -> run tools in Python -> tool messages -> model ...
until the model answers in plain text or the step limit is hit.

This is the only module that knows which LLM provider is used. Tools,
data, policies and the system prompt are provider-agnostic.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from openai import OpenAI, OpenAIError

from .prompts import build_system_prompt
from .tools import STORE_TZ, TOOL_SCHEMAS, StoreTools

DEFAULT_MODEL = "gpt-4o-mini"
MAX_TOOL_STEPS = 8
MAX_TURNS_IN_CONTEXT = 12  # user turns kept in the prompt; older ones are dropped
FALLBACK_REPLY = (
    "Desculpa, tive um problema técnico agora e não consegui consultar o sistema. "
    "Pode tentar de novo em instantes? Se preferir, fale com a equipe pelo (67) 3341-4444."
)

# Tool schemas are written once (name, description, JSON Schema) and adapted
# to the OpenAI "function" format here.
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
    }
    for t in TOOL_SCHEMAS
]


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
    cached_input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0


def to_whatsapp(text: str) -> str:
    """WhatsApp bold is *one* asterisk; markdown **bold** shows up with the asterisks.
    gpt-4o-mini kept using ** in about 1 of 5 conversations despite the prompt, so it is
    converted here instead of relying on the model."""
    return re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)


def trim_history(messages: list[dict], max_user_turns: int = MAX_TURNS_IN_CONTEXT) -> list[dict]:
    """Keep the last N user turns. Cuts only at user messages, so an assistant
    message with tool_calls is never separated from its tool results."""
    starts = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(starts) <= max_user_turns:
        return messages
    return messages[starts[-max_user_turns]:]


class StoreAgent:
    def __init__(
        self,
        tools: StoreTools,
        client: OpenAI | None = None,
        model: str | None = None,
        history: list[dict] | None = None,
    ):
        self.tools = tools
        self.client = client or OpenAI()
        self.model = model or os.getenv("EMPORIO_MODEL", DEFAULT_MODEL)
        self.history: list[dict] = history or []

    def _system(self) -> dict:
        # The system prompt and tool schemas form a stable prefix of ~3k tokens;
        # OpenAI caches repeated prefixes automatically, so later calls bill it at a discount.
        now = self.tools.now_fn().astimezone(STORE_TZ)
        return {"role": "system", "content": build_system_prompt(now)}

    def reply(self, user_text: str) -> Turn:
        self.history.append({"role": "user", "content": user_text})
        messages = list(trim_history(self.history))  # copy: this list grows inside the loop
        turn = Turn(reply="")

        for _ in range(MAX_TOOL_STEPS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[self._system(), *messages],
                    tools=OPENAI_TOOLS,
                    max_completion_tokens=1024,
                )
            except OpenAIError as exc:
                turn.reply = FALLBACK_REPLY
                turn.tool_calls.append(ToolCall("api_error", {}, str(exc)))
                self.history.append({"role": "assistant", "content": FALLBACK_REPLY})
                return turn

            turn.model_calls += 1
            usage = response.usage
            if usage:
                turn.input_tokens += usage.prompt_tokens
                turn.output_tokens += usage.completion_tokens
                details = getattr(usage, "prompt_tokens_details", None)
                turn.cached_input_tokens += getattr(details, "cached_tokens", 0) or 0

            msg = response.choices[0].message
            if not msg.tool_calls:
                turn.reply = to_whatsapp((msg.content or "").strip())
                self.history.append({"role": "assistant", "content": turn.reply})
                return turn

            assistant_msg = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in msg.tool_calls
                ],
            }
            messages.append(assistant_msg)
            self.history.append(assistant_msg)

            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                    output = json.dumps({"erro": "Argumentos inválidos (JSON malformado)."}, ensure_ascii=False)
                else:
                    output = self.tools.call(call.function.name, args)
                turn.tool_calls.append(ToolCall(call.function.name, args, output))
                tool_msg = {"role": "tool", "tool_call_id": call.id, "content": output}
                messages.append(tool_msg)
                self.history.append(tool_msg)

        # Step limit reached: answer safely instead of looping forever
        turn.reply = ("Essa ficou mais complexa do que consigo resolver por aqui. "
                      "Quer que eu encaminhe para alguém da equipe?")
        self.history.append({"role": "assistant", "content": turn.reply})
        return turn
