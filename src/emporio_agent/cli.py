"""Terminal chat with the store agent.

    python -m emporio_agent.cli                 # new conversation
    python -m emporio_agent.cli --session ana   # resume / persist conversation "ana"
    python -m emporio_agent.cli --debug         # also print every tool call

Inside the chat: /sair to quit, /nova to clear the current session.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .agent import StoreAgent
from .data import load_data
from .policies import load_sections
from .tools import StoreTools

SESSIONS_DIR = Path("sessions")


def load_session(name: str) -> list[dict]:
    path = SESSIONS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_session(name: str, history: list[dict]) -> None:
    SESSIONS_DIR.mkdir(exist_ok=True)
    (SESSIONS_DIR / f"{name}.json").write_text(json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Empório da Música customer service agent")
    parser.add_argument("--session", help="persist the conversation under this name (sessions/<name>.json)")
    parser.add_argument("--model", help="OpenAI model id (default: EMPORIO_MODEL or gpt-4o-mini)")
    parser.add_argument("--debug", action="store_true", help="print tool calls and token usage")
    args = parser.parse_args(argv)

    history = load_session(args.session) if args.session else []
    agent = StoreAgent(StoreTools(load_data(), load_sections()), model=args.model, history=history)

    print("Empório da Música | assistente virtual (digite /sair para encerrar)\n")
    if history:
        print(f"[sessão '{args.session}' retomada com {sum(1 for m in history if isinstance(m['content'], str) and m['role'] == 'user')} mensagens anteriores]\n")

    while True:
        try:
            text = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/sair":
            break
        if text == "/nova":
            agent.history.clear()
            print("[conversa reiniciada]\n")
            continue

        turn = agent.reply(text)
        if args.debug:
            for call in turn.tool_calls:
                print(f"  [tool] {call.name}({json.dumps(call.input, ensure_ascii=False)})", file=sys.stderr)
                print(f"         -> {call.output[:300]}{'...' if len(call.output) > 300 else ''}", file=sys.stderr)
            print(f"  [tokens] in={turn.input_tokens} (cached {turn.cached_input_tokens}) out={turn.output_tokens}", file=sys.stderr)
        print(f"\nDuda: {turn.reply}\n")
        if args.session:
            save_session(args.session, agent.history)


if __name__ == "__main__":
    main()
