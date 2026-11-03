"""Run a scripted text conversation through Morgan and print each turn.

Usage:
    uv run python bench/conversation.py
    CLASSIFIER=none uv run python bench/conversation.py

The script uses session.run(), which doesn't call on_user_turn_completed. Spoken turns
do, so the script calls the hook itself before each turn.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env.local")

from livekit.agents import AgentSession, StopResponse, inference, llm, utils  # noqa: E402

import agent  # noqa: E402

TURNS = [
    "Hi, I want to get insurance for my bakery. It's Riverside Bakery LLC.",
    "Five five five, two one two, four four eight nine.",
    "Close, but it ends in four four nine eight.",
    "Yep, that's it.",
    "General liability.",
    "Twelve employees.",
    "About eight hundred thousand a year.",
    "Nine four one one oh.",
    "Yeah, wait, no. It's nine four one one two.",
    "Mm-hmm, correct.",
]


async def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="  %(message)s")
    logging.getLogger("agent").setLevel(logging.INFO)

    async with (
        utils.http_context.open(),
        inference.LLM(model="google/gemma-4-31b-it") as reply_llm,
        AgentSession(llm=reply_llm) as session,
    ):
        morgan = agent.Morgan()
        await session.start(morgan)
        for text in TURNS:
            print(f"\nUSER: {text}")
            try:
                await morgan.on_user_turn_completed(
                    morgan.chat_ctx.copy(), llm.ChatMessage(role="user", content=[text])
                )
            except StopResponse:
                print("  (transferred, no reply)")
                break
            result = await session.run(user_input=text)
            for ev in result.events:
                if ev.type == "message":
                    print(f"AGENT: {ev.item.text_content}")
                elif ev.type == "function_call":
                    print(f"  -> {ev.item.name}({ev.item.arguments})")
                elif ev.type == "function_call_output":
                    print(f"  <- {ev.item.output}")


if __name__ == "__main__":
    asyncio.run(main())
