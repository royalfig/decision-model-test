"""Per-turn signals for the onboarding agent, from a decision model or an LLM.

Both classifiers answer the same four questions about the conversation so the
benchmark can compare them directly:

- handoff: probability that the caller asks to speak with a person.
- intent: what the caller is doing in their latest turn.
- frustration: how frustrated the caller sounds, on a 0 to 2 scale.
- confirmed: probability that the caller confirms the details the agent just read back.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from livekit.agents import inference, llm

logger = logging.getLogger("signals")

HANDOFF_PROMPT = "In their latest turn, the caller asks to speak with a person instead of the automated agent."

INTENT_PROMPT = "What is the caller doing in their latest turn?"
INTENT_OPTIONS = {
    "provide_info": "Answers the agent's question or gives details about their business.",
    "ask_question": "Asks what a question means or how the application process works.",
    "coverage_advice": "Asks which coverage to buy, how much coverage they need, or whether something is covered.",
    "needs_time": "Asks for a moment to find information or step away.",
    "off_topic": "Talks about something unrelated to the insurance application.",
}

FRUSTRATION_PROMPT = "How frustrated does the caller sound in their latest turn?"
FRUSTRATION_LEVELS = ["calm", "impatient", "frustrated"]

CONFIRMED_PROMPT = (
    "The agent's last turn reads details back to the caller, and in their latest turn "
    "the caller confirms that those details are correct, without a correction."
)

DECISIONS: dict[str, llm.Decision] = {
    "handoff": llm.Probability(HANDOFF_PROMPT),
    "intent": llm.Choice(INTENT_PROMPT, INTENT_OPTIONS),
    "frustration": llm.Score(FRUSTRATION_PROMPT, FRUSTRATION_LEVELS),
    "confirmed": llm.Probability(CONFIRMED_PROMPT),
}


@dataclass
class TurnSignals:
    handoff: float
    intent: str
    frustration: float
    confirmed: float
    latency: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    source: str = ""


def transcript(chat_ctx: llm.ChatContext) -> str:
    """Render user and assistant text as ``role: text`` lines.

    This matches the transcript that DecisionModel builds from a ChatContext, so
    both classifiers see the same input.
    """
    lines = []
    for item in chat_ctx.items:
        if item.type != "message" or item.role not in ("user", "assistant"):
            continue
        if text := item.text_content:
            lines.append(f"{item.role}: {text}")
    return "\n".join(lines)


@dataclass
class Evaluation:
    """Answers to a set of decisions. Probabilities and scores are floats; choices are
    option keys."""

    values: dict[str, float | str]
    latency: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    errors: dict[str, str] = field(default_factory=dict)
    """Decisions the model declined or failed, keyed by name. They're absent from values."""


def _signals(e: Evaluation, source: str) -> TurnSignals:
    # A declined decision mustn't drop the others, so fall back to neutral values.
    if e.errors:
        logger.warning(f"[{source}] decisions without an answer: {e.errors}")
    return TurnSignals(
        handoff=float(e.values.get("handoff", 0.0)),
        intent=str(e.values.get("intent", "unknown")),
        frustration=float(e.values.get("frustration", 0.0)),
        confirmed=float(e.values.get("confirmed", 0.0)),
        latency=e.latency,
        input_tokens=e.input_tokens,
        output_tokens=e.output_tokens,
        source=source,
    )


class DecisionClassifier:
    def __init__(self, model: str = "openai/gpt-6-luna-decisions") -> None:
        self.name = model
        self._model = inference.DecisionModel(model)

    async def evaluate(
        self, conversation: str, decisions: dict[str, llm.Decision]
    ) -> Evaluation:
        started = time.perf_counter()
        response = await self._model.evaluate(input=conversation, decisions=decisions)
        latency = time.perf_counter() - started
        return Evaluation(
            values={name: r.value for name, r in response.results.items()},
            latency=latency,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            errors=dict(response.errors),
        )

    async def classify(self, conversation: str) -> TurnSignals:
        return _signals(await self.evaluate(conversation, DECISIONS), self.name)


def _llm_instructions(decisions: dict[str, llm.Decision]) -> str:
    """Describe the decisions as a JSON reply format for a general-purpose LLM."""
    fields = []
    lines = []
    for name, d in decisions.items():
        if isinstance(d, llm.Probability):
            fields.append(f'"{name}": <number from 0 to 1>')
            lines.append(f"- {name}: the probability that this is true: {d.prompt}")
        elif isinstance(d, llm.Choice):
            fields.append(f'"{name}": <string>')
            options = "\n".join(f'  - "{k}": {v}' for k, v in d.options.items())
            lines.append(
                f"- {name}: {d.prompt} Use exactly one of these keys:\n{options}"
            )
        else:
            top = len(d.levels) - 1
            fields.append(f'"{name}": <number from 0 to {top}>')
            levels = ", ".join(f"{i} = {level}" for i, level in enumerate(d.levels))
            lines.append(f"- {name}: {d.prompt} {levels}.")
    return (
        "You classify the latest caller turn in a phone call with an insurance "
        "onboarding agent.\nRead the transcript and reply with a single JSON object "
        "and nothing else:\n\n{" + ", ".join(fields) + "}\n\n" + "\n".join(lines) + "\n"
    )


def _parse(text: str, decisions: dict[str, llm.Decision]) -> dict[str, float | str]:
    """Parse and validate an LLM's JSON answer. Raises ValueError on a bad answer."""
    data = json.loads(_FENCE.sub("", text.strip()))
    values: dict[str, float | str] = {}
    for name, d in decisions.items():
        if name not in data:
            raise ValueError(f"missing {name}")
        if isinstance(d, llm.Choice):
            if data[name] not in d.options:
                raise ValueError(f"{name}: invalid option {data[name]!r}")
            values[name] = str(data[name])
        else:
            value = float(data[name])
            top = 1 if isinstance(d, llm.Probability) else len(d.levels) - 1
            if not 0 <= value <= top:
                raise ValueError(f"{name}: {value} out of range")
            values[name] = value
    return values


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class LLMClassifier:
    """The same decisions answered by a general-purpose LLM with a JSON prompt."""

    def __init__(self, model: str = "google/gemma-4-31b-it") -> None:
        self.name = model
        self._llm = inference.LLM(model=model)

    async def evaluate(
        self, conversation: str, decisions: dict[str, llm.Decision]
    ) -> Evaluation:
        ctx = llm.ChatContext()
        ctx.add_message(role="system", content=_llm_instructions(decisions))
        ctx.add_message(role="user", content=f"Transcript:\n{conversation}")

        started = time.perf_counter()
        text = ""
        input_tokens = output_tokens = None
        async with self._llm.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
        latency = time.perf_counter() - started
        return Evaluation(
            values=_parse(text, decisions),
            latency=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def classify(self, conversation: str) -> TurnSignals:
        return _signals(await self.evaluate(conversation, DECISIONS), self.name)
