import logging
import os
import textwrap
from typing import Literal

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    StopResponse,
    STTContextOptions,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    llm,
    room_io,
)
from livekit.plugins import ai_coustics

from signals import DecisionClassifier, LLMClassifier, transcript

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Set CLASSIFIER in .env.local or the shell to compare approaches:
#   decision  Luna decision model classifies each user turn (default)
#   llm       the same classification with a general-purpose LLM
#   none      no classifier; the main LLM handles handoff and confirmation with tools
CLASSIFIER = os.getenv("CLASSIFIER", "decision")
DECISION_MODEL = os.getenv("DECISION_MODEL", "openai/gpt-6-luna-decisions")
CLASSIFIER_LLM = os.getenv("CLASSIFIER_LLM", "google/gemma-4-31b-it")

HANDOFF_THRESHOLD = 0.7
FRUSTRATION_THRESHOLD = 1.5
CONFIRMED_THRESHOLD = 0.8

TRANSFER_MESSAGE = (
    "Of course. I'm connecting you with a licensed member of our team now. "
    "Please stay on the line."
)

Field = Literal[
    "business_name", "phone", "coverage", "employees", "annual_revenue", "zip_code"
]
REQUIRED_FIELDS: tuple[str, ...] = Field.__args__
# Details that the caller must confirm after a readback before the application can go in.
READBACK_FIELDS = ("phone", "zip_code")

INSTRUCTIONS = textwrap.dedent(
    """\
    You are Morgan, the onboarding assistant for Acme Commercial Insurance. You talk with
    small business owners on the phone and collect the details for their insurance
    application. A licensed team member reviews the application and follows up with quotes.

    # Output rules

    You are interacting with the user via voice. Respond in plain text only, with no
    markdown, lists, or emojis. Keep replies to one or two sentences and ask one question
    at a time. Spell out numbers, phone numbers, and email addresses.

    # Application flow

    Collect these details in order. Skip any detail the caller already gave.

    1. The legal name of the business.
    2. The business phone number.
    3. The coverage they want: general liability, workers' compensation, commercial auto,
       or a business owner's policy.
    4. The number of employees.
    5. The approximate annual revenue.
    6. The ZIP code of the main business location.

    Call record_detail each time the caller gives a detail, or corrects one. For the phone
    number and the ZIP code, read the value back and wait for the caller to confirm it
    before you ask the next question.

    When you have every detail, call submit_application. If it reports a problem, fix it
    with the caller and try again. When it succeeds, tell the caller that a licensed team
    member will follow up with quotes.

    # Guardrails

    - You can explain what a question means, but never recommend coverage, say how much
      coverage they need, or confirm that something is covered. Offer to have a licensed
      team member answer those questions.
    - If the caller needs time to find information, tell them to take their time.
    - Keep the conversation on the application. Politely steer back if it drifts.
    """
)


# Without a classifier, the LLM decides when to hand off and when a readback is
# confirmed, with these tools.
@function_tool
async def transfer_to_licensed_agent(context: RunContext) -> str:
    """Transfer the caller to a licensed team member. Use this when the caller asks to
    speak with a person instead of the automated assistant."""
    logger.info("TRANSFER (tool call from LLM)")
    return "Tell the caller you're connecting them with a licensed team member now."


@function_tool
async def confirm_detail(context: RunContext, field: Field) -> str:
    """Record that the caller confirmed a detail you read back to them.

    Args:
        field: The detail the caller confirmed.
    """
    agent = context.session.current_agent
    assert isinstance(agent, Morgan)
    return agent.mark_confirmed(field, source="LLM tool call")


class Morgan(Agent):
    def __init__(self) -> None:
        if CLASSIFIER == "decision":
            self._classifier = DecisionClassifier(DECISION_MODEL)
        elif CLASSIFIER == "llm":
            self._classifier = LLMClassifier(CLASSIFIER_LLM)
        else:
            self._classifier = None
        logger.info(f"classifier mode: {CLASSIFIER}")

        self._details: dict[str, str] = {}
        self._confirmed: set[str] = set()
        # The readback detail that waits for the caller's yes.
        self._pending: str | None = None

        super().__init__(
            # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
            # See all available models at https://docs.livekit.io/agents/models/llm/
            llm=inference.LLM(model="google/gemma-4-31b-it"),
            instructions=INSTRUCTIONS,
            tools=(
                [transfer_to_licensed_agent, confirm_detail]
                if self._classifier is None
                else None
            ),
        )

    def mark_confirmed(self, field: str, *, source: str) -> str:
        if field not in self._details:
            return f"There's no {field} on record to confirm. Record it first."
        self._confirmed.add(field)
        if self._pending == field:
            self._pending = None
        logger.info(f"CONFIRMED {field}={self._details[field]!r} ({source})")
        return f"The {field} is confirmed."

    @function_tool
    async def record_detail(self, context: RunContext, field: Field, value: str) -> str:
        """Record a detail for the application, or replace one the caller corrected.

        Args:
            field: The detail to record.
            value: The value the caller gave, as digits where the value is a number.
        """
        self._details[field] = value
        self._confirmed.discard(field)
        logger.info(f"RECORDED {field}={value!r}")
        if field in READBACK_FIELDS:
            self._pending = field
            return "Recorded. Read the value back to the caller and wait for a yes."
        return "Recorded."

    @function_tool
    async def submit_application(self, context: RunContext) -> str:
        """Submit the application after you have collected every detail."""
        # The hard gate: code decides whether the application can go in, not the LLM.
        missing = [f for f in REQUIRED_FIELDS if f not in self._details]
        unconfirmed = [f for f in READBACK_FIELDS if f not in self._confirmed]
        if missing or unconfirmed:
            logger.info(f"SUBMIT BLOCKED missing={missing} unconfirmed={unconfirmed}")
            problems = []
            if missing:
                problems.append(f"these details are missing: {', '.join(missing)}")
            if unconfirmed:
                problems.append(
                    "the caller hasn't confirmed the readback of: "
                    + ", ".join(unconfirmed)
                )
            return (
                "The application was NOT submitted, because "
                + " and ".join(problems)
                + ". Don't tell the caller it was submitted. Read back each unconfirmed "
                "detail, wait for the caller to confirm it, then call submit_application "
                "again."
            )
        logger.info(f"SUBMITTED {self._details}")
        return "Submitted."

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        if self._classifier is None:
            return

        # Classify the conversation including the turn that just ended.
        ctx = turn_ctx.copy()
        ctx.insert(new_message)
        try:
            signals = await self._classifier.classify(transcript(ctx))
        except Exception:
            # A failed classification shouldn't break the call.
            logger.exception("classification failed; continuing without signals")
            return

        logger.info(
            f"[{signals.source}] {signals.latency * 1000:.0f} ms "
            f"handoff={signals.handoff:.2f} intent={signals.intent} "
            f"frustration={signals.frustration:.2f} confirmed={signals.confirmed:.2f}"
        )

        if signals.handoff >= HANDOFF_THRESHOLD:
            logger.info("TRANSFER (classifier)")
            self.session.say(TRANSFER_MESSAGE)
            raise StopResponse()

        # The classifier is the authority on whether the caller said yes to a readback.
        if self._pending and signals.confirmed >= CONFIRMED_THRESHOLD:
            self.mark_confirmed(self._pending, source=signals.source)

        guidance = []
        if signals.intent == "coverage_advice":
            guidance.append(
                "The caller wants coverage advice. Don't give it. Offer to have a "
                "licensed team member answer, then continue the application."
            )
        elif signals.intent == "needs_time":
            guidance.append(
                "The caller needs a moment. Tell them to take their time, and don't ask "
                "the next question yet."
            )
        if signals.frustration >= FRUSTRATION_THRESHOLD:
            guidance.append(
                "The caller sounds frustrated. Acknowledge it briefly and offer to "
                "connect them with a person."
            )
        if guidance:
            turn_ctx.add_message(role="system", content=" ".join(guidance))


server = AgentServer()


@server.rtc_session(agent_name="insurance-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using AssemblyAI, Fish Audio, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-3-6-pro", language="en"),
        # Keyterms bias the STT toward distinctive words it would otherwise misspell.
        # List your own names, brands, and jargon in `keyterms`. Detection additionally
        # extracts terms from the live conversation, such as a caller's name, and applies
        # them once the transcript corroborates the spelling.
        # See more at https://docs.livekit.io/agents/models/stt/keyterms/
        stt_context_options=STTContextOptions(
            keyterms=["LiveKit"],
            keyterm_detection={"enabled": True},
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"
        ),
        turn_handling=TurnHandlingOptions(
            # The LiveKit turn detector determines when the user is done speaking and the agent should respond.
            # TurnDetector is an end-of-turn model that listens to the user's audio directly, combining
            # semantic understanding with acoustic cues (intonation, pitch, rhythm) for state-of-the-art accuracy.
            # AgentSession supplies the required VAD automatically.
            # See more at https://docs.livekit.io/agents/build/turns
            turn_detection=inference.TurnDetector(),
            # Adaptive interruptions use the turn detector to tell a real interruption from a
            # backchannel like "mhm" or "right", so the agent keeps talking through the latter.
            # This model is on by default on LiveKit Cloud and in dev mode. To use the model elsewhere,
            # uncomment the `interruption` line. Self-hosted usage past the monthly allowance is billed.
            # See https://docs.livekit.io/deploy/admin/quotas-and-limits/#adaptive-interruption-handling for more information.
            # interruption={"mode": "adaptive"},
            # allow the LLM to generate a response while waiting for the end of turn
            # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
            preemptive_generation={"enabled": True},
        ),
        # Expressive mode injects the TTS provider's markup guide into the LLM prompt, so the model
        # emits inline delivery tags (emotion, pacing, non-verbal sounds) that the TTS renders and
        # the transcript never shows. Requires a TTS model that supports markup, such as the Fish
        # Audio model above.
        expressive=True,
    )

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Morgan(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = anam.AvatarSession(
    #     persona_config=anam.PersonaConfig(
    #         name="...",
    #         avatarId="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/anam
    #     ),
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
