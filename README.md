<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Decision model test

A voice agent and a benchmark for trying decision models in LiveKit Agents. The project starts from the LiveKit Agents Python starter.

The project has these parts:

- `src/agent.py`: Morgan, an insurance onboarding agent. Morgan collects business details for an application, and a classifier checks each caller turn.
- `src/signals.py`: Two classifiers that answer the same four questions about the caller's latest turn:
  - `handoff`: The probability that the caller asks to speak with a person.
  - `intent`: What the caller does: `provide_info`, `ask_question`, `coverage_advice`, `needs_time`, or `off_topic`.
  - `frustration`: A score from 0 (calm) to 2 (frustrated).
  - `confirmed`: The probability that the caller confirms the details the agent just read back, with no correction.
- `bench/cases.json`: Labeled caller turns for the benchmark.
- `bench/run.py`: The benchmark. It sends each case to each classifier and reports accuracy, latency, and tokens.
- `bench/hard.py` and `bench/cases_hard.json`: A harder benchmark with 16 decisions per call, long transcripts, and concurrent load.
- `bench/conversation.py`: A scripted text conversation through the agent, with each tool call and gate result.

## Setup

Install dependencies and write your LiveKit Cloud credentials to `.env.local`:

```console
uv sync
lk cloud auth
lk app env --write --destination .env.local
```

The decision model and the LLM run through LiveKit Inference, so you don't need provider API keys.

## Run the agent

The `CLASSIFIER` variable selects how the agent classifies each caller turn:

| Value | Behavior |
|-------|----------|
| `decision` | Default. A decision model classifies each turn. |
| `llm` | A general-purpose LLM does the same classification with a JSON prompt. |
| `none` | No classifier. The main LLM transfers the caller and confirms readbacks with tools. |

To talk to the agent in your terminal, set `CLASSIFIER` when you start the console:

```console
lk agent console
CLASSIFIER=llm lk agent console
CLASSIFIER=none lk agent console
```

You can also set `CLASSIFIER` in `.env.local`. A value in the shell takes priority.

Speak to the agent to test the classifier. The classifier runs in `on_user_turn_completed`, and typed input in `--text` mode doesn't call that hook. To test with text, use the [scripted conversation](#run-a-scripted-conversation).

To change the models, set these variables the same way:

- `DECISION_MODEL`: The decision model for `decision` mode. The default is `openai/gpt-6-luna-decisions`.
- `CLASSIFIER_LLM`: The LLM for `llm` mode. The default is `google/gemma-4-31b-it`.

Each classified turn writes a log line with the latency and the four signals:

```console
[openai/gpt-6-luna-decisions] 246 ms handoff=0.02 intent=provide_info frustration=0.00 confirmed=0.00
```

The agent acts on the signals in code:

- **Handoff**: When `handoff` is 0.7 or more, the agent says that it transfers the caller and doesn't generate a reply. The transfer is simulated, so it shows only in the logs.
- **Guidance**: Coverage advice requests, requests for time, and frustration add an instruction to the LLM context for that turn.
- **Confirmation**: The LLM records each detail with the `record_detail` tool. After it records the phone number or the ZIP code, the agent waits for a readback confirmation. When `confirmed` is 0.8 or more on a later turn, code marks that detail as confirmed. A correction records the new value, and the agent waits for a new confirmation.

The `submit_application` tool is a hard gate. It refuses to submit until every detail is recorded and both readbacks are confirmed, no matter what the LLM decides. The log shows `CONFIRMED`, `SUBMIT BLOCKED`, and `SUBMITTED` lines as this happens.

The classifier runs before the agent replies, so its latency adds to each reply.

## Run a scripted conversation

To run a fixed conversation through the agent and print each turn, tool call, and log line, run:

```console
uv run python bench/conversation.py
CLASSIFIER=none uv run python bench/conversation.py
```

The conversation includes a correction to each readback, so you can see whether the agent confirms the wrong value. Edit `TURNS` in the script to try other conversations.

## Run the benchmark

To compare the default decision model and LLM over all cases once, run:

```console
uv run python bench/run.py
```

The benchmark accepts these options:

| Option | Description |
|--------|-------------|
| `--runs N` | Number of passes over the cases. The default is 1. |
| `--verbose` | Print the result and latency for each case, with the reason for each miss. |
| `--decision-models` | Comma-separated decision model IDs. The default is `openai/gpt-6-luna-decisions`. |
| `--llms` | Comma-separated LLM model IDs. The default is `google/gemma-4-31b-it`. |
| `--cases PATH` | A different case file. The default is `bench/cases.json`. |

For example, to run five passes with output for each case, or to compare more than one LLM:

```console
uv run python bench/run.py --runs 5 --verbose
uv run python bench/run.py --llms google/gemma-4-31b-it,openai/gpt-4.1-mini
```

Each run writes the full results to `bench/results/<timestamp>.json`.

The summary table has these columns:

- `handoff`: The share of calls where the result (a probability of 0.7 or more) matches the label.
- `intent`: The share of exact matches. Cases with a `null` intent label aren't scored.
- `confirm`: The share of readback cases where the result (a probability of 0.8 or more) matches the label. Cases with a `null` confirmed label aren't scored.
- `frust MAE`: The mean absolute error of the frustration score against the 0 to 2 label. Lower is better.
- `p50 ms` and `p95 ms`: The median and 95th percentile latency for each call.
- `in tok` and `out tok`: The mean input and output tokens for each call.

Before it measures, the benchmark makes one call to each classifier to open the connection. If that call fails, the summary shows the error for that classifier. For example, a model that your project can't access returns a 403 error.

### Add cases

Each case in `bench/cases.json` is a transcript that ends on a caller turn, with these labels:

```json
{
  "id": "grab-docs",
  "transcript": "assistant: What's the approximate annual revenue?\nuser: Hang on, let me grab my tax return.",
  "handoff": false,
  "intent": "needs_time",
  "frustration": 0,
  "confirmed": null
}
```

Set `intent` to `null` for a case where more than one intent is correct. Set `confirmed` to `true` or `false` for a case where the agent's last turn reads details back, and to `null` for other cases.

## Run the harder benchmark

`bench/hard.py` tests conditions closer to production. Each call answers 16 decisions at once: the four agent signals plus guardrails (abuse, threats, prompt injection, sensitive numbers), routing checks, and a `next_action` choice. Most cases follow a long conversation with distractors, such as an earlier request for a person that the caller withdrew. Some cases also have a short version without the conversation.

The script has three modes:

```console
uv run python bench/hard.py accuracy --runs 3 --verbose
uv run python bench/hard.py batch --runs 4 --sizes 1,2,4,8,16
uv run python bench/hard.py load --concurrency 1,8,24 --rounds 3
```

- `accuracy`: Scores every decision against the labels in `bench/cases_hard.json`. It also reports the Brier score of the probabilities, how often answers change between runs (`flips`), and decisions the model declined (`decl`). A declined decision counts as a miss.
- `batch`: Measures latency and output tokens as the number of decisions in one call grows.
- `load`: Sends many 16-decision calls at the same time and measures latency.

Every mode accepts `--decision-models` and `--llms`. Results go to `bench/results/`.

In `bench/cases_hard.json`, `true` lists the probability decisions that are true for a case. All others are false. Add a decision to `skip` to leave it unscored for that case.

## Using coding agents

This project works with coding agents like [Claude Code](https://claude.com/product/claude-code), [Cursor](https://www.cursor.com/), and [Codex](https://openai.com/codex/).

LiveKit offers both a CLI and an [MCP server](https://docs.livekit.io/reference/developer-tools/docs-mcp/) for browsing and searching its documentation. Search returns short excerpts, so fetch the full page to read the details:

```console
lk docs search "testing my agent"
lk docs get-page /testing/unit-tests
```

The project also includes an [`AGENTS.md`](AGENTS.md) file, so your coding agent follows LiveKit's best practices for workflows, handoffs, and testing, and tries its changes with the [agent debugger](https://docs.livekit.io/testing/debugger/).

For more guidance at each stage of the work, from building and debugging to testing and deploying, install LiveKit's [agent skills](https://docs.livekit.io/intro/coding-agents/#agent-skills). The CLI installs them for the coding agents on your machine and adds the Docs MCP server to their config. `lk agent init` offers to do this when it creates your project, or you can run it yourself:

```console
lk skills install
```

Run `lk skills update` to get the latest skills, and commit `skills-lock.json` along with them. See the [coding agents guide](https://docs.livekit.io/intro/coding-agents/) for more details.

## Dev setup

Install the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/), version 2.18.8 or later:

- **macOS:** `brew install livekit-cli`
- **Linux:** `curl -sSL https://get.livekit.io/cli | bash`
- **Windows:** `winget install LiveKit.LiveKitCLI`

Check your version with `lk --version`. To update an existing install, see [Update the CLI](https://docs.livekit.io/reference/developer-tools/livekit-cli/#updates).

Then create a project from this template. The CLI clones the template and configures your environment:

```console
lk cloud auth
lk agent init my-agent --template agent-starter-python
```

<details>
<summary>Set up the project manually</summary>

Clone the repository and install dependencies into a virtual environment with [uv](https://docs.astral.sh/uv/):

```console
git clone https://github.com/livekit-examples/agent-starter-python.git
cd agent-starter-python
uv sync
```

Sign up for [LiveKit Cloud](https://cloud.livekit.io/), then copy `.env.example` to `.env.local` and fill it in. To have the CLI write your project's URL and API keys into the file instead, run:

```console
lk cloud auth
lk app env --write --destination .env.local
```

</details>

## Run the agent with the LiveKit CLI

The `lk agent console`, `lk agent dev`, and `lk agent debugger` commands run your agent on your own machine. To talk to it in your terminal:

```console
lk agent console
```

To connect it to LiveKit Cloud so a frontend, a phone call, or the [Agent Console](https://docs.livekit.io/testing/agent-console/) can reach it:

```console
lk agent dev
```

To let a coding agent or a script test it one text turn at a time, use the [agent debugger](https://docs.livekit.io/testing/debugger/). Each turn prints the agent's reply along with the tool calls and handoffs behind it:

```console
lk agent debugger start
lk agent debugger say "Hi, what can you do?"
lk agent debugger stop
```

In production, run the agent directly:

```console
uv run src/agent.py start
```

## Frontends and telephony

Pair the agent with a prebuilt frontend starter, or add telephony:

| Platform | Link | Description |
|----------|----------|-------------|
| **Web** | [`livekit-examples/agent-starter-react`](https://github.com/livekit-examples/agent-starter-react) | Web voice AI assistant with React & Next.js |
| **iOS/macOS** | [`livekit-examples/agent-starter-swift`](https://github.com/livekit-examples/agent-starter-swift) | Native iOS, macOS, and visionOS voice AI assistant |
| **Flutter** | [`livekit-examples/agent-starter-flutter`](https://github.com/livekit-examples/agent-starter-flutter) | Cross-platform voice AI assistant app |
| **React Native** | [`livekit-examples/voice-assistant-react-native`](https://github.com/livekit-examples/voice-assistant-react-native) | Native mobile app with React Native & Expo |
| **Android** | [`livekit-examples/agent-starter-android`](https://github.com/livekit-examples/agent-starter-android) | Native Android app with Kotlin & Jetpack Compose |
| **Web Embed** | [`livekit-examples/agent-starter-embed`](https://github.com/livekit-examples/agent-starter-embed) | Voice AI widget for any website |
| **Telephony** | [Documentation](https://docs.livekit.io/telephony/) | Add inbound or outbound calling to your agent |

For more options, see the [frontend guide](https://docs.livekit.io/frontends/).

## Testing and debugging

Simulations run full multi-turn conversations between a simulated user and your agent on LiveKit Cloud, then judge each transcript. The scenarios live in [`scenarios.yaml`](scenarios.yaml). Run them locally with the CLI:

```console
lk agent simulate text --scenarios scenarios.yaml
```

The `Simulations` workflow in [`.github/workflows/simulations.yml`](.github/workflows/simulations.yml) runs the same file on every merge to `main`, and on demand from the Actions tab. It doesn't run on every pull request push because each run uses real inference. See the [simulations guide](https://docs.livekit.io/testing/simulations/) for how to write scenarios and read results.

To check a change turn by turn without a live session, use the [agent debugger](https://docs.livekit.io/testing/debugger/) shown in [Run the agent](#run-the-agent).

To debug a running agent, open it in the [Agent Console](https://docs.livekit.io/testing/agent-console/). It shows events, tool calls, and model timing as you talk to the agent. To stream logs from a deployed agent, run `lk agent logs`.

## Using this template for your own project

After you create your own project from this template:

- **Commit `uv.lock`.** The template doesn't track it, but your project should, for reproducible builds. If you deploy to LiveKit Cloud, commit `livekit.toml` too.
- **Add repository secrets.** Add `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` as [repository secrets](https://docs.github.com/en/actions/how-tos/writing-workflows/choosing-what-your-workflow-does/using-secrets-in-github-actions) so the simulations can run in CI.

## Deploying to production

To deploy the agent to LiveKit Cloud or another environment with the included `Dockerfile`, see the [deployment guide](https://docs.livekit.io/deploy/agents/).

## Self-hosted LiveKit

You can self-host LiveKit instead of using LiveKit Cloud. See the [self-hosting guide](https://docs.livekit.io/transport/self-hosting/local/). If you self-host, use [model plugins](https://docs.livekit.io/agents/models/#plugins) instead of LiveKit Inference, and remove the [LiveKit Cloud noise cancellation](https://docs.livekit.io/transport/media/noise-cancellation/) plugin.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
