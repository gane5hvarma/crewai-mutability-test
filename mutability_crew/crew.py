"""A CrewAI crew for checking live session updates on Managed Agents.

Every turn is one kickoff of `crew` below, with the turn's text as the
`message` input (crewai.json). The platform's mediator imports this module
once, at start, so the environment it reads is the one the mediator process
started with. A live env, secret, model or tools update restarts that process,
which is what the replies make visible.

Send `/env` (or `printenv`), optionally with a name prefix (`/env TEST_`), for
the mediator's environment, without the model. Values of secret-looking names
are masked.

Send `/tools` for the tools the `do.actions` Action Gateway advertises, listed
with an MCP client on HARNESS_MCP_URL_DO_ACTIONS, without the model.

With HARNESS_INFERENCE_* set, other messages go to a one-agent Crew. Without
them the crew answers deterministically.

Every reply ends with "currently I'm using model: <name>", and for a model
reply "(reply from: <name>)", the model the endpoint says answered. Every
reply also names the mediator pid and how many kickoffs this process has run:
both change when the process restarts.
"""

from __future__ import annotations

import hashlib
import itertools
import os
from dataclasses import dataclass
from typing import Any

MODEL_KEYS = ("HARNESS_INFERENCE_MODEL", "HARNESS_INFERENCE_BASE_URL", "HARNESS_INFERENCE_API_KEY")
DO_ACTIONS_URL_ENV = "HARNESS_MCP_URL_DO_ACTIONS"

# A name containing any of these is masked by /env.
SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")

_kickoffs = itertools.count(1)
# The model the endpoint named in its last reply this turn.
_served: str | None = None


def _note_served(reply: Any) -> None:
    global _served
    model = getattr(reply, "model", None)
    if isinstance(model, str) and model:
        _served = model


class _RecordedStream:
    """An OpenAI stream that notes the served model as chunks go by."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __iter__(self):
        for chunk in self._inner:
            _note_served(chunk)
            yield chunk

    async def __aiter__(self):
        async for chunk in self._inner:
            _note_served(chunk)
            yield chunk


def _record_served_model(llm: Any) -> None:
    """Wrap the LLM's OpenAI clients so each reply's model is noted.

    Uses crewai's private client getters (crewai 1.15.11). If they change,
    only the "reply from" part of the model line is lost.
    """
    try:
        sync, aio = llm._get_sync_client(), llm._get_async_client()
    except Exception:  # noqa: BLE001
        return
    create, acreate = sync.chat.completions.create, aio.chat.completions.create

    def recorded(*args: Any, **kwargs: Any) -> Any:
        reply = create(*args, **kwargs)
        _note_served(reply)
        return _RecordedStream(reply) if kwargs.get("stream") else reply

    async def arecorded(*args: Any, **kwargs: Any) -> Any:
        reply = await acreate(*args, **kwargs)
        _note_served(reply)
        return _RecordedStream(reply) if kwargs.get("stream") else reply

    sync.chat.completions.create = recorded
    aio.chat.completions.create = arecorded


def _llm() -> Any:
    """The crew's LLM, or None when inference is not configured."""
    if not all(os.environ.get(k, "").strip() for k in MODEL_KEYS):
        return None
    from crewai import LLM

    # provider="openai" keeps any model name on the OpenAI-compatible wire.
    llm = LLM(
        model=os.environ["HARNESS_INFERENCE_MODEL"],
        base_url=os.environ["HARNESS_INFERENCE_BASE_URL"],
        api_key=os.environ["HARNESS_INFERENCE_API_KEY"],
        provider="openai",
        temperature=0,
    )
    _record_served_model(llm)
    return llm


llm = _llm()


def _chat_crew() -> Any:
    from crewai import Agent, Crew, Process, Task

    assistant = Agent(
        role="Assistant",
        goal="Answer the user's message briefly and accurately",
        backstory="You are a concise, helpful assistant.",
        llm=llm,
        allow_delegation=False,
    )
    answer = Task(
        description="Reply to this message in at most three sentences: {message}",
        expected_output="A short, direct reply.",
        agent=assistant,
    )
    return Crew(agents=[assistant], tasks=[answer], process=Process.sequential, stream=True)


def _process_line(kickoff: int) -> str:
    return f"mediator pid: {os.getpid()}, kickoff {kickoff} in this process"


def _env_report(prefix: str = "") -> str:
    """The mediator's environment, as this process started with it.

    Values of secret-looking names are replaced by their length and a short
    sha256: enough to see a rotation, without the value. Use test values: an
    8-hex digest of a short value can be guessed.
    """
    lines = []
    for name in sorted(os.environ):
        if prefix and not name.startswith(prefix):
            continue
        value = os.environ[name]
        if any(hint in name.upper() for hint in SECRET_HINTS):
            digest = hashlib.sha256(value.encode()).hexdigest()[:8]
            value = f"<redacted len={len(value)} sha256={digest}>"
        lines.append(f"{name}={value}")
    if not lines:
        lines.append(f"(no variable starts with {prefix!r})" if prefix else "(empty environment)")
    return "\n".join(lines)


async def _list_gateway_tools(url: str) -> list[str]:
    """The tool names the Action Gateway advertises in tools/list."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
    return sorted(tool.name for tool in listed.tools)


async def _tools_report() -> str:
    url = os.environ.get(DO_ACTIONS_URL_ENV, "").strip()
    if not url:
        return f"do.actions is not attached ({DO_ACTIONS_URL_ENV} is unset)"
    lines = [f"do_actions url: {url}"]
    try:
        names = await _list_gateway_tools(url)
    except Exception as e:  # noqa: BLE001  report it; a failed listing must not fail the turn
        lines.append(f"could not list tools: {type(e).__name__}: {e}")
        return "\n".join(lines)
    lines.append(f"tools: {len(names)}")
    lines += [f"  - {name}" for name in names]
    return "\n".join(lines)


def _model_line() -> str:
    configured = os.environ.get("HARNESS_INFERENCE_MODEL", "").strip()
    if llm is not None:
        line = f"currently I'm using model: {getattr(llm, 'model', None) or configured}"
        if _served:
            line += f" (reply from: {_served})"
    elif configured:
        line = f"currently I'm using model: none ({configured} is set, but inference is not fully configured)"
    else:
        line = "currently I'm using model: none (HARNESS_INFERENCE_MODEL is unset)"
    return f"\n\n{line}"


@dataclass
class Reply:
    """A finished reply; the mediator reads `.raw`."""

    raw: str

    def __str__(self) -> str:
        return self.raw


@dataclass
class Chunk:
    """A streamed text chunk, shaped like crewai's StreamChunk."""

    content: str
    chunk_type: str = "TEXT"
    agent_role: str | None = None
    task_name: str | None = None


class _WithFooter:
    """The Crew's stream, then one chunk with the footer. `.result` is set
    once the stream ends, as on crewai's CrewStreamingOutput."""

    def __init__(self, inner: Any, kickoff: int) -> None:
        self._inner = inner
        self._kickoff = kickoff
        self.result: Reply | None = None

    async def __aiter__(self):
        async for chunk in self._inner:
            yield chunk
        footer = _model_line() + "\n" + _process_line(self._kickoff)
        yield Chunk(footer)
        self.result = Reply(str(getattr(self._inner, "result", "") or "") + footer)


class MutabilityCrew:
    """The object crewai.json names. The mediator needs only akickoff."""

    async def akickoff(self, inputs: dict[str, Any] | None = None) -> Any:
        global _served
        _served = None
        kickoff = next(_kickoffs)
        inputs = dict(inputs or {})
        message = str(inputs.get("message", "")).strip()
        words = message.split(maxsplit=1)
        command = words[0].lower() if words else ""
        rest = words[1].strip() if len(words) > 1 else ""

        if command in ("/env", "printenv"):
            text = _env_report(rest)
        elif command == "/tools":
            text = await _tools_report()
        elif llm is not None:
            inputs["message"] = message
            return _WithFooter(await _chat_crew().akickoff(inputs=inputs), kickoff)
        else:
            text = f"You said: {message!r}. Set HARNESS_INFERENCE_* for a model reply."
        return Reply(text + _model_line() + "\n" + _process_line(kickoff))

    def kickoff(self, inputs: dict[str, Any] | None = None) -> Reply:
        """Sync kickoff for local runs: the whole reply at once."""
        import asyncio

        async def run() -> Reply:
            out = await self.akickoff(inputs)
            if isinstance(out, Reply):
                return out
            async for _ in out:
                pass
            return out.result

        return asyncio.run(run())


crew = MutabilityCrew()
