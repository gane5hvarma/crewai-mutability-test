"""Local checks; no model is called.

    pip install -r requirements-dev.txt && pytest -q
"""

import asyncio
import json
import os
import sys
from pathlib import Path

for k in ("HARNESS_INFERENCE_MODEL", "HARNESS_INFERENCE_BASE_URL", "HARNESS_INFERENCE_API_KEY"):
    os.environ.pop(k, None)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mutability_crew import crew as module  # noqa: E402
from mutability_crew.crew import Chunk, Reply, crew  # noqa: E402


def say(text: str) -> str:
    return crew.kickoff(inputs={"message": text}).raw


def test_crewai_json_names_the_crew():
    data = json.loads((ROOT / "crewai.json").read_text())
    module_name, _, attr = data["crew"].partition(":")
    assert (module_name, attr) == ("mutability_crew.crew", "crew")
    assert data["inputs"]["prompt_key"] == "message"
    assert callable(getattr(crew, "akickoff"))


def test_env_lists_the_process_env_and_masks_secrets(monkeypatch):
    monkeypatch.setenv("TEST_FLAG", "hello")
    monkeypatch.setenv("TEST_SECRET", "s3cret-v1")
    report = say("/env TEST_")
    assert "TEST_FLAG=hello" in report
    assert "s3cret-v1" not in report
    assert "TEST_SECRET=<redacted len=9 sha256=" in report
    assert f"mediator pid: {os.getpid()}" in report


def test_env_with_an_unknown_prefix_says_so():
    assert "(no variable starts with 'NO_SUCH_PREFIX_')" in say("printenv NO_SUCH_PREFIX_")


def test_every_reply_ends_with_the_model_line_and_kickoff_count():
    first = say("hello")
    second = say("hello again")
    assert "\n\ncurrently I'm using model: none (HARNESS_INFERENCE_MODEL is unset)\n" in first
    n = int(first.rsplit("kickoff ", 1)[1].split()[0])
    assert second.endswith(f"kickoff {n + 1} in this process")


def test_the_model_line_says_when_inference_is_half_configured(monkeypatch):
    monkeypatch.setenv("HARNESS_INFERENCE_MODEL", "kimi-k3")
    assert "currently I'm using model: none (kimi-k3 is set, but inference is not fully configured)" in say("hi")


def test_tools_without_do_actions_says_so(monkeypatch):
    monkeypatch.delenv("HARNESS_MCP_URL_DO_ACTIONS", raising=False)
    assert "do.actions is not attached" in say("/tools")


def test_tools_lists_what_the_gateway_advertises(monkeypatch):
    seen = {}

    async def fake_list(url):
        seen["url"] = url
        return ["action_invoke", "action_search", "web_search"]

    monkeypatch.setenv("HARNESS_MCP_URL_DO_ACTIONS", "https://ag.example/mcp/endpoint/e1/version/2")
    monkeypatch.setattr(module, "_list_gateway_tools", fake_list)
    report = say("/tools")
    assert seen["url"] == "https://ag.example/mcp/endpoint/e1/version/2"
    assert "do_actions url: https://ag.example/mcp/endpoint/e1/version/2" in report
    assert "tools: 3" in report
    assert "  - web_search" in report


def test_tools_reports_a_listing_failure_without_failing_the_turn(monkeypatch):
    async def boom(url):
        raise ConnectionError("gateway unreachable")

    monkeypatch.setenv("HARNESS_MCP_URL_DO_ACTIONS", "https://ag.example/mcp")
    monkeypatch.setattr(module, "_list_gateway_tools", boom)
    assert "could not list tools: ConnectionError: gateway unreachable" in say("/tools")


class FakeLLM:
    model = "kimi-k3"


class FakeStream:
    """Shaped like CrewStreamingOutput: chunks, then `.result`."""

    def __init__(self):
        self.result = None

    async def __aiter__(self):
        module._served = "kimi-k3-0711"
        yield Chunk("the answer")
        self.result = "the answer"


class FakeCrew:
    def __init__(self, prompts):
        self.prompts = prompts

    async def akickoff(self, inputs):
        self.prompts.append(inputs)
        return FakeStream()


def test_a_model_reply_streams_once_then_the_model_line(monkeypatch):
    prompts = []
    monkeypatch.setattr(module, "llm", FakeLLM())
    monkeypatch.setattr(module, "_chat_crew", lambda: FakeCrew(prompts))

    async def run():
        out = await crew.akickoff({"message": "  what is 2+2?  "})
        chunks = [c.content async for c in out]
        return chunks, out.result

    chunks, result = asyncio.run(run())
    assert prompts == [{"message": "what is 2+2?"}]
    assert chunks[0] == "the answer"
    assert chunks[1].startswith("\n\ncurrently I'm using model: kimi-k3 (reply from: kimi-k3-0711)\nmediator pid: ")
    assert isinstance(result, Reply)
    assert result.raw == "".join(chunks)
